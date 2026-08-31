#!/usr/bin/env python3
"""Run flow-switch and residual-patching interventions for one saved pair."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from openpi.models import model as model_types
from openpi.policies import policy_config
from openpi.training import config as training_config

from action_chunking.metrics import LIBERO_ACTION_GROUPS, summarize_transfer, target_direction_affinity
from action_chunking.pairs import file_digest, load_instruction_pair
from action_chunking.sampling import PreparedCondition, SamplingTrace, prepare_condition, sample_actions
from action_chunking.tracing import PatchSpec, ResidualTrace, ResidualTracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--steps", default="all", help="Residual-patch steps, e.g. all or 0,4,9")
    parser.add_argument("--layers", default="all", help="Residual-patch layers, e.g. all or 0,8,17")
    parser.add_argument("--skip-residual-patches", action="store_true")
    parser.add_argument("--position-mode", choices=("all", "single"), default="all")
    parser.add_argument("--dimension-mode", choices=("none", "groups", "scalar", "both"), default="none")
    parser.add_argument("--identity-sites", choices=("none", "anchors", "all"), default="anchors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entry = _manifest_entry(manifest, args.pair_id)
    fixture_path = args.manifest.parent / entry["fixture"]
    if file_digest(fixture_path) != entry["fixture_sha256"]:
        raise ValueError("pair fixture hash does not match its manifest")
    pair = load_instruction_pair(fixture_path)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.noise_seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    model = policy._model
    model.eval()
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive")

    base_condition, base_transformed = _condition(policy, pair.raw_observation("base"), args.device)
    donor_condition, donor_transformed = _condition(policy, pair.raw_observation("donor"), args.device)
    rng = np.random.default_rng(args.noise_seed)
    noise_array = rng.standard_normal(
        (model.config.action_horizon, model.config.action_dim),
        dtype=np.float32,
    )
    noise = torch.from_numpy(noise_array).to(args.device)[None, ...]
    np.save(args.output / "noise.npy", noise_array)

    layers = model.paligemma_with_expert.gemma_expert.model.layers
    base_actions_t, base_sampling, base_residual = _clean_run(model, noise, base_condition, layers, args.num_steps)
    donor_actions_t, donor_sampling, donor_residual = _clean_run(model, noise, donor_condition, layers, args.num_steps)
    base_actions = _physical_actions(policy, base_transformed, base_actions_t)
    donor_actions = _physical_actions(policy, donor_transformed, donor_actions_t)
    endpoint_contrast = float(np.linalg.norm(donor_actions - base_actions))
    if endpoint_contrast <= 1e-8:
        raise ValueError("clean pair endpoints have a degenerate action contrast")

    np.savez_compressed(
        args.output / "clean_trace.npz",
        base_actions=base_actions,
        donor_actions=donor_actions,
        base_x_t=_stack_trace(base_sampling.x_t),
        donor_x_t=_stack_trace(donor_sampling.x_t),
        base_v_t=_stack_trace(base_sampling.v_t),
        donor_v_t=_stack_trace(donor_sampling.v_t),
        base_clean_action_estimates=_stack_trace(base_sampling.clean_action_estimates),
        donor_clean_action_estimates=_stack_trace(donor_sampling.clean_action_estimates),
        flow_times=np.asarray(base_sampling.times, dtype=np.float32),
    )

    records: list[dict[str, Any]] = []
    endpoint_context = _endpoint_context(entry)
    records.extend(
        [
            _record("clean", "base", base_actions, base_actions, donor_actions, endpoint_context),
            _record("clean", "donor", donor_actions, base_actions, donor_actions, endpoint_context),
        ]
    )
    records.extend(
        _formation_records(
            policy,
            base_transformed,
            donor_transformed,
            base_sampling,
            donor_sampling,
            base_actions,
            donor_actions,
            endpoint_context,
        )
    )

    directions = (
        (
            "base_to_donor",
            base_condition,
            donor_condition,
            base_actions,
            donor_actions,
            base_sampling,
            donor_sampling,
            base_residual,
            donor_residual,
            base_transformed,
            endpoint_context,
        ),
        (
            "donor_to_base",
            donor_condition,
            base_condition,
            donor_actions,
            base_actions,
            donor_sampling,
            base_sampling,
            donor_residual,
            base_residual,
            donor_transformed,
            _reverse_context(endpoint_context),
        ),
    )
    for (
        direction,
        source_condition,
        destination_condition,
        source_actions,
        destination_actions,
        source_sampling,
        destination_sampling,
        source_residual,
        destination_residual,
        source_transformed,
        context,
    ) in directions:
        records.extend(
            _flow_switch_records(
                model,
                policy,
                noise,
                source_condition,
                destination_condition,
                source_transformed,
                source_actions,
                destination_actions,
                context,
                direction,
                args.num_steps,
            )
        )
        if not args.skip_residual_patches:
            patch_steps = _indices(args.steps, args.num_steps)
            patch_layers = _indices(args.layers, len(layers))
            position_sets = _position_sets(args.position_mode, model.config.action_horizon)
            records.extend(
                _patch_records(
                    model,
                    policy,
                    noise,
                    source_condition,
                    source_transformed,
                    source_actions,
                    destination_actions,
                    source_residual,
                    destination_residual,
                    layers,
                    patch_steps,
                    patch_layers,
                    context,
                    direction,
                    args.num_steps,
                    family="residual_patch" if args.position_mode == "all" else "residual_patch_position",
                    position_sets=position_sets,
                )
            )
        if args.dimension_mode != "none":
            records.extend(
                _dimension_patch_records(
                    model,
                    policy,
                    noise,
                    source_condition,
                    source_transformed,
                    source_actions,
                    destination_actions,
                    source_sampling,
                    destination_sampling,
                    context,
                    direction,
                    args.num_steps,
                    _dimension_specs(args.dimension_mode),
                    family="action_dimension_patch",
                )
            )

    identity_sites = _identity_sites(args.identity_sites, args.num_steps, len(layers))
    if identity_sites:
        records.extend(
            _patch_records(
                model,
                policy,
                noise,
                base_condition,
                base_transformed,
                base_actions,
                base_actions,
                base_residual,
                base_residual,
                layers,
                sorted({site[0] for site in identity_sites}),
                sorted({site[1] for site in identity_sites}),
                endpoint_context,
                "base_to_base",
                args.num_steps,
                family="identity_patch",
                allowed_sites=identity_sites,
                position_sets=(None,),
            )
        )
    if args.dimension_mode != "none":
        records.extend(
            _dimension_patch_records(
                model,
                policy,
                noise,
                base_condition,
                base_transformed,
                base_actions,
                base_actions,
                base_sampling,
                base_sampling,
                endpoint_context,
                "base_to_base",
                args.num_steps,
                _dimension_specs(args.dimension_mode),
                family="action_dimension_identity",
            )
        )

    records_path = args.output / "records.jsonl"
    with records_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")

    metadata = {
        "schema_version": 1,
        "pair_id": args.pair_id,
        "pair_fixture": str(fixture_path),
        "pair_fixture_sha256": entry["fixture_sha256"],
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "noise_seed": args.noise_seed,
        "num_steps": args.num_steps,
        "residual_patch_steps": args.steps,
        "residual_patch_layers": args.layers,
        "skip_residual_patches": args.skip_residual_patches,
        "position_mode": args.position_mode,
        "dimension_mode": args.dimension_mode,
        "identity_sites": args.identity_sites,
        "action_horizon": model.config.action_horizon,
        "model_action_dim": model.config.action_dim,
        "physical_action_dim": 7,
        "layers": len(layers),
        "endpoint_l2_contrast": endpoint_contrast,
        "endpoint_group_l2_contrasts": {
            name: float(np.linalg.norm(donor_actions[:, indices] - base_actions[:, indices]))
            for name, indices in LIBERO_ACTION_GROUPS.items()
        },
        "record_count": len(records),
        "openpi_commit": _git_revision(Path(__file__).resolve().parents[1] / "third_party" / "openpi"),
        "controls": _control_summary(records, donor_actions),
    }
    (args.output / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(metadata), indent=2, sort_keys=True))
    return 0


def _condition(policy: Any, raw: dict[str, Any], device: str) -> tuple[PreparedCondition, dict[str, Any]]:
    transformed = policy._input_transform(jax.tree.map(lambda value: value, raw))
    transformed_torch = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to(device)[None, ...],
        transformed,
    )
    observation = model_types.Observation.from_dict(transformed_torch)
    return prepare_condition(policy._model, observation), transformed_torch


def _clean_run(
    model: Any,
    noise: torch.Tensor,
    condition: PreparedCondition,
    layers: Any,
    num_steps: int,
) -> tuple[torch.Tensor, SamplingTrace, ResidualTrace]:
    with ResidualTracer(layers, action_horizon=model.config.action_horizon) as tracer:
        actions, sampling_trace = sample_actions(
            model,
            noise,
            lambda _step: condition,
            num_steps=num_steps,
            tracer=tracer,
        )
    return actions, sampling_trace.cpu(), tracer.trace.cpu()


def _physical_actions(policy: Any, transformed: dict[str, Any], actions: torch.Tensor) -> np.ndarray:
    outputs = {
        "state": np.asarray(transformed["state"][0].detach().cpu()),
        "actions": np.asarray(actions[0].detach().cpu()),
    }
    return np.asarray(policy._output_transform(outputs)["actions"])


def _flow_switch_records(
    model: Any,
    policy: Any,
    noise: torch.Tensor,
    source: PreparedCondition,
    destination: PreparedCondition,
    transformed: dict[str, Any],
    source_actions: np.ndarray,
    destination_actions: np.ndarray,
    context: dict[str, Any],
    direction: str,
    num_steps: int,
) -> list[dict[str, Any]]:
    records = []
    for switch_after in range(num_steps + 1):
        actions_t, _ = sample_actions(
            model,
            noise,
            lambda step, boundary=switch_after: source if step < boundary else destination,
            num_steps=num_steps,
        )
        actions = _physical_actions(policy, transformed, actions_t)
        record = _record("flow_switch", direction, actions, source_actions, destination_actions, context)
        record["switch_after_steps"] = switch_after
        record["retention"] = {
            key: 1.0 - value if value is not None else None for key, value in record["metrics"]["ncte"].items()
        }
        record["destination_endpoint_max_abs_error"] = float(np.max(np.abs(actions - destination_actions)))
        records.append(record)
    return records


def _patch_records(
    model: Any,
    policy: Any,
    noise: torch.Tensor,
    condition: PreparedCondition,
    transformed: dict[str, Any],
    source_actions: np.ndarray,
    destination_actions: np.ndarray,
    source_residual: ResidualTrace,
    destination_residual: ResidualTrace,
    layers: Any,
    steps: list[int],
    layer_indices: list[int],
    context: dict[str, Any],
    direction: str,
    num_steps: int,
    *,
    family: str,
    position_sets: tuple[tuple[int, ...] | None, ...],
    allowed_sites: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for step in steps:
        for layer in layer_indices:
            if allowed_sites is not None and (step, layer) not in allowed_sites:
                continue
            for positions in position_sets:
                patch = PatchSpec(step=step, layer=layer, positions=positions)
                with ResidualTracer(
                    layers,
                    action_horizon=model.config.action_horizon,
                    patch=patch,
                    donor=destination_residual.values,
                    capture=False,
                ) as tracer:
                    actions_t, _ = sample_actions(
                        model,
                        noise,
                        lambda _step: condition,
                        num_steps=num_steps,
                        tracer=tracer,
                    )
                actions = _physical_actions(policy, transformed, actions_t)
                record = _record(family, direction, actions, source_actions, destination_actions, context)
                record.update(
                    {
                        "flow_step": step,
                        "layer": layer,
                        "action_positions": "all" if positions is None else list(positions),
                        "source_endpoint_max_abs_error": float(np.max(np.abs(actions - source_actions))),
                        "activation_intervention": _activation_intervention_metrics(
                            source_residual.values[(step, layer)],
                            destination_residual.values[(step, layer)],
                            positions,
                        ),
                    }
                )
                records.append(record)
    return records


def _dimension_patch_records(
    model: Any,
    policy: Any,
    noise: torch.Tensor,
    condition: PreparedCondition,
    transformed: dict[str, Any],
    source_actions: np.ndarray,
    destination_actions: np.ndarray,
    source_trace: SamplingTrace,
    destination_trace: SamplingTrace,
    context: dict[str, Any],
    direction: str,
    num_steps: int,
    dimension_specs: list[tuple[str, tuple[int, ...]]],
    *,
    family: str,
) -> list[dict[str, Any]]:
    records = []
    for step in range(num_steps):
        for group, dimensions in dimension_specs:
            for tensor_name in ("x_t", "v_t"):
                source_tensor = getattr(source_trace, tensor_name)[step]
                destination_tensor = getattr(destination_trace, tensor_name)[step]

                def interchange(
                    current_step: int,
                    value: torch.Tensor,
                    target_step: int = step,
                    donor_tensor: torch.Tensor = destination_tensor,
                    selected_dimensions: tuple[int, ...] = dimensions,
                ) -> torch.Tensor:
                    if current_step != target_step:
                        return value
                    patched = value.clone()
                    donor = donor_tensor.to(device=value.device, dtype=value.dtype)
                    patched[..., selected_dimensions] = donor[..., selected_dimensions]
                    return patched

                sampler_kwargs = (
                    {"state_intervention": interchange}
                    if tensor_name == "x_t"
                    else {"velocity_intervention": interchange}
                )
                actions_t, _ = sample_actions(
                    model,
                    noise,
                    lambda _step: condition,
                    num_steps=num_steps,
                    **sampler_kwargs,
                )
                actions = _physical_actions(policy, transformed, actions_t)
                record = _record(family, direction, actions, source_actions, destination_actions, context)
                record.update(
                    {
                        "flow_step": step,
                        "patched_tensor": tensor_name,
                        "action_dimension_group": group,
                        "action_dimensions": list(dimensions),
                        "source_endpoint_max_abs_error": float(np.max(np.abs(actions - source_actions))),
                        "tensor_intervention": _tensor_intervention_metrics(
                            source_tensor,
                            destination_tensor,
                            dimensions,
                        ),
                    }
                )
                records.append(record)
    return records


def _formation_records(
    policy: Any,
    base_transformed: dict[str, Any],
    donor_transformed: dict[str, Any],
    base_trace: SamplingTrace,
    donor_trace: SamplingTrace,
    base_final: np.ndarray,
    donor_final: np.ndarray,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for step, (base_t, donor_t) in enumerate(
        zip(base_trace.clean_action_estimates, donor_trace.clean_action_estimates, strict=True)
    ):
        base_estimate = _physical_actions(policy, base_transformed, base_t)
        donor_estimate = _physical_actions(policy, donor_transformed, donor_t)
        for side, estimate, final in (
            ("base", base_estimate, base_final),
            ("donor", donor_estimate, donor_final),
        ):
            records.append(
                {
                    "family": "formation",
                    "side": side,
                    "flow_step": step,
                    "flow_time": base_trace.times[step],
                    "actions": estimate.tolist(),
                    "final_relative_l2_error": float(
                        np.linalg.norm(estimate - final) / max(np.linalg.norm(final), 1e-12)
                    ),
                    "final_max_abs_error": float(np.max(np.abs(estimate - final))),
                    "target_direction_affinity": target_direction_affinity(
                        estimate,
                        context["end_effector_position"],
                        context["source_target_position"],
                        context["destination_target_position"],
                    ),
                }
            )
    return records


def _record(
    family: str,
    direction: str,
    actions: np.ndarray,
    source_actions: np.ndarray,
    destination_actions: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "direction": direction,
        "actions": actions.tolist(),
        "metrics": summarize_transfer(actions, source_actions, destination_actions),
        "target_direction_affinity": target_direction_affinity(
            actions,
            context["end_effector_position"],
            context["source_target_position"],
            context["destination_target_position"],
        ),
    }


def _endpoint_context(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "end_effector_position": entry["end_effector_position"],
        "source_target_position": entry["base_target_position"],
        "destination_target_position": entry["donor_target_position"],
    }


def _reverse_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "end_effector_position": context["end_effector_position"],
        "source_target_position": context["destination_target_position"],
        "destination_target_position": context["source_target_position"],
    }


def _indices(value: str, upper: int) -> list[int]:
    if value == "all":
        return list(range(upper))
    result = sorted({int(item) for item in value.split(",")})
    if not result or result[0] < 0 or result[-1] >= upper:
        raise ValueError(f"indices must lie within [0, {upper})")
    return result


def _identity_sites(mode: str, steps: int, layers: int) -> set[tuple[int, int]]:
    if mode == "none":
        return set()
    if mode == "all":
        return {(step, layer) for step in range(steps) for layer in range(layers)}
    anchor_steps = {0, steps // 2, steps - 1}
    anchor_layers = {0, layers // 2, layers - 1}
    return {(step, layer) for step in anchor_steps for layer in anchor_layers}


def _position_sets(mode: str, action_horizon: int) -> tuple[tuple[int, ...] | None, ...]:
    if mode == "all":
        return (None,)
    return tuple((position,) for position in range(action_horizon))


def _dimension_specs(mode: str) -> list[tuple[str, tuple[int, ...]]]:
    specs = []
    if mode in {"groups", "both"}:
        specs.extend((name, indices) for name, indices in LIBERO_ACTION_GROUPS.items() if name != "all")
    if mode in {"scalar", "both"}:
        specs.extend((f"dimension_{index}", (index,)) for index in range(7))
    return specs


def _stack_trace(values: list[torch.Tensor]) -> np.ndarray:
    return np.stack([np.asarray(value[0]) for value in values])


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


def _control_summary(records: list[dict[str, Any]], donor_actions: np.ndarray) -> dict[str, Any]:
    identity = [record["source_endpoint_max_abs_error"] for record in records if record["family"] == "identity_patch"]
    dimension_identity = [
        record["source_endpoint_max_abs_error"]
        for record in records
        if record["family"] == "action_dimension_identity"
    ]
    donor_ceiling = [
        record
        for record in records
        if record["family"] == "flow_switch"
        and record["direction"] == "base_to_donor"
        and record["switch_after_steps"] == 0
    ]
    return {
        "identity_patch_sites": len(identity),
        "identity_patch_max_abs_error": max(identity, default=None),
        "dimension_identity_sites": len(dimension_identity),
        "dimension_identity_max_abs_error": max(dimension_identity, default=None),
        "full_donor_switch_max_abs_error": (
            float(np.max(np.abs(np.asarray(donor_ceiling[0]["actions"]) - donor_actions))) if donor_ceiling else None
        ),
    }


def _activation_intervention_metrics(
    source: torch.Tensor,
    destination: torch.Tensor,
    positions: tuple[int, ...] | None,
) -> dict[str, float]:
    if positions is not None:
        source = source[:, positions, :]
        destination = destination[:, positions, :]
    return _tensor_intervention_metrics(source, destination, tuple(range(source.shape[-1])))


def _tensor_intervention_metrics(
    source: torch.Tensor,
    destination: torch.Tensor,
    dimensions: tuple[int, ...],
) -> dict[str, float]:
    source_flat = source[..., dimensions].float().reshape(-1)
    destination_flat = destination[..., dimensions].float().reshape(-1)
    delta = destination_flat - source_flat
    source_norm = torch.linalg.vector_norm(source_flat)
    destination_norm = torch.linalg.vector_norm(destination_flat)
    return {
        "source_l2_norm": float(source_norm),
        "destination_l2_norm": float(destination_norm),
        "delta_l2_norm": float(torch.linalg.vector_norm(delta)),
        "relative_delta_l2": float(torch.linalg.vector_norm(delta) / source_norm.clamp_min(1e-12)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(source_flat, destination_flat, dim=0)),
    }


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
