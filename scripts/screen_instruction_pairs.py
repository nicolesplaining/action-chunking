#!/usr/bin/env python3
"""Screen clean paired chunks before any causal intervention is inspected."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from openpi.models import model as model_types
from openpi.policies import policy_config
from openpi.training import config as training_config

from action_chunking.metrics import LIBERO_ACTION_GROUPS, target_direction_affinity
from action_chunking.pairs import file_digest, load_instruction_pair
from action_chunking.sampling import prepare_condition, sample_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seeds", default="0,1,2,3")
    parser.add_argument("--minimum-direction-margin", type=float, default=0.005)
    parser.add_argument("--minimum-translation-l2", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.noise_seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("noise seeds must be a nonempty unique list")
    manifest_paths = sorted(args.manifests_root.rglob("manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"no manifests under {args.manifests_root}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    model = policy._model
    model.eval()
    noises = {
        seed: np.random.default_rng(seed).standard_normal(
            (model.config.action_horizon, model.config.action_dim),
            dtype=np.float32,
        )
        for seed in seeds
    }

    records = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["pairs"]:
            fixture_path = manifest_path.parent / entry["fixture"]
            if file_digest(fixture_path) != entry["fixture_sha256"]:
                raise ValueError(f"fixture hash mismatch: {fixture_path}")
            pair = load_instruction_pair(fixture_path)
            base_condition, base_transformed = _condition(policy, pair.raw_observation("base"), args.device)
            donor_condition, donor_transformed = _condition(policy, pair.raw_observation("donor"), args.device)
            for seed in seeds:
                noise = torch.from_numpy(noises[seed]).to(args.device)[None, ...]
                base_t, _ = sample_actions(model, noise, lambda _step, condition=base_condition: condition)
                donor_t, _ = sample_actions(model, noise, lambda _step, condition=donor_condition: condition)
                base_actions = _physical_actions(policy, base_transformed, base_t)
                donor_actions = _physical_actions(policy, donor_transformed, donor_t)
                context = (
                    entry["end_effector_position"],
                    entry["base_target_position"],
                    entry["donor_target_position"],
                )
                base_affinity = target_direction_affinity(base_actions, *context)
                donor_affinity = target_direction_affinity(donor_actions, *context)
                contrasts = {
                    name: float(np.linalg.norm(donor_actions[:, indices] - base_actions[:, indices]))
                    for name, indices in LIBERO_ACTION_GROUPS.items()
                }
                records.append(
                    {
                        "pair_id": entry["pair_id"],
                        "manifest": str(manifest_path),
                        "fixture_sha256": entry["fixture_sha256"],
                        "scene_state_sha256": entry["identity_hashes"]["sim_state"],
                        "init_index": entry["init_index"],
                        "base_target": entry["base_target"],
                        "donor_target": entry["donor_target"],
                        "noise_seed": seed,
                        "base_target_direction_affinity": base_affinity,
                        "donor_target_direction_affinity": donor_affinity,
                        "direction_contrast": donor_affinity - base_affinity,
                        "endpoint_group_l2_contrasts": contrasts,
                        "direction_screen_pass": (
                            base_affinity <= -args.minimum_direction_margin
                            and donor_affinity >= args.minimum_direction_margin
                            and contrasts["translation"] >= args.minimum_translation_l2
                        ),
                        "base_actions": base_actions.tolist(),
                        "donor_actions": donor_actions.tolist(),
                    }
                )
            print(f"screened {entry['pair_id']}", flush=True)

    records_path = args.output / "clean_screen.jsonl"
    with records_path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pair_groups.setdefault(record["pair_id"], []).append(record)
    passing_all_seeds = sorted(
        pair_id for pair_id, pair_records in pair_groups.items() if all(row["direction_screen_pass"] for row in pair_records)
    )
    summary = {
        "schema_version": 1,
        "manifests": len(manifest_paths),
        "scene_pairs": len(pair_groups),
        "independent_serialized_states": len({record["scene_state_sha256"] for record in records}),
        "noise_seeds": seeds,
        "records": len(records),
        "screen_definition": {
            "base_affinity_maximum": -args.minimum_direction_margin,
            "donor_affinity_minimum": args.minimum_direction_margin,
            "translation_l2_minimum": args.minimum_translation_l2,
            "requires_all_noise_seeds": True,
            "uses_patched_outcomes": False,
        },
        "pairs_passing_all_seeds": len(passing_all_seeds),
        "passing_pair_ids": passing_all_seeds,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _condition(policy: Any, raw: dict[str, Any], device: str):
    transformed = policy._input_transform(jax.tree.map(lambda value: value, raw))
    transformed_torch = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to(device)[None, ...],
        transformed,
    )
    observation = model_types.Observation.from_dict(transformed_torch)
    return prepare_condition(policy._model, observation), transformed_torch


def _physical_actions(policy: Any, transformed: dict[str, Any], actions: torch.Tensor) -> np.ndarray:
    outputs = {
        "state": np.asarray(transformed["state"][0].detach().cpu()),
        "actions": np.asarray(actions[0].detach().cpu()),
    }
    return np.asarray(policy._output_transform(outputs)["actions"])


if __name__ == "__main__":
    raise SystemExit(main())
