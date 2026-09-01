#!/usr/bin/env python3
"""Serve deterministic OpenPI inference with per-request causal interventions."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from typing import Any

import jax
import numpy as np
import torch
from openpi.models import model as model_types
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config
from openpi_client import base_policy
from typing_extensions import override

from action_chunking.sampling import PreparedCondition, SamplingTrace, prepare_condition, sample_actions
from action_chunking.tracing import PatchSpec, ResidualTrace, ResidualTracer


class InterventionPolicy(base_policy.BasePolicy):
    """Apply a requested flow, residual, or action-state interchange online."""

    def __init__(self, policy: Any, device: str, num_steps: int):
        self.policy = policy
        self.model = policy._model
        self.device = device
        self.num_steps = num_steps
        self.layers = self.model.paligemma_with_expert.gemma_expert.model.layers

    @override
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = dict(obs)
        noise_array = np.asarray(request.pop("_action_noise"), dtype=np.float32)
        donor_prompt = request.pop("_donor_prompt", None)
        raw_spec = request.pop("_intervention", None)
        noise = torch.from_numpy(noise_array).to(self.device)[None, ...]
        source, source_transformed = _condition(self.policy, request, self.device)

        if raw_spec is None:
            actions_t, _ = sample_actions(
                self.model,
                noise,
                lambda _step: source,
                num_steps=self.num_steps,
            )
            family = "clean"
        else:
            if not isinstance(donor_prompt, str) or not donor_prompt.strip():
                raise ValueError("an intervention request requires a nonempty _donor_prompt")
            donor_request = {**request, "prompt": donor_prompt}
            donor, _ = _condition(self.policy, donor_request, self.device)
            family = str(raw_spec.get("family"))
            actions_t = self._intervene(noise, source, donor, raw_spec)

        return {
            "actions": _physical_actions(self.policy, source_transformed, actions_t),
            "intervention_family": family,
        }

    def _intervene(
        self,
        noise: torch.Tensor,
        source: PreparedCondition,
        donor: PreparedCondition,
        spec: dict[str, Any],
    ) -> torch.Tensor:
        family = str(spec.get("family"))
        if family == "flow_switch":
            boundary = _bounded_int(spec, "switch_after_steps", 0, self.num_steps + 1)
            actions, _ = sample_actions(
                self.model,
                noise,
                lambda step: source if step < boundary else donor,
                num_steps=self.num_steps,
            )
            return actions

        _source_trace, donor_trace, donor_residual = self._clean_traces(noise, source, donor)
        step = _bounded_int(spec, "flow_step", 0, self.num_steps)
        if family == "residual_patch":
            layer = _bounded_int(spec, "layer", 0, len(self.layers))
            positions = _optional_indices(spec.get("action_positions"), self.model.config.action_horizon)
            patch = PatchSpec(step=step, layer=layer, positions=positions)
            with ResidualTracer(
                self.layers,
                action_horizon=self.model.config.action_horizon,
                patch=patch,
                donor=donor_residual.values,
                capture=False,
            ) as tracer:
                actions, _ = sample_actions(
                    self.model,
                    noise,
                    lambda _step: source,
                    num_steps=self.num_steps,
                    tracer=tracer,
                )
            return actions

        if family == "action_dimension_patch":
            tensor_name = str(spec.get("patched_tensor"))
            if tensor_name not in {"x_t", "v_t"}:
                raise ValueError("patched_tensor must be x_t or v_t")
            dimensions = _indices(spec.get("action_dimensions"), self.model.config.action_dim)
            donor_tensor = getattr(donor_trace, tensor_name)[step]

            def interchange(current_step: int, value: torch.Tensor) -> torch.Tensor:
                if current_step != step:
                    return value
                patched = value.clone()
                selected = torch.as_tensor(dimensions, device=value.device, dtype=torch.long)
                donor_value = donor_tensor.to(device=value.device, dtype=value.dtype)
                patched.index_copy_(2, selected, donor_value.index_select(2, selected))
                return patched

            kwargs = (
                {"state_intervention": interchange}
                if tensor_name == "x_t"
                else {"velocity_intervention": interchange}
            )
            actions, _ = sample_actions(
                self.model,
                noise,
                lambda _step: source,
                num_steps=self.num_steps,
                **kwargs,
            )
            return actions

        raise ValueError(f"unsupported intervention family {family!r}")

    def _clean_traces(
        self,
        noise: torch.Tensor,
        source: PreparedCondition,
        donor: PreparedCondition,
    ) -> tuple[SamplingTrace, SamplingTrace, ResidualTrace]:
        with ResidualTracer(self.layers, action_horizon=self.model.config.action_horizon) as tracer:
            _, source_trace = sample_actions(
                self.model,
                noise,
                lambda _step: source,
                num_steps=self.num_steps,
                tracer=tracer,
            )
        with ResidualTracer(self.layers, action_horizon=self.model.config.action_horizon) as tracer:
            _, donor_trace = sample_actions(
                self.model,
                noise,
                lambda _step: donor,
                num_steps=self.num_steps,
                tracer=tracer,
            )
            donor_residual = tracer.trace
        return source_trace, donor_trace, donor_residual

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.policy.metadata,
            "accepts_action_noise": True,
            "accepts_causal_intervention": True,
            "causal_intervention_families": ["flow_switch", "residual_patch", "action_dimension_patch"],
            "flow_steps": self.num_steps,
            "action_expert_layers": len(self.layers),
        }

    @override
    def reset(self) -> None:
        return None


def _condition(policy: Any, raw: dict[str, Any], device: str) -> tuple[PreparedCondition, dict[str, Any]]:
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


def _bounded_int(spec: dict[str, Any], key: str, lower: int, upper: int) -> int:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value < upper:
        raise ValueError(f"{key} must be an integer in [{lower}, {upper})")
    return value


def _indices(value: Any, upper: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("intervention indices must be a nonempty list")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in value):
        raise ValueError("intervention indices must be integers")
    result = tuple(sorted(set(value)))
    if result[0] < 0 or result[-1] >= upper:
        raise ValueError(f"intervention indices must lie within [0, {upper})")
    return result


def _optional_indices(value: Any, upper: int) -> tuple[int, ...] | None:
    return None if value is None or value == "all" else _indices(value, upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--num-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("num-steps must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    policy._model.eval()
    wrapped = InterventionPolicy(policy, args.device, args.num_steps)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=wrapped.metadata,
    )
    logging.info("serving causal-intervention policy on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
