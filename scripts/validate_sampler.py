#!/usr/bin/env python3
"""Validate the explicit sampler and trace-only hooks against OpenPI inference."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import torch
from openpi.models import model as model_types
from openpi.policies import policy_config
from openpi.training import config as training_config

from action_chunking.sampling import prepare_condition, sample_actions
from action_chunking.tracing import ResidualTracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture = np.load(args.fixture)
    raw_observation = {
        "observation/image": fixture["image"],
        "observation/wrist_image": fixture["wrist_image"],
        "observation/state": fixture["state"],
        "prompt": str(fixture["prompt"]),
    }
    noise = fixture["noise"]

    config = training_config.get_config("pi05_libero")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device="cuda:0")
    official_actions = policy.infer(raw_observation, noise=noise)["actions"]

    transformed = policy._input_transform(jax.tree.map(lambda value: value, raw_observation))
    transformed_torch = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to("cuda:0")[None, ...], transformed
    )
    observation = model_types.Observation.from_dict(transformed_torch)
    model = policy._model
    condition = prepare_condition(model, observation)
    noise_torch = torch.from_numpy(noise).to("cuda:0")[None, ...]

    adapter_actions, _ = sample_actions(model, noise_torch, lambda _step: condition)
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    with ResidualTracer(layers, action_horizon=model.config.action_horizon) as tracer:
        traced_actions, _ = sample_actions(model, noise_torch, lambda _step: condition, tracer=tracer)

    def output_actions(actions: torch.Tensor) -> np.ndarray:
        outputs = {
            "state": np.asarray(transformed_torch["state"][0].detach().cpu()),
            "actions": np.asarray(actions[0].detach().cpu()),
        }
        return policy._output_transform(outputs)["actions"]

    adapter_output = output_actions(adapter_actions)
    traced_output = output_actions(traced_actions)
    official_error = float(np.max(np.abs(adapter_output - official_actions)))
    hook_error = float(np.max(np.abs(traced_output - adapter_output)))
    expected_sites = 10 * len(layers)
    summary = {
        "official_vs_adapter_max_abs_error": official_error,
        "adapter_vs_trace_hook_max_abs_error": hook_error,
        "captured_sites": len(tracer.trace.values),
        "expected_sites": expected_sites,
        "tolerance": args.tolerance,
        "passed": official_error <= args.tolerance
        and hook_error <= args.tolerance
        and len(tracer.trace.values) == expected_sites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
