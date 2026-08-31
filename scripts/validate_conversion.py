#!/usr/bin/env python3
"""Compare official JAX and converted PyTorch policies on an identical input."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-abs-tolerance", type=float, default=0.15)
    parser.add_argument("--worker", choices=("jax", "pytorch"))
    return parser.parse_args()


def make_fixture(path: Path) -> None:
    rng = np.random.default_rng(20260831)
    np.savez_compressed(
        path,
        image=rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        wrist_image=rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        state=rng.uniform(-0.25, 0.25, size=8).astype(np.float32),
        noise=rng.standard_normal(size=(10, 32)).astype(np.float32),
        prompt=np.asarray("pick up the black bowl and place it on the plate"),
    )


def run_worker(args: argparse.Namespace) -> None:
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    fixture = np.load(args.output_dir / "fixture.npz")
    observation = {
        "observation/image": fixture["image"],
        "observation/wrist_image": fixture["wrist_image"],
        "observation/state": fixture["state"],
        "prompt": str(fixture["prompt"]),
    }
    noise = fixture["noise"]
    config = training_config.get_config("pi05_libero")
    checkpoint = args.jax_checkpoint
    kwargs = {}
    if args.worker == "pytorch":
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
        checkpoint = args.pytorch_checkpoint
        kwargs["pytorch_device"] = "cuda:0"

    policy = policy_config.create_trained_policy(config, checkpoint, **kwargs)
    actions = policy.infer(observation, noise=noise)["actions"]
    np.save(args.output_dir / f"actions_{args.worker}.npy", actions)


def run_parent(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_fixture(args.output_dir / "fixture.npz")
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--jax-checkpoint",
        str(args.jax_checkpoint),
        "--pytorch-checkpoint",
        str(args.pytorch_checkpoint),
        "--output-dir",
        str(args.output_dir),
        "--gpu",
        args.gpu,
        "--max-abs-tolerance",
        str(args.max_abs_tolerance),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for backend in ("jax", "pytorch"):
        subprocess.run([*common, "--worker", backend], check=True, env=env)

    jax_actions = np.load(args.output_dir / "actions_jax.npy")
    pytorch_actions = np.load(args.output_dir / "actions_pytorch.npy")
    difference = pytorch_actions - jax_actions
    summary = {
        "shape": list(jax_actions.shape),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine_similarity": float(
            np.dot(jax_actions.ravel(), pytorch_actions.ravel())
            / (np.linalg.norm(jax_actions) * np.linalg.norm(pytorch_actions))
        ),
        "max_abs_tolerance": args.max_abs_tolerance,
    }
    summary["passed"] = summary["max_abs_error"] <= args.max_abs_tolerance
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


def main() -> int:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
