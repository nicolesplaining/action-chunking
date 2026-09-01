#!/usr/bin/env python3
"""Validate JAX/PyTorch conversion on exact held-out LIBERO pair inputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from action_chunking.conversion import (
    conversion_parity_summary,
    validate_prior_conversion_failure,
)
from action_chunking.pairs import file_digest
from action_chunking.pi0_checkpoint import validate_pi0_final_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-converter", type=Path, required=True)
    parser.add_argument("--prior-failed-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="pi0_libero")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--max-abs-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--worker", choices=("jax", "pytorch"))
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> None:
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    from action_chunking.pairs import load_instruction_pair

    manifest = json.loads(args.manifest.read_text())
    config = training_config.get_config(args.config)
    checkpoint = args.jax_checkpoint
    kwargs = {}
    if args.worker == "pytorch":
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, pytorch_compile_mode=None),
        )
        checkpoint = args.pytorch_checkpoint
        kwargs["pytorch_device"] = "cuda:0"
    policy = policy_config.create_trained_policy(config, checkpoint, **kwargs)
    identifiers = []
    actions = []
    for entry in manifest["pairs"]:
        pair = load_instruction_pair(args.manifest.parent / entry["fixture"])
        noise = np.random.default_rng(args.noise_seed).standard_normal(
            (config.model.action_horizon, config.model.action_dim), dtype=np.float32
        )
        for side in ("base", "donor"):
            identifiers.append(f"{entry['pair_id']}:{side}")
            actions.append(np.asarray(policy.infer(pair.raw_observation(side), noise=noise)["actions"]))
    np.save(args.output / f"actions_{args.worker}.npy", np.stack(actions))
    (args.output / f"identifiers_{args.worker}.json").write_text(json.dumps(identifiers, indent=2) + "\n")


def run_parent(args: argparse.Namespace) -> int:
    if args.noise_seed < 0:
        raise ValueError("noise seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    prior_failure = validate_prior_conversion_failure(
        args.prior_failed_summary,
        args.jax_checkpoint,
        args.manifest,
    )
    conversion_provenance = _validate_conversion_provenance(
        args.pytorch_checkpoint,
        args.upstream_converter,
    )
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--jax-checkpoint",
        str(args.jax_checkpoint),
        "--pytorch-checkpoint",
        str(args.pytorch_checkpoint),
        "--upstream-converter",
        str(args.upstream_converter),
        "--prior-failed-summary",
        str(args.prior_failed_summary),
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output),
        "--config",
        args.config,
        "--gpu",
        args.gpu,
        "--noise-seed",
        str(args.noise_seed),
        "--max-abs-tolerance",
        str(args.max_abs_tolerance),
        "--minimum-cosine-similarity",
        str(args.minimum_cosine_similarity),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for backend in ("jax", "pytorch"):
        subprocess.run([*common, "--worker", backend], check=True, env=env)
    jax_ids = json.loads((args.output / "identifiers_jax.json").read_text())
    pytorch_ids = json.loads((args.output / "identifiers_pytorch.json").read_text())
    if jax_ids != pytorch_ids:
        raise ValueError("conversion workers produced different case identifiers")
    result = conversion_parity_summary(
        jax_ids,
        np.load(args.output / "actions_jax.npy"),
        np.load(args.output / "actions_pytorch.npy"),
        max_abs_tolerance=args.max_abs_tolerance,
        minimum_cosine_similarity=args.minimum_cosine_similarity,
    )
    result.update(
        {
            "config": args.config,
            "noise_seed": args.noise_seed,
            "manifest": str(args.manifest),
            "manifest_sha256": file_digest(args.manifest),
            "jax_checkpoint": str(args.jax_checkpoint),
            "pytorch_checkpoint": str(args.pytorch_checkpoint),
            "jax_checkpoint_identity": validate_pi0_final_checkpoint(args.jax_checkpoint),
            "pytorch_checkpoint_artifact_sha256": _checkpoint_hashes(args.pytorch_checkpoint),
            "conversion_provenance": conversion_provenance,
            "prior_failed_conversion": prior_failure,
        }
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def _checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    required = ("config.json", "conversion_provenance.json", "model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"converted checkpoint artifacts are missing: {missing}")
    return {name: file_digest(checkpoint / name) for name in required}


def _validate_conversion_provenance(
    checkpoint: Path,
    upstream_converter: Path,
) -> dict:
    path = checkpoint / "conversion_provenance.json"
    if not path.is_file():
        raise FileNotFoundError("converted checkpoint lacks conversion_provenance.json")
    provenance = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "adapter": "openpi_pr978_float32_intermediate",
        "source_precision_repair_commit": "e5fe45e2c6784f315ffa59c207457701fb906c05",
        "upstream_openpi_revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "intermediate_model_config_dtype": "float32",
        "saved_checkpoint_precision": "float32",
        "policy_loader_precision_behavior": "unchanged_openpi_mixed_precision",
    }
    mismatched = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in required.items()
        if provenance.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"conversion provenance mismatch: {mismatched}")
    if not upstream_converter.is_file():
        raise FileNotFoundError(f"upstream converter is missing: {upstream_converter}")
    actual_digest = file_digest(upstream_converter)
    if provenance.get("upstream_converter_sha256") != actual_digest:
        raise ValueError("conversion provenance has the wrong upstream converter digest")
    return provenance


def main() -> int:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
