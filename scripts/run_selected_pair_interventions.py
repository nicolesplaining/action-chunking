#!/usr/bin/env python3
"""Run a frozen offline intervention grid over clean-selected paired states."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

MODES = {
    "flow_only": {
        "steps": "all",
        "layers": "all",
        "skip_residual_patches": True,
        "position_mode": "all",
        "dimension_mode": "none",
        "identity_sites": "none",
    },
    "coarse": {
        "steps": "all",
        "layers": "all",
        "skip_residual_patches": False,
        "position_mode": "all",
        "dimension_mode": "groups",
        "identity_sites": "anchors",
    },
    "positions": {
        "steps": "7,8,9",
        "layers": "0,1,8,17",
        "skip_residual_patches": False,
        "position_mode": "single",
        "dimension_mode": "none",
        "identity_sites": "anchors",
    },
    "population_positions": {
        "steps": "0,7,8,9",
        "layers": "0,8,14,17",
        "skip_residual_patches": False,
        "position_mode": "single",
        "dimension_mode": "none",
        "identity_sites": "anchors",
    },
    "full": {
        "steps": "all",
        "layers": "all",
        "skip_residual_patches": False,
        "position_mode": "all",
        "dimension_mode": "groups",
        "identity_sites": "all",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODES), default="coarse")
    parser.add_argument("--eligibility", choices=("dual_success", "contact_valid"), default="dual_success")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seeds", default="0")
    parser.add_argument("--num-steps", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_steps <= 0:
        raise ValueError("num-steps must be positive")
    seeds = _seeds(args.noise_seeds)
    selected = _select_pairs(args.clean_validation, args.eligibility)
    mode = MODES[args.mode]
    manifest = json.loads(args.manifest.read_text())
    manifest_ids = {entry["pair_id"] for entry in manifest["pairs"]}
    if not set(selected) <= manifest_ids:
        raise ValueError("clean-selected pair is absent from the intervention manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "clean_validation": str(args.clean_validation),
        "eligibility": args.eligibility,
        "mode": args.mode,
        "mode_parameters": mode,
        "pairs": selected,
        "noise_seeds": seeds,
    }
    selection_path = args.output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("existing selection differs from the requested clean-only intervention grid")
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    script = Path(__file__).with_name("run_pair_interventions.py")
    jobs = []
    for pair_id in selected:
        for seed in seeds:
            job_output = args.output / pair_id / f"noise_{seed}"
            metadata_path = job_output / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                _validate_metadata(metadata, args, mode, pair_id, seed)
            else:
                command = [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--manifest",
                    str(args.manifest),
                    "--pair-id",
                    pair_id,
                    "--output",
                    str(job_output),
                    "--config",
                    args.config,
                    "--device",
                    args.device,
                    "--noise-seed",
                    str(seed),
                    "--num-steps",
                    str(args.num_steps),
                    "--steps",
                    mode["steps"],
                    "--layers",
                    mode["layers"],
                    "--position-mode",
                    mode["position_mode"],
                    "--dimension-mode",
                    mode["dimension_mode"],
                    "--identity-sites",
                    mode["identity_sites"],
                ]
                if mode["skip_residual_patches"]:
                    command.append("--skip-residual-patches")
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0 or not metadata_path.exists():
                    raise RuntimeError(f"offline intervention grid failed for {pair_id} noise seed {seed}")
                metadata = json.loads(metadata_path.read_text())
                _validate_metadata(metadata, args, mode, pair_id, seed)
            jobs.append(
                {
                    "pair_id": pair_id,
                    "noise_seed": seed,
                    "records": metadata["record_count"],
                    "metadata": str(metadata_path),
                }
            )
            _write_summary(args.output, selection, jobs)
    return 0


def _validate_metadata(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    mode: dict[str, Any],
    pair_id: str,
    seed: int,
) -> None:
    expected = {
        "pair_id": pair_id,
        "noise_seed": seed,
        "config": args.config,
        "num_steps": args.num_steps,
        "residual_patch_steps": mode["steps"],
        "residual_patch_layers": mode["layers"],
        "skip_residual_patches": mode["skip_residual_patches"],
        "position_mode": mode["position_mode"],
        "dimension_mode": mode["dimension_mode"],
        "identity_sites": mode["identity_sites"],
    }
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise ValueError(f"existing intervention metadata mismatch: {mismatched}")


def _select_pairs(root: Path, eligibility: str) -> list[str]:
    selected = []
    for path in sorted(root.glob("*/noise_*/summary.json")):
        summary = json.loads(path.read_text())
        if int(summary["noise_seed"]) != 0:
            continue
        results = summary["results"]
        strict = all(
            result.get("initial_input_mode") == "strict"
            and result.get("restored_sim_state_max_abs_error") == 0.0
            and result.get("first_chunk_max_abs_error") == 0.0
            and all(value["array_equal"] for value in result["live_initial_input_diagnostics"].values())
            for result in results
        )
        contact_valid = all(_first_contact_is_target(result) for result in results)
        dual_success = all(result["success"] for result in results)
        if strict and contact_valid and (dual_success or eligibility == "contact_valid"):
            selected.append(summary["pair_id"])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("clean validation yields an empty or duplicate intervention selection")
    return selected


def _first_contact_is_target(result: dict[str, Any]) -> bool:
    contacts = result["first_contact_step_by_object"]
    return bool(contacts) and min(contacts, key=contacts.get) == result["target"]


def _seeds(value: str) -> list[int]:
    try:
        seeds = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise ValueError("noise-seeds must be comma-separated integers") from error
    if not seeds or seeds[0] < 0:
        raise ValueError("noise-seeds must be nonnegative")
    return seeds


def _write_summary(output: Path, selection: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    expected = len(selection["pairs"]) * len(selection["noise_seeds"])
    summary = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "eligibility": selection["eligibility"],
        "mode": selection["mode"],
        "expected_jobs": expected,
        "completed_jobs": len(jobs),
        "complete": len(jobs) == expected,
        "jobs": jobs,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"completed {len(jobs)}/{expected} offline intervention jobs", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
