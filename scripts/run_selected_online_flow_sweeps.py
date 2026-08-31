#!/usr/bin/env python3
"""Select clean-valid pairs and run closed-loop flow commitment sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eligibility", choices=("dual_success", "contact_valid"), default="dual_success")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seeds", default="0")
    parser.add_argument("--boundaries", default="all")
    parser.add_argument("--intervene-replans", default="all")
    parser.add_argument("--rollout-endpoint", choices=("first_contact", "full"), default="first_contact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = _seeds(args.noise_seeds)
    selected = _select_pairs(args.clean_validation, args.eligibility)
    manifest = json.loads(args.manifest.read_text())
    manifest_ids = {entry["pair_id"] for entry in manifest["pairs"]}
    unknown = sorted(set(selected) - manifest_ids)
    if unknown:
        raise ValueError(f"clean validation contains pairs absent from manifest: {unknown}")
    args.output.mkdir(parents=True, exist_ok=True)
    selection = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "eligibility": args.eligibility,
        "clean_validation": str(args.clean_validation),
        "pairs": selected,
        "noise_seeds": seeds,
    }
    selection_path = args.output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("existing selection differs from the requested clean-only selection")
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    runner = Path(__file__).with_name("run_online_flow_sweep.py")
    jobs = []
    for pair_id in selected:
        for seed in seeds:
            job_output = args.output / pair_id / f"noise_{seed}"
            command = [
                sys.executable,
                str(runner),
                "--manifest",
                str(args.manifest),
                "--pair-id",
                pair_id,
                "--output",
                str(job_output),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(seed),
                "--boundaries",
                args.boundaries,
                "--intervene-replans",
                args.intervene_replans,
                "--rollout-endpoint",
                args.rollout_endpoint,
            ]
            completed = subprocess.run(command, check=False)
            summary_path = job_output / "summary.json"
            if completed.returncode != 0 or not summary_path.exists():
                raise RuntimeError(f"online flow sweep failed for {pair_id} noise seed {seed}")
            summary = json.loads(summary_path.read_text())
            jobs.append(
                {
                    "pair_id": pair_id,
                    "noise_seed": seed,
                    "summary": str(summary_path),
                    "boundaries_completed": len(summary["boundaries"]),
                }
            )
            _write_run_summary(args.output, selection, jobs)
    return 0


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
    if not selected:
        raise ValueError("no pairs satisfy the requested clean-only eligibility rule")
    if len(selected) != len(set(selected)):
        raise ValueError("clean validation contains duplicate seed-0 pair summaries")
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


def _write_run_summary(output: Path, selection: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    expected = len(selection["pairs"]) * len(selection["noise_seeds"])
    summary = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "eligibility": selection["eligibility"],
        "expected_jobs": expected,
        "completed_jobs": len(jobs),
        "complete": len(jobs) == expected,
        "jobs": jobs,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"completed {len(jobs)}/{expected} closed-loop flow jobs", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
