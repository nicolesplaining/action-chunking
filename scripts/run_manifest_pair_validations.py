#!/usr/bin/env python3
"""Run exact clean closed-loop validation for every pair in a manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--save-sim-states", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest.get("pairs", [])
    if not entries:
        raise ValueError("manifest contains no pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    jobs = []
    for entry in entries:
        pair_id = entry["pair_id"]
        job_output = args.output / pair_id
        summary_path = job_output / "summary.json"
        if not summary_path.is_file():
            command = [
                str(launcher),
                str(args.manifest),
                pair_id,
                str(args.gpu),
                str(args.port),
                str(args.noise_seed),
                str(job_output),
                "",
                "strict",
                str(args.save_sim_states).lower(),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode not in {0, 1} or not summary_path.is_file():
                raise RuntimeError(f"clean validation produced no summary: {pair_id}")
        summary = json.loads(summary_path.read_text())
        jobs.append(_job_record(entry, summary, job_output, args))
        _write_summary(args.output, jobs, len(entries), args)
    return 0


def _job_record(
    entry: dict[str, Any],
    summary: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if summary.get("pair_id") != entry.get("pair_id"):
        raise ValueError("existing clean validation has a different pair id")
    if int(summary.get("noise_seed", -1)) != args.noise_seed:
        raise ValueError("existing clean validation has a different noise seed")
    exact = all(
        result.get("restored_sim_state_max_abs_error") == 0.0
        and all(
            field.get("array_equal")
            for field in result.get("live_initial_input_diagnostics", {}).values()
        )
        for result in summary["results"]
    )
    if not exact:
        raise ValueError("clean validation failed exact initial-state restoration")
    traces_present = all((output / f"{side}_sim_states.npz").is_file() for side in ("base", "donor"))
    replan_inputs_present = all(
        (output / f"{side}_replan_inputs.npz").is_file() for side in ("base", "donor")
    )
    if args.save_sim_states and (not traces_present or not replan_inputs_present):
        raise ValueError("clean validation omitted requested simulator or replan-input traces")
    return {
        "pair_id": entry["pair_id"],
        "init_index": int(entry["init_index"]),
        "both_successful": bool(summary["both_successful"]),
        "exact_initial_state": exact,
        "simulator_traces_present": traces_present,
        "replan_input_traces_present": replan_inputs_present,
        "summary": str(output / "summary.json"),
    }


def _write_summary(
    output: Path,
    jobs: list[dict[str, Any]],
    expected: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema_version": 1,
        "noise_seed": args.noise_seed,
        "expected_pairs": expected,
        "completed_pairs": len(jobs),
        "dual_success_pairs": sum(job["both_successful"] for job in jobs),
        "all_initial_states_exact": all(job["exact_initial_state"] for job in jobs),
        "all_requested_traces_present": (
            not args.save_sim_states
            or all(
                job["simulator_traces_present"] and job["replan_input_traces_present"]
                for job in jobs
            )
        ),
        "jobs": jobs,
    }
    (output / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"validated {len(jobs)}/{expected}: {payload['dual_success_pairs']} dual-success",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
