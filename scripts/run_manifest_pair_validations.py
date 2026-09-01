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
    parser.add_argument("--intervention", type=Path)
    parser.add_argument("--intervene-replans", default="all")
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
                str(args.intervention) if args.intervention is not None else "",
                args.intervene_replans,
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
    expected_intervention = (
        json.loads(args.intervention.read_text()) if args.intervention is not None else None
    )
    if summary.get("intervention") != expected_intervention:
        raise ValueError("existing validation has a different intervention")
    expected_replans = args.intervene_replans if expected_intervention is not None else None
    if summary.get("intervene_replans") != expected_replans:
        raise ValueError("existing validation has a different intervention-replan set")
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
    by_side = {result["side"]: result for result in summary["results"]}
    instructed_target_first = bool(
        set(by_side) == {"base", "donor"}
        and all(
            _first_contact(by_side[side]) == entry[f"{side}_target"]
            for side in ("base", "donor")
        )
    )
    exact_dual_success_target_first = bool(
        exact and summary["both_successful"] and instructed_target_first
    )
    early_exit_compute_exact = None
    if expected_intervention is not None and expected_intervention.get("family") == "early_exit":
        after_steps = int(expected_intervention["after_steps"])
        total_steps = int(expected_intervention["total_flow_steps"])
        expected_savings = total_steps - after_steps
        expected_fraction = expected_savings / total_steps
        early_exit_compute_exact = all(
            bool(result.get("early_exit_diagnostics", []))
            and len(result.get("early_exit_diagnostics", []))
            == len(result.get("intervention_replans_applied", []))
            and all(
                int(diagnostic["after_steps"]) == after_steps
                and int(diagnostic["total_flow_steps"]) == total_steps
                and int(diagnostic["velocity_field_evaluations"]) == after_steps
                and int(diagnostic["velocity_field_evaluation_savings"])
                == expected_savings
                and float(diagnostic["velocity_field_evaluation_savings_fraction"])
                == expected_fraction
                for diagnostic in result.get("early_exit_diagnostics", [])
            )
            for result in summary["results"]
        )
        if not early_exit_compute_exact:
            raise ValueError("early-exit validation failed exact compute accounting")
    return {
        "pair_id": entry["pair_id"],
        "init_index": int(entry["init_index"]),
        "both_successful": bool(summary["both_successful"]),
        "exact_initial_state": exact,
        "simulator_traces_present": traces_present,
        "replan_input_traces_present": replan_inputs_present,
        "instructed_target_first_both": instructed_target_first,
        "exact_dual_success_target_first": exact_dual_success_target_first,
        "early_exit_compute_exact": early_exit_compute_exact,
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
        "intervention": (
            json.loads(args.intervention.read_text())
            if args.intervention is not None
            else None
        ),
        "intervene_replans": (
            args.intervene_replans if args.intervention is not None else None
        ),
        "expected_pairs": expected,
        "completed_pairs": len(jobs),
        "dual_success_pairs": sum(job["both_successful"] for job in jobs),
        "exact_dual_success_target_first_pairs": sum(
            job["exact_dual_success_target_first"] for job in jobs
        ),
        "all_initial_states_exact": all(job["exact_initial_state"] for job in jobs),
        "all_requested_traces_present": (
            not args.save_sim_states
            or all(
                job["simulator_traces_present"] and job["replan_input_traces_present"]
                for job in jobs
            )
        ),
        "early_exit_compute_exact_pairs": sum(
            job["early_exit_compute_exact"] is True for job in jobs
        ),
        "jobs": jobs,
    }
    (output / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"validated {len(jobs)}/{expected}: {payload['dual_success_pairs']} dual-success, "
        f"{payload['exact_dual_success_target_first_pairs']} exact target-first",
        flush=True,
    )


def _first_contact(result: dict[str, Any]) -> str | None:
    contacts = result.get("first_contact_step_by_object", {})
    if not contacts:
        return None
    first_step = min(contacts.values())
    first_objects = [name for name, step in contacts.items() if step == first_step]
    return first_objects[0] if len(first_objects) == 1 else None


if __name__ == "__main__":
    raise SystemExit(main())
