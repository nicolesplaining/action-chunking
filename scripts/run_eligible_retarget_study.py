#!/usr/bin/env python3
"""Freeze all action-only predictions, then run held-out retargeting rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate = json.loads(args.gate_summary.read_text())
    if gate.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("eligibility selection must explicitly exclude continuation outcomes")
    eligible = [row for row in gate["rows"] if row["eligible"]]
    if not eligible:
        raise ValueError("endpoint gate contains no eligible retargeting directions")
    manifest_by_pair = _candidate_manifests(args.candidate_root)
    missing = sorted({row["pair_id"] for row in eligible} - set(manifest_by_pair))
    if missing:
        raise ValueError(f"eligible pairs are absent from candidate manifests: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)

    prediction_entries = []
    prediction_script = Path(__file__).with_name("sample_retarget_prediction.py")
    for row in eligible:
        pair_id = row["pair_id"]
        prediction_output = args.output / "predictions" / pair_id
        prediction_path = prediction_output / "prediction.json"
        if not prediction_path.is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(prediction_script),
                    "--manifest",
                    str(manifest_by_pair[pair_id]),
                    "--pair-id",
                    pair_id,
                    "--new-side",
                    row["new_side"],
                    "--output",
                    str(prediction_output),
                    "--port",
                    str(args.port),
                    "--noise-seed",
                    str(args.noise_seed),
                ],
                check=True,
            )
        prediction = json.loads(prediction_path.read_text())
        if prediction.get("pair_id") != pair_id or prediction.get("new_side") != row["new_side"]:
            raise ValueError("existing prediction does not match the eligible direction")
        prediction_entries.append(
            {
                "pair_id": pair_id,
                "new_side": row["new_side"],
                "manifest": str(manifest_by_pair[pair_id]),
                "prediction": str(prediction_path),
                "prediction_sha256": _digest(prediction_path),
                "valid": bool(prediction["valid"]),
                "predicted_last_successful_boundary": prediction.get(
                    "predicted_last_successful_boundary"
                ),
            }
        )
    frozen_manifest = {
        "schema_version": 1,
        "all_predictions_frozen_before_closed_loop": True,
        "selection_uses_continuation_outcomes": False,
        "noise_seed": args.noise_seed,
        "entries": prediction_entries,
    }
    frozen_path = args.output / "frozen_predictions.json"
    frozen_path.write_text(json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")
    frozen_digest = _digest(frozen_path)

    sweep_script = Path(__file__).with_name("run_dynamic_retarget_sweep.py")
    jobs = []
    for row in eligible:
        pair_id = row["pair_id"]
        rollout_output = args.output / "rollouts" / pair_id
        subprocess.run(
            [
                sys.executable,
                str(sweep_script),
                "--manifest",
                str(manifest_by_pair[pair_id]),
                "--pair-id",
                pair_id,
                "--output",
                str(rollout_output),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
                "--boundaries",
                ",".join(str(boundary) for boundary in range(11)),
                "--sides",
                row["new_side"],
            ],
            check=True,
        )
        if _digest(frozen_path) != frozen_digest:
            raise ValueError("frozen prediction manifest changed after closed-loop rollout began")
        jobs.append(_job_summary(row, rollout_output, prediction_entries))
        _write_summary(args.output, jobs, len(eligible), frozen_path, frozen_digest)
    return 0


def _candidate_manifests(root: Path) -> dict[str, Path]:
    result = {}
    paths = {
        *root.glob("**/offset_*/manifest.json"),
        *root.glob("**/aligned/manifest.json"),
    }
    for path in sorted(paths):
        manifest = json.loads(path.read_text())
        for entry in manifest["pairs"]:
            pair_id = entry["pair_id"]
            if pair_id in result:
                raise ValueError(f"duplicate candidate pair id: {pair_id}")
            result[pair_id] = path
    return result


def _job_summary(
    gate_row: dict[str, Any],
    rollout_output: Path,
    prediction_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    with (rollout_output / "rollouts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    continuation = [row for row in rows if row["strategy"] == "continue"]
    by_boundary = {int(row["switch_after_steps"]): row for row in continuation}
    if set(by_boundary) != set(range(11)):
        raise ValueError("completed retarget sweep does not contain all boundaries 0..10")
    success_curve = [
        _boolean(by_boundary[boundary]["new_target_first"])
        and _boolean(by_boundary[boundary]["eventual_new_task_success"])
        for boundary in range(11)
    ]
    observed_last = max((boundary for boundary, success in enumerate(success_curve) if success), default=None)
    prediction = next(
        entry for entry in prediction_entries if entry["pair_id"] == gate_row["pair_id"]
    )
    boundary7 = by_boundary[7]
    restart = next(row for row in rows if row["strategy"] == "restart")
    return {
        "pair_id": gate_row["pair_id"],
        "new_side": gate_row["new_side"],
        "prediction_valid": prediction["valid"],
        "predicted_last_successful_boundary": prediction[
            "predicted_last_successful_boundary"
        ],
        "observed_last_successful_boundary": observed_last,
        "prediction_exact": (
            prediction["valid"]
            and prediction["predicted_last_successful_boundary"] == observed_last
        ),
        "success_curve": success_curve,
        "boundary7_new_target_first": _boolean(boundary7["new_target_first"]),
        "boundary7_new_task_success": _boolean(boundary7["eventual_new_task_success"]),
        "boundary7_post_event_velocity_evaluations": int(
            boundary7["post_event_velocity_evaluations"]
        ),
        "boundary7_post_event_total_ms": float(boundary7["post_event_total_ms"]),
        "restart_new_target_first": _boolean(restart["new_target_first"]),
        "restart_new_task_success": _boolean(restart["eventual_new_task_success"]),
        "restart_post_event_total_ms": float(restart["post_event_total_ms"]),
    }


def _write_summary(
    output: Path,
    jobs: list[dict[str, Any]],
    expected: int,
    frozen_path: Path,
    frozen_digest: str,
) -> None:
    valid = [job for job in jobs if job["prediction_valid"]]
    payload = {
        "schema_version": 1,
        "expected_eligible_directions": expected,
        "completed_directions": len(jobs),
        "frozen_predictions": str(frozen_path),
        "frozen_predictions_sha256": frozen_digest,
        "valid_predictions": len(valid),
        "exact_prediction_rate": (
            sum(job["prediction_exact"] for job in valid) / len(valid) if valid else None
        ),
        "boundary7_new_target_first_rate": _rate(jobs, "boundary7_new_target_first"),
        "boundary7_new_task_success_rate": _rate(jobs, "boundary7_new_task_success"),
        "restart_new_target_first_rate": _rate(jobs, "restart_new_target_first"),
        "restart_new_task_success_rate": _rate(jobs, "restart_new_task_success"),
        "jobs": jobs,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"completed {len(jobs)}/{expected} eligible directions", flush=True)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


if __name__ == "__main__":
    raise SystemExit(main())
