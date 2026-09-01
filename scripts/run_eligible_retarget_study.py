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

from action_chunking.retarget_controls import boundary_zero_behavior_exact
from action_chunking.utility_analysis import summarize_utility_jobs
from action_chunking.utility_prediction import validate_eligible_retarget_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-summary", type=Path, required=True)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-root", type=Path)
    candidates.add_argument("--candidate-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orientation-calibration", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.noise_seed != 0:
        raise ValueError("registered retarget utility requires noise seed zero")
    gate = json.loads(args.gate_summary.read_text())
    if gate.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("eligibility selection must explicitly exclude continuation outcomes")
    endpoint_eligible = [row for row in gate["rows"] if row["eligible"]]
    if not endpoint_eligible:
        raise ValueError("endpoint gate contains no eligible retargeting directions")
    for row in endpoint_eligible:
        validate_eligible_retarget_row(row)
    orientation_calibration = json.loads(args.orientation_calibration.read_text())
    if orientation_calibration.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("orientation calibration must exclude continuation outcomes")
    if not orientation_calibration.get("all_pairs_pass_contrast"):
        raise ValueError("orientation calibration did not pass its clean-control gate")
    orientation_calibration_digest = _digest(args.orientation_calibration)
    eligible, selection = _select_primary_directions(endpoint_eligible)
    gate_digest = _digest(args.gate_summary)
    manifest_by_pair = _candidate_manifests(args.candidate_root, args.candidate_index)
    missing = sorted({row["pair_id"] for row in eligible} - set(manifest_by_pair))
    if missing:
        raise ValueError(f"eligible pairs are absent from candidate manifests: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)

    prediction_entries = []
    prediction_script = Path(__file__).with_name("sample_retarget_prediction.py")
    for row in eligible:
        pair_id = row["pair_id"]
        prediction_output = args.output / "predictions" / pair_id / row["new_side"]
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
                "cluster_id": _cluster_id(row),
                "manifest": str(manifest_by_pair[pair_id]),
                "manifest_sha256": _digest(manifest_by_pair[pair_id]),
                "prediction": str(prediction_path),
                "prediction_sha256": _digest(prediction_path),
                "valid": bool(prediction["valid"]),
                "predicted_last_successful_boundary": prediction.get("predicted_last_successful_boundary"),
            }
        )
    frozen_manifest = {
        "schema_version": 1,
        "all_predictions_frozen_before_closed_loop": True,
        "selection_uses_continuation_outcomes": False,
        "primary_direction_selection_rule": "first_endpoint_eligible_in_frozen_gate_order",
        "endpoint_eligible_directions": len(endpoint_eligible),
        "selected_independent_clusters": len(eligible),
        "direction_selection": selection,
        "noise_seed": args.noise_seed,
        "gate_summary": str(args.gate_summary),
        "gate_summary_sha256": gate_digest,
        "orientation_calibration": str(args.orientation_calibration),
        "orientation_calibration_sha256": orientation_calibration_digest,
        "entries": prediction_entries,
    }
    frozen_path = args.output / "frozen_predictions.json"
    serialized = json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n"
    if frozen_path.is_file():
        if frozen_path.read_text() != serialized:
            raise ValueError("existing frozen prediction manifest differs from current inputs")
    else:
        frozen_path.write_text(serialized)
    frozen_digest = _digest(frozen_path)

    sweep_script = Path(__file__).with_name("run_dynamic_retarget_sweep.py")
    orientation_script = Path(__file__).with_name("analyze_grasp_orientation_sweep.py")
    jobs = []
    for row in eligible:
        _validate_frozen_inputs(
            args.gate_summary,
            gate_digest,
            prediction_entries,
            frozen_path,
            frozen_digest,
            args.orientation_calibration,
            orientation_calibration_digest,
        )
        pair_id = row["pair_id"]
        rollout_output = args.output / "rollouts" / pair_id / row["new_side"]
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
        orientation_path = rollout_output / "grasp_orientation.json"
        subprocess.run(
            [
                sys.executable,
                str(orientation_script),
                "--sweep",
                str(rollout_output),
                "--manifest",
                str(manifest_by_pair[pair_id]),
                "--pair-id",
                pair_id,
                "--calibration",
                str(args.orientation_calibration),
                "--output",
                str(orientation_path),
            ],
            check=True,
        )
        _validate_frozen_inputs(
            args.gate_summary,
            gate_digest,
            prediction_entries,
            frozen_path,
            frozen_digest,
            args.orientation_calibration,
            orientation_calibration_digest,
        )
        jobs.append(_job_summary(row, rollout_output, prediction_entries, orientation_path))
        _write_summary(args.output, jobs, len(eligible), frozen_path, frozen_digest)
    return 0


def _candidate_manifests(root: Path | None, index_path: Path | None) -> dict[str, Path]:
    if index_path is not None:
        index = json.loads(index_path.read_text())
        if index.get("selection_uses_continuation_outcomes") is not False:
            raise ValueError("candidate index must exclude continuation outcomes")
        result = {pair_id: Path(path) for pair_id, path in index["manifest_by_pair"].items()}
        expected_digests = index.get("manifest_sha256_by_pair")
        if not isinstance(expected_digests, dict) or set(expected_digests) != set(result):
            raise ValueError("candidate index must contain one frozen digest per manifest")
        missing = [str(path) for path in result.values() if not path.is_file()]
        if missing:
            raise ValueError(f"candidate index contains missing manifests: {missing}")
        changed = [pair_id for pair_id, path in result.items() if _digest(path) != expected_digests[pair_id]]
        if changed:
            raise ValueError(f"candidate manifests changed after catalog handoff: {changed}")
        return result
    if root is None:
        raise ValueError("candidate root or index is required")
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
    orientation_path: Path,
) -> dict[str, Any]:
    sweep_summary = json.loads((rollout_output / "summary.json").read_text())
    _validate_completed_sweep(sweep_summary, gate_row["pair_id"])
    with (rollout_output / "rollouts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if boundary_zero_behavior_exact(rows, [str(gate_row["new_side"])]) is not True:
        raise ValueError("boundary-zero behavior differs between continue and restart rows")
    continuation = [row for row in rows if row["strategy"] == "continue"]
    restart_rows = [row for row in rows if row["strategy"] == "restart"]
    if len(continuation) != 11 or len(restart_rows) != 1:
        raise ValueError("completed retarget sweep has duplicate or missing strategy rows")
    by_boundary = {int(row["switch_after_steps"]): row for row in continuation}
    if len(by_boundary) != len(continuation) or set(by_boundary) != set(range(11)):
        raise ValueError("completed retarget sweep does not contain all boundaries 0..10")
    new_target_first_curve = [
        _boolean(by_boundary[boundary]["new_target_first"])
        for boundary in range(11)
    ]
    new_task_success_curve = [
        _boolean(by_boundary[boundary]["eventual_new_task_success"])
        for boundary in range(11)
    ]
    success_curve = [
        target_first and task_success
        for target_first, task_success in zip(
            new_target_first_curve,
            new_task_success_curve,
            strict=True,
        )
    ]
    first_chunk_old_event_curve = [_boolean(by_boundary[boundary]["first_chunk_old_event"]) for boundary in range(11)]
    old_target_first_curve = [_boolean(by_boundary[boundary]["old_target_first"]) for boundary in range(11)]
    clean_replanning_rescue_curve = [
        _boolean(by_boundary[boundary]["clean_replanning_rescue"]) for boundary in range(11)
    ]
    first_contact_replan_index_curve = [
        _optional_int(by_boundary[boundary]["first_contact_replan_index"]) for boundary in range(11)
    ]
    post_event_velocity_evaluations_curve = [
        int(by_boundary[boundary]["post_event_velocity_evaluations"])
        for boundary in range(11)
    ]
    observed_last = max((boundary for boundary, success in enumerate(success_curve) if success), default=None)
    prediction = next(
        entry
        for entry in prediction_entries
        if entry["pair_id"] == gate_row["pair_id"] and entry["new_side"] == gate_row["new_side"]
    )
    boundary7 = by_boundary[7]
    restart = restart_rows[0]
    orientation = json.loads(orientation_path.read_text())
    if orientation.get("pair_id") != gate_row["pair_id"]:
        raise ValueError("grasp-orientation result has the wrong pair id")
    return {
        "pair_id": gate_row["pair_id"],
        "new_side": gate_row["new_side"],
        "cluster_id": _cluster_id(gate_row),
        "prediction_valid": prediction["valid"],
        "predicted_last_successful_boundary": prediction["predicted_last_successful_boundary"],
        "observed_last_successful_boundary": observed_last,
        "prediction_exact": (prediction["valid"] and prediction["predicted_last_successful_boundary"] == observed_last),
        "new_target_first_curve": new_target_first_curve,
        "new_task_success_curve": new_task_success_curve,
        "success_curve": success_curve,
        "first_chunk_old_event_curve": first_chunk_old_event_curve,
        "old_target_first_curve": old_target_first_curve,
        "clean_replanning_rescue_curve": clean_replanning_rescue_curve,
        "first_contact_replan_index_curve": first_contact_replan_index_curve,
        "post_event_velocity_evaluations_curve": post_event_velocity_evaluations_curve,
        "boundary7_new_target_first": _boolean(boundary7["new_target_first"]),
        "boundary7_new_task_success": _boolean(boundary7["eventual_new_task_success"]),
        "boundary7_first_chunk_old_event": _boolean(boundary7["first_chunk_old_event"]),
        "boundary7_clean_replanning_rescue": _boolean(boundary7["clean_replanning_rescue"]),
        "boundary7_post_event_velocity_evaluations": int(boundary7["post_event_velocity_evaluations"]),
        "boundary7_post_event_total_ms": float(boundary7["post_event_total_ms"]),
        "restart_new_target_first": _boolean(restart["new_target_first"]),
        "restart_new_task_success": _boolean(restart["eventual_new_task_success"]),
        "restart_first_chunk_old_event": _boolean(restart["first_chunk_old_event"]),
        "restart_clean_replanning_rescue": _boolean(restart["clean_replanning_rescue"]),
        "restart_post_event_velocity_evaluations": int(restart["post_event_velocity_evaluations"]),
        "restart_post_event_total_ms": float(restart["post_event_total_ms"]),
        "grasp_orientation": str(orientation_path),
        "grasp_orientation_sha256": _digest(orientation_path),
        "orientation_editability_boundary": orientation["orientation_editability_boundary"],
        "predicted_last_orientation_correction_boundary": orientation["predicted_last_orientation_correction_boundary"],
        "orientation_curve_complete": orientation["all_boundaries_have_registered_target_contact"],
        "orientation_correct_target_first_curve": [bool(row["correct_target_first"]) for row in orientation["rows"]],
    }


def _validate_completed_sweep(summary: dict[str, Any], pair_id: str) -> None:
    required_true = (
        "all_initial_inputs_exact",
        "all_simulator_states_exact",
        "all_controller_replays_exact",
        "all_retargets_only_at_first_replan",
        "boundary_zero_continue_restart_actions_exact",
        "boundary_zero_continue_restart_behavior_exact",
    )
    if summary.get("schema_version") != 1 or summary.get("pair_id") != pair_id:
        raise ValueError("completed retarget sweep has the wrong identity")
    if int(summary.get("noise_seed", -1)) != 0:
        raise ValueError("completed retarget sweep has the wrong noise seed")
    if summary.get("registered_boundaries") != list(range(11)):
        raise ValueError("completed retarget sweep lacks a registered boundary")
    if int(summary.get("directions", -1)) != 1 or int(summary.get("source_summaries", -1)) != 12:
        raise ValueError("completed retarget sweep has the wrong condition count")
    failed = [field for field in required_true if summary.get(field) is not True]
    if failed:
        raise ValueError(f"completed retarget sweep failed exact controls: {failed}")


def _optional_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def _write_summary(
    output: Path,
    jobs: list[dict[str, Any]],
    expected: int,
    frozen_path: Path,
    frozen_digest: str,
) -> None:
    statistics = summarize_utility_jobs(jobs)
    decision = _utility_decision(statistics, len(jobs), expected)
    payload = {
        "schema_version": 1,
        "expected_primary_clusters": expected,
        "completed_primary_clusters": len(jobs),
        **decision,
        "frozen_predictions": str(frozen_path),
        "frozen_predictions_sha256": frozen_digest,
        **statistics,
        "boundary7_new_target_first_rate": _rate(jobs, "boundary7_new_target_first"),
        "boundary7_new_task_success_rate": _rate(jobs, "boundary7_new_task_success"),
        "restart_new_target_first_rate": _rate(jobs, "restart_new_target_first"),
        "restart_new_task_success_rate": _rate(jobs, "restart_new_task_success"),
        "jobs": jobs,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"completed {len(jobs)}/{expected} primary scene clusters", flush=True)


def _utility_decision(
    statistics: dict[str, Any], completed: int, expected: int
) -> dict[str, bool | str | None]:
    if expected <= 0 or completed < 0 or completed > expected:
        raise ValueError("utility completion counts are inconsistent")
    study_complete = completed == expected
    if not study_complete:
        return {
            "study_complete": False,
            "prediction_utility_positive": None,
            "practical_utility_positive": None,
            "adaptive_policy_utility_positive": None,
            "utility_inference_status": "pending",
        }
    prediction_positive = bool(statistics["prediction_utility_gate_passed"])
    practical_positive = bool(statistics["boundary7_practical_gate_passed"])
    adaptive_positive = bool(statistics["adaptive_policy_utility_gate_passed"])
    return {
        "study_complete": True,
        "prediction_utility_positive": prediction_positive,
        "practical_utility_positive": practical_positive,
        "adaptive_policy_utility_positive": adaptive_positive,
        "utility_inference_status": (
            "positive"
            if prediction_positive or practical_positive or adaptive_positive
            else "negative"
        ),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_frozen_inputs(
    gate_path: Path,
    gate_digest: str,
    prediction_entries: list[dict[str, Any]],
    frozen_path: Path,
    frozen_digest: str,
    orientation_calibration_path: Path,
    orientation_calibration_digest: str,
) -> None:
    if _digest(gate_path) != gate_digest:
        raise ValueError("endpoint gate changed after predictions were frozen")
    if _digest(frozen_path) != frozen_digest:
        raise ValueError("frozen prediction manifest changed after closed-loop rollout began")
    if _digest(orientation_calibration_path) != orientation_calibration_digest:
        raise ValueError("orientation calibration changed after closed-loop rollout began")
    for entry in prediction_entries:
        if _digest(Path(entry["manifest"])) != entry["manifest_sha256"]:
            raise ValueError("candidate manifest changed after predictions were frozen")
        if _digest(Path(entry["prediction"])) != entry["prediction_sha256"]:
            raise ValueError("action-only prediction changed after it was frozen")


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def _cluster_id(row: dict[str, Any]) -> str:
    return str(row.get("cluster_id") or row.get("source_pair_id") or row["pair_id"])


def _select_primary_directions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    decisions = []
    seen = set()
    for row in rows:
        cluster_id = _cluster_id(row)
        is_selected = cluster_id not in seen
        if is_selected:
            selected.append(row)
            seen.add(cluster_id)
        decisions.append(
            {
                "pair_id": row["pair_id"],
                "new_side": row["new_side"],
                "cluster_id": cluster_id,
                "selected": is_selected,
                "reason": (
                    "first_endpoint_eligible_in_frozen_gate_order"
                    if is_selected
                    else "additional_direction_in_selected_cluster"
                ),
            }
        )
    return selected, decisions


if __name__ == "__main__":
    raise SystemExit(main())
