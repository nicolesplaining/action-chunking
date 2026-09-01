"""Reconstruct and summarize frozen held-out retargeting artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest
from action_chunking.retarget_controls import boundary_zero_behavior_exact
from action_chunking.utility_analysis import summarize_utility_jobs
from action_chunking.utility_prediction import validate_eligible_retarget_row


def build_utility_job(
    gate_row: dict[str, Any],
    rollout_output: Path,
    prediction_entries: list[dict[str, Any]],
    orientation_path: Path,
) -> dict[str, Any]:
    """Reconstruct one cluster-level job from raw sweep artifacts."""
    sweep_path = rollout_output / "summary.json"
    rollouts_path = rollout_output / "rollouts.csv"
    sweep_summary = json.loads(sweep_path.read_text())
    validate_completed_sweep(sweep_summary, gate_row["pair_id"])
    with rollouts_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required_columns = {
        "strategy",
        "switch_after_steps",
        "side",
        "new_target_first",
        "old_target_first",
        "first_chunk_old_event",
        "eventual_new_task_success",
        "clean_replanning_rescue",
        "first_contact_replan_index",
        "post_event_velocity_evaluations",
        "post_event_total_ms",
    }
    if (
        len(rows) != 12
        or any(required_columns - set(row) for row in rows)
        or any(row[column] is None for row in rows for column in required_columns)
    ):
        raise ValueError("utility raw rollout table is malformed")
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
        strict_boolean(by_boundary[boundary]["new_target_first"])
        for boundary in range(11)
    ]
    new_task_success_curve = [
        strict_boolean(by_boundary[boundary]["eventual_new_task_success"])
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
    first_chunk_old_event_curve = [
        strict_boolean(by_boundary[boundary]["first_chunk_old_event"])
        for boundary in range(11)
    ]
    old_target_first_curve = [
        strict_boolean(by_boundary[boundary]["old_target_first"])
        for boundary in range(11)
    ]
    clean_replanning_rescue_curve = [
        strict_boolean(by_boundary[boundary]["clean_replanning_rescue"])
        for boundary in range(11)
    ]
    first_contact_replan_index_curve = [
        _optional_int(by_boundary[boundary]["first_contact_replan_index"])
        for boundary in range(11)
    ]
    post_event_velocity_evaluations_curve = [
        int(by_boundary[boundary]["post_event_velocity_evaluations"])
        for boundary in range(11)
    ]
    observed_last = max(
        (boundary for boundary, success in enumerate(success_curve) if success),
        default=None,
    )
    matching_predictions = [
        entry
        for entry in prediction_entries
        if entry["pair_id"] == gate_row["pair_id"]
        and entry["new_side"] == gate_row["new_side"]
    ]
    if len(matching_predictions) != 1:
        raise ValueError("utility job does not have exactly one frozen prediction")
    prediction = matching_predictions[0]
    boundary7 = by_boundary[7]
    restart = restart_rows[0]
    orientation = json.loads(orientation_path.read_text())
    if orientation.get("pair_id") != gate_row["pair_id"]:
        raise ValueError("grasp-orientation result has the wrong pair id")
    orientation_rows = orientation.get("rows")
    if (
        not isinstance(orientation_rows, list)
        or len(orientation_rows) != 11
        or any(type(row.get("correct_target_first")) is not bool for row in orientation_rows)
        or type(orientation.get("all_boundaries_have_registered_target_contact"))
        is not bool
    ):
        raise ValueError("grasp-orientation result has an invalid registered curve")
    return {
        "pair_id": gate_row["pair_id"],
        "new_side": gate_row["new_side"],
        "cluster_id": cluster_id(gate_row),
        "prediction_valid": prediction["valid"],
        "predicted_last_successful_boundary": prediction[
            "predicted_last_successful_boundary"
        ],
        "observed_last_successful_boundary": observed_last,
        "prediction_exact": bool(
            prediction["valid"]
            and prediction["predicted_last_successful_boundary"] == observed_last
        ),
        "new_target_first_curve": new_target_first_curve,
        "new_task_success_curve": new_task_success_curve,
        "success_curve": success_curve,
        "first_chunk_old_event_curve": first_chunk_old_event_curve,
        "old_target_first_curve": old_target_first_curve,
        "clean_replanning_rescue_curve": clean_replanning_rescue_curve,
        "first_contact_replan_index_curve": first_contact_replan_index_curve,
        "post_event_velocity_evaluations_curve": post_event_velocity_evaluations_curve,
        "boundary7_new_target_first": strict_boolean(boundary7["new_target_first"]),
        "boundary7_new_task_success": strict_boolean(
            boundary7["eventual_new_task_success"]
        ),
        "boundary7_first_chunk_old_event": strict_boolean(
            boundary7["first_chunk_old_event"]
        ),
        "boundary7_clean_replanning_rescue": strict_boolean(
            boundary7["clean_replanning_rescue"]
        ),
        "boundary7_post_event_velocity_evaluations": int(
            boundary7["post_event_velocity_evaluations"]
        ),
        "boundary7_post_event_total_ms": float(boundary7["post_event_total_ms"]),
        "restart_new_target_first": strict_boolean(restart["new_target_first"]),
        "restart_new_task_success": strict_boolean(
            restart["eventual_new_task_success"]
        ),
        "restart_first_chunk_old_event": strict_boolean(
            restart["first_chunk_old_event"]
        ),
        "restart_clean_replanning_rescue": strict_boolean(
            restart["clean_replanning_rescue"]
        ),
        "restart_post_event_velocity_evaluations": int(
            restart["post_event_velocity_evaluations"]
        ),
        "restart_post_event_total_ms": float(restart["post_event_total_ms"]),
        "sweep_summary": str(sweep_path),
        "sweep_summary_sha256": file_digest(sweep_path),
        "rollouts_csv": str(rollouts_path),
        "rollouts_csv_sha256": file_digest(rollouts_path),
        "grasp_orientation": str(orientation_path),
        "grasp_orientation_sha256": file_digest(orientation_path),
        "orientation_editability_boundary": orientation[
            "orientation_editability_boundary"
        ],
        "predicted_last_orientation_correction_boundary": orientation[
            "predicted_last_orientation_correction_boundary"
        ],
        "orientation_curve_complete": orientation[
            "all_boundaries_have_registered_target_contact"
        ],
        "orientation_correct_target_first_curve": [
            row["correct_target_first"] for row in orientation_rows
        ],
    }


def validate_completed_sweep(summary: dict[str, Any], pair_id: str) -> None:
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
    if int(summary.get("directions", -1)) != 1 or int(
        summary.get("source_summaries", -1)
    ) != 12:
        raise ValueError("completed retarget sweep has the wrong condition count")
    failed = [field for field in required_true if summary.get(field) is not True]
    if failed:
        raise ValueError(f"completed retarget sweep failed exact controls: {failed}")


def build_utility_summary(
    jobs: list[dict[str, Any]],
    expected: int,
    frozen_path: Path,
    frozen_digest: str,
) -> dict[str, Any]:
    statistics = summarize_utility_jobs(jobs)
    decision = utility_decision(statistics, len(jobs), expected)
    return {
        "schema_version": 1,
        "expected_primary_clusters": expected,
        "completed_primary_clusters": len(jobs),
        **decision,
        "frozen_predictions": str(frozen_path),
        "frozen_predictions_sha256": frozen_digest,
        **statistics,
        "boundary7_new_target_first_rate": _rate(
            jobs, "boundary7_new_target_first"
        ),
        "boundary7_new_task_success_rate": _rate(
            jobs, "boundary7_new_task_success"
        ),
        "restart_new_target_first_rate": _rate(jobs, "restart_new_target_first"),
        "restart_new_task_success_rate": _rate(jobs, "restart_new_task_success"),
        "jobs": jobs,
    }


def audit_utility_study(root: Path) -> dict[str, Any]:
    """Independently reconstruct a completed held-out utility study."""
    summary_path = root / "summary.json"
    frozen_path = root / "frozen_predictions.json"
    summary = _read_json(summary_path)
    frozen = _read_json(frozen_path)
    if summary.get("study_complete") is not True:
        raise ValueError("utility audit requires a completed frozen population")
    if (
        summary.get("frozen_predictions") != str(frozen_path)
        or summary.get("frozen_predictions_sha256") != file_digest(frozen_path)
        or frozen.get("schema_version") != 1
        or frozen.get("all_predictions_frozen_before_closed_loop") is not True
        or frozen.get("selection_uses_continuation_outcomes") is not False
        or int(frozen.get("noise_seed", -1)) != 0
    ):
        raise ValueError("utility summary has an invalid frozen-prediction binding")
    action_chunking_commit = str(frozen.get("action_chunking_commit", ""))
    code_binding = root / "code_commit.txt"
    if (
        re.fullmatch(r"[0-9a-f]{40}", action_chunking_commit) is None
        or not code_binding.is_file()
        or code_binding.read_text().strip() != action_chunking_commit
    ):
        raise ValueError("utility study lacks a valid code-commit binding")

    gate_path = Path(str(frozen.get("gate_summary", "")))
    calibration_path = Path(str(frozen.get("orientation_calibration", "")))
    if (
        not gate_path.is_file()
        or frozen.get("gate_summary_sha256") != file_digest(gate_path)
        or not calibration_path.is_file()
        or frozen.get("orientation_calibration_sha256")
        != file_digest(calibration_path)
    ):
        raise ValueError("utility frozen inputs changed after prediction")
    gate = _read_json(gate_path)
    if gate.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("utility endpoint gate used continuation outcomes")
    endpoint_rows = [row for row in gate.get("rows", []) if row.get("eligible")]
    for row in endpoint_rows:
        validate_eligible_retarget_row(row)
    selected, decisions = select_primary_directions(endpoint_rows)
    entries = frozen.get("entries")
    if (
        not isinstance(entries, list)
        or frozen.get("direction_selection") != decisions
        or int(frozen.get("endpoint_eligible_directions", -1)) != len(endpoint_rows)
        or int(frozen.get("selected_independent_clusters", -1)) != len(selected)
        or len(entries) != len(selected)
    ):
        raise ValueError("utility frozen population differs from the endpoint gate")
    selected_keys = [
        (str(row["pair_id"]), str(row["new_side"]), cluster_id(row))
        for row in selected
    ]
    entry_keys = [
        (str(entry["pair_id"]), str(entry["new_side"]), str(entry["cluster_id"]))
        for entry in entries
    ]
    if entry_keys != selected_keys or len(set(entry_keys)) != len(entry_keys):
        raise ValueError("utility frozen entries are not the selected independent population")
    for entry in entries:
        for path_field, digest_field in (
            ("manifest", "manifest_sha256"),
            ("prediction", "prediction_sha256"),
        ):
            path = Path(str(entry.get(path_field, "")))
            if not path.is_file() or entry.get(digest_field) != file_digest(path):
                raise ValueError(f"utility frozen {path_field} changed after prediction")

    gate_by_key = {
        (str(row["pair_id"]), str(row["new_side"])): row for row in selected
    }
    jobs = []
    for entry in entries:
        key = (str(entry["pair_id"]), str(entry["new_side"]))
        rollout = root / "rollouts" / key[0] / key[1]
        jobs.append(
            build_utility_job(
                gate_by_key[key],
                rollout,
                entries,
                rollout / "grasp_orientation.json",
            )
        )
    expected = build_utility_summary(
        jobs,
        len(selected),
        frozen_path,
        file_digest(frozen_path),
    )
    if summary != expected:
        raise ValueError("utility summary differs from independently reconstructed artifacts")
    return {
        "schema_version": 1,
        "passed": True,
        "utility_summary_sha256": file_digest(summary_path),
        "frozen_predictions_sha256": file_digest(frozen_path),
        "independent_scene_clusters": len(jobs),
        "action_chunking_commit": action_chunking_commit,
        "raw_sweep_files": len(jobs) * 3,
        "utility_inference_status": summary["utility_inference_status"],
        "prediction_utility_positive": summary["prediction_utility_positive"],
        "practical_utility_positive": summary["practical_utility_positive"],
        "adaptive_policy_utility_positive": summary[
            "adaptive_policy_utility_positive"
        ],
    }


def utility_decision(
    statistics: dict[str, Any], completed: int, expected: int
) -> dict[str, bool | str | None]:
    if expected <= 0 or completed < 0 or completed > expected:
        raise ValueError("utility completion counts are inconsistent")
    if completed != expected:
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


def select_primary_directions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    decisions = []
    seen = set()
    for row in rows:
        row_cluster_id = cluster_id(row)
        is_selected = row_cluster_id not in seen
        if is_selected:
            selected.append(row)
            seen.add(row_cluster_id)
        decisions.append(
            {
                "pair_id": row["pair_id"],
                "new_side": row["new_side"],
                "cluster_id": row_cluster_id,
                "selected": is_selected,
                "reason": (
                    "first_endpoint_eligible_in_frozen_gate_order"
                    if is_selected
                    else "additional_direction_in_selected_cluster"
                ),
            }
        )
    return selected, decisions


def cluster_id(row: dict[str, Any]) -> str:
    return str(row.get("cluster_id") or row.get("source_pair_id") or row["pair_id"])


def strict_boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def _optional_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
