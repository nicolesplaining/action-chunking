from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from action_chunking.metrics import target_direction_affinity
from action_chunking.pairs import file_digest
from action_chunking.retarget_controls import BOUNDARY_ZERO_BEHAVIOR_FIELDS
from action_chunking.utility_artifacts import (
    audit_utility_study,
    build_utility_job,
    build_utility_summary,
    select_primary_directions,
)
from action_chunking.utility_prediction import (
    audit_prediction_artifacts,
    predict_last_successful_boundary,
)


def test_utility_audit_reconstructs_raw_sweeps_and_detects_tampering(
    tmp_path: Path,
) -> None:
    root, rollouts_path = _study(tmp_path)

    audit = audit_utility_study(root)

    assert audit["passed"] is True
    assert audit["independent_scene_clusters"] == 1
    assert audit["raw_sweep_files"] == 3

    with rollouts_path.open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="raw rollout table"):
        audit_utility_study(root)

    bound_root, _ = _study(tmp_path / "binding")
    (bound_root / "code_commit.txt").write_text("b" * 40 + "\n")
    with pytest.raises(ValueError, match="code-commit binding"):
        audit_utility_study(bound_root)

    prediction_root, _ = _study(tmp_path / "prediction")
    frozen = json.loads((prediction_root / "frozen_predictions.json").read_text())
    entry = frozen["entries"][0]
    assert audit_prediction_artifacts(
        Path(entry["prediction"]),
        Path(entry["prediction_actions"]),
        Path(entry["manifest"]),
        entry["pair_id"],
        entry["new_side"],
    )["valid"] is True
    changed_prediction = json.loads(Path(entry["prediction"]).read_text())
    changed_prediction["predicted_last_successful_boundary"] -= 1
    Path(entry["prediction"]).write_text(json.dumps(changed_prediction))
    with pytest.raises(ValueError, match="reconstructed curve"):
        audit_prediction_artifacts(
            Path(entry["prediction"]),
            Path(entry["prediction_actions"]),
            Path(entry["manifest"]),
            entry["pair_id"],
            entry["new_side"],
        )


def _study(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    action_chunking_commit = "a" * 40
    (root / "code_commit.txt").write_text(action_chunking_commit + "\n")
    pair_id = "pair-a"
    side = "donor"
    gate_row = {
        "pair_id": pair_id,
        "new_side": side,
        "cluster_id": "cluster-a",
        "eligible": True,
        "event_exact_initial_state": True,
        "controller_replay_exact": True,
        "source_chunk_exact": True,
        "source_input_exact": True,
        "old_event_induced": True,
        "restart_avoids_old_event": True,
        "event_gate_pass": True,
        "competence_exact_initial_state": True,
        "restart_new_target_first": True,
        "clean_tasks_competent": True,
    }
    source_gate = root / "source_gate.json"
    source_gate.write_text(json.dumps({"selection_uses_continuation_outcomes": False}))
    catalog = root / "catalog_summary.json"
    catalog.write_text(
        json.dumps(
            {
                "code_commit": action_chunking_commit,
                "selection_uses_continuation_outcomes": False,
                "selection_uses_action_only_prediction_validity": True,
                "stop_threshold_reached": True,
                "catalog_exhausted": False,
                "eligible_clusters": 1,
                "valid_prediction_clusters": 1,
            }
        )
    )
    gate = root / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "selection_uses_continuation_outcomes": False,
                "source_catalog_summary": str(catalog),
                "source_catalog_summary_sha256": file_digest(catalog),
                "catalog_stop_threshold_reached": True,
                "catalog_exhausted": False,
                "confirmatory_population_complete": True,
                "eligible_clusters": 1,
                "valid_prediction_clusters": 1,
                "source_gates": [
                    {
                        "path": str(source_gate),
                        "sha256": file_digest(source_gate),
                    }
                ],
                "rows": [gate_row],
            }
        )
    )
    calibration = root / "calibration.json"
    calibration.write_text(json.dumps({"selection_uses_continuation_outcomes": False}))
    manifest = root / "candidate.json"
    manifest_entry = {
        "pair_id": pair_id,
        "end_effector_position": [0.0, 0.0, 0.0],
        "base_target_position": [1.0, 0.0, 0.0],
        "donor_target_position": [0.0, 1.0, 0.0],
    }
    manifest.write_text(json.dumps({"pairs": [manifest_entry]}))
    prediction = root / "prediction.json"
    prediction_actions = root / "prediction_actions.npz"
    actions_by_boundary = {}
    records = []
    for boundary in range(11):
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:5, 0] = boundary / 10
        actions[:5, 1] = 1 - boundary / 10
        actions_by_boundary[boundary] = actions
        records.append(
            {
                "family": "flow_switch",
                "direction": "base_to_donor",
                "switch_after_steps": boundary,
                "target_direction_affinity": target_direction_affinity(
                    actions,
                    manifest_entry["end_effector_position"],
                    manifest_entry["base_target_position"],
                    manifest_entry["donor_target_position"],
                    executed_horizon=5,
                ),
            }
        )
    restart = actions_by_boundary[0].copy()
    noise = np.zeros((10, 32), dtype=np.float32)
    np.savez_compressed(
        prediction_actions,
        **{
            f"continue_after_{boundary}": actions
            for boundary, actions in actions_by_boundary.items()
        },
        restart=restart,
        noise=noise,
    )
    prediction_payload = predict_last_successful_boundary(
        records,
        "base_to_donor",
    )
    prediction_payload.update(
        {
            "valid": True,
            "invalid_reason": None,
            "pair_id": pair_id,
            "noise_seed": 0,
            "noise_start_index": 0,
            "action_noise_shape": list(noise.shape),
            "action_shape": list(restart.shape),
            "old_side": "base",
            "new_side": side,
            "executed_action_horizon": 5,
            "boundary_zero_restart_exact": True,
            "action_sha256_by_boundary": {
                str(boundary): hashlib.sha256(actions.tobytes()).hexdigest()
                for boundary, actions in actions_by_boundary.items()
            },
            "restart_action_sha256": hashlib.sha256(restart.tobytes()).hexdigest(),
            "action_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        }
    )
    prediction.write_text(json.dumps(prediction_payload))
    entry = {
        "pair_id": pair_id,
        "new_side": side,
        "cluster_id": "cluster-a",
        "manifest": str(manifest),
        "manifest_sha256": file_digest(manifest),
        "prediction": str(prediction),
        "prediction_sha256": file_digest(prediction),
        "prediction_actions": str(prediction_actions),
        "prediction_actions_sha256": file_digest(prediction_actions),
        "valid": True,
        "predicted_last_successful_boundary": prediction_payload[
            "predicted_last_successful_boundary"
        ],
    }
    _selected, decisions = select_primary_directions([gate_row])
    frozen = root / "frozen_predictions.json"
    frozen.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "all_predictions_frozen_before_closed_loop": True,
                "selection_uses_continuation_outcomes": False,
                "primary_direction_selection_rule": (
                    "first_endpoint_eligible_in_frozen_gate_order"
                ),
                "endpoint_eligible_directions": 1,
                "selected_independent_clusters": 1,
                "direction_selection": decisions,
                "noise_seed": 0,
                "action_chunking_commit": action_chunking_commit,
                "gate_summary": str(gate),
                "gate_summary_sha256": file_digest(gate),
                "orientation_calibration": str(calibration),
                "orientation_calibration_sha256": file_digest(calibration),
                "entries": [entry],
            }
        )
    )
    rollout = root / "rollouts" / pair_id / side
    rollout.mkdir(parents=True)
    sweep = {
        "schema_version": 1,
        "pair_id": pair_id,
        "noise_seed": 0,
        "registered_boundaries": list(range(11)),
        "directions": 1,
        "source_summaries": 12,
        "all_initial_inputs_exact": True,
        "all_simulator_states_exact": True,
        "all_controller_replays_exact": True,
        "all_retargets_only_at_first_replan": True,
        "boundary_zero_continue_restart_actions_exact": True,
        "boundary_zero_continue_restart_behavior_exact": True,
    }
    (rollout / "summary.json").write_text(json.dumps(sweep))
    rows = [_row("restart", 0), *[_row("continue", boundary) for boundary in range(11)]]
    rollouts_path = rollout / "rollouts.csv"
    with rollouts_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    orientation = {
        "pair_id": pair_id,
        "orientation_editability_boundary": 8,
        "predicted_last_orientation_correction_boundary": 7,
        "all_boundaries_have_registered_target_contact": True,
        "rows": [{"correct_target_first": True} for _ in range(11)],
    }
    orientation_path = rollout / "grasp_orientation.json"
    orientation_path.write_text(json.dumps(orientation))
    job = build_utility_job(gate_row, rollout, [entry], orientation_path)
    summary = build_utility_summary([job], 1, frozen, file_digest(frozen))
    (root / "summary.json").write_text(json.dumps(summary))
    return root, rollouts_path


def _row(strategy: str, boundary: int) -> dict[str, object]:
    behavior = {
        "first_contact_object": "new_target",
        "first_contact_step": 7,
        "first_contact_replan_index": 1,
        "new_target_contact_step": 7,
        "old_target_contact_step": "",
        "new_target_first": True,
        "old_target_first": False,
        "first_chunk_new_target_contact": False,
        "first_chunk_old_event": False,
        "no_registered_contact_first_chunk": True,
        "eventual_new_task_success": True,
        "clean_replanning_rescue": True,
        "first_chunk_correction_survives": False,
        "completion_steps": 80,
    }
    assert set(behavior) == set(BOUNDARY_ZERO_BEHAVIOR_FIELDS)
    return {
        "strategy": strategy,
        "switch_after_steps": boundary,
        "side": "donor",
        **behavior,
        "post_event_velocity_evaluations": (
            10 if strategy == "restart" else 10 - boundary
        ),
        "post_event_total_ms": 400.0 if strategy == "restart" else 180.0,
    }
