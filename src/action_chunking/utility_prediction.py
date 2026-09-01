"""Pre-outcome prediction of the last useful retargeting boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.analysis import commitment_step
from action_chunking.metrics import target_direction_affinity

PI05_ACTION_SHAPE = (10, 7)
PI05_ACTION_NOISE_SHAPE = (10, 32)


def validate_pi05_prediction_arrays(
    actions_by_boundary: dict[int, np.ndarray],
    restart: np.ndarray,
    noise: np.ndarray,
) -> None:
    """Validate complete finite pi0.5 arrays before prediction or audit."""
    if set(actions_by_boundary) != set(range(11)):
        raise ValueError("pi0.5 prediction requires action arrays for boundaries 0..10")
    action_arrays = [*actions_by_boundary.values(), restart]
    if any(array.shape != PI05_ACTION_SHAPE for array in action_arrays):
        raise ValueError(
            f"pi0.5 prediction requires physical action shape {PI05_ACTION_SHAPE}"
        )
    if noise.shape != PI05_ACTION_NOISE_SHAPE:
        raise ValueError(
            f"pi0.5 prediction requires action noise shape {PI05_ACTION_NOISE_SHAPE}"
        )
    if any(not np.issubdtype(array.dtype, np.number) for array in [*action_arrays, noise]):
        raise ValueError("pi0.5 prediction arrays must be numeric")
    if any(not np.all(np.isfinite(array)) for array in [*action_arrays, noise]):
        raise ValueError("pi0.5 prediction arrays must be finite")


def validate_eligible_retarget_row(row: dict[str, Any]) -> None:
    """Reject an eligible row unless every frozen endpoint control passed."""
    if not row.get("eligible"):
        raise ValueError("retarget utility requires an eligible endpoint row")
    required = (
        "event_exact_initial_state",
        "controller_replay_exact",
        "source_chunk_exact",
        "source_input_exact",
        "old_event_induced",
        "restart_avoids_old_event",
        "event_gate_pass",
        "competence_exact_initial_state",
        "restart_new_target_first",
        "clean_tasks_competent",
    )
    failed = [field for field in required if row.get(field) is not True]
    if failed:
        raise ValueError(f"eligible retarget row failed frozen controls: {failed}")


def predict_last_successful_boundary(
    records: list[dict[str, Any]],
    direction: str,
    *,
    threshold: float = 0.8,
    minimum_target_contrast: float = 0.01,
) -> dict[str, Any]:
    """Predict utility from direction-specific offline target-affinity retention."""
    if direction not in {"base_to_donor", "donor_to_base"}:
        raise ValueError("direction must be base_to_donor or donor_to_base")
    selected = sorted(
        (
            record
            for record in records
            if record.get("family") == "flow_switch" and record.get("direction") == direction
        ),
        key=lambda record: int(record["switch_after_steps"]),
    )
    boundaries = [int(record["switch_after_steps"]) for record in selected]
    if boundaries != list(range(11)):
        raise ValueError("prediction requires exactly one flow-switch record for boundaries 0..10")
    affinities = np.asarray(
        [float(record["target_direction_affinity"]) for record in selected],
        dtype=np.float64,
    )
    source = affinities[-1]
    destination = affinities[0]
    contrast = destination - source
    if abs(contrast) < minimum_target_contrast:
        raise ValueError("target-direction endpoint contrast is below the frozen validity threshold")
    retention = 1.0 - (affinities - source) / contrast
    boundary, fitted = commitment_step(retention, threshold)
    if boundary is None:
        raise ValueError("valid retention curve has no threshold-crossing boundary")
    return {
        "schema_version": 1,
        "direction": direction,
        "metric": "target_direction_affinity",
        "threshold": threshold,
        "minimum_target_contrast": minimum_target_contrast,
        "endpoint_target_contrast": abs(float(contrast)),
        "editability_boundary": boundary,
        "predicted_last_successful_boundary": boundary - 1,
        "raw_retention": retention.tolist(),
        "isotonic_retention": fitted.tolist(),
    }


def audit_prediction_artifacts(
    prediction_path: Path,
    actions_path: Path,
    manifest_path: Path,
    pair_id: str,
    new_side: str,
) -> dict[str, Any]:
    """Rebuild a frozen action-only utility prediction from its saved arrays."""
    prediction = json.loads(prediction_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1 or new_side not in {"base", "donor"}:
        raise ValueError("prediction audit has the wrong pair identity")
    entry = matches[0]
    old_side = "donor" if new_side == "base" else "base"
    direction = f"{old_side}_to_{new_side}"
    if entry.get("origin_side") is not None and entry["origin_side"] != old_side:
        raise ValueError("prediction direction differs from the screened origin side")
    required_arrays = {
        *(f"continue_after_{boundary}" for boundary in range(11)),
        "restart",
        "noise",
    }
    with np.load(actions_path, allow_pickle=False) as archive:
        if set(archive.files) != required_arrays:
            raise ValueError("prediction action archive has missing or extra arrays")
        actions_by_boundary = {
            boundary: np.asarray(archive[f"continue_after_{boundary}"])
            for boundary in range(11)
        }
        restart = np.asarray(archive["restart"])
        noise = np.asarray(archive["noise"])
    validate_pi05_prediction_arrays(actions_by_boundary, restart, noise)
    if (
        tuple(restart.shape) != tuple(prediction.get("action_shape", []))
        or tuple(noise.shape) != tuple(prediction.get("action_noise_shape", []))
    ):
        raise ValueError("prediction action archive has invalid shapes")
    if not np.array_equal(restart, actions_by_boundary[0]):
        raise ValueError("prediction boundary-zero action differs from restart")
    action_hashes = {
        str(boundary): hashlib.sha256(actions.tobytes()).hexdigest()
        for boundary, actions in actions_by_boundary.items()
    }
    if (
        prediction.get("action_sha256_by_boundary") != action_hashes
        or prediction.get("restart_action_sha256")
        != hashlib.sha256(restart.tobytes()).hexdigest()
        or prediction.get("action_noise_sha256")
        != hashlib.sha256(noise.tobytes()).hexdigest()
    ):
        raise ValueError("prediction array hashes differ from the frozen metadata")
    required_metadata = {
        "pair_id": pair_id,
        "old_side": old_side,
        "new_side": new_side,
        "direction": direction,
        "noise_seed": 0,
        "noise_start_index": int(entry.get("source_replan_index") or 0),
        "executed_action_horizon": 5,
        "boundary_zero_restart_exact": True,
    }
    mismatched = {
        key: {"expected": value, "actual": prediction.get(key)}
        for key, value in required_metadata.items()
        if prediction.get(key) != value
    }
    if mismatched:
        raise ValueError(f"prediction metadata mismatch: {mismatched}")

    source_position = np.asarray(entry[f"{old_side}_target_position"], dtype=np.float64)
    destination_position = np.asarray(
        entry[f"{new_side}_target_position"], dtype=np.float64
    )
    records = [
        {
            "family": "flow_switch",
            "direction": direction,
            "switch_after_steps": boundary,
            "target_direction_affinity": target_direction_affinity(
                actions_by_boundary[boundary],
                entry["end_effector_position"],
                source_position,
                destination_position,
                executed_horizon=int(prediction["executed_action_horizon"]),
            ),
        }
        for boundary in range(11)
    ]
    try:
        rebuilt = predict_last_successful_boundary(
            records,
            direction,
            threshold=float(prediction["threshold"]),
            minimum_target_contrast=float(prediction["minimum_target_contrast"]),
        )
        rebuilt.update({"valid": True, "invalid_reason": None})
    except ValueError as error:
        rebuilt = {
            "schema_version": 1,
            "direction": direction,
            "metric": "target_direction_affinity",
            "threshold": float(prediction["threshold"]),
            "minimum_target_contrast": float(prediction["minimum_target_contrast"]),
            "valid": False,
            "invalid_reason": str(error),
            "editability_boundary": None,
            "predicted_last_successful_boundary": None,
            "raw_target_direction_affinity": [
                float(record["target_direction_affinity"]) for record in records
            ],
        }
    differing = {
        key: {"expected": value, "actual": prediction.get(key)}
        for key, value in rebuilt.items()
        if prediction.get(key) != value
    }
    if differing:
        raise ValueError(f"prediction differs from its reconstructed curve: {differing}")
    return {
        "valid": bool(rebuilt["valid"]),
        "predicted_last_successful_boundary": rebuilt[
            "predicted_last_successful_boundary"
        ],
    }
