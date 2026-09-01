"""Pre-outcome prediction of the last useful retargeting boundary."""

from __future__ import annotations

from typing import Any

import numpy as np

from action_chunking.analysis import commitment_step


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
        "predicted_last_successful_boundary": boundary - 1 if boundary > 0 else None,
        "raw_retention": retention.tolist(),
        "isotonic_retention": fitted.tolist(),
    }
