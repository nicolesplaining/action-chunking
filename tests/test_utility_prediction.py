from __future__ import annotations

import pytest

from action_chunking.utility_prediction import predict_last_successful_boundary


def test_predicts_last_successful_boundary_before_retention_crossing() -> None:
    retention = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]
    records = [_record(boundary, 1.0 - value) for boundary, value in enumerate(retention)]

    result = predict_last_successful_boundary(records, "base_to_donor")

    assert result["editability_boundary"] == 8
    assert result["predicted_last_successful_boundary"] == 7
    assert result["isotonic_retention"] == pytest.approx(retention)


def test_rejects_degenerate_target_direction_contrast() -> None:
    records = [_record(boundary, boundary * 0.0001) for boundary in range(11)]

    with pytest.raises(ValueError, match="contrast"):
        predict_last_successful_boundary(records, "base_to_donor")


def _record(boundary: int, affinity: float) -> dict:
    return {
        "family": "flow_switch",
        "direction": "base_to_donor",
        "switch_after_steps": boundary,
        "target_direction_affinity": affinity,
    }
