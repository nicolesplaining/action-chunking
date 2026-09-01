from __future__ import annotations

import runpy

import pytest

from action_chunking.model_comparison import (
    aggregate_paired_cells,
    normalized_position_rows,
    paired_cell_rows,
    paired_flow_shape_rows,
    paired_timing_rows,
    paired_timing_summary,
)

_script = runpy.run_path("scripts/compare_pi0_models.py")


def test_paired_timing_uses_common_eligible_scene_states() -> None:
    pi05 = [_unit("a", formation=0, boundary=9), _unit("b", formation=1, boundary=8)]
    pi0 = [_unit("a", formation=1, boundary=7), _unit("b", formation=1, boundary=8)]

    rows = paired_timing_rows(pi05, pi0)
    summary = paired_timing_summary(rows, bootstrap_replicates=100)

    assert len(rows) == 2
    assert rows[0]["editability_boundary_difference_pi05_minus_pi0"] == 2.0
    assert summary["translation"]["editability_boundary_difference_pi05_minus_pi0"]["mean_difference"] == 1.0


def test_paired_cells_and_normalized_positions() -> None:
    pi05 = [_cell("a", position, float(position)) for position in range(10)]
    pi0 = [_cell("a", position, 0.0) for position in range(50)]
    normalized_pi05 = normalized_position_rows(pi05, 10)
    normalized_pi0 = normalized_position_rows(pi0, 50)

    paired = paired_cell_rows(
        normalized_pi05,
        normalized_pi0,
        ("flow_step", "layer", "normalized_position_bin"),
    )
    cells = aggregate_paired_cells(
        paired,
        ("flow_step", "layer", "normalized_position_bin"),
        bootstrap_replicates=100,
    )

    assert len(cells) == 10
    assert cells[-1]["mean_difference_pi05_minus_pi0"] == pytest.approx(9.0)


def test_flow_shape_compares_transition_width_and_swap_asymmetry() -> None:
    pi05 = [_flow(boundary, min(1.0, boundary / 8.0), 0.05) for boundary in range(11)]
    pi0 = [_flow(boundary, boundary / 10.0, 0.10) for boundary in range(11)]

    rows = paired_flow_shape_rows(pi05, pi0)

    assert len(rows) == 1
    assert rows[0]["directional_asymmetry_auc_difference_pi05_minus_pi0"] == pytest.approx(-0.05)
    assert rows[0]["transition_width_10_to_90_difference_pi05_minus_pi0"] < 0


def test_comparison_rejects_incomplete_registered_position_grid() -> None:
    rows = [
        {
            "flow_step": flow_step,
            "layer": layer,
            "action_position": position,
        }
        for flow_step in (0, 7, 8, 9)
        for layer in (0, 8, 14, 17)
        for position in range(10)
    ]
    rows.pop()

    with pytest.raises(ValueError, match="position grid is incomplete"):
        _script["_validate_position_grid"](rows, 10, "pi0.5")


def _unit(cluster: str, *, formation: int, boundary: int) -> dict:
    return {
        "pair_id": f"pair-{cluster}",
        "scene_state_sha256": cluster,
        "noise_seed": 0,
        "metric": "translation",
        "eligible": True,
        "formation_step": formation,
        "commitment_step": boundary,
    }


def _cell(cluster: str, position: int, value: float) -> dict:
    return {
        "scene_state_sha256": cluster,
        "metric": "translation",
        "eligible": True,
        "flow_step": 9,
        "layer": 17,
        "action_position": position,
        "symmetric_ncte": value,
    }


def _flow(boundary: int, retention: float, asymmetry: float) -> dict:
    return {
        "scene_state_sha256": "a",
        "metric": "target_direction",
        "eligible": True,
        "switch_after_steps": boundary,
        "symmetric_retention": retention,
        "directional_asymmetry": asymmetry,
    }
