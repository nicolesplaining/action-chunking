from __future__ import annotations

import json
import runpy
from pathlib import Path

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_dynamic_obstacle_sweep.py")
)
_write_tables = _module["_write_tables"]


def test_dynamic_obstacle_requires_induced_collision_and_isolated_speedup(tmp_path) -> None:
    for strategy, boundary in [("restart", 0), *(("continue", value) for value in range(11))]:
        root = tmp_path / f"{strategy}_after_{boundary}"
        root.mkdir()
        collision = strategy == "continue" and boundary >= 8
        post = 10 if strategy == "restart" else 10 - boundary
        latency = 100.0 if strategy == "restart" else 100.0 - 5.0 * boundary
        result = _result(collision, post, latency)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dynamic_retarget": {
                        "family": "dynamic_retarget",
                        "strategy": strategy,
                        "switch_after_steps": boundary,
                    },
                    "results": [result],
                }
            )
        )
        (root / "donor_actions.json").write_text(json.dumps([[[1.0, 2.0]]]))
        with (root / "donor_trajectory_records.jsonl").open("w") as stream:
            for step in range(5):
                stream.write(json.dumps({"eef_pos": [0.2 + 0.01 * step, 0.0, 0.1]}) + "\n")

    entry = {
        "pair_id": "obstacle-pair",
        "obstacle": "bowl",
        "donor_obstacle_position": [0.25, 0.0, 0.05],
        "obstacle_bounding_radius_m": 0.03,
    }
    _write_tables(tmp_path, entry, 0, True)
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["behaviorally_eligible"] is True
    assert summary["endpoint_gate_pass"] is True
    assert summary["velocity_evaluation_counts_exact"] is True
    assert summary["successful_continued_boundaries"] == list(range(8))
    assert summary["last_successful_continued_boundary"] == 7
    assert summary["efficient_continued_boundaries"] == list(range(1, 8))
    assert summary["practical_positive"] is True
    assert summary["population_timing_claim_allowed"] is False

    _write_tables(tmp_path, entry, 0, False)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["practical_positive"] is False


def test_dynamic_obstacle_endpoint_gate_can_stop_before_intermediates(tmp_path) -> None:
    for strategy, boundary in [("restart", 0), ("continue", 10)]:
        root = tmp_path / f"{strategy}_after_{boundary}"
        root.mkdir()
        result = _result(False, 10 if strategy == "restart" else 0, 100.0)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dynamic_retarget": {
                        "family": "dynamic_retarget",
                        "strategy": strategy,
                        "switch_after_steps": boundary,
                    },
                    "results": [result],
                }
            )
        )
        (root / "donor_actions.json").write_text(json.dumps([[[1.0, 2.0]]]))
        with (root / "donor_trajectory_records.jsonl").open("w") as stream:
            for step in range(5):
                stream.write(json.dumps({"eef_pos": [0.2 + 0.01 * step, 0.0, 0.1]}) + "\n")

    entry = {
        "pair_id": "obstacle-pair",
        "obstacle": "bowl",
        "donor_obstacle_position": [0.25, 0.0, 0.05],
        "obstacle_bounding_radius_m": 0.03,
    }
    _write_tables(tmp_path, entry, 0, True)
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["endpoint_gate_complete"] is True
    assert summary["endpoint_gate_pass"] is False
    assert summary["registered_boundaries_complete"] is False
    assert summary["practical_positive"] is False


def _result(collision: bool, post: int, latency: float) -> dict:
    return {
        "side": "donor",
        "success": True,
        "first_contact_step_by_object": {"bowl": 3} if collision else {},
        "retarget_diagnostics": {
            "post_event_velocity_evaluations": post,
            "post_event_total_ms": latency,
        },
        "live_initial_input_diagnostics": {
            "observation/image": {"array_equal": True},
            "observation/wrist_image": {"array_equal": True},
            "observation/state": {"array_equal": True},
        },
        "restored_sim_state_max_abs_error": 0.0,
        "visual_condition_switch": True,
        "source_condition_is_frozen_fixture": True,
        "donor_live_input_is_frozen_fixture": True,
        "intervention_replans_applied": [0],
    }
