from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_dynamic_retarget_sweep.py")
)
_write_tables = _module["_write_tables"]
_sides = _module["_sides"]
_validate_run = _module["_validate_run"]
_boundary_zero_behavior_exact = _module["_boundary_zero_behavior_exact"]


def test_dynamic_retarget_summary_preserves_compute_and_behavior_controls(tmp_path) -> None:
    for strategy, boundary in [
        ("restart", 0),
        ("continue", 0),
        ("continue", 7),
        ("continue", 8),
        ("continue", 9),
        ("continue", 10),
    ]:
        root = tmp_path / f"{strategy}_after_{boundary}"
        root.mkdir()
        results = []
        for side, target in (("base", "wine"), ("donor", "bowl")):
            post = 10 if strategy == "restart" else 10 - boundary
            results.append(_result(side, target, post, strategy == "restart" and boundary > 0))
            (root / f"{side}_actions.json").write_text(json.dumps([[[1.0, 2.0]]]))
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dynamic_retarget": {
                        "family": "dynamic_retarget",
                        "strategy": strategy,
                        "switch_after_steps": boundary,
                    },
                    "results": results,
                }
            )
        )

    entry = {"base_target": "wine", "donor_target": "bowl"}
    _write_tables(tmp_path, "pair", entry, 0)
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["boundary_zero_continue_restart_actions_exact"] is True
    assert summary["boundary_zero_continue_restart_behavior_exact"] is True
    assert summary["restart_new_target_first_rate"] == 1.0
    assert summary["restart_new_task_success_rate"] == 1.0
    by_boundary = {row["switch_after_steps"]: row for row in summary["continuation"]}
    assert by_boundary[7]["mean_post_event_velocity_evaluations"] == 3.0
    assert by_boundary[10]["mean_post_event_velocity_evaluations"] == 0.0
    assert summary["all_initial_inputs_exact"] is True
    assert summary["all_simulator_states_exact"] is True
    assert summary["all_controller_replays_exact"] is True
    assert by_boundary[7]["first_chunk_old_event_rate"] == 0.0
    assert by_boundary[7]["no_registered_contact_first_chunk_rate"] == 1.0
    assert by_boundary[7]["clean_replanning_rescue_rate"] == 1.0


def test_dynamic_retarget_summary_allows_restart_only_partial_result(tmp_path) -> None:
    root = tmp_path / "restart_after_0"
    root.mkdir()
    results = []
    for side, target in (("base", "wine"), ("donor", "bowl")):
        results.append(_result(side, target, 10, False))
        (root / f"{side}_actions.json").write_text(json.dumps([[[1.0, 2.0]]]))
    (root / "summary.json").write_text(
        json.dumps(
            {
                "dynamic_retarget": {
                    "family": "dynamic_retarget",
                    "strategy": "restart",
                    "switch_after_steps": 0,
                },
                "results": results,
            }
        )
    )

    _write_tables(tmp_path, "pair", {"base_target": "wine", "donor_target": "bowl"}, 0)
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["boundary_zero_continue_restart_actions_exact"] is None
    assert summary["boundary_zero_continue_restart_behavior_exact"] is None
    assert summary["restart_new_task_success_rate"] == 1.0
    assert summary["continuation"] == []


def test_dynamic_retarget_side_parser_accepts_one_direction() -> None:
    assert _sides("donor") == ["donor"]


def test_boundary_zero_behavior_control_detects_downstream_divergence() -> None:
    fields = {
        "first_contact_object": "bowl",
        "first_contact_step": 7,
        "first_contact_replan_index": 1,
        "new_target_contact_step": 7,
        "old_target_contact_step": None,
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
    rows = [
        {**fields, "strategy": strategy, "switch_after_steps": 0, "side": "donor"}
        for strategy in ("restart", "continue")
    ]
    assert _boundary_zero_behavior_exact(rows, ["donor"]) is True
    rows[1]["eventual_new_task_success"] = False
    assert _boundary_zero_behavior_exact(rows, ["donor"]) is False


def test_dynamic_retarget_requires_registered_controller_replay(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "pair",
                        "source_replan_index": 2,
                        "controller_replay_required": True,
                    }
                ]
            }
        )
    )
    result = _result("donor", "bowl", 3, False)
    result["first_policy_input_is_fixture"] = True
    summary = {
        "pair_id": "pair",
        "noise_seed": 0,
        "noise_start_index": 2,
        "dynamic_retarget": {
            "family": "dynamic_retarget",
            "strategy": "continue",
            "switch_after_steps": 7,
        },
        "results": [result],
    }
    args = SimpleNamespace(pair_id="pair", noise_seed=0, manifest=manifest, sides="donor")

    with pytest.raises(ValueError, match="controller state"):
        _validate_run(summary, "continue", 7, args)

    result.update(
        {
            "controller_replay_required": True,
            "controller_replay_applied": True,
            "controller_replay_trajectory_max_abs_error": 0.0,
        }
    )
    _validate_run(summary, "continue", 7, args)


def _result(side: str, target: str, post: int, discarded: bool) -> dict:
    return {
        "side": side,
        "success": True,
        "steps": 50,
        "first_contact_step_by_object": {target: 10},
        "retarget_diagnostics": {
            "post_event_velocity_evaluations": post,
            "discarded_velocity_evaluations": 1 if discarded else 0,
            "post_event_evaluation_savings_fraction": (10 - post) / 10,
            "donor_condition_ms": 4.0,
            "post_event_integration_ms": float(post),
            "post_event_total_ms": 4.0 + post,
        },
        "live_initial_input_diagnostics": {
            "observation/image": {"array_equal": True},
            "observation/wrist_image": {"array_equal": True},
            "observation/state": {"array_equal": True},
        },
        "restored_sim_state_max_abs_error": 0.0,
        "intervention_replans_applied": [0],
    }
