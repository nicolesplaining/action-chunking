from __future__ import annotations

import json
import runpy
from pathlib import Path

from action_chunking.pairs import file_digest

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "build_obstacle_screening_plan.py")
)
build_plan = _module["build_plan"]
_runner = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_public_obstacle_catalog_screen.py")
)
_validate_plans = _runner["_validate_plans"]
_base_endpoint = _runner["_base_endpoint"]


def test_obstacle_plan_round_robins_target_pair_families(tmp_path) -> None:
    source_path = tmp_path / "source.json"
    rows = [
        _row("a0", "a", 0),
        _row("a1", "a", 1),
        _row("b0", "b", 0),
        _row("b1", "b", 1),
    ]
    source = {"selection_uses_intervention_outcomes": False, "rows": rows}
    source_path.write_text(json.dumps(source))

    plan = build_plan(source, source_path)

    assert [row["screen_id"] for row in plan["rows"]] == ["a0", "b0", "a1", "b1"]
    assert [row["target_pair_rank"] for row in plan["rows"]] == [0, 1, 0, 1]
    assert plan["selection_uses_obstacle_intervention_outcomes"] is False


def test_frozen_public_obstacle_plan_hash_and_source_contract() -> None:
    repo = Path(__file__).parents[1]
    plan_path = repo / "catalogs" / "obstacle_screening_plan.json"
    source_path = repo / "catalogs" / "retarget_screening_plan.json"
    plan = json.loads(plan_path.read_text())
    source = json.loads(source_path.read_text())

    _validate_plans(plan, source, source_path)

    assert file_digest(plan_path) == "c6e4ada21bd348791a699c870fbba6d2bae2e82714129225bbe7718cba9499ae"
    assert plan["candidate_rows"] == 2218
    assert plan["target_pair_families"] == 45
    assert plan["rows"][0]["source_row_index"] == 0
    assert plan["rows"][1]["source_row_index"] == 50


def test_source_base_endpoint_gate_requires_exact_target_first_success() -> None:
    summary = {
        "requested_sides": ["base"],
        "results": [
            {
                "success": True,
                "first_contact_step_by_object": {"distractor": 3, "target": 5},
                "live_initial_input_diagnostics": {
                    "observation/image": {"array_equal": True},
                    "observation/wrist_image": {"array_equal": True},
                    "observation/state": {"array_equal": True},
                },
                "restored_sim_state_max_abs_error": 0.0,
            }
        ],
    }

    result = _base_endpoint(summary, "target")

    assert result["source_base_first_contact_object"] == "distractor"
    assert result["source_base_task_success"] is True
    assert result["source_base_endpoint_eligible"] is False


def _row(screen_id: str, family: str, init_index: int) -> dict:
    return {
        "screen_id": screen_id,
        "suite": "libero_90",
        "canonical_scene_sha256": f"scene-{family}",
        "base_task": f"base-{family}.bddl",
        "donor_task": f"donor-{family}.bddl",
        "base_target": f"base-{family}",
        "donor_target": f"donor-{family}",
        "init_index": init_index,
    }
