from __future__ import annotations

from action_chunking.obstacle_screening import obstacle_screen_row


def test_obstacle_screen_requires_safe_clean_detour() -> None:
    row = obstacle_screen_row(
        _entry(),
        _summary(),
        {
            "base": _trajectory([0.02, 0.04, 0.06, 0.08, 0.10]),
            "donor": _trajectory([0.02, 0.04, 0.06, 0.08, 0.10], y=0.09),
        },
    )

    assert row["nominal_path_intersects_corridor"] is True
    assert row["donor_first_chunk_avoids_obstacle"] is True
    assert row["clearance_gain_m"] > 0.015
    assert row["eligible"] is True


def test_obstacle_screen_rejects_contact_within_first_chunk() -> None:
    summary = _summary()
    summary["results"][1]["first_contact_step_by_object"]["bowl"] = 3
    row = obstacle_screen_row(
        _entry(),
        summary,
        {
            "base": _trajectory([0.02, 0.04, 0.06, 0.08, 0.10]),
            "donor": _trajectory([0.02, 0.04, 0.06, 0.08, 0.10], y=0.09),
        },
    )

    assert row["donor_first_chunk_avoids_obstacle"] is False
    assert row["eligible"] is False


def _entry() -> dict:
    return {
        "pair_id": "obstacle-a",
        "source_pair_id": "source-a",
        "init_index": 0,
        "base_target": "wine",
        "donor_target": "wine",
        "obstacle": "bowl",
        "path_fraction": 0.5,
        "lateral_offset_m": 0.0,
        "end_effector_position": [0.0, 0.0, 0.2],
        "donor_obstacle_position": [0.06, 0.0, 0.05],
        "obstacle_bounding_radius_m": 0.04,
    }


def _summary() -> dict:
    return {
        "pair_id": "obstacle-a",
        "results": [
            _result("base", {"wine": 20}),
            _result("donor", {"wine": 22}),
        ],
    }


def _result(side: str, contacts: dict[str, int]) -> dict:
    return {
        "side": side,
        "success": True,
        "first_contact_step_by_object": contacts,
        "live_initial_input_diagnostics": {
            "image": {"array_equal": True},
            "wrist": {"array_equal": True},
            "state": {"array_equal": True},
        },
        "restored_sim_state_max_abs_error": 0.0,
    }


def _trajectory(xs: list[float], y: float = 0.0) -> list[dict]:
    return [{"eef_pos": [x, y, 0.2]} for x in xs]
