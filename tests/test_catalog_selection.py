from __future__ import annotations

import json

from action_chunking.catalog_selection import build_retarget_screening_plan


def test_screening_plan_is_ordered_and_excludes_pilots(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "suite": "suite",
                "pairs": [
                    _pair("b", 2, 3, "manipulated_object"),
                    _pair("a", 5, 6, "destination"),
                    _pair("a", 1, 4, "manipulated_object"),
                ],
            }
        )
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(
        json.dumps(
            {
                "exclusions": [
                    {
                        "suite": "suite",
                        "canonical_scene_sha256": "a",
                        "base_task_id": 1,
                        "donor_task_id": 4,
                        "init_index_start": 0,
                        "init_index_stop": 1,
                        "reason": "pilot",
                    }
                ]
            }
        )
    )

    plan = build_retarget_screening_plan(
        [catalog], exclusions, initial_states_per_pair=2, minimum_eligible_clusters=1
    )

    assert plan["target_pair_definitions"] == 2
    assert plan["candidate_rows"] == 3
    assert plan["excluded_rows"] == 1
    assert [(row["canonical_scene_sha256"], row["init_index"]) for row in plan["rows"]] == [
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]
    assert len({row["screen_id"] for row in plan["rows"]}) == 3


def _pair(scene: str, base: int, donor: int, role: str) -> dict:
    return {
        "canonical_scene_sha256": scene,
        "base_task_id": base,
        "donor_task_id": donor,
        "base_task": f"base_{base}.bddl",
        "donor_task": f"donor_{donor}.bddl",
        "base_target": "base",
        "donor_target": "donor",
        "semantic_role": role,
    }
