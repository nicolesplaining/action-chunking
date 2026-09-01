from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_eligible_retarget_study.py")
)
_boolean = _module["_boolean"]
_candidate_manifests = _module["_candidate_manifests"]
_select_primary_directions = _module["_select_primary_directions"]
_utility_decision = _module["_utility_decision"]


def test_strict_serialized_boolean_parser() -> None:
    assert _boolean("True") is True
    assert _boolean("False") is False


def test_selects_only_first_frozen_direction_per_scene_cluster() -> None:
    rows = [
        {"pair_id": "a", "new_side": "base", "cluster_id": "scene-1"},
        {"pair_id": "a", "new_side": "donor", "cluster_id": "scene-1"},
        {"pair_id": "b", "new_side": "base", "cluster_id": "scene-2"},
    ]

    selected, decisions = _select_primary_directions(rows)

    assert [(row["pair_id"], row["new_side"]) for row in selected] == [
        ("a", "base"),
        ("b", "base"),
    ]
    assert [decision["selected"] for decision in decisions] == [True, False, True]


def test_reads_frozen_candidate_index(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    index = tmp_path / "index.json"
    index.write_text(
        __import__("json").dumps(
            {
                "selection_uses_continuation_outcomes": False,
                "manifest_by_pair": {"pair": str(manifest)},
                "manifest_sha256_by_pair": {
                    "pair": hashlib.sha256(manifest.read_bytes()).hexdigest()
                },
            }
        )
    )

    assert _candidate_manifests(None, index) == {"pair": manifest}


def test_utility_decision_is_withheld_until_every_cluster_finishes() -> None:
    statistics = {
        "prediction_beats_fixed_boundary_mae": True,
        "boundary7_practical_gate_passed": True,
    }

    pending = _utility_decision(statistics, 99, 100)
    positive = _utility_decision(statistics, 100, 100)

    assert pending == {
        "study_complete": False,
        "prediction_utility_positive": None,
        "practical_utility_positive": None,
        "utility_inference_status": "pending",
    }
    assert positive["study_complete"] is True
    assert positive["prediction_utility_positive"] is True
    assert positive["practical_utility_positive"] is True
    assert positive["utility_inference_status"] == "positive"
