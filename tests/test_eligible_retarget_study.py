from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

from action_chunking.utility_artifacts import (
    select_primary_directions,
    strict_boolean,
    utility_decision,
    validate_completed_sweep,
)

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_eligible_retarget_study.py")
)
_candidate_manifests = _module["_candidate_manifests"]
_validate_execution_binding = _module["_validate_execution_binding"]


def test_strict_serialized_boolean_parser() -> None:
    assert strict_boolean("True") is True
    assert strict_boolean("False") is False


def test_selects_only_first_frozen_direction_per_scene_cluster() -> None:
    rows = [
        {"pair_id": "a", "new_side": "base", "cluster_id": "scene-1"},
        {"pair_id": "a", "new_side": "donor", "cluster_id": "scene-1"},
        {"pair_id": "b", "new_side": "base", "cluster_id": "scene-2"},
    ]

    selected, decisions = select_primary_directions(rows)

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


def test_utility_execution_binding_requires_full_commit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full lowercase code commit"):
        _validate_execution_binding(tmp_path / "output", "short")


def test_utility_decision_is_withheld_until_every_cluster_finishes() -> None:
    statistics = {
        "prediction_utility_gate_passed": True,
        "boundary7_practical_gate_passed": True,
        "adaptive_policy_utility_gate_passed": True,
    }

    pending = utility_decision(statistics, 99, 100)
    positive = utility_decision(statistics, 100, 100)

    assert pending == {
        "study_complete": False,
        "prediction_utility_positive": None,
        "practical_utility_positive": None,
        "adaptive_policy_utility_positive": None,
        "utility_inference_status": "pending",
    }
    assert positive["study_complete"] is True
    assert positive["prediction_utility_positive"] is True
    assert positive["practical_utility_positive"] is True
    assert positive["adaptive_policy_utility_positive"] is True
    assert positive["utility_inference_status"] == "positive"


def test_completed_sweep_requires_every_condition_and_exact_control() -> None:
    summary = {
        "schema_version": 1,
        "pair_id": "pair",
        "noise_seed": 0,
        "registered_boundaries": list(range(11)),
        "directions": 1,
        "source_summaries": 12,
        "all_initial_inputs_exact": True,
        "all_simulator_states_exact": True,
        "all_controller_replays_exact": True,
        "all_retargets_only_at_first_replan": True,
        "boundary_zero_continue_restart_actions_exact": True,
        "boundary_zero_continue_restart_behavior_exact": True,
    }

    validate_completed_sweep(summary, "pair")
    summary["boundary_zero_continue_restart_actions_exact"] = False
    with pytest.raises(ValueError, match="exact controls"):
        validate_completed_sweep(summary, "pair")
