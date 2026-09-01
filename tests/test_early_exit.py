from __future__ import annotations

import json
import runpy
from types import SimpleNamespace

import pytest

_job_record = runpy.run_path("scripts/run_manifest_pair_validations.py")["_job_record"]


def test_early_exit_validation_requires_exact_compute_accounting(tmp_path) -> None:
    intervention = tmp_path / "early_exit.json"
    intervention.write_text(
        json.dumps(
            {
                "after_steps": 7,
                "family": "early_exit",
                "schema_version": 1,
                "total_flow_steps": 10,
            }
        )
    )
    args = SimpleNamespace(
        intervention=intervention,
        intervene_replans="all",
        noise_seed=0,
        save_sim_states=False,
    )

    job = _job_record(_entry(), _summary(), tmp_path, args)

    assert job["exact_dual_success_target_first"] is True
    assert job["early_exit_compute_exact"] is True


def test_early_exit_validation_rejects_wrong_evaluation_count(tmp_path) -> None:
    intervention = tmp_path / "early_exit.json"
    intervention.write_text(
        json.dumps(
            {
                "after_steps": 7,
                "family": "early_exit",
                "schema_version": 1,
                "total_flow_steps": 10,
            }
        )
    )
    args = SimpleNamespace(
        intervention=intervention,
        intervene_replans="all",
        noise_seed=0,
        save_sim_states=False,
    )
    summary = _summary()
    summary["results"][0]["early_exit_diagnostics"][0][
        "velocity_field_evaluations"
    ] = 8

    with pytest.raises(ValueError, match="exact compute accounting"):
        _job_record(_entry(), summary, tmp_path, args)


def _entry() -> dict:
    return {
        "pair_id": "pair",
        "init_index": 0,
        "base_target": "mug",
        "donor_target": "bowl",
    }


def _summary() -> dict:
    intervention = {
        "after_steps": 7,
        "family": "early_exit",
        "schema_version": 1,
        "total_flow_steps": 10,
    }
    return {
        "pair_id": "pair",
        "noise_seed": 0,
        "both_successful": True,
        "intervention": intervention,
        "intervene_replans": "all",
        "results": [
            _result("base", "mug"),
            _result("donor", "bowl"),
        ],
    }


def _result(side: str, target: str) -> dict:
    diagnostic = {
        "after_steps": 7,
        "total_flow_steps": 10,
        "velocity_field_evaluations": 7,
        "velocity_field_evaluation_savings": 3,
        "velocity_field_evaluation_savings_fraction": 0.3,
    }
    return {
        "side": side,
        "success": True,
        "restored_sim_state_max_abs_error": 0.0,
        "live_initial_input_diagnostics": {"image": {"array_equal": True}},
        "first_contact_step_by_object": {target: 5},
        "intervention_replans_applied": [0, 1],
        "early_exit_diagnostics": [
            {"replan_index": 0, **diagnostic},
            {"replan_index": 1, **diagnostic},
        ],
    }
