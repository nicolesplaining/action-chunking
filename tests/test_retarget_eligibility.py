from __future__ import annotations

from action_chunking.retarget_eligibility import eligibility_row


def test_eligibility_requires_old_event_and_clean_restart_avoidance() -> None:
    event = _summary(old_step=4, restart_old_step=None, success=False)
    row = eligibility_row(_entry(), event, 5, _summary(old_step=4, restart_old_step=None))

    assert row["old_event_induced"] is True
    assert row["restart_avoids_old_event"] is True
    assert row["restart_new_target_first"] is True
    assert row["eligible"] is True


def test_eligibility_rejects_restart_that_contacts_old_target_within_horizon() -> None:
    row = eligibility_row(_entry(), _summary(old_step=4, restart_old_step=5), 5)

    assert row["restart_avoids_old_event"] is False
    assert row["eligible"] is False


def test_eligibility_rejects_old_event_after_execution_horizon() -> None:
    row = eligibility_row(_entry(), _summary(old_step=6, restart_old_step=None), 5)

    assert row["old_event_induced"] is False
    assert row["eligible"] is False


def test_event_gate_does_not_claim_eligibility_before_competence_run() -> None:
    row = eligibility_row(_entry(), _summary(old_step=4, restart_old_step=None, success=False), 5)

    assert row["event_gate_pass"] is True
    assert row["competence_run_completed"] is False
    assert row["eligible"] is False


def _entry() -> dict:
    return {
        "pair_id": "pair_precontact_base_010",
        "source_pair_id": "pair",
        "origin_side": "base",
        "snapshot_step": 10,
        "precontact_offset_steps": 4,
        "base_target": "wine",
        "donor_target": "bowl",
    }


def _summary(old_step: int, restart_old_step: int | None, success: bool = True) -> dict:
    restart_contacts = {"bowl": 8}
    if restart_old_step is not None:
        restart_contacts["wine"] = restart_old_step
    return {
        "pair_id": "pair_precontact_base_010",
        "noise_seed": 0,
        "results": [
            _result("base", {"wine": old_step}, success),
            _result("donor", restart_contacts, success),
        ],
    }


def _result(side: str, contacts: dict[str, int], success: bool) -> dict:
    return {
        "side": side,
        "success": success,
        "first_contact_step_by_object": contacts,
        "live_initial_input_diagnostics": {
            "observation/image": {"array_equal": True},
            "observation/wrist_image": {"array_equal": True},
            "observation/state": {"array_equal": True},
        },
        "restored_sim_state_max_abs_error": 0.0,
    }
