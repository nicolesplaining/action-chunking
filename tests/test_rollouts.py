from __future__ import annotations

from action_chunking.rollouts import paired_rollout_rows, paired_rollout_summary


def test_paired_rollout_summary_preserves_pairing_and_first_contact() -> None:
    summary = {
        "pair_id": "pair_0",
        "noise_seed": 3,
        "both_successful": False,
        "results": [
            {
                "side": "base",
                "target": "mug_a",
                "success": True,
                "steps": 100,
                "first_chunk_max_abs_error": 0.0,
                "first_contact_step_by_object": {"mug_a": 20},
                "initial_input_mode": "strict",
                "live_initial_input_diagnostics": {"observation/image": {"array_equal": True}},
                "restored_sim_state_max_abs_error": 0.0,
            },
            {
                "side": "donor",
                "target": "mug_b",
                "success": False,
                "steps": 400,
                "first_chunk_max_abs_error": 0.0,
                "first_contact_step_by_object": {"mug_a": 18, "mug_b": 30},
                "initial_input_mode": "strict",
                "live_initial_input_diagnostics": {"observation/image": {"array_equal": True}},
                "restored_sim_state_max_abs_error": 0.0,
            },
        ],
    }
    rows = paired_rollout_rows([summary])
    aggregate = paired_rollout_summary(rows)
    assert rows[0]["first_contact_is_target"] is True
    assert rows[1]["first_contact_object"] == "mug_a"
    assert rows[1]["first_contact_is_target"] is False
    assert aggregate["paired_noise_jobs"] == 1
    assert aggregate["paired_noise_jobs_both_successful"] == 0
    assert aggregate["successful_side_rollouts"] == 1
    assert aggregate["all_first_chunks_exact"] is True
    assert aggregate["paired_noise_jobs_both_first_contacts_target"] == 0
    assert aggregate["paired_noise_jobs_strictly_eligible"] == 0
    assert aggregate["initial_input_exact_side_rollouts"] == 2
    assert aggregate["simulator_state_exact_side_rollouts"] == 2
