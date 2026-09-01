from __future__ import annotations

import runpy

import numpy as np
import pytest

analysis = runpy.run_path("scripts/analyze_early_exit_confirmation.py")
runner = runpy.run_path("scripts/run_early_exit_suite_confirmation.py")
analyze_confirmation = analysis["analyze_confirmation"]


def test_confirmation_passes_with_four_paired_losses() -> None:
    rows, summary = analyze_confirmation(_pairs(losses=4))

    assert len(rows) == 500
    assert summary["paired_losses"] == 4
    assert summary["paired_loss_clopper_pearson_upper95"] < 0.02
    assert summary["condition_order_counts"] == {
        "early_exit_first": 250,
        "full_control_first": 250,
    }
    assert summary["velocity_evaluation_savings_fraction"] == 0.3
    assert summary["median_first_replan_latency_savings_fraction"] == 0.3
    assert summary["confirmation_positive"] is True


def test_confirmation_fails_with_five_paired_losses() -> None:
    _, summary = analyze_confirmation(_pairs(losses=5))

    assert summary["paired_losses"] == 5
    assert summary["paired_loss_clopper_pearson_upper95"] > 0.02
    assert summary["confirmation_positive"] is False


def test_confirmation_rejects_nonexact_initial_pair() -> None:
    pairs = _pairs(losses=0)
    pairs[0]["early_exit_7"]["initial_input_sha256"]["image"] = "different"

    with pytest.raises(ValueError, match="initial inputs differ"):
        analyze_confirmation(pairs)


def test_runner_noise_and_condition_order_are_deterministic_and_balanced() -> None:
    first = runner["_noise_for_replan"](0, 2, 3, 4)
    second = runner["_noise_for_replan"](0, 2, 3, 4)
    other = runner["_noise_for_replan"](0, 2, 3, 5)
    assert first.dtype == np.float32
    assert first.shape == (10, 32)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    orders = [
        runner["_condition_order"](task_id, trial_index)[0]
        for task_id in range(10)
        for trial_index in range(50)
    ]
    assert orders.count("early_exit_7") == 250
    assert orders.count("full_control_10") == 250


def _pairs(*, losses: int) -> list[dict]:
    pairs = []
    loss_keys = {(0, trial) for trial in range(losses)}
    for task_id in range(10):
        for trial_index in range(50):
            early_success = (task_id, trial_index) not in loss_keys
            early = _condition(
                "early_exit_7", 7, early_success, 7.0, task_id, trial_index
            )
            full = _condition(
                "full_control_10", 10, True, 10.0, task_id, trial_index
            )
            pairs.append(
                {
                    "schema_version": 1,
                    "suite": "libero_goal",
                    "task_id": task_id,
                    "trial_index": trial_index,
                    "pair_key": f"task_{task_id:02d}_trial_{trial_index:02d}",
                    "condition_order": analysis["_condition_order"](
                        task_id, trial_index
                    ),
                    "order_digest_sha256": analysis["_order_digest"](
                        task_id, trial_index
                    ),
                    "initial_inputs_exact": True,
                    "initial_sim_state_exact": True,
                    "shared_noise_common_replans": 1,
                    "shared_noise_exact": True,
                    "early_exit_7": early,
                    "full_control_10": full,
                    "paired_loss": not early_success,
                }
            )
    return pairs


def _condition(
    name: str,
    after_steps: int,
    success: bool,
    integration_ms: float,
    task_id: int,
    trial_index: int,
) -> dict:
    savings = 10 - after_steps
    noise = analysis["_noise_for_replan"](0, task_id, trial_index, 0)
    noise_hash = analysis["_array_digest"](noise)
    return {
        "condition": name,
        "environment_seed": 7,
        "noise_seed": 0,
        "after_steps": after_steps,
        "total_flow_steps": 10,
        "success": success,
        "replans": 1,
        "initial_input_sha256": {
            "observation/image": "same",
            "observation/state": "same",
            "observation/wrist_image": "same",
            "prompt": "same",
        },
        "initial_state_fixture_sha256": "same-fixture",
        "initial_sim_state_sha256": "same",
        "noise_sha256_by_replan": [noise_hash],
        "action_sha256_by_replan": [f"action-{name}-{task_id}-{trial_index}"],
        "early_exit_diagnostics": [
            {
                "replan_index": 0,
                "after_steps": after_steps,
                "total_flow_steps": 10,
                "velocity_field_evaluations": after_steps,
                "velocity_field_evaluation_savings": savings,
                "velocity_field_evaluation_savings_fraction": savings / 10,
                "integration_ms": integration_ms,
            }
        ],
    }
