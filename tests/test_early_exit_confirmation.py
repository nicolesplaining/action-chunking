from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

analysis = runpy.run_path("scripts/analyze_early_exit_confirmation.py")
runner = runpy.run_path("scripts/run_early_exit_suite_confirmation.py")
analyze_confirmation = analysis["analyze_confirmation"]
CODE_COMMIT = "a" * 40


def test_confirmation_passes_with_four_paired_losses() -> None:
    rows, summary = analyze_confirmation(_pairs(losses=4))

    assert len(rows) == 500
    assert summary["episode_pairs"] == 500
    assert summary["episodes_per_condition"] == 500
    assert summary["condition_rollouts"] == 1000
    assert len(summary["per_task"]) == 10
    assert summary["per_task"][0]["paired_losses"] == 4
    assert all(
        task["condition_order_counts"] == {"early_exit_first": 25, "full_control_first": 25}
        for task in summary["per_task"]
    )
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


def test_confirmation_fails_without_positive_latency_interval() -> None:
    pairs = _pairs(losses=0)
    for index, pair in enumerate(pairs):
        pair["early_exit_7"]["early_exit_diagnostics"][0]["integration_ms"] = (
            11.0 if index < 300 else 7.0
        )

    _, summary = analyze_confirmation(pairs)

    assert summary["paired_loss_clopper_pearson_upper95"] < 0.02
    assert summary["median_first_replan_latency_savings_fraction_bootstrap_ci95"][0] <= 0.0
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
    orders = [runner["_condition_order"](task_id, trial_index)[0] for task_id in range(10) for trial_index in range(50)]
    assert orders.count("early_exit_7") == 250
    assert orders.count("full_control_10") == 250


def test_runner_rejects_resuming_pair_from_another_commit() -> None:
    pair = _pairs(losses=0)[0]

    with pytest.raises(ValueError, match="incompatible"):
        runner["_validate_existing_pair"](pair, 0, 0, "b" * 40)


def test_runner_rejects_warmup_from_another_commit(tmp_path) -> None:
    warmup = {
        "schema_version": 1,
        "scored": False,
        "session_index": 0,
        "code_commit": CODE_COMMIT,
        "records": [
            {
                "condition": name,
                "diagnostic": {
                    "after_steps": after_steps,
                    "total_flow_steps": 10,
                    "velocity_field_evaluations": after_steps,
                    "velocity_field_evaluation_savings": 10 - after_steps,
                    "velocity_field_evaluation_savings_fraction": (10 - after_steps) / 10,
                    "integration_ms": 1.0,
                },
            }
            for name, after_steps in (
                ("full_control_10", 10),
                ("early_exit_7", 7),
            )
        ],
    }
    (tmp_path / "warmup_sessions.jsonl").write_text(json.dumps(warmup) + "\n")

    with pytest.raises(ValueError, match="warm-up session log"):
        runner["_existing_warmup_sessions"](tmp_path, "b" * 40)


def test_runner_rejects_full_control_without_exact_sampler_flag() -> None:
    diagnostic = _condition("full_control_10", 10, True, 1.0, 0, 0)["early_exit_diagnostics"][0]
    diagnostic["full_step_output_exact"] = False

    with pytest.raises(ValueError, match="exact compute or latency"):
        runner["_validate_diagnostic"](diagnostic, 10)


def test_independent_analysis_rejects_progress_counter_drift() -> None:
    rows, summary = analyze_confirmation(_pairs(losses=4))
    progress = _progress(rows, summary)
    progress["paired_losses_so_far"] = 3

    with pytest.raises(ValueError, match="progress counters"):
        analysis["_validate_progress"](progress, rows, summary)


def test_independent_analysis_validates_run_binding(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "pilot_positive": True,
                "eligible_scene_clusters": 15,
                "composite_preserved_clusters": 14,
                "all_compute_counts_exact": True,
            }
        )
    )
    digest = hashlib.sha256(pilot.read_bytes()).hexdigest()
    (tmp_path / "pilot_summary.sha256").write_text(f"{digest}  {pilot}\n")
    (tmp_path / "code_commit.txt").write_text(CODE_COMMIT + "\n")
    (tmp_path / "gpu_preflight.csv").write_text(
        "index, uuid, name, driver_version, memory.total [MiB]\n"
        "0, GPU-a, NVIDIA H100 80GB HBM3, 570.1, 81559 MiB\n"
        "1, GPU-b, NVIDIA H100 80GB HBM3, 570.1, 81559 MiB\n"
    )

    analysis["_validate_run_binding"](tmp_path, CODE_COMMIT)

    (tmp_path / "code_commit.txt").write_text("b" * 40 + "\n")
    with pytest.raises(ValueError, match="code-commit binding"):
        analysis["_validate_run_binding"](tmp_path, CODE_COMMIT)


def test_independent_analysis_reconciles_condition_files(tmp_path: Path) -> None:
    pair = _pairs(losses=0)[0]
    for condition in ("early_exit_7", "full_control_10"):
        (tmp_path / f"{condition}.json").write_text(json.dumps(pair[condition]))

    analysis["_validate_condition_files"](tmp_path, pair)

    changed = dict(pair["early_exit_7"])
    changed["success"] = False
    (tmp_path / "early_exit_7.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from pair summary"):
        analysis["_validate_condition_files"](tmp_path, pair)


def _pairs(*, losses: int) -> list[dict]:
    pairs = []
    loss_keys = {(0, trial) for trial in range(losses)}
    for task_id in range(10):
        for trial_index in range(50):
            early_success = (task_id, trial_index) not in loss_keys
            early = _condition("early_exit_7", 7, early_success, 7.0, task_id, trial_index)
            full = _condition("full_control_10", 10, True, 10.0, task_id, trial_index)
            pairs.append(
                {
                    "schema_version": 1,
                    "suite": "libero_goal",
                    "code_commit": CODE_COMMIT,
                    "task_id": task_id,
                    "trial_index": trial_index,
                    "pair_key": f"task_{task_id:02d}_trial_{trial_index:02d}",
                    "task_description": f"task {task_id}",
                    "condition_order": analysis["_condition_order"](task_id, trial_index),
                    "order_digest_sha256": analysis["_order_digest"](task_id, trial_index),
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
        "code_commit": CODE_COMMIT,
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
                "full_step_output_exact": after_steps == 10,
                "full_step_estimate_max_abs_error": (1e-7 if after_steps == 10 else None),
            }
        ],
    }


def _progress(rows: list[dict], summary: dict) -> dict:
    return {
        "completed_pairs": 500,
        "completed_condition_rollouts": 1000,
        "early_exit_successes_so_far": summary["early_exit_successes"],
        "full_control_successes_so_far": summary["full_control_successes"],
        "paired_losses_so_far": summary["paired_losses"],
        "jobs": [
            {
                "pair_key": row["pair_key"],
                "task_id": row["task_id"],
                "trial_index": row["trial_index"],
                "condition_order": row["condition_order"],
                "early_exit_success": row["early_exit_success"],
                "full_control_success": row["full_control_success"],
                "paired_loss": row["paired_loss"],
                "pair_summary": f"/data/pairs/{row['pair_key']}/pair_summary.json",
            }
            for row in rows
        ],
    }
