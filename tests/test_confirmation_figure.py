from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from action_chunking.noninferiority import binomial_upper_bound

_module = runpy.run_path("scripts/make_confirmation_figure.py")
validate = _module["validate_confirmation_summary"]
make_figure = _module["make_confirmation_figure"]


def test_confirmation_figure_reconstructs_all_paired_counts(tmp_path: Path) -> None:
    summary = _summary(losses=4, positive=True)
    audited = validate(summary)

    assert audited["early_exit_successes"] == 496
    assert audited["full_control_successes"] == 500
    assert audited["paired_losses"] == 4
    assert audited["paired_gains"] == 0
    assert audited["median_latency"] == pytest.approx(0.3)

    make_figure(summary, tmp_path)
    assert (tmp_path / "fig_early_exit_confirmation.pdf").is_file()
    assert (tmp_path / "fig_early_exit_confirmation.png").is_file()


def test_confirmation_figure_rejects_edited_aggregate() -> None:
    summary = _summary(losses=4, positive=True)
    summary["paired_losses"] = 3

    with pytest.raises(ValueError, match="counts differ from rows"):
        validate(summary)


def test_confirmation_figure_keeps_negative_confirmation_reportable(tmp_path: Path) -> None:
    summary = _summary(losses=5, positive=False)

    audited = make_figure(summary, tmp_path)

    assert audited["paired_losses"] == 5
    assert (tmp_path / "fig_early_exit_confirmation.pdf").is_file()


def _summary(*, losses: int, positive: bool) -> dict:
    rows = []
    tasks = []
    for task_id in range(10):
        task_losses = losses if task_id == 0 else 0
        task_rows = []
        for trial_index in range(50):
            early = not (task_id == 0 and trial_index < task_losses)
            row = {
                "task_id": task_id,
                "trial_index": trial_index,
                "early_exit_success": early,
                "full_control_success": True,
                "paired_loss": not early,
                "paired_gain": False,
                "first_replan_latency_savings_fraction": 0.3,
            }
            rows.append(row)
            task_rows.append(row)
        tasks.append(
            {
                "task_id": task_id,
                "trials": 50,
                "early_exit_successes": sum(row["early_exit_success"] for row in task_rows),
                "full_control_successes": 50,
                "paired_losses": task_losses,
                "paired_gains": 0,
                "condition_order_counts": {
                    "early_exit_first": 25,
                    "full_control_first": 25,
                },
            }
        )
    return {
        "schema_version": 1,
        "analysis_unit": "paired_libero_episode_pair",
        "suite": "libero_goal",
        "episode_pairs": 500,
        "episodes_per_condition": 500,
        "condition_rollouts": 1000,
        "condition_order_counts": {
            "early_exit_first": 250,
            "full_control_first": 250,
        },
        "paired_loss_margin": 0.02,
        "maximum_passing_losses": 4,
        "all_compute_counts_exact": True,
        "velocity_evaluation_savings_fraction": 0.3,
        "confirmation_positive": positive,
        "early_exit_successes": 500 - losses,
        "full_control_successes": 500,
        "paired_losses": losses,
        "paired_gains": 0,
        "paired_loss_clopper_pearson_upper95": binomial_upper_bound(losses, 500),
        "median_first_replan_latency_savings_fraction": 0.3,
        "median_first_replan_latency_savings_fraction_bootstrap_ci95": [0.29, 0.31],
        "per_task": tasks,
        "rows": rows,
    }
