from __future__ import annotations

import pytest

from action_chunking.utility_analysis import summarize_utility_jobs


def test_cluster_summary_reports_prediction_and_noninferiority_statistics() -> None:
    jobs = [
        _job("a", predicted=7, observed=7, boundary7=True, latency=180.0),
        _job("b", predicted=8, observed=7, boundary7=True, latency=200.0),
    ]

    result = summarize_utility_jobs(jobs, bootstrap_samples=200)

    assert result["analysis_unit"] == "independent_scene_cluster"
    assert result["prediction_exact_rate"] == 0.5
    assert result["prediction_within_one_rate"] == 1.0
    assert result["prediction_mean_absolute_error"] == 0.5
    assert result["prediction_fixed_boundary_baseline"] == 7
    assert result["prediction_fixed_boundary_baseline_exact_rate"] == 1.0
    assert result["prediction_fixed_boundary_baseline_mean_absolute_error"] == 0.0
    assert result["prediction_mae_difference_vs_fixed_boundary"] == 0.5
    assert result["prediction_beats_fixed_boundary_mae"] is False
    assert result["boundary7_paired_losses"] == 0
    assert result["boundary7_velocity_evaluation_counts_exact"] is True
    assert result["boundary7_post_event_velocity_evaluation_savings_fraction"] == 0.7
    assert result["boundary7_noninferior"] is False
    assert result["boundary7_first_chunk_old_events"] == 0
    assert result["wrong_target_failure_first_contact_replan_histogram"] == {"0": 6}


def test_cluster_summary_rejects_direction_pseudoreplication() -> None:
    with pytest.raises(ValueError, match="one direction per cluster"):
        summarize_utility_jobs([_job("a"), _job("a")], bootstrap_samples=10)


def test_cluster_summary_requires_valid_fixed_boundary_baseline() -> None:
    with pytest.raises(ValueError, match=r"baseline must lie in 0\.\.10"):
        summarize_utility_jobs([_job("a")], fixed_boundary_baseline=11)


def test_cluster_summary_detects_prediction_advantage_over_fixed_boundary() -> None:
    jobs = [
        _job(f"cluster-{index}", predicted=observed, observed=observed)
        for index, observed in enumerate([2, 3, 4, 5, 9, 10] * 3)
    ]

    result = summarize_utility_jobs(jobs, bootstrap_samples=1_000)

    assert result["prediction_mean_absolute_error"] == 0.0
    assert result["prediction_fixed_boundary_baseline_mean_absolute_error"] > 0.0
    assert result["prediction_mae_difference_vs_fixed_boundary_ci95"][1] < 0.0
    assert result["prediction_beats_fixed_boundary_mae"] is True


def _job(
    cluster: str,
    *,
    predicted: int = 7,
    observed: int = 7,
    boundary7: bool = True,
    latency: float = 180.0,
) -> dict:
    curve = [boundary <= observed for boundary in range(11)]
    old_target_first = [not value for value in curve]
    return {
        "cluster_id": cluster,
        "prediction_valid": True,
        "predicted_last_successful_boundary": predicted,
        "observed_last_successful_boundary": observed,
        "prediction_exact": predicted == observed,
        "success_curve": curve,
        "first_chunk_old_event_curve": old_target_first,
        "old_target_first_curve": old_target_first,
        "clean_replanning_rescue_curve": [False] * 11,
        "first_contact_replan_index_curve": [1 if value else 0 for value in curve],
        "boundary7_new_target_first": boundary7,
        "boundary7_new_task_success": boundary7,
        "boundary7_first_chunk_old_event": False,
        "boundary7_clean_replanning_rescue": False,
        "boundary7_post_event_velocity_evaluations": 3,
        "boundary7_post_event_total_ms": latency,
        "restart_new_target_first": True,
        "restart_new_task_success": True,
        "restart_first_chunk_old_event": False,
        "restart_clean_replanning_rescue": False,
        "restart_post_event_velocity_evaluations": 10,
        "restart_post_event_total_ms": 400.0,
    }
