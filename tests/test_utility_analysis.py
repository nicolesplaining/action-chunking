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
    assert result["prediction_sample_size_gate_passed"] is False
    assert result["prediction_fixed_boundary_baseline"] == 7
    assert result["prediction_fixed_boundary_baseline_exact_rate"] == 1.0
    assert result["prediction_fixed_boundary_baseline_mean_absolute_error"] == 0.0
    assert result["prediction_mae_difference_vs_fixed_boundary"] == 0.5
    assert result["prediction_beats_fixed_boundary_mae"] is False
    assert result["boundary7_paired_losses"] == 0
    assert result["boundary7_target_first_paired_losses"] == 0
    assert result["boundary7_task_success_paired_losses"] == 0
    assert result["boundary7_composite_paired_losses"] == 0
    assert result["boundary7_velocity_evaluation_counts_exact"] is True
    assert result["boundary7_post_event_velocity_evaluation_savings_fraction"] == 0.7
    assert result["boundary7_noninferior"] is False
    assert result["boundary7_target_first_noninferior"] is False
    assert result["boundary7_task_success_noninferior"] is False
    assert result["boundary7_practical_gate_passed"] is False
    assert result["boundary7_first_chunk_old_events"] == 0
    assert result["wrong_target_failure_first_contact_replan_histogram"] == {"0": 6}


def test_cluster_summary_rejects_direction_pseudoreplication() -> None:
    with pytest.raises(ValueError, match="one direction per cluster"):
        summarize_utility_jobs([_job("a"), _job("a")], bootstrap_samples=10)


def test_cluster_summary_requires_valid_fixed_boundary_baseline() -> None:
    with pytest.raises(ValueError, match=r"baseline must lie in 0\.\.10"):
        summarize_utility_jobs([_job("a")], fixed_boundary_baseline=11)


def test_cluster_summary_reconstructs_serialized_prediction_targets() -> None:
    job = _job("a", predicted=7, observed=7)
    job["observed_last_successful_boundary"] = 8
    with pytest.raises(ValueError, match="differs from the success curve"):
        summarize_utility_jobs([job], bootstrap_samples=10)

    job = _job("a", predicted=7, observed=7)
    job["prediction_exact"] = False
    with pytest.raises(ValueError, match="prediction-exact"):
        summarize_utility_jobs([job], bootstrap_samples=10)


def test_cluster_summary_rejects_curve_and_registered_boundary_mismatch() -> None:
    job = _job("a", predicted=7, observed=7)
    job["boundary7_new_task_success"] = False
    with pytest.raises(ValueError, match="boundary-zero or boundary-seven"):
        summarize_utility_jobs([job], bootstrap_samples=10)


def test_cluster_summary_detects_prediction_advantage_over_fixed_boundary() -> None:
    jobs = [
        _job(f"cluster-{index}", predicted=observed, observed=observed)
        for index, observed in enumerate([2, 3, 4, 5, 9, 10] * 10)
    ]

    result = summarize_utility_jobs(
        jobs,
        bootstrap_samples=1_000,
        minimum_valid_predictions=len(jobs),
    )

    assert result["prediction_mean_absolute_error"] == 0.0
    assert result["prediction_fixed_boundary_baseline_mean_absolute_error"] > 0.0
    assert result["prediction_mae_difference_vs_fixed_boundary_ci95"][1] < 0.0
    assert result["prediction_beats_fixed_boundary_mae"] is True
    assert result["prediction_spearman_rho"] == 1.0
    assert result["prediction_rank_association_positive"] is True
    assert result["prediction_selected_boundary_noninferior"] is True
    assert result["prediction_utility_gate_passed"] is True


def test_prediction_advantage_requires_frozen_sample_size() -> None:
    jobs = [_job(f"cluster-{index}", predicted=2, observed=2) for index in range(10)]

    result = summarize_utility_jobs(jobs, bootstrap_samples=100)

    assert result["prediction_mae_difference_vs_fixed_boundary_ci95"][1] < 0.0
    assert result["prediction_sample_size_gate_passed"] is False
    assert result["prediction_beats_fixed_boundary_mae"] is False


def test_prediction_utility_requires_chosen_boundary_behavior() -> None:
    jobs = [
        _job(f"cluster-{index}", predicted=9, observed=10)
        for index in range(59)
    ]
    for job in jobs:
        job["success_curve"][9] = False
        job["old_target_first_curve"][9] = True
        job["first_chunk_old_event_curve"][9] = True

    result = summarize_utility_jobs(
        jobs,
        bootstrap_samples=1_000,
        minimum_valid_predictions=len(jobs),
    )

    assert result["prediction_beats_fixed_boundary_mae"] is True
    assert result["prediction_rank_association_positive"] is False
    assert result["prediction_selected_boundary_paired_losses"] == len(jobs)
    assert result["prediction_selected_boundary_noninferior"] is False
    assert result["prediction_utility_gate_passed"] is False


def test_total_behavioral_failure_remains_in_prediction_denominator() -> None:
    result = summarize_utility_jobs(
        [_job("failed", predicted=0, observed=None)],
        bootstrap_samples=100,
        minimum_valid_predictions=1,
    )

    assert result["behavioral_boundary_no_success_sentinel"] == -1
    assert result["valid_predictions_with_no_successful_boundary"] == 1
    assert result["prediction_mean_absolute_error"] == 1.0
    assert result["prediction_fixed_boundary_baseline_mean_absolute_error"] == 8.0
    assert result["predicted_boundary_composite_success_rate"] == 0.0


def test_practical_gate_requires_each_behavioral_outcome_and_latency() -> None:
    jobs = [_job(f"cluster-{index}") for index in range(100)]
    passing = summarize_utility_jobs(jobs, bootstrap_samples=1_000)

    assert passing["boundary7_target_first_noninferior"] is True
    assert passing["boundary7_task_success_noninferior"] is True
    assert passing["boundary7_composite_noninferior"] is True
    assert passing["boundary7_latency_savings_positive"] is True
    assert passing["boundary7_practical_gate_passed"] is True

    jobs[0]["boundary7_new_target_first"] = False
    jobs[0]["boundary7_new_task_success"] = True
    jobs[0]["success_curve"][7] = False
    jobs[0]["observed_last_successful_boundary"] = 6
    jobs[0]["prediction_exact"] = False
    failing = summarize_utility_jobs(
        jobs,
        bootstrap_samples=1_000,
        noninferiority_margin=0.04,
    )

    assert failing["boundary7_target_first_noninferior"] is False
    assert failing["boundary7_task_success_noninferior"] is True
    assert failing["boundary7_composite_noninferior"] is False
    assert failing["boundary7_practical_gate_passed"] is False


def _job(
    cluster: str,
    *,
    predicted: int = 7,
    observed: int | None = 7,
    boundary7: bool | None = None,
    latency: float = 180.0,
) -> dict:
    curve = (
        [False] * 11
        if observed is None
        else [boundary <= observed for boundary in range(11)]
    )
    old_target_first = [not value for value in curve]
    boundary7 = curve[7] if boundary7 is None else boundary7
    return {
        "cluster_id": cluster,
        "prediction_valid": True,
        "predicted_last_successful_boundary": predicted,
        "observed_last_successful_boundary": observed,
        "prediction_exact": observed is not None and predicted == observed,
        "success_curve": curve,
        "first_chunk_old_event_curve": old_target_first,
        "old_target_first_curve": old_target_first,
        "clean_replanning_rescue_curve": [False] * 11,
        "first_contact_replan_index_curve": [1 if value else 0 for value in curve],
        "boundary7_new_target_first": boundary7,
        "boundary7_new_task_success": boundary7,
        "boundary7_first_chunk_old_event": old_target_first[7],
        "boundary7_clean_replanning_rescue": False,
        "boundary7_post_event_velocity_evaluations": 3,
        "boundary7_post_event_total_ms": latency,
        "restart_new_target_first": curve[0],
        "restart_new_task_success": curve[0],
        "restart_first_chunk_old_event": old_target_first[0],
        "restart_clean_replanning_rescue": False,
        "restart_post_event_velocity_evaluations": 10,
        "restart_post_event_total_ms": 400.0,
    }
