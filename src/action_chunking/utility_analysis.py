"""Cluster-level analysis for the frozen retargeting-utility experiment."""

from __future__ import annotations

from typing import Any

import numpy as np

from action_chunking.noninferiority import binomial_upper_bound


def summarize_utility_jobs(
    jobs: list[dict[str, Any]],
    *,
    fixed_boundary_baseline: int = 7,
    noninferiority_margin: float = 0.05,
    alpha: float = 0.05,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    minimum_valid_predictions: int = 59,
) -> dict[str, Any]:
    """Summarize one frozen direction per independent scene cluster."""
    if not 0 <= fixed_boundary_baseline <= 10:
        raise ValueError("fixed boundary baseline must lie in 0..10")
    if not 0.0 < noninferiority_margin < 1.0:
        raise ValueError("noninferiority margin must lie strictly between zero and one")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if minimum_valid_predictions <= 0:
        raise ValueError("minimum valid predictions must be positive")
    cluster_ids = [str(job["cluster_id"]) for job in jobs]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("primary utility jobs must contain at most one direction per cluster")
    _validate_derived_job_fields(jobs)

    valid = [job for job in jobs if job["prediction_valid"]]
    predicted = np.asarray([int(job["predicted_last_successful_boundary"]) for job in valid], dtype=np.float64)
    observed = np.asarray(
        [
            -1
            if job["observed_last_successful_boundary"] is None
            else int(job["observed_last_successful_boundary"])
            for job in valid
        ],
        dtype=np.float64,
    )
    errors = predicted - observed
    baseline_errors = fixed_boundary_baseline - observed
    absolute_error_advantage = np.abs(errors) - np.abs(baseline_errors)
    absolute_error_advantage_interval = _cluster_bootstrap_mean(
        absolute_error_advantage, bootstrap_samples, bootstrap_seed
    )
    prediction_sample_size_gate_passed = len(valid) >= minimum_valid_predictions
    prediction_spearman_rho = _spearman(predicted, observed)
    prediction_spearman_p = (
        _spearman_permutation_p(
            predicted,
            observed,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )
        if prediction_sample_size_gate_passed
        else None
    )
    prediction_rank_association_positive = bool(
        prediction_sample_size_gate_passed
        and prediction_spearman_rho is not None
        and prediction_spearman_rho > 0.0
        and prediction_spearman_p is not None
        and prediction_spearman_p < alpha
    )
    predicted_boundary_success = [
        bool(job["success_curve"][int(job["predicted_last_successful_boundary"])])
        for job in valid
    ]
    fixed_boundary_success = [
        bool(job["success_curve"][fixed_boundary_baseline]) for job in valid
    ]
    prediction_selection_noninferiority = _paired_loss_noninferiority(
        fixed_boundary_success,
        predicted_boundary_success,
        margin=noninferiority_margin,
        alpha=alpha,
    )

    outcome_pairs = {
        "target_first": (
            [bool(job["restart_new_target_first"]) for job in jobs],
            [bool(job["boundary7_new_target_first"]) for job in jobs],
        ),
        "task_success": (
            [bool(job["restart_new_task_success"]) for job in jobs],
            [bool(job["boundary7_new_task_success"]) for job in jobs],
        ),
        "composite": (
            [bool(job["restart_new_target_first"] and job["restart_new_task_success"]) for job in jobs],
            [bool(job["boundary7_new_target_first"] and job["boundary7_new_task_success"]) for job in jobs],
        ),
    }
    noninferiority = {
        name: _paired_loss_noninferiority(
            restart,
            continued,
            margin=noninferiority_margin,
            alpha=alpha,
        )
        for name, (restart, continued) in outcome_pairs.items()
    }

    latency_savings = np.asarray(
        [
            (float(job["restart_post_event_total_ms"]) - float(job["boundary7_post_event_total_ms"]))
            / float(job["restart_post_event_total_ms"])
            for job in jobs
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(latency_savings)) or np.any(
        np.asarray([job["restart_post_event_total_ms"] for job in jobs], dtype=np.float64) <= 0
    ):
        raise ValueError("post-event latency must be finite and strictly positive")
    latency_interval = _cluster_bootstrap_median(latency_savings, bootstrap_samples, bootstrap_seed)
    expected_boundary7_evaluations = 3
    evaluation_counts_exact = all(
        int(job["boundary7_post_event_velocity_evaluations"]) == expected_boundary7_evaluations
        and int(job["restart_post_event_velocity_evaluations"]) == 10
        for job in jobs
    )
    latency_positive = bool(latency_interval is not None and latency_interval[0] > 0.0)
    practical_gate_passed = bool(
        jobs
        and all(result["noninferior"] for result in noninferiority.values())
        and evaluation_counts_exact
        and latency_positive
    )

    next_boundary_failure = []
    predicted_boundary_first_chunk_old_event = []
    next_boundary_first_chunk_old_event = []
    for job in valid:
        boundary = int(job["predicted_last_successful_boundary"])
        curve = [bool(value) for value in job["success_curve"]]
        first_chunk_old_events = [bool(value) for value in job["first_chunk_old_event_curve"]]
        predicted_boundary_first_chunk_old_event.append(first_chunk_old_events[boundary])
        if boundary < 10:
            next_boundary_failure.append(not curve[boundary + 1])
            next_boundary_first_chunk_old_event.append(first_chunk_old_events[boundary + 1])

    wrong_target_failure_replan_histogram: dict[str, int] = {}
    eventual_failures_after_new_target_first = 0
    for job in jobs:
        for success, old_first, replan_index in zip(
            job["success_curve"],
            job["old_target_first_curve"],
            job["first_contact_replan_index_curve"],
            strict=True,
        ):
            if old_first:
                key = "none" if replan_index is None else str(int(replan_index))
                wrong_target_failure_replan_histogram[key] = wrong_target_failure_replan_histogram.get(key, 0) + 1
            elif not success:
                eventual_failures_after_new_target_first += 1

    prediction_beats_fixed_boundary_mae = bool(
        prediction_sample_size_gate_passed
        and absolute_error_advantage_interval is not None
        and absolute_error_advantage_interval[1] < 0.0
    )
    prediction_utility_gate_passed = bool(
        prediction_beats_fixed_boundary_mae
        and prediction_rank_association_positive
        and prediction_selection_noninferiority["noninferior"]
    )
    nonmonotone_success_curves = sum(
        any(not left and right for left, right in zip(job["success_curve"][:-1], job["success_curve"][1:], strict=True))
        for job in jobs
    )
    return {
        "analysis_unit": "independent_scene_cluster",
        "independent_clusters": len(jobs),
        "valid_predictions": len(valid),
        "prediction_minimum_valid_clusters": minimum_valid_predictions,
        "prediction_sample_size_gate_passed": prediction_sample_size_gate_passed,
        "behavioral_boundary_no_success_sentinel": -1,
        "behavioral_success_curve_nonmonotone_clusters": nonmonotone_success_curves,
        "behavioral_success_curve_nonmonotone_fraction": (
            nonmonotone_success_curves / len(jobs) if jobs else None
        ),
        "valid_predictions_with_no_successful_boundary": sum(
            job["observed_last_successful_boundary"] is None for job in valid
        ),
        "prediction_exact_rate": _mean_boolean([job["prediction_exact"] for job in valid]),
        "prediction_within_one_rate": (float(np.mean(np.abs(errors) <= 1)) if len(errors) else None),
        "prediction_mean_absolute_error": (float(np.mean(np.abs(errors))) if len(errors) else None),
        "prediction_fixed_boundary_baseline": fixed_boundary_baseline,
        "prediction_fixed_boundary_baseline_exact_rate": (
            float(np.mean(baseline_errors == 0)) if len(baseline_errors) else None
        ),
        "prediction_fixed_boundary_baseline_within_one_rate": (
            float(np.mean(np.abs(baseline_errors) <= 1)) if len(baseline_errors) else None
        ),
        "prediction_fixed_boundary_baseline_mean_absolute_error": (
            float(np.mean(np.abs(baseline_errors))) if len(baseline_errors) else None
        ),
        "prediction_mae_difference_vs_fixed_boundary": (
            float(np.mean(absolute_error_advantage)) if len(absolute_error_advantage) else None
        ),
        "prediction_mae_difference_vs_fixed_boundary_ci95": (absolute_error_advantage_interval),
        "prediction_beats_fixed_boundary_mae": prediction_beats_fixed_boundary_mae,
        "prediction_spearman_rho": prediction_spearman_rho,
        "prediction_spearman_p_one_sided_permutation": prediction_spearman_p,
        "prediction_rank_permutation_samples": bootstrap_samples,
        "prediction_rank_association_positive": prediction_rank_association_positive,
        "predicted_boundary_composite_success_rate": _mean_boolean(predicted_boundary_success),
        "fixed_boundary_composite_success_rate_on_valid_predictions": _mean_boolean(
            fixed_boundary_success
        ),
        **{
            f"prediction_selected_boundary_{key}": value
            for key, value in prediction_selection_noninferiority.items()
        },
        "prediction_utility_gate_requires": [
            "minimum_valid_independent_clusters",
            "mae_bootstrap_ci_upper_below_fixed_boundary",
            "positive_scene_rank_association_permutation_p_below_alpha",
            "selected_boundary_composite_noninferior_to_fixed_boundary",
        ],
        "prediction_utility_gate_passed": prediction_utility_gate_passed,
        "next_boundary_composite_failure_rate": _mean_boolean(next_boundary_failure),
        "predicted_boundary_first_chunk_old_event_rate": _mean_boolean(predicted_boundary_first_chunk_old_event),
        "next_boundary_first_chunk_old_event_rate": _mean_boolean(next_boundary_first_chunk_old_event),
        "wrong_target_failure_first_contact_replan_histogram": (wrong_target_failure_replan_histogram),
        "eventual_failures_without_wrong_target_first": (eventual_failures_after_new_target_first),
        **{
            f"boundary7_{name}_{key}": value for name, result in noninferiority.items() for key, value in result.items()
        },
        # Backward-compatible aliases for the primary composite outcome.
        "boundary7_restart_composite_successes": noninferiority["composite"]["restart_successes"],
        "boundary7_continue_composite_successes": noninferiority["composite"]["continue_successes"],
        "boundary7_paired_losses": noninferiority["composite"]["paired_losses"],
        "boundary7_paired_gains": noninferiority["composite"]["paired_gains"],
        "boundary7_first_chunk_old_events": sum(bool(job["boundary7_first_chunk_old_event"]) for job in jobs),
        "restart_first_chunk_old_events": sum(bool(job["restart_first_chunk_old_event"]) for job in jobs),
        "boundary7_clean_replanning_rescues": sum(bool(job["boundary7_clean_replanning_rescue"]) for job in jobs),
        "restart_clean_replanning_rescues": sum(bool(job["restart_clean_replanning_rescue"]) for job in jobs),
        "boundary7_paired_loss_upper_bound_one_sided": noninferiority["composite"]["paired_loss_upper_bound_one_sided"],
        "noninferiority_margin": noninferiority_margin,
        "noninferiority_alpha_one_sided": alpha,
        "boundary7_noninferior": noninferiority["composite"]["noninferior"],
        "boundary7_velocity_evaluation_counts_exact": evaluation_counts_exact,
        "boundary7_post_event_velocity_evaluation_savings_fraction": (
            0.7 if evaluation_counts_exact and jobs else None
        ),
        "boundary7_median_latency_savings_fraction": (
            float(np.median(latency_savings)) if len(latency_savings) else None
        ),
        "boundary7_median_latency_savings_ci95": latency_interval,
        "boundary7_latency_savings_positive": latency_positive,
        "boundary7_practical_gate_requires": [
            "target_first_noninferior",
            "task_success_noninferior",
            "composite_noninferior",
            "exact_velocity_evaluation_counts",
            "latency_savings_ci95_low_above_zero",
        ],
        "boundary7_practical_gate_passed": practical_gate_passed,
        "latency_bootstrap_samples": bootstrap_samples,
        "latency_bootstrap_seed": bootstrap_seed,
    }


def _paired_loss_noninferiority(
    restart: list[bool],
    continued: list[bool],
    *,
    margin: float,
    alpha: float,
) -> dict[str, int | float | bool | None]:
    if len(restart) != len(continued):
        raise ValueError("paired outcomes must have equal lengths")
    paired_losses = sum(control and not intervention for control, intervention in zip(restart, continued, strict=True))
    paired_gains = sum(intervention and not control for control, intervention in zip(restart, continued, strict=True))
    upper_bound = binomial_upper_bound(paired_losses, len(restart), alpha) if restart else None
    return {
        "restart_successes": sum(restart),
        "continue_successes": sum(continued),
        "paired_losses": int(paired_losses),
        "paired_gains": int(paired_gains),
        "paired_loss_upper_bound_one_sided": upper_bound,
        "noninferior": bool(upper_bound < margin) if upper_bound is not None else None,
    }


def _validate_derived_job_fields(jobs: list[dict[str, Any]]) -> None:
    """Reconstruct prediction targets and registered curve cells before inference."""
    boolean_curves = (
        "success_curve",
        "first_chunk_old_event_curve",
        "old_target_first_curve",
        "clean_replanning_rescue_curve",
    )
    for job in jobs:
        for field in boolean_curves:
            curve = job.get(field)
            if (
                not isinstance(curve, list)
                or len(curve) != 11
                or any(type(value) is not bool for value in curve)
            ):
                raise ValueError(f"utility job {field} must contain 11 booleans")
        replan_curve = job.get("first_contact_replan_index_curve")
        if (
            not isinstance(replan_curve, list)
            or len(replan_curve) != 11
            or any(value is not None and (type(value) is not int or value < 0) for value in replan_curve)
        ):
            raise ValueError("utility job first-contact replan curve is invalid")

        success_curve = job["success_curve"]
        observed = max((boundary for boundary, success in enumerate(success_curve) if success), default=None)
        if job.get("observed_last_successful_boundary") != observed:
            raise ValueError("observed last successful boundary differs from the success curve")
        prediction_valid = job.get("prediction_valid")
        if type(prediction_valid) is not bool:
            raise ValueError("utility job prediction-valid flag must be boolean")
        predicted = job.get("predicted_last_successful_boundary")
        if prediction_valid:
            if type(predicted) is not int or not 0 <= predicted <= 10:
                raise ValueError("valid utility prediction must be an integer boundary in 0..10")
        elif predicted is not None:
            raise ValueError("invalid utility prediction must not contain a boundary")
        if job.get("prediction_exact") is not bool(prediction_valid and predicted == observed):
            raise ValueError("serialized prediction-exact flag differs from reconstructed value")

        boundary7_composite = bool(
            job.get("boundary7_new_target_first")
            and job.get("boundary7_new_task_success")
        )
        restart_composite = bool(
            job.get("restart_new_target_first")
            and job.get("restart_new_task_success")
        )
        if (
            success_curve[7] is not boundary7_composite
            or success_curve[0] is not restart_composite
            or job["first_chunk_old_event_curve"][7]
            is not job.get("boundary7_first_chunk_old_event")
            or job["clean_replanning_rescue_curve"][7]
            is not job.get("boundary7_clean_replanning_rescue")
        ):
            raise ValueError("registered boundary-zero or boundary-seven fields differ from curves")


def _mean_boolean(values: list[bool]) -> float | None:
    return sum(bool(value) for value in values) / len(values) if values else None


def _cluster_bootstrap_median(values: np.ndarray, samples: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def _cluster_bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[draws], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _spearman_permutation_p(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float | None:
    observed = _spearman(left, right)
    if observed is None:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(samples):
        null_rho = float(
            np.dot(rng.permutation(left_centered), right_centered) / denominator
        )
        exceedances += null_rho >= observed
    return (exceedances + 1) / (samples + 1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks
