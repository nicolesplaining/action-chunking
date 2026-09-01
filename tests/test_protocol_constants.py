from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from action_chunking.catalog_selection import build_retarget_screening_plan
from action_chunking.utility_analysis import summarize_utility_jobs


def test_machine_readable_protocol_matches_executable_frozen_defaults() -> None:
    protocol = yaml.safe_load(Path("configs/study.yaml").read_text())
    utility = protocol["retarget_utility"]
    confirmation = protocol["early_exit_confirmation"]
    comparison = protocol["matched_pi0"]
    analysis = protocol["analysis"]

    utility_defaults = inspect.signature(summarize_utility_jobs).parameters
    catalog_defaults = inspect.signature(build_retarget_screening_plan).parameters
    assert protocol["study"]["protocol_version"] == "0.34"
    assert protocol["inference"]["flow_steps"] == 10
    assert protocol["inference"]["receding_horizon_steps"] == 5
    assert protocol["study"]["editability_retention_threshold"] == 0.8
    assert utility["dense_prediction_boundaries"] == list(range(11))
    assert utility["primary_efficiency_boundary"] == 7
    assert utility["prediction_fixed_boundary_comparator"] == utility_defaults[
        "fixed_boundary_baseline"
    ].default
    assert utility["paired_noninferiority_margin"] == utility_defaults[
        "noninferiority_margin"
    ].default
    assert utility["paired_noninferiority_alpha_one_sided"] == utility_defaults[
        "alpha"
    ].default
    assert utility["minimum_eligible_scene_clusters"] == utility_defaults[
        "minimum_valid_predictions"
    ].default
    assert utility["minimum_eligible_scene_clusters"] == catalog_defaults[
        "minimum_eligible_clusters"
    ].default
    assert utility["prediction_bootstrap_replicates"] == utility_defaults[
        "bootstrap_samples"
    ].default
    assert utility["prediction_bootstrap_seed"] == utility_defaults[
        "bootstrap_seed"
    ].default
    assert utility["prediction_rank_test"] == "one_sided_spearman_permutation"
    assert utility["prediction_rank_alpha"] == utility_defaults["alpha"].default
    assert utility["prediction_no_success_boundary_sentinel"] == -1
    assert utility["prediction_positive_requires_selected_boundary_noninferiority"] is True
    assert utility["report_nonmonotone_behavioral_success_curves"] is True
    assert utility["adaptive_policy_event_boundary_weighting"] == (
        "uniform_over_0_to_10_design_estimand"
    )
    assert utility["adaptive_policy_invalid_prediction_fallback"] == "restart"
    assert utility["adaptive_policy_comparators"] == ["always_restart", "fixed_cutoff_7"]
    assert utility["adaptive_policy_outcomes"] == utility["noninferiority_outcomes"]
    assert utility["adaptive_policy_bootstrap_replicates"] == utility[
        "prediction_bootstrap_replicates"
    ]
    assert utility["adaptive_policy_bootstrap_seed"] == 20
    assert utility["adaptive_policy_requires_noninferiority_to_both_comparators"] is True
    assert utility["adaptive_policy_requires_compute_savings_ci_low_above_zero_vs_both"] is True
    assert utility["noninferiority_outcomes"] == [
        "new_target_first",
        "eventual_new_task_success",
        "target_first_and_task_success_composite",
    ]
    assert utility["practical_gate_requires_boundary_zero_behavior_equivalence"] is True

    assert confirmation["paired_episode_keys"] == 500
    assert confirmation["condition_rollouts"] == 1000
    assert confirmation["maximum_passing_paired_losses"] == 4
    assert confirmation["paired_loss_margin"] == 0.02
    assert confirmation["conditions"] == {
        "early_exit_flow_steps": 7,
        "full_control_flow_steps": 10,
    }
    assert confirmation["publication_audit_requires_completely_clean_worktree"] is True
    assert confirmation["publication_figure_requires_hardened_source_digest"] is True
    assert confirmation["publication_outputs_require_new_paths"] is True
    assert comparison["optimizer_updates"] == 30_000
    assert comparison["final_checkpoint_label"] == 29_999
    assert comparison["parity_cases"] == 32
    assert comparison["parity_shape_per_case"] == [50, 7]
    assert comparison["maximum_absolute_error"] == 0.02
    assert comparison["minimum_cosine_similarity"] == 0.999
    assert comparison["minimum_common_scene_pairs"] == 12
    assert comparison["parallel_intervention_gpus"] == 2
    assert comparison["require_distinct_h100_preflight"] is True
    assert comparison["require_complete_worktree_cleanliness"] is True
    assert comparison["resume_requires_code_commit_binding"] is True
    assert comparison["require_complete_grid_per_intervention_unit"] is True
    assert comparison["require_prior_failure_digest_in_conversion_provenance"] is True
    assert comparison["require_parity_reconstruction_from_worker_artifacts"] is True
    assert comparison["require_prior_failure_digest_equality"] is True
    assert comparison["require_frozen_normalization_assets"] is True
    assert comparison["require_new_conversion_output_directories"] is True
    assert comparison["require_final_output_lineage_audit"] is True
    assert comparison["comparison_generated_files"] == 10
    assert comparison["require_comparison_output_hashes"] is True
    assert comparison["publication_figure_requires_final_audit"] is True
    assert comparison["publication_figure_requires_new_path"] is True
    assert comparison["publication_figure_primary_metric"] == "all"
    assert comparison["publication_figure_panels"] == [
        "paired_formation_and_editability_timing",
        "all_metric_residual_flow_by_layer",
        "all_metric_action_state_dimension_groups",
        "all_metric_normalized_position_at_layer_17",
    ]
    assert analysis["population_position_flow_steps"] == [0, 7, 8, 9]
    assert analysis["population_position_layers"] == [0, 8, 14, 17]
    assert analysis["pi05_action_horizon"] == 10
    assert analysis["pi0_action_horizon"] == 50
    assert protocol["obstacle_pilot"]["whole_robot_collision_measured"] is False
    assert protocol["obstacle_pilot"]["practical_gate_requires_boundary_zero_actions_exact"] is True
    assert (
        protocol["obstacle_pilot"]["practical_gate_requires_boundary_zero_behavior_equivalence"]
        is True
    )
