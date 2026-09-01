#!/usr/bin/env python3
"""Render the preregistered held-out retargeting utility figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from action_chunking.utility_artifacts import audit_utility_study

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 40
BOUNDARIES = tuple(range(11))
POLICIES = ("predicted", "fixed", "restart")
POLICY_LABELS = {
    "predicted": "boundary-adaptive",
    "fixed": "fixed cutoff 7",
    "restart": "always restart",
}
OUTCOMES = (
    ("target first", "new_target_first_curve", "restart_new_target_first"),
    ("task success", "new_task_success_curve", "restart_new_task_success"),
    ("composite", "success_curve", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_audited_utility(study_root: Path, final_audit_path: Path) -> dict[str, Any]:
    observed_audit = _read_json(final_audit_path)
    reconstructed_audit = audit_utility_study(study_root)
    if observed_audit != reconstructed_audit:
        raise ValueError("utility final audit differs from current raw-artifact reconstruction")
    summary_path = study_root / "summary.json"
    summary = _read_json(summary_path)
    jobs = summary.get("jobs")
    if (
        summary.get("study_complete") is not True
        or summary.get("utility_inference_status") not in {"positive", "negative"}
        or not isinstance(jobs, list)
        or not jobs
        or len(jobs) != int(summary.get("completed_primary_clusters", -1))
        or len(jobs) != int(summary.get("expected_primary_clusters", -2))
        or observed_audit.get("utility_summary_sha256") != _digest(summary_path)
    ):
        raise ValueError("utility figure requires one complete audited population")
    return {
        "summary": summary,
        "summary_path": summary_path,
        "audit": observed_audit,
        "audit_path": final_audit_path,
        "jobs": jobs,
    }


def make_utility_figure(
    study_root: Path,
    final_audit_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"utility figure output already exists: {output}")
    data = load_audited_utility(study_root, final_audit_path)
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.2, 3.45), constrained_layout=True)
    _plot_prediction(axes[0], data["jobs"])
    _plot_boundary_curves(axes[1], data["jobs"])
    _plot_policy_outcomes(axes[2], data["jobs"])
    _plot_policy_compute(axes[3], data["jobs"])
    status = data["summary"]["utility_inference_status"]
    figure.suptitle(f"Held-out retargeting utility (registered decision: {status})", fontsize=10.3)
    stem = output / "fig_retarget_utility"
    for suffix in ("pdf", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)
    outputs = ["fig_retarget_utility.pdf", "fig_retarget_utility.png"]
    manifest = {
        "schema_version": 1,
        "artifact": "held_out_retarget_utility_figure",
        "utility_summary_sha256": _digest(data["summary_path"]),
        "final_audit_sha256": _digest(data["audit_path"]),
        "frozen_predictions_sha256": data["audit"]["frozen_predictions_sha256"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "panels": [
            "predicted_vs_observed_last_successful_boundary",
            "behavior_by_update_boundary",
            "adaptive_policy_behavior_vs_controls",
            "adaptive_policy_velocity_evaluations_vs_controls",
        ],
        "outputs": outputs,
        "output_sha256": {name: _digest(output / name) for name in outputs},
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _plot_prediction(axis: Any, jobs: list[dict[str, Any]]) -> None:
    valid = [job for job in jobs if job["prediction_valid"]]
    if valid:
        observed = np.asarray(
            [
                -1
                if job["observed_last_successful_boundary"] is None
                else int(job["observed_last_successful_boundary"])
                for job in valid
            ],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [int(job["predicted_last_successful_boundary"]) for job in valid],
            dtype=np.float64,
        )
        axis.scatter(observed, predicted, s=15, alpha=0.65, color="#377eb8")
    else:
        axis.text(0.5, 0.5, "no valid predictions", ha="center", transform=axis.transAxes)
    axis.plot([-1, 10], [-1, 10], color="0.35", linestyle="--", linewidth=0.9, label="identity")
    axis.axhline(7, color="#e41a1c", linestyle=":", linewidth=1, label="fixed cutoff 7")
    axis.set(
        title="a  Boundary prediction",
        xlabel="observed last successful boundary",
        ylabel="predicted boundary",
        xlim=(-1.5, 10.5),
        ylim=(-1.5, 10.5),
        xticks=(-1, 0, 2, 4, 6, 8, 10),
        yticks=(-1, 0, 2, 4, 6, 8, 10),
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_boundary_curves(axis: Any, jobs: list[dict[str, Any]]) -> None:
    colors = ("#1b9e77", "#7570b3", "#d95f02")
    for (label, curve_field, _restart_field), color in zip(OUTCOMES, colors, strict=True):
        matrix = np.asarray([job[curve_field] for job in jobs], dtype=np.float64)
        means, low, high = _mean_interval(matrix, BOOTSTRAP_SEED)
        axis.plot(BOUNDARIES, means, marker="o", markersize=2.5, linewidth=1, color=color, label=label)
        axis.fill_between(BOUNDARIES, low, high, color=color, alpha=0.14, linewidth=0)
    axis.axvline(7, color="0.45", linestyle="--", linewidth=0.8)
    axis.set(
        title="b  Recovery by update time",
        xlabel="instruction-update boundary",
        ylabel="cluster success rate",
        xlim=(-0.2, 10.2),
        ylim=(-0.03, 1.03),
        xticks=BOUNDARIES,
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_policy_outcomes(axis: Any, jobs: list[dict[str, Any]]) -> None:
    x = np.arange(len(OUTCOMES), dtype=np.float64)
    width = 0.24
    colors = {"predicted": "#377eb8", "fixed": "#984ea3", "restart": "#4daf4a"}
    for policy_index, policy in enumerate(POLICIES):
        values = []
        errors = []
        for outcome_index, (_label, curve_field, restart_field) in enumerate(OUTCOMES):
            cluster_values = _policy_values(jobs, curve_field, restart_field, policy)
            mean, low, high = _scalar_interval(
                cluster_values,
                BOOTSTRAP_SEED + 10 + policy_index * 3 + outcome_index,
            )
            values.append(mean)
            errors.append((mean - low, high - mean))
        position = x + (policy_index - 1) * width
        axis.bar(position, values, width, color=colors[policy], label=POLICY_LABELS[policy])
        axis.errorbar(
            position,
            values,
            yerr=np.asarray(errors).T,
            fmt="none",
            ecolor="0.2",
            elinewidth=0.8,
            capsize=2,
        )
    axis.set(
        title="c  Adaptive-policy behavior",
        ylabel="uniform-boundary success",
        xticks=x,
        xticklabels=[spec[0] for spec in OUTCOMES],
        ylim=(0.0, 1.04),
    )
    axis.set_title("c  Adaptive-policy behavior", pad=29)
    axis.tick_params(axis="x", rotation=18)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=0.7,
        handlelength=1.3,
    )
    axis.spines[["top", "right"]].set_visible(False)


def _plot_policy_compute(axis: Any, jobs: list[dict[str, Any]]) -> None:
    colors = ("#377eb8", "#984ea3", "#4daf4a")
    means = []
    errors = []
    for policy_index, policy in enumerate(POLICIES):
        values = _evaluation_values(jobs, policy)
        mean, low, high = _scalar_interval(values, BOOTSTRAP_SEED + 30 + policy_index)
        means.append(mean)
        errors.append((mean - low, high - mean))
    x = np.arange(len(POLICIES), dtype=np.float64)
    axis.bar(x, means, color=colors, width=0.62)
    axis.errorbar(
        x,
        means,
        yerr=np.asarray(errors).T,
        fmt="none",
        ecolor="0.2",
        elinewidth=0.8,
        capsize=2,
    )
    axis.set(
        title="d  Post-update compute",
        ylabel="mean velocity evaluations",
        xticks=x,
        xticklabels=[POLICY_LABELS[policy] for policy in POLICIES],
        ylim=(0.0, 10.5),
    )
    axis.tick_params(axis="x", rotation=18)
    axis.spines[["top", "right"]].set_visible(False)


def _policy_values(
    jobs: list[dict[str, Any]],
    curve_field: str,
    restart_field: str | None,
    policy: str,
) -> np.ndarray:
    values = []
    for job in jobs:
        restart = (
            bool(job[restart_field])
            if restart_field is not None
            else bool(job["restart_new_target_first"] and job["restart_new_task_success"])
        )
        cutoff = _policy_cutoff(job, policy)
        curve = [bool(value) for value in job[curve_field]]
        selected = [curve[boundary] if boundary <= cutoff else restart for boundary in BOUNDARIES]
        values.append(float(np.mean(selected)))
    return np.asarray(values, dtype=np.float64)


def _evaluation_values(jobs: list[dict[str, Any]], policy: str) -> np.ndarray:
    values = []
    for job in jobs:
        cutoff = _policy_cutoff(job, policy)
        curve = [int(value) for value in job["post_event_velocity_evaluations_curve"]]
        restart = int(job["restart_post_event_velocity_evaluations"])
        selected = [curve[boundary] if boundary <= cutoff else restart for boundary in BOUNDARIES]
        values.append(float(np.mean(selected)))
    return np.asarray(values, dtype=np.float64)


def _policy_cutoff(job: dict[str, Any], policy: str) -> int:
    if policy == "predicted":
        return int(job["predicted_last_successful_boundary"]) if job["prediction_valid"] else -1
    if policy == "fixed":
        return 7
    if policy == "restart":
        return -1
    raise ValueError(f"unknown utility policy: {policy}")


def _mean_interval(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(matrix), size=(BOOTSTRAP_REPLICATES, len(matrix)))
    sampled = matrix[draws].mean(axis=1)
    return (
        matrix.mean(axis=0),
        np.quantile(sampled, 0.025, axis=0),
        np.quantile(sampled, 0.975, axis=0),
    )


def _scalar_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    mean, low, high = _mean_interval(values[:, None], seed)
    return float(mean[0]), float(low[0]), float(high[0])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    manifest = make_utility_figure(args.study_root, args.final_audit, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
