#!/usr/bin/env python3
"""Render the sealed early-exit confirmation without hiding a negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from action_chunking.noninferiority import binomial_upper_bound


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_confirmation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every quantity shown in the confirmation figure."""
    required = {
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
        "source_artifact_files": 1505,
    }
    mismatched = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"confirmation figure received an incompatible summary: {mismatched}")
    if type(summary.get("confirmation_positive")) is not bool:
        raise ValueError("confirmation summary has no boolean decision")
    source_digest = summary.get("source_artifact_manifest_sha256")
    if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise ValueError("confirmation figure requires a hardened source-artifact digest")

    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError("confirmation figure requires exactly 500 paired rows")
    keys = {(int(row["task_id"]), int(row["trial_index"])) for row in rows}
    expected_keys = {(task_id, trial) for task_id in range(10) for trial in range(50)}
    if keys != expected_keys:
        raise ValueError("confirmation figure rows do not contain the exact task/trial grid")

    reconstructed = []
    latencies = []
    for row in rows:
        early = row.get("early_exit_success")
        full = row.get("full_control_success")
        if type(early) is not bool or type(full) is not bool:
            raise ValueError("confirmation figure success outcomes must be boolean")
        loss = bool(full and not early)
        gain = bool(early and not full)
        if row.get("paired_loss") is not loss or row.get("paired_gain") is not gain:
            raise ValueError("confirmation row paired outcome label is inconsistent")
        latency = float(row["first_replan_latency_savings_fraction"])
        if not np.isfinite(latency):
            raise ValueError("confirmation figure latency savings must be finite")
        latencies.append(latency)
        reconstructed.append((early, full, loss, gain))

    early_successes = sum(value[0] for value in reconstructed)
    full_successes = sum(value[1] for value in reconstructed)
    losses = sum(value[2] for value in reconstructed)
    gains = sum(value[3] for value in reconstructed)
    expected_counts = {
        "early_exit_successes": early_successes,
        "full_control_successes": full_successes,
        "paired_losses": losses,
        "paired_gains": gains,
    }
    count_mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in expected_counts.items()
        if summary.get(key) != expected
    }
    if count_mismatches:
        raise ValueError(f"confirmation summary counts differ from rows: {count_mismatches}")
    loss_upper = float(binomial_upper_bound(losses, 500, alpha=0.05))
    if summary.get("paired_loss_clopper_pearson_upper95") != loss_upper:
        raise ValueError("confirmation paired-loss bound differs from reconstructed count")
    median_latency = float(np.median(np.asarray(latencies, dtype=np.float64)))
    if summary.get("median_first_replan_latency_savings_fraction") != median_latency:
        raise ValueError("confirmation latency median differs from paired rows")
    latency_interval = summary.get(
        "median_first_replan_latency_savings_fraction_bootstrap_ci95"
    )
    if (
        not isinstance(latency_interval, list)
        or len(latency_interval) != 2
        or any(not np.isfinite(float(value)) for value in latency_interval)
        or float(latency_interval[0]) > float(latency_interval[1])
    ):
        raise ValueError("confirmation latency interval is invalid")
    reconstructed_positive = bool(
        losses <= 4 and loss_upper < 0.02 and float(latency_interval[0]) > 0.0
    )
    if summary["confirmation_positive"] is not reconstructed_positive:
        raise ValueError("confirmation decision differs from its registered gates")

    task_rows = summary.get("per_task")
    if not isinstance(task_rows, list) or len(task_rows) != 10:
        raise ValueError("confirmation figure requires all ten task strata")
    by_task = {int(row["task_id"]): row for row in task_rows}
    if set(by_task) != set(range(10)):
        raise ValueError("confirmation task strata are incomplete or duplicated")
    for task_id, task in by_task.items():
        selected = [row for row in rows if int(row["task_id"]) == task_id]
        task_expected = {
            "trials": 50,
            "early_exit_successes": sum(bool(row["early_exit_success"]) for row in selected),
            "full_control_successes": sum(bool(row["full_control_success"]) for row in selected),
            "paired_losses": sum(bool(row["paired_loss"]) for row in selected),
            "paired_gains": sum(bool(row["paired_gain"]) for row in selected),
            "condition_order_counts": {
                "early_exit_first": 25,
                "full_control_first": 25,
            },
        }
        if any(task.get(key) != value for key, value in task_expected.items()):
            raise ValueError(f"confirmation task stratum differs from paired rows: {task_id}")

    return {
        "rows": rows,
        "tasks": [by_task[task_id] for task_id in range(10)],
        "latency": latencies,
        "early_exit_successes": early_successes,
        "full_control_successes": full_successes,
        "paired_losses": losses,
        "paired_gains": gains,
        "median_latency": median_latency,
    }


def make_confirmation_figure(
    summary: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    audited = validate_confirmation_summary(summary)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.1), constrained_layout=True)

    task_ids = np.arange(10)
    full_rates = np.asarray([task["full_control_successes"] / 50 for task in audited["tasks"]])
    early_rates = np.asarray([task["early_exit_successes"] / 50 for task in audited["tasks"]])
    for task_id, full, early in zip(task_ids, full_rates, early_rates, strict=True):
        axes[0].plot([task_id - 0.08, task_id + 0.08], [full, early], color="0.65", linewidth=0.8)
    axes[0].scatter(task_ids - 0.08, full_rates, color="#4d4d4d", s=20, label="10 evaluations")
    axes[0].scatter(task_ids + 0.08, early_rates, color="#1b9e77", s=20, label="7 evaluations")
    axes[0].set(
        title="a  Success by frozen task stratum",
        xlabel="LIBERO Goal task",
        ylabel="success rate",
        xticks=task_ids,
        ylim=(0, 1.04),
    )
    axes[0].legend(frameon=False, loc="lower left")

    both = sum(
        bool(row["early_exit_success"]) and bool(row["full_control_success"])
        for row in audited["rows"]
    )
    neither = 500 - both - audited["paired_losses"] - audited["paired_gains"]
    labels = ["both\nsucceed", "full only\n(loss)", "early only\n(gain)", "neither"]
    counts = [both, audited["paired_losses"], audited["paired_gains"], neither]
    colors = ["#4daf4a", "#e41a1c", "#377eb8", "#999999"]
    axes[1].bar(np.arange(4), counts, color=colors, width=0.7)
    for index, count in enumerate(counts):
        axes[1].text(index, count + 5, str(count), ha="center", va="bottom")
    axes[1].set(
        title="b  Paired behavioral outcomes",
        ylabel="episode pairs",
        xticks=np.arange(4),
        xticklabels=labels,
        ylim=(0, max(counts) * 1.12 if max(counts) else 1),
    )

    latency = np.asarray(audited["latency"], dtype=np.float64)
    axes[2].hist(latency, bins=30, color="#7570b3", alpha=0.82)
    axes[2].axvline(0.0, color="0.25", linestyle="--", linewidth=1)
    axes[2].axvline(audited["median_latency"], color="#b2182b", linewidth=1.2)
    low, high = summary["median_first_replan_latency_savings_fraction_bootstrap_ci95"]
    axes[2].set(
        title="c  Paired first-replan latency",
        xlabel="fractional latency savings",
        ylabel="episode pairs",
    )
    axes[2].text(
        0.03,
        0.96,
        f"median {audited['median_latency']:.1%}\n95% bootstrap CI [{low:.1%}, {high:.1%}]",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    decision = "positive" if summary["confirmation_positive"] else "negative"
    figure.suptitle(f"Sealed 500-pair early-exit confirmation: {decision}", fontsize=10.5)

    stem = output / "fig_early_exit_confirmation"
    for suffix in ("pdf", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)
    return audited


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"confirmation figure output already exists: {args.output}"
        )
    summary = json.loads(args.summary.read_text())
    audited = make_confirmation_figure(summary, args.output)
    outputs = ["fig_early_exit_confirmation.pdf", "fig_early_exit_confirmation.png"]
    manifest = {
        "schema_version": 1,
        "source_summary": str(args.summary),
        "source_summary_sha256": _sha256(args.summary),
        "source_artifact_files": summary["source_artifact_files"],
        "source_artifact_manifest_sha256": summary[
            "source_artifact_manifest_sha256"
        ],
        "confirmation_positive": summary["confirmation_positive"],
        "episode_pairs": 500,
        "early_exit_successes": audited["early_exit_successes"],
        "full_control_successes": audited["full_control_successes"],
        "paired_losses": audited["paired_losses"],
        "paired_gains": audited["paired_gains"],
        "median_latency_savings_fraction": audited["median_latency"],
        "median_latency_savings_ci95": summary[
            "median_first_replan_latency_savings_fraction_bootstrap_ci95"
        ],
        "outputs": outputs,
        "output_sha256": {name: _sha256(args.output / name) for name in outputs},
    }
    (args.output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
