#!/usr/bin/env python3
"""Aggregate clean-selected intervention jobs with scene-state clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from action_chunking.analysis import benjamini_hochberg, commitment_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--formation-relative-error-tolerance", type=float, default=0.2)
    parser.add_argument("--minimum-action-contrast", type=float, default=0.01)
    parser.add_argument("--minimum-target-contrast", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=23)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    manifest = json.loads(args.manifest.read_text())
    clusters = {entry["pair_id"]: entry["identity_hashes"]["sim_state"] for entry in manifest["pairs"]}
    jobs = sorted(path.parent for path in args.input.glob("*/noise_*/metadata.json"))
    if not jobs:
        raise ValueError("no completed intervention jobs found")
    if args.expected_jobs is not None and len(jobs) != args.expected_jobs:
        raise ValueError(f"expected {args.expected_jobs} intervention jobs, found {len(jobs)}")
    args.output.mkdir(parents=True, exist_ok=True)

    units = []
    flow_rows = []
    formation_rows = []
    residual_rows = []
    dimension_rows = []
    analyzer = Path(__file__).with_name("analyze_pair.py")
    for job in jobs:
        metadata = json.loads((job / "metadata.json").read_text())
        pair_id = metadata["pair_id"]
        if pair_id not in clusters:
            raise ValueError(f"pair {pair_id!r} is absent from the manifest")
        noise_seed = int(metadata["noise_seed"])
        job_analysis = args.output / "jobs" / pair_id / f"noise_{noise_seed}"
        command = [
            sys.executable,
            str(analyzer),
            "--input",
            str(job),
            "--output",
            str(job_analysis),
            "--threshold",
            str(args.threshold),
            "--minimum-action-contrast",
            str(args.minimum_action_contrast),
            "--minimum-target-contrast",
            str(args.minimum_target_contrast),
            "--formation-relative-error-tolerance",
            str(args.formation_relative_error_tolerance),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        summary = json.loads((job_analysis / "summary.json").read_text())
        for metric, valid in summary["valid_contrasts"].items():
            units.append(
                {
                    "pair_id": pair_id,
                    "scene_state_sha256": clusters[pair_id],
                    "noise_seed": noise_seed,
                    "metric": metric,
                    "eligible": bool(valid),
                    "endpoint_contrast": summary["endpoint_contrasts"][metric],
                    "commitment_step": summary["commitment_steps"][metric],
                    "formation_step": summary["formation_steps"][metric],
                }
            )
        for row in _read_csv(job_analysis / "flow_retention.csv"):
            flow_rows.append(
                {
                    "pair_id": pair_id,
                    "scene_state_sha256": clusters[pair_id],
                    "noise_seed": noise_seed,
                    "metric": row["metric"],
                    "switch_after_steps": int(row["switch_after_steps"]),
                    "eligible": bool(summary["valid_contrasts"][row["metric"]]),
                    "symmetric_retention": float(row["symmetric_retention"]),
                }
            )
        for row in _read_csv(job_analysis / "formation_contrast.csv"):
            formation_rows.append(
                {
                    "pair_id": pair_id,
                    "scene_state_sha256": clusters[pair_id],
                    "noise_seed": noise_seed,
                    "metric": row["metric"],
                    "flow_step": int(row["flow_step"]),
                    "eligible": row["eligible"] == "True",
                    "contrast_alignment": float(row["contrast_alignment"]),
                    "contrast_relative_error": float(row["contrast_relative_error"]),
                    "contrast_cosine": float(row["contrast_cosine"]),
                }
            )
        residual_path = job_analysis / "residual_transfer.csv"
        if residual_path.exists():
            residual_rows.extend(
                _intervention_rows(
                    residual_path,
                    pair_id,
                    clusters[pair_id],
                    noise_seed,
                    summary,
                    ("flow_step", "layer"),
                )
            )
        dimension_path = job_analysis / "dimension_transfer.csv"
        if dimension_path.exists():
            dimension_rows.extend(
                _intervention_rows(
                    dimension_path,
                    pair_id,
                    clusters[pair_id],
                    noise_seed,
                    summary,
                    ("flow_step", "patched_tensor", "patched_dimension_group"),
                )
            )

    flow_curve, flow_statistics = _aggregate_flow(flow_rows, args)
    formation_curve, formation_statistics = _aggregate_formation(formation_rows, args)
    timing_separation_statistics = _timing_separation(units, args)
    residual_cells = _aggregate_intervention_cells(
        residual_rows,
        ("flow_step", "layer"),
        args,
        seed_offset=2,
    )
    dimension_cells = _aggregate_intervention_cells(
        dimension_rows,
        ("flow_step", "patched_tensor", "patched_dimension_group"),
        args,
        seed_offset=3,
    )
    _write_csv(args.output / "units.csv", units)
    _write_csv(args.output / "flow_units.csv", flow_rows)
    _write_csv(args.output / "flow_curve.csv", flow_curve)
    _write_csv(args.output / "formation_units.csv", formation_rows)
    _write_csv(args.output / "formation_curve.csv", formation_curve)
    if residual_rows:
        _write_csv(args.output / "residual_units.csv", residual_rows)
        _write_csv(args.output / "residual_cells.csv", residual_cells)
    if dimension_rows:
        _write_csv(args.output / "dimension_units.csv", dimension_rows)
        _write_csv(args.output / "dimension_cells.csv", dimension_cells)
    _plot_flow(flow_curve, args.output / "flow_commitment")
    _plot_formation(
        formation_curve,
        args.formation_relative_error_tolerance,
        args.output / "contrast_formation",
    )
    if residual_cells:
        _plot_residual_cells(residual_cells, args.output / "residual_heatmap")
    if dimension_cells:
        _plot_dimension_cells(dimension_cells, args.output / "dimension_heatmap")
    summary = {
        "schema_version": 1,
        "jobs": len(jobs),
        "pairs": len({row["pair_id"] for row in units}),
        "state_clusters": len({row["scene_state_sha256"] for row in units}),
        "noise_seeds": sorted({row["noise_seed"] for row in units}),
        "commitment_threshold": args.threshold,
        "formation_relative_error_tolerance": args.formation_relative_error_tolerance,
        "flow_statistics": flow_statistics,
        "formation_statistics": formation_statistics,
        "timing_separation_statistics": timing_separation_statistics,
        "intervention_statistics": {
            "residual_patch": _intervention_statistics(residual_cells, ("flow_step", "layer")),
            "dimension_patch": _intervention_statistics(
                dimension_cells,
                ("flow_step", "patched_tensor", "patched_dimension_group"),
            ),
        },
        "eligible_units_by_metric": {
            metric: sum(row["eligible"] for row in units if row["metric"] == metric)
            for metric in sorted({row["metric"] for row in units})
        },
        "bootstrap": {
            "unit": "scene_state_sha256",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _intervention_rows(
    path: Path,
    pair_id: str,
    cluster: str,
    noise_seed: int,
    summary: dict[str, Any],
    cell_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    output = []
    integer_fields = {"flow_step", "layer"}
    for row in _read_csv(path):
        record = {
            "pair_id": pair_id,
            "scene_state_sha256": cluster,
            "noise_seed": noise_seed,
            "metric": row["metric"],
            "eligible": bool(summary["valid_contrasts"][row["metric"]]),
            "base_to_donor_ncte": float(row["base_to_donor_ncte"]),
            "donor_to_base_ncte": float(row["donor_to_base_ncte"]),
            "symmetric_ncte": float(row["symmetric_ncte"]),
            "directional_asymmetry": float(row["directional_asymmetry"]),
        }
        record.update(
            {
                field: int(row[field]) if field in integer_fields else row[field]
                for field in cell_fields
            }
        )
        output.append(record)
    return output


def _aggregate_intervention_cells(
    rows: list[dict[str, Any]],
    cell_fields: tuple[str, ...],
    args: argparse.Namespace,
    seed_offset: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    value_fields = (
        "base_to_donor_ncte",
        "donor_to_base_ncte",
        "symmetric_ncte",
        "directional_asymmetry",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["scene_state_sha256"], row["metric"], *(row[field] for field in cell_fields))
        grouped[key].append(row)
    state_rows = []
    for key, selected in sorted(grouped.items()):
        eligibility = {row["eligible"] for row in selected}
        if len(eligibility) != 1:
            raise ValueError("intervention-cell eligibility differs across noise seeds within a state")
        record = {
            "scene_state_sha256": key[0],
            "metric": key[1],
            "eligible": eligibility.pop(),
        }
        record.update(dict(zip(cell_fields, key[2:], strict=True)))
        record.update({field: float(np.mean([row[field] for row in selected])) for field in value_fields})
        state_rows.append(record)

    rng = np.random.default_rng(args.bootstrap_seed + seed_offset)
    output = []
    for metric in sorted({row["metric"] for row in state_rows}):
        selected = [row for row in state_rows if row["metric"] == metric and row["eligible"]]
        if not selected:
            continue
        clusters = sorted({row["scene_state_sha256"] for row in selected})
        cells = sorted({tuple(row[field] for field in cell_fields) for row in selected})
        lookup = {
            (row["scene_state_sha256"], *(row[field] for field in cell_fields)): row
            for row in selected
        }
        expected = len(clusters) * len(cells)
        if len(lookup) != expected:
            raise ValueError(
                f"incomplete {metric} intervention grid: expected {expected} state cells, found {len(lookup)}"
            )
        matrices = {
            field: np.asarray(
                [[lookup[(cluster, *cell)][field] for cell in cells] for cluster in clusters],
                dtype=np.float64,
            )
            for field in value_fields
        }
        bootstrap_indices = rng.integers(
            0,
            len(clusters),
            size=(args.bootstrap_replicates, len(clusters)),
        )
        bootstrap = {
            field: matrix[bootstrap_indices].mean(axis=1)
            for field, matrix in matrices.items()
        }
        null_means, exact = _sign_flip_null(
            matrices["symmetric_ncte"],
            rng,
            args.bootstrap_replicates,
        )
        observed = matrices["symmetric_ncte"].mean(axis=0)
        exceedances = np.sum(np.abs(null_means) >= np.abs(observed)[None, :], axis=0)
        p_values = (
            exceedances / len(null_means)
            if exact
            else (exceedances + 1) / (len(null_means) + 1)
        )
        q_values = benjamini_hochberg(p_values)
        for index, cell in enumerate(cells):
            record = {
                "metric": metric,
                "eligible_state_clusters": len(clusters),
                "mean_base_to_donor_ncte": float(matrices["base_to_donor_ncte"][:, index].mean()),
                "mean_donor_to_base_ncte": float(matrices["donor_to_base_ncte"][:, index].mean()),
                "mean_symmetric_ncte": float(observed[index]),
                "symmetric_ci95_low": float(np.quantile(bootstrap["symmetric_ncte"][:, index], 0.025)),
                "symmetric_ci95_high": float(np.quantile(bootstrap["symmetric_ncte"][:, index], 0.975)),
                "mean_directional_asymmetry": float(matrices["directional_asymmetry"][:, index].mean()),
                "asymmetry_ci95_low": float(
                    np.quantile(bootstrap["directional_asymmetry"][:, index], 0.025)
                ),
                "asymmetry_ci95_high": float(
                    np.quantile(bootstrap["directional_asymmetry"][:, index], 0.975)
                ),
                "positive_state_fraction": float(np.mean(matrices["symmetric_ncte"][:, index] > 0.0)),
                "p_two_sided_sign_flip": float(p_values[index]),
                "q_bh_within_metric_family": float(q_values[index]),
                "sign_flip_exact": exact,
            }
            record.update(dict(zip(cell_fields, cell, strict=True)))
            output.append(record)
    return output


def _sign_flip_null(
    matrix: np.ndarray,
    rng: np.random.Generator,
    monte_carlo_replicates: int,
) -> tuple[np.ndarray, bool]:
    clusters = matrix.shape[0]
    if clusters <= 20:
        assignments = np.arange(1 << clusters, dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(clusters, dtype=np.uint64)[None, :]) & 1
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        exact = True
    else:
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(monte_carlo_replicates, clusters),
        )
        exact = False
    return signs @ matrix / clusters, exact


def _intervention_statistics(
    rows: list[dict[str, Any]], cell_fields: tuple[str, ...]
) -> dict[str, Any] | None:
    if not rows:
        return None
    output = {}
    for metric in sorted({row["metric"] for row in rows}):
        selected = [row for row in rows if row["metric"] == metric]
        peak = max(selected, key=lambda row: row["mean_symmetric_ncte"])
        output[metric] = {
            "cells": len(selected),
            "eligible_state_clusters": peak["eligible_state_clusters"],
            "fdr_significant_positive_cells": sum(
                row["q_bh_within_metric_family"] < 0.05 and row["mean_symmetric_ncte"] > 0.0
                for row in selected
            ),
            "peak": {
                **{field: peak[field] for field in cell_fields},
                "mean_symmetric_ncte": peak["mean_symmetric_ncte"],
                "symmetric_ci95_low": peak["symmetric_ci95_low"],
                "symmetric_ci95_high": peak["symmetric_ci95_high"],
                "q_bh_within_metric_family": peak["q_bh_within_metric_family"],
            },
        }
    return output


def _timing_separation(units: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for unit in units:
        if not unit["eligible"] or unit["commitment_step"] is None or unit["formation_step"] is None:
            continue
        grouped[(unit["scene_state_sha256"], unit["metric"])].append(
            float(unit["commitment_step"] - unit["formation_step"])
        )
    state_gaps = [
        {
            "scene_state_sha256": cluster,
            "metric": metric,
            "gap_steps": float(np.mean(values)),
        }
        for (cluster, metric), values in sorted(grouped.items())
    ]
    rng = np.random.default_rng(args.bootstrap_seed + 4)
    output = {}
    for metric in sorted({row["metric"] for row in state_gaps}):
        values = np.asarray(
            [row["gap_steps"] for row in state_gaps if row["metric"] == metric],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(values), size=(args.bootstrap_replicates, len(values)))
        sampled = values[indices]
        bootstrap_means = sampled.mean(axis=1)
        bootstrap_medians = np.median(sampled, axis=1)
        output[metric] = {
            "eligible_state_clusters": len(values),
            "mean_commitment_minus_formation_steps": float(values.mean()),
            "mean_gap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "mean_gap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
            "median_commitment_minus_formation_steps": float(np.median(values)),
            "median_gap_ci95_low": float(np.quantile(bootstrap_medians, 0.025)),
            "median_gap_ci95_high": float(np.quantile(bootstrap_medians, 0.975)),
            "positive_gap_state_fraction": float(np.mean(values > 0.0)),
        }
    return output


def _aggregate_flow(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_rows = _state_means(rows, "switch_after_steps", ("symmetric_retention",))
    rng = np.random.default_rng(args.bootstrap_seed)
    output = []
    statistics = {}
    for metric in sorted({row["metric"] for row in state_rows}):
        selected = [row for row in state_rows if row["metric"] == metric and row["eligible"]]
        clusters = sorted({row["scene_state_sha256"] for row in selected})
        if not clusters:
            statistics[metric] = None
            continue
        matrix = np.asarray(
            [
                [
                    next(
                        row["symmetric_retention"]
                        for row in selected
                        if row["scene_state_sha256"] == cluster and row["switch_after_steps"] == boundary
                    )
                    for boundary in range(11)
                ]
                for cluster in clusters
            ]
        )
        bootstrap_indices = rng.integers(0, len(clusters), size=(args.bootstrap_replicates, len(clusters)))
        bootstrap_curves = matrix[bootstrap_indices].mean(axis=1)
        mean_curve = matrix.mean(axis=0)
        aggregate_step, fitted = commitment_step(mean_curve, args.threshold)
        bootstrap_steps = np.asarray(
            [
                step if (step := commitment_step(curve, args.threshold)[0]) is not None else 11
                for curve in bootstrap_curves
            ]
        )
        auc = float(np.trapz(mean_curve, dx=0.1))
        statistics[metric] = {
            "eligible_state_clusters": len(clusters),
            "aggregate_commitment_step": aggregate_step,
            "commitment_step_ci95_low": float(np.quantile(bootstrap_steps, 0.025)),
            "commitment_step_ci95_high": float(np.quantile(bootstrap_steps, 0.975)),
            "retention_auc": auc,
            "late_weighting_index": 0.5 - auc,
        }
        for boundary in range(11):
            values = bootstrap_curves[:, boundary]
            output.append(
                {
                    "metric": metric,
                    "switch_after_steps": boundary,
                    "state_clusters": len(clusters),
                    "mean_symmetric_retention": float(mean_curve[boundary]),
                    "isotonic_mean_retention": float(fitted[boundary]),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                }
            )
    return output, statistics


def _aggregate_formation(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = ("contrast_alignment", "contrast_relative_error", "contrast_cosine")
    state_rows = _state_means(rows, "flow_step", fields)
    rng = np.random.default_rng(args.bootstrap_seed + 1)
    output = []
    statistics = {}
    for metric in sorted({row["metric"] for row in state_rows}):
        selected = [row for row in state_rows if row["metric"] == metric and row["eligible"]]
        clusters = sorted({row["scene_state_sha256"] for row in selected})
        if not clusters:
            statistics[metric] = None
            continue
        matrices = {
            field: np.asarray(
                [
                    [
                        next(
                            row[field]
                            for row in selected
                            if row["scene_state_sha256"] == cluster and row["flow_step"] == step
                        )
                        for step in range(10)
                    ]
                    for cluster in clusters
                ]
            )
            for field in fields
        }
        bootstrap_indices = rng.integers(0, len(clusters), size=(args.bootstrap_replicates, len(clusters)))
        error_bootstrap = matrices["contrast_relative_error"][bootstrap_indices].mean(axis=1)
        mean_error = matrices["contrast_relative_error"].mean(axis=0)
        formation_step = _persistent_step(mean_error, args.formation_relative_error_tolerance)
        bootstrap_steps = np.asarray(
            [
                step if (step := _persistent_step(curve, args.formation_relative_error_tolerance)) is not None else 10
                for curve in error_bootstrap
            ]
        )
        statistics[metric] = {
            "eligible_state_clusters": len(clusters),
            "aggregate_formation_step": formation_step,
            "formation_step_ci95_low": float(np.quantile(bootstrap_steps, 0.025)),
            "formation_step_ci95_high": float(np.quantile(bootstrap_steps, 0.975)),
        }
        for step in range(10):
            errors = error_bootstrap[:, step]
            output.append(
                {
                    "metric": metric,
                    "flow_step": step,
                    "state_clusters": len(clusters),
                    "mean_contrast_alignment": float(matrices["contrast_alignment"].mean(axis=0)[step]),
                    "mean_contrast_relative_error": float(mean_error[step]),
                    "mean_contrast_cosine": float(matrices["contrast_cosine"].mean(axis=0)[step]),
                    "relative_error_ci95_low": float(np.quantile(errors, 0.025)),
                    "relative_error_ci95_high": float(np.quantile(errors, 0.975)),
                }
            )
    return output, statistics


def _state_means(
    rows: list[dict[str, Any]], index_field: str, value_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_state_sha256"], row["metric"], row[index_field])].append(row)
    output = []
    for (cluster, metric, index), selected in sorted(grouped.items()):
        eligibility = {row["eligible"] for row in selected}
        if len(eligibility) != 1:
            raise ValueError("metric eligibility differs across noise seeds within a state")
        record = {
            "scene_state_sha256": cluster,
            "metric": metric,
            index_field: index,
            "eligible": eligibility.pop(),
        }
        record.update({field: float(np.mean([row[field] for row in selected])) for field in value_fields})
        output.append(record)
    return output


def _persistent_step(values: np.ndarray, tolerance: float) -> int | None:
    for index, value in enumerate(values):
        if np.isfinite(value) and value <= tolerance and np.all(values[index:] <= tolerance):
            return index
    return None


def _plot_flow(rows: list[dict[str, Any]], stem: Path) -> None:
    metrics = sorted({row["metric"] for row in rows})
    figure, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.3), sharey=True, squeeze=False)
    for axis, metric in zip(axes[0], metrics, strict=True):
        selected = [row for row in rows if row["metric"] == metric]
        x = np.asarray([row["switch_after_steps"] for row in selected])
        y = np.asarray([row["mean_symmetric_retention"] for row in selected])
        low = np.asarray([row["ci95_low"] for row in selected])
        high = np.asarray([row["ci95_high"] for row in selected])
        axis.plot(x, y, "o-", color="#2166ac")
        axis.fill_between(x, low, high, color="#2166ac", alpha=0.18)
        axis.axhline(0.8, color="0.55", linewidth=0.8, linestyle="--")
        axis.set(title=metric.replace("_", " "), xlabel="source flow updates", ylim=(-0.1, 1.1))
    axes[0, 0].set_ylabel("source retention")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)


def _plot_formation(rows: list[dict[str, Any]], tolerance: float, stem: Path) -> None:
    metrics = sorted({row["metric"] for row in rows})
    figure, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.3), sharey=True, squeeze=False)
    for axis, metric in zip(axes[0], metrics, strict=True):
        selected = [row for row in rows if row["metric"] == metric]
        x = np.asarray([row["flow_step"] for row in selected])
        y = np.asarray([row["mean_contrast_relative_error"] for row in selected])
        low = np.asarray([row["relative_error_ci95_low"] for row in selected])
        high = np.asarray([row["relative_error_ci95_high"] for row in selected])
        axis.plot(x, y, "o-", color="#762a83")
        axis.fill_between(x, low, high, color="#762a83", alpha=0.18)
        axis.axhline(tolerance, color="0.55", linewidth=0.8, linestyle="--")
        axis.set(title=metric.replace("_", " "), xlabel="flow step")
    axes[0, 0].set_ylabel("contrast relative error")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)


def _plot_residual_cells(rows: list[dict[str, Any]], stem: Path) -> None:
    metrics = sorted({row["metric"] for row in rows})
    steps = sorted({row["flow_step"] for row in rows})
    layers = sorted({row["layer"] for row in rows})
    limit = max(abs(row["mean_symmetric_ncte"]) for row in rows) or 1.0
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(3.5 * len(metrics), 3.6),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, metric in zip(axes[0], metrics, strict=True):
        selected = {(row["layer"], row["flow_step"]): row for row in rows if row["metric"] == metric}
        values = np.asarray([[selected[(layer, step)]["mean_symmetric_ncte"] for step in steps] for layer in layers])
        image = axis.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        for y, layer in enumerate(layers):
            for x, step in enumerate(steps):
                row = selected[(layer, step)]
                if row["q_bh_within_metric_family"] < 0.05:
                    axis.plot(x, y, ".", color="black", markersize=3)
        axis.set(
            title=metric.replace("_", " "),
            xlabel="flow step",
            xticks=np.arange(len(steps)),
            xticklabels=steps,
            yticks=np.arange(len(layers)),
            yticklabels=layers,
        )
    axes[0, 0].set_ylabel("transformer layer")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="mean normalized causal transfer")
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)


def _plot_dimension_cells(rows: list[dict[str, Any]], stem: Path) -> None:
    metrics = sorted({row["metric"] for row in rows})
    tensors = sorted({row["patched_tensor"] for row in rows})
    groups = sorted({row["patched_dimension_group"] for row in rows})
    steps = sorted({row["flow_step"] for row in rows})
    limit = max(abs(row["mean_symmetric_ncte"]) for row in rows) or 1.0
    figure, axes = plt.subplots(
        len(metrics),
        len(tensors),
        figsize=(3.4 * len(tensors), 2.7 * len(metrics)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for metric_index, metric in enumerate(metrics):
        for tensor_index, tensor in enumerate(tensors):
            axis = axes[metric_index, tensor_index]
            selected = {
                (row["patched_dimension_group"], row["flow_step"]): row
                for row in rows
                if row["metric"] == metric and row["patched_tensor"] == tensor
            }
            values = np.asarray([[selected[(group, step)]["mean_symmetric_ncte"] for step in steps] for group in groups])
            image = axis.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            for y, group in enumerate(groups):
                for x, step in enumerate(steps):
                    if selected[(group, step)]["q_bh_within_metric_family"] < 0.05:
                        axis.plot(x, y, ".", color="black", markersize=3)
            axis.set(
                title=f"{metric.replace('_', ' ')} · {tensor}",
                xlabel="flow step" if metric_index == len(metrics) - 1 else None,
                xticks=np.arange(len(steps)),
                xticklabels=steps if metric_index == len(metrics) - 1 else [],
                yticks=np.arange(len(groups)),
                yticklabels=groups if tensor_index == 0 else [],
            )
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="mean normalized causal transfer")
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
