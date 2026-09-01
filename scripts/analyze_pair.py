#!/usr/bin/env python3
"""Summarize and plot a completed single-pair intervention screen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from action_chunking.analysis import commitment_step, symmetric_mean
from action_chunking.metrics import (
    LIBERO_ACTION_GROUPS,
    gripper_closure_position,
    gripper_closure_time,
    normalized_causal_transfer,
)

GROUPS = ("all", "translation", "rotation", "gripper")
DIRECTIONS = ("base_to_donor", "donor_to_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-action-contrast", type=float, default=0.01)
    parser.add_argument("--minimum-target-contrast", type=float, default=0.01)
    parser.add_argument("--gripper-closure-threshold", type=float, default=0.0)
    parser.add_argument("--formation-relative-error-tolerance", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.input / "metadata.json").read_text())
    records = [json.loads(line) for line in (args.input / "records.jsonl").read_text().splitlines()]

    contrasts, closure_positions = _endpoint_contrasts(
        args.input,
        records,
        args.gripper_closure_threshold,
    )
    validity = {
        metric: contrasts[metric]
        >= (args.minimum_target_contrast if metric == "target_direction" else args.minimum_action_contrast)
        for metric in (*GROUPS, "target_direction")
    }
    validity["gripper"] = (
        validity["gripper"]
        and closure_positions["base"] != closure_positions["donor"]
        and any(position is not None for position in closure_positions.values())
    )
    flow_rows, commitments, curve_statistics = _flow_summary(records, args.threshold, validity)
    _write_csv(args.output / "flow_retention.csv", flow_rows)
    _plot_flow(flow_rows, validity, args.output / "flow_retention")

    formation_rows, formation_steps = _formation_contrast_summary(
        records,
        validity,
        args.formation_relative_error_tolerance,
    )
    _write_csv(args.output / "formation_contrast.csv", formation_rows)
    _plot_formation_contrast(formation_rows, validity, args.output / "formation_contrast")

    closure_analysis, closure_tables = _gripper_closure_analysis(
        records,
        args.gripper_closure_threshold,
        args.threshold,
        args.formation_relative_error_tolerance,
    )
    for name, rows in closure_tables.items():
        if rows:
            _write_csv(args.output / f"gripper_closure_{name}.csv", rows)

    residual_rows = _residual_summary(records)
    intervention_peaks: dict[str, Any] = {}
    token_mixing = []
    if residual_rows:
        _write_csv(args.output / "residual_transfer.csv", residual_rows)
        _plot_residual(
            residual_rows,
            metadata["num_steps"],
            metadata["layers"],
            validity,
            args.output / "residual_transfer",
        )
        intervention_peaks["residual_patch"] = _peak_summary(residual_rows, "symmetric_ncte", validity)

    position_rows = _position_summary(records)
    if position_rows:
        _write_csv(args.output / "position_transfer.csv", position_rows)
        _plot_positions(position_rows, validity, args.output / "position_transfer")
        intervention_peaks["position_patch"] = _peak_summary(position_rows, "symmetric_ncte", validity)
        position_output_rows = _position_output_summary(records, args.input, args.minimum_action_contrast)
        _write_csv(args.output / "position_to_output_transfer.csv", position_output_rows)
        _plot_position_to_output(position_output_rows, args.output / "position_to_output_transfer")
        token_mixing = _position_mixing_summary(position_output_rows)
        _write_csv(args.output / "position_mixing.csv", token_mixing)

    dimension_rows = _dimension_summary(records)
    if dimension_rows:
        _write_csv(args.output / "dimension_transfer.csv", dimension_rows)
        _plot_dimensions(dimension_rows, validity, args.output / "dimension_transfer")
        intervention_peaks["dimension_patch"] = _peak_summary(dimension_rows, "symmetric_ncte", validity)

    _plot_formation(records, args.output / "formation")
    summary = {
        "schema_version": 1,
        "pair_id": metadata["pair_id"],
        "commitment_threshold": args.threshold,
        "commitment_steps": commitments,
        "curve_statistics": curve_statistics,
        "formation_relative_error_tolerance": args.formation_relative_error_tolerance,
        "formation_steps": formation_steps,
        "endpoint_contrasts": contrasts,
        "minimum_action_contrast": args.minimum_action_contrast,
        "minimum_target_contrast": args.minimum_target_contrast,
        "valid_contrasts": validity,
        "gripper_closure_threshold": args.gripper_closure_threshold,
        "gripper_closure_positions": closure_positions,
        "gripper_closure_timing": closure_analysis,
        "intervention_peaks": intervention_peaks,
        "token_mixing": token_mixing,
        "endpoint_l2_contrast": metadata["endpoint_l2_contrast"],
        "controls": metadata["controls"],
        "interpretation_scope": "single-pair pilot; descriptive only",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _flow_summary(
    records: list[dict[str, Any]],
    threshold: float,
    validity: dict[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    flow = [record for record in records if record["family"] == "flow_switch"]
    by_direction = {direction: sorted((r for r in flow if r["direction"] == direction), key=_switch_key) for direction in DIRECTIONS}
    boundaries = [record["switch_after_steps"] for record in by_direction[DIRECTIONS[0]]]
    if boundaries != list(range(len(boundaries))):
        raise ValueError("flow-switch boundaries must form a complete zero-based grid")
    if [record["switch_after_steps"] for record in by_direction[DIRECTIONS[1]]] != boundaries:
        raise ValueError("flow-switch directions have different grids")

    rows = []
    commitments: dict[str, Any] = {}
    curve_statistics: dict[str, Any] = {}
    metrics = (*GROUPS, "target_direction")
    for metric in metrics:
        directional = {}
        for direction in DIRECTIONS:
            if metric == "target_direction":
                values = [record["target_direction_affinity"] for record in by_direction[direction]]
                source, destination = values[-1], values[0]
                contrast = destination - source
                directional[direction] = [
                    1.0 - (value - source) / contrast if abs(contrast) > 1e-12 else np.nan for value in values
                ]
            else:
                directional[direction] = [record["retention"][metric] for record in by_direction[direction]]
        symmetric = symmetric_mean(directional[DIRECTIONS[0]], directional[DIRECTIONS[1]])
        finite = np.all(np.isfinite(symmetric))
        if finite and validity[metric]:
            step, fitted = commitment_step(symmetric, threshold)
            half_step, _ = commitment_step(symmetric, 0.5)
            normalized_x = np.asarray(boundaries, dtype=np.float64) / boundaries[-1]
            auc = float(np.trapz(symmetric, normalized_x))
            curve_statistics[metric] = {
                "half_commitment_step": half_step,
                "retention_auc": auc,
                "late_weighting_index": 0.5 - auc,
                "final_step_marginal_retention": float(fitted[-1] - fitted[-2]),
            }
        else:
            step, fitted = None, np.full_like(symmetric, np.nan)
            curve_statistics[metric] = None
        commitments[metric] = step
        for index, boundary in enumerate(boundaries):
            rows.append(
                {
                    "metric": metric,
                    "switch_after_steps": boundary,
                    "base_to_donor_retention": directional["base_to_donor"][index],
                    "donor_to_base_retention": directional["donor_to_base"][index],
                    "symmetric_retention": float(symmetric[index]),
                    "isotonic_retention": float(fitted[index]),
                    "directional_asymmetry": abs(
                        directional["base_to_donor"][index] - directional["donor_to_base"][index]
                    ),
                }
            )
    return rows, commitments, curve_statistics


def _residual_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patch = [record for record in records if record["family"] == "residual_patch"]
    if not patch:
        return []
    sites: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in patch:
        sites[(record["flow_step"], record["layer"])][record["direction"]] = record
    rows = []
    for (step, layer), directions in sorted(sites.items()):
        if set(directions) != set(DIRECTIONS):
            raise ValueError(f"residual site {(step, layer)} lacks a patch direction")
        for group in GROUPS:
            values = [directions[direction]["metrics"]["ncte"][group] for direction in DIRECTIONS]
            rows.append(
                {
                    "flow_step": step,
                    "layer": layer,
                    "metric": group,
                    "base_to_donor_ncte": values[0],
                    "donor_to_base_ncte": values[1],
                    "symmetric_ncte": float(np.mean(values)),
                    "directional_asymmetry": abs(values[0] - values[1]),
                }
            )
    return rows


def _position_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patch = [record for record in records if record["family"] == "residual_patch_position"]
    if not patch:
        return []
    sites: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in patch:
        position = record["action_positions"]
        if not isinstance(position, list) or len(position) != 1:
            raise ValueError("position-patch records must contain one action position")
        sites[(record["flow_step"], record["layer"], position[0])][record["direction"]] = record
    rows = []
    for (step, layer, position), directions in sorted(sites.items()):
        if set(directions) != set(DIRECTIONS):
            raise ValueError(f"position site {(step, layer, position)} lacks a patch direction")
        for group in GROUPS:
            values = [directions[direction]["metrics"]["ncte"][group] for direction in DIRECTIONS]
            rows.append(
                {
                    "flow_step": step,
                    "layer": layer,
                    "action_position": position,
                    "metric": group,
                    "base_to_donor_ncte": values[0],
                    "donor_to_base_ncte": values[1],
                    "symmetric_ncte": float(np.mean(values)),
                    "directional_asymmetry": abs(values[0] - values[1]),
                }
            )
    return rows


def _position_output_summary(
    records: list[dict[str, Any]],
    input_directory: Path,
    minimum_contrast: float,
) -> list[dict[str, Any]]:
    patch = [record for record in records if record["family"] == "residual_patch_position"]
    sites: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in patch:
        position = record["action_positions"]
        if not isinstance(position, list) or len(position) != 1:
            raise ValueError("position-patch records must contain one action position")
        sites[(record["flow_step"], record["layer"], position[0])][record["direction"]] = record
    with np.load(input_directory / "clean_trace.npz") as trace:
        base = trace["base_actions"]
        donor = trace["donor_actions"]
    output_contrasts = np.linalg.norm(donor - base, axis=1)
    rows = []
    for (step, layer, patched_position), directions in sorted(sites.items()):
        if set(directions) != set(DIRECTIONS):
            raise ValueError(f"position site {(step, layer, patched_position)} lacks a patch direction")
        directional = [directions[direction]["metrics"]["per_position_ncte"] for direction in DIRECTIONS]
        if len(directional[0]) != len(output_contrasts) or len(directional[1]) != len(output_contrasts):
            raise ValueError("per-position effects do not match the clean action horizon")
        for output_position, contrast in enumerate(output_contrasts):
            values = [float(direction[output_position]) for direction in directional]
            eligible = bool(contrast >= minimum_contrast and np.all(np.isfinite(values)))
            rows.append(
                {
                    "flow_step": step,
                    "layer": layer,
                    "patched_action_position": patched_position,
                    "output_action_position": output_position,
                    "output_position_l2_contrast": float(contrast),
                    "eligible_output_position": eligible,
                    "base_to_donor_ncte": values[0],
                    "donor_to_base_ncte": values[1],
                    "symmetric_ncte": float(np.mean(values)) if eligible else np.nan,
                    "directional_asymmetry": abs(values[0] - values[1]) if eligible else np.nan,
                }
            )
    return rows


def _position_mixing_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sites = sorted({(row["flow_step"], row["layer"]) for row in rows})
    summaries = []
    for step, layer in sites:
        selected = [
            row
            for row in rows
            if row["flow_step"] == step and row["layer"] == layer and row["eligible_output_position"]
        ]
        diagonal = [
            float(row["symmetric_ncte"])
            for row in selected
            if row["patched_action_position"] == row["output_action_position"]
        ]
        off_diagonal = [
            float(row["symmetric_ncte"])
            for row in selected
            if row["patched_action_position"] != row["output_action_position"]
        ]
        total_absolute = sum(abs(value) for value in (*diagonal, *off_diagonal))
        summaries.append(
            {
                "flow_step": step,
                "layer": layer,
                "eligible_cells": len(selected),
                "mean_diagonal_ncte": float(np.mean(diagonal)),
                "mean_absolute_diagonal_ncte": float(np.mean(np.abs(diagonal))),
                "mean_absolute_off_diagonal_ncte": float(np.mean(np.abs(off_diagonal))),
                "off_diagonal_absolute_share": (
                    sum(abs(value) for value in off_diagonal) / total_absolute if total_absolute else np.nan
                ),
                "maximum_absolute_cell_ncte": max(abs(value) for value in (*diagonal, *off_diagonal)),
            }
        )
    return summaries


def _formation_contrast_summary(
    records: list[dict[str, Any]],
    validity: dict[str, bool],
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    if tolerance <= 0:
        raise ValueError("formation relative-error tolerance must be positive")
    formation = [record for record in records if record["family"] == "formation"]
    by_side = {
        side: {record["flow_step"]: record for record in formation if record["side"] == side}
        for side in ("base", "donor")
    }
    if set(by_side["base"]) != set(by_side["donor"]):
        raise ValueError("formation sides have different flow-step grids")
    clean = {record["direction"]: record for record in records if record["family"] == "clean"}
    final_actions = {
        side: np.asarray(clean[side]["actions"], dtype=np.float64) for side in ("base", "donor")
    }
    rows = []
    for step in sorted(by_side["base"]):
        estimates = {
            side: np.asarray(by_side[side][step]["actions"], dtype=np.float64)
            for side in ("base", "donor")
        }
        for metric in GROUPS:
            indices = LIBERO_ACTION_GROUPS[metric]
            final_contrast = final_actions["donor"][:, indices] - final_actions["base"][:, indices]
            current_contrast = estimates["donor"][:, indices] - estimates["base"][:, indices]
            rows.append(_formation_row(step, metric, current_contrast, final_contrast, validity[metric]))
        final_target = (
            clean["donor"]["target_direction_affinity"] - clean["base"]["target_direction_affinity"]
        )
        current_target = (
            by_side["donor"][step]["target_direction_affinity"]
            - by_side["base"][step]["target_direction_affinity"]
        )
        rows.append(
            _formation_row(
                step,
                "target_direction",
                np.asarray([current_target]),
                np.asarray([final_target]),
                validity["target_direction"],
            )
        )
    formation_steps = {}
    for metric in (*GROUPS, "target_direction"):
        selected = [row for row in rows if row["metric"] == metric]
        formation_steps[metric] = _persistent_formation_step(selected, tolerance) if validity[metric] else None
    return rows, formation_steps


def _formation_row(
    step: int,
    metric: str,
    current_contrast: np.ndarray,
    final_contrast: np.ndarray,
    eligible: bool,
) -> dict[str, Any]:
    current = current_contrast.reshape(-1)
    final = final_contrast.reshape(-1)
    denominator = float(np.dot(final, final))
    if not eligible or denominator <= 1e-12:
        alignment = relative_error = cosine = np.nan
    else:
        alignment = float(np.dot(current, final) / denominator)
        relative_error = float(np.linalg.norm(current - final) / np.sqrt(denominator))
        current_norm = float(np.linalg.norm(current))
        cosine = float(np.dot(current, final) / (current_norm * np.sqrt(denominator))) if current_norm else np.nan
    return {
        "flow_step": step,
        "metric": metric,
        "eligible": eligible,
        "contrast_alignment": alignment,
        "contrast_relative_error": relative_error,
        "contrast_cosine": cosine,
    }


def _persistent_formation_step(rows: list[dict[str, Any]], tolerance: float) -> int | None:
    errors = [float(row["contrast_relative_error"]) for row in rows]
    for index, error in enumerate(errors):
        if np.isfinite(error) and error <= tolerance and all(value <= tolerance for value in errors[index:]):
            return int(rows[index]["flow_step"])
    return None


def _dimension_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patch = [record for record in records if record["family"] == "action_dimension_patch"]
    if not patch:
        return []
    sites: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in patch:
        key = (record["flow_step"], record["patched_tensor"], record["action_dimension_group"])
        sites[key][record["direction"]] = record
    rows = []
    for (step, tensor_name, patched_group), directions in sorted(sites.items()):
        if set(directions) != set(DIRECTIONS):
            raise ValueError(f"dimension site {(step, tensor_name, patched_group)} lacks a patch direction")
        for outcome_group in GROUPS:
            values = [directions[direction]["metrics"]["ncte"][outcome_group] for direction in DIRECTIONS]
            rows.append(
                {
                    "flow_step": step,
                    "patched_tensor": tensor_name,
                    "patched_dimension_group": patched_group,
                    "metric": outcome_group,
                    "base_to_donor_ncte": values[0],
                    "donor_to_base_ncte": values[1],
                    "symmetric_ncte": float(np.mean(values)),
                    "directional_asymmetry": abs(values[0] - values[1]),
                }
            )
    return rows


def _plot_flow(rows: list[dict[str, Any]], validity: dict[str, bool], output_stem: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.3), sharex=True, sharey=True)
    for axis, metric in zip(axes.flat, (*GROUPS, "target_direction"), strict=False):
        selected = [row for row in rows if row["metric"] == metric]
        x = [row["switch_after_steps"] for row in selected]
        axis.plot([x[0], x[-1]], [0.0, 1.0], ":", color="0.65", label="uniform-step null")
        axis.plot(x, [row["base_to_donor_retention"] for row in selected], "--", alpha=0.6, label="A→B")
        axis.plot(x, [row["donor_to_base_retention"] for row in selected], "--", alpha=0.6, label="B→A")
        axis.plot(x, [row["isotonic_retention"] for row in selected], "o-", color="black", label="symmetric isotonic")
        axis.axhline(0.8, color="0.6", linewidth=0.8)
        suffix = "" if validity[metric] else " (ineligible clean outcome)"
        axis.set_title(metric.replace("_", " ") + suffix)
        axis.set_ylim(-0.15, 1.15)
        axis.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes[1, 0].set_xlabel("flow updates before condition switch")
    axes[0, 0].set_ylabel("source-condition retention")
    axes[1, 0].set_ylabel("source-condition retention")
    axes[0, 0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_stem)


def _plot_residual(
    rows: list[dict[str, Any]],
    steps: int,
    layers: int,
    validity: dict[str, bool],
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True, sharey=True, layout="constrained")
    values = np.asarray(
        [row["symmetric_ncte"] for row in rows if validity[row["metric"]]],
        dtype=np.float64,
    )
    limit = max(float(np.nanpercentile(np.abs(values), 98)), 0.05)
    image = None
    for axis, metric in zip(axes.flat, GROUPS, strict=True):
        if not validity[metric]:
            axis.text(
                0.5,
                0.5,
                "not analyzed\nlow clean endpoint contrast",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(metric)
            axis.set_axis_off()
            continue
        heatmap = np.full((steps, layers), np.nan)
        for row in rows:
            if row["metric"] == metric:
                heatmap[row["flow_step"], row["layer"]] = row["symmetric_ncte"]
        image = axis.imshow(heatmap, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(metric)
        axis.set_xlabel("action-expert layer")
        axis.set_ylabel("flow step")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label="symmetric normalized causal transfer", shrink=0.85)
    _save_figure(figure, output_stem)


def _plot_formation(records: list[dict[str, Any]], output_stem: Path) -> None:
    formation = [record for record in records if record["family"] == "formation"]
    figure, axis = plt.subplots(figsize=(5.8, 3.6))
    for side in ("base", "donor"):
        selected = sorted((record for record in formation if record["side"] == side), key=lambda row: row["flow_step"])
        axis.plot(
            [record["flow_step"] for record in selected],
            [record["final_relative_l2_error"] for record in selected],
            "o-",
            label=side,
        )
    axis.set_xlabel("flow step")
    axis.set_ylabel("relative L2 error to final chunk")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_stem)


def _plot_formation_contrast(
    rows: list[dict[str, Any]],
    validity: dict[str, bool],
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, layout="constrained")
    for metric in (*GROUPS, "target_direction"):
        if not validity[metric]:
            continue
        selected = [row for row in rows if row["metric"] == metric]
        steps = [row["flow_step"] for row in selected]
        axes[0].plot(steps, [row["contrast_alignment"] for row in selected], "o-", label=metric)
        axes[1].plot(steps, [row["contrast_relative_error"] for row in selected], "o-", label=metric)
    axes[0].axhline(1.0, color="0.6", linewidth=0.8)
    axes[0].set_ylabel("alignment with final paired contrast")
    axes[1].set_ylabel("relative error to final paired contrast")
    for axis in axes:
        axis.set_xlabel("flow step")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, frameon=False)
    _save_figure(figure, output_stem)


def _plot_positions(rows: list[dict[str, Any]], validity: dict[str, bool], output_stem: Path) -> None:
    sites = sorted({(row["flow_step"], row["layer"]) for row in rows})
    positions = sorted({row["action_position"] for row in rows})
    valid_values = [row["symmetric_ncte"] for row in rows if validity[row["metric"]]]
    limit = max(float(np.nanpercentile(np.abs(valid_values), 98)), 0.05)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True, sharey=True, layout="constrained")
    image = None
    for axis, metric in zip(axes.flat, GROUPS, strict=True):
        if not validity[metric]:
            axis.text(
                0.5,
                0.5,
                "not analyzed\nlow clean endpoint contrast",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(metric)
            axis.set_axis_off()
            continue
        heatmap = np.full((len(sites), len(positions)), np.nan)
        for row in rows:
            if row["metric"] == metric:
                heatmap[sites.index((row["flow_step"], row["layer"])), positions.index(row["action_position"])] = row[
                    "symmetric_ncte"
                ]
        image = axis.imshow(heatmap, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(metric)
        axis.set_yticks(range(len(sites)), [f"s{s} l{layer}" for s, layer in sites])
        axis.set_xticks(range(len(positions)), positions)
        axis.set_xlabel("future action position")
        axis.set_ylabel("flow/layer site")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label="symmetric normalized causal transfer", shrink=0.85)
    _save_figure(figure, output_stem)


def _plot_position_to_output(rows: list[dict[str, Any]], output_stem: Path) -> None:
    sites = sorted({(row["flow_step"], row["layer"]) for row in rows})
    patched_positions = sorted({row["patched_action_position"] for row in rows})
    output_positions = sorted({row["output_action_position"] for row in rows})
    values = np.asarray(
        [row["symmetric_ncte"] for row in rows if row["eligible_output_position"]],
        dtype=np.float64,
    )
    limit = max(float(np.nanpercentile(np.abs(values), 98)), 0.05)
    columns = min(4, len(sites))
    row_count = (len(sites) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(3.1 * columns, 2.7 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for axis, site in zip(axes.flat, sites, strict=False):
        heatmap = np.full((len(output_positions), len(patched_positions)), np.nan)
        for row in rows:
            if (row["flow_step"], row["layer"]) == site:
                output_index = output_positions.index(row["output_action_position"])
                patched_index = patched_positions.index(row["patched_action_position"])
                heatmap[output_index, patched_index] = row["symmetric_ncte"]
        image = axis.imshow(heatmap, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"flow {site[0]}, layer {site[1]}")
        axis.set_xticks(range(len(patched_positions)), patched_positions)
        axis.set_yticks(range(len(output_positions)), output_positions)
        axis.set_xlabel("patched token")
        axis.set_ylabel("output token")
    for axis in axes.flat[len(sites) :]:
        axis.set_axis_off()
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label="symmetric per-output NCTE", shrink=0.85)
    _save_figure(figure, output_stem)


def _plot_dimensions(rows: list[dict[str, Any]], validity: dict[str, bool], output_stem: Path) -> None:
    tensors = sorted({row["patched_tensor"] for row in rows})
    patched_groups = sorted({row["patched_dimension_group"] for row in rows})
    steps = sorted({row["flow_step"] for row in rows})
    valid_outcomes = [group for group in GROUPS if validity[group]]
    valid_values = [row["symmetric_ncte"] for row in rows if validity[row["metric"]]]
    limit = max(float(np.nanpercentile(np.abs(valid_values), 98)), 0.05)
    figure, axes = plt.subplots(
        len(tensors),
        len(patched_groups),
        figsize=(3.4 * len(patched_groups), 2.8 * len(tensors)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    image = None
    for tensor_index, tensor_name in enumerate(tensors):
        for group_index, patched_group in enumerate(patched_groups):
            axis = axes[tensor_index, group_index]
            heatmap = np.full((len(valid_outcomes), len(steps)), np.nan)
            for row in rows:
                if (
                    row["patched_tensor"] == tensor_name
                    and row["patched_dimension_group"] == patched_group
                    and row["metric"] in valid_outcomes
                ):
                    heatmap[valid_outcomes.index(row["metric"]), steps.index(row["flow_step"])] = row[
                        "symmetric_ncte"
                    ]
            image = axis.imshow(heatmap, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set_title(f"patch {tensor_name}: {patched_group}")
            axis.set_xticks(range(len(steps)), steps)
            axis.set_yticks(range(len(valid_outcomes)), valid_outcomes)
            axis.set_xlabel("flow step")
            axis.set_ylabel("output metric")
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), label="symmetric normalized causal transfer", shrink=0.85)
    _save_figure(figure, output_stem)


def _save_figure(figure: Any, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".png"), dpi=200)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _switch_key(record: dict[str, Any]) -> int:
    return int(record["switch_after_steps"])


def _endpoint_contrasts(
    input_directory: Path,
    records: list[dict[str, Any]],
    gripper_closure_threshold: float,
) -> tuple[dict[str, float], dict[str, int | None]]:
    with np.load(input_directory / "clean_trace.npz") as trace:
        base = trace["base_actions"]
        donor = trace["donor_actions"]
    contrasts = {
        name: float(np.linalg.norm(donor[:, indices] - base[:, indices]))
        for name, indices in LIBERO_ACTION_GROUPS.items()
    }
    clean = {record["direction"]: record for record in records if record["family"] == "clean"}
    contrasts["target_direction"] = abs(
        clean["donor"]["target_direction_affinity"] - clean["base"]["target_direction_affinity"]
    )
    closure_positions = {
        "base": gripper_closure_position(base, threshold=gripper_closure_threshold),
        "donor": gripper_closure_position(donor, threshold=gripper_closure_threshold),
    }
    return contrasts, closure_positions


def _gripper_closure_analysis(
    records: list[dict[str, Any]],
    closure_threshold: float,
    commitment_threshold: float,
    formation_tolerance: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    clean_records = {record["direction"]: record for record in records if record["family"] == "clean"}
    clean_actions = {
        side: np.asarray(clean_records[side]["actions"], dtype=np.float64)
        for side in ("base", "donor")
    }
    clean_positions = {
        side: gripper_closure_position(actions, threshold=closure_threshold)
        for side, actions in clean_actions.items()
    }
    clean_times = {
        side: gripper_closure_time(actions, threshold=closure_threshold)
        for side, actions in clean_actions.items()
    }
    eligible = clean_times["base"] != clean_times["donor"]
    tables: dict[str, list[dict[str, Any]]] = {
        "flow_retention": [],
        "formation": [],
        "residual_transfer": [],
        "position_transfer": [],
        "dimension_transfer": [],
    }
    summary: dict[str, Any] = {
        "eligible": eligible,
        "right_censoring_value": clean_actions["base"].shape[0],
        "clean_positions": clean_positions,
        "clean_times": clean_times,
        "commitment_step": None,
        "formation_step": None,
        "curve_statistics": None,
        "intervention_peaks": {},
    }
    if not eligible:
        return summary, tables

    flow = [record for record in records if record["family"] == "flow_switch"]
    directional_retention: dict[str, list[float]] = {}
    boundaries = None
    for direction in DIRECTIONS:
        selected = sorted((record for record in flow if record["direction"] == direction), key=_switch_key)
        current_boundaries = [int(record["switch_after_steps"]) for record in selected]
        if boundaries is None:
            boundaries = current_boundaries
        elif current_boundaries != boundaries:
            raise ValueError("gripper-closure flow directions have different grids")
        source_side, destination_side = (
            ("base", "donor") if direction == "base_to_donor" else ("donor", "base")
        )
        directional_retention[direction] = [
            1.0
            - normalized_causal_transfer(
                [gripper_closure_time(record["actions"], threshold=closure_threshold)],
                [clean_times[source_side]],
                [clean_times[destination_side]],
            )
            for record in selected
        ]
    if boundaries != list(range(len(boundaries or []))):
        raise ValueError("gripper-closure flow boundaries must form a complete zero-based grid")
    symmetric = symmetric_mean(
        directional_retention["base_to_donor"],
        directional_retention["donor_to_base"],
    )
    commitment, fitted = commitment_step(symmetric, commitment_threshold)
    half_commitment, _ = commitment_step(symmetric, 0.5)
    normalized_x = np.asarray(boundaries, dtype=np.float64) / boundaries[-1]
    auc = float(np.trapz(symmetric, normalized_x))
    summary["commitment_step"] = commitment
    summary["curve_statistics"] = {
        "half_commitment_step": half_commitment,
        "retention_auc": auc,
        "late_weighting_index": 0.5 - auc,
        "final_step_marginal_retention": float(fitted[-1] - fitted[-2]),
    }
    for index, boundary in enumerate(boundaries):
        tables["flow_retention"].append(
            {
                "switch_after_steps": boundary,
                "base_to_donor_retention": directional_retention["base_to_donor"][index],
                "donor_to_base_retention": directional_retention["donor_to_base"][index],
                "symmetric_retention": float(symmetric[index]),
                "isotonic_retention": float(fitted[index]),
                "directional_asymmetry": abs(
                    directional_retention["base_to_donor"][index]
                    - directional_retention["donor_to_base"][index]
                ),
            }
        )

    formation = [record for record in records if record["family"] == "formation"]
    by_side = {
        side: {int(record["flow_step"]): record for record in formation if record["side"] == side}
        for side in ("base", "donor")
    }
    if set(by_side["base"]) != set(by_side["donor"]):
        raise ValueError("gripper-closure formation sides have different grids")
    final_contrast = float(clean_times["donor"] - clean_times["base"])
    for step in sorted(by_side["base"]):
        current_contrast = float(
            gripper_closure_time(by_side["donor"][step]["actions"], threshold=closure_threshold)
            - gripper_closure_time(by_side["base"][step]["actions"], threshold=closure_threshold)
        )
        relative_error = abs(current_contrast - final_contrast) / abs(final_contrast)
        tables["formation"].append(
            {
                "flow_step": step,
                "base_closure_time": gripper_closure_time(
                    by_side["base"][step]["actions"], threshold=closure_threshold
                ),
                "donor_closure_time": gripper_closure_time(
                    by_side["donor"][step]["actions"], threshold=closure_threshold
                ),
                "contrast": current_contrast,
                "final_contrast": final_contrast,
                "contrast_alignment": current_contrast / final_contrast,
                "contrast_relative_error": relative_error,
            }
        )
    summary["formation_step"] = _persistent_formation_step(
        [
            {"flow_step": row["flow_step"], "contrast_relative_error": row["contrast_relative_error"]}
            for row in tables["formation"]
        ],
        formation_tolerance,
    )

    family_specs = {
        "residual_transfer": ("residual_patch", ("flow_step", "layer")),
        "position_transfer": ("residual_patch_position", ("flow_step", "layer", "action_position")),
        "dimension_transfer": (
            "action_dimension_patch",
            ("flow_step", "patched_tensor", "action_dimension_group"),
        ),
    }
    for table_name, (family, fields) in family_specs.items():
        rows = _gripper_closure_patch_rows(
            records,
            family,
            fields,
            clean_times,
            closure_threshold,
        )
        tables[table_name] = rows
        if rows:
            summary["intervention_peaks"][family] = {
                "maximum": max(rows, key=lambda row: row["symmetric_ncte"]),
                "minimum": min(rows, key=lambda row: row["symmetric_ncte"]),
            }
    return summary, tables


def _gripper_closure_patch_rows(
    records: list[dict[str, Any]],
    family: str,
    fields: tuple[str, ...],
    clean_times: dict[str, int],
    threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["family"] != family:
            continue
        values = []
        for field in fields:
            if field == "action_position":
                positions = record["action_positions"]
                if not isinstance(positions, list) or len(positions) != 1:
                    raise ValueError("closure position patch must identify one action position")
                values.append(int(positions[0]))
            else:
                values.append(record[field])
        grouped[tuple(values)][record["direction"]] = record
    output = []
    for site, directions in sorted(grouped.items()):
        if set(directions) != set(DIRECTIONS):
            raise ValueError(f"gripper-closure site {site} lacks a patch direction")
        values = []
        for direction in DIRECTIONS:
            source_side, destination_side = (
                ("base", "donor") if direction == "base_to_donor" else ("donor", "base")
            )
            current = gripper_closure_time(directions[direction]["actions"], threshold=threshold)
            values.append(
                normalized_causal_transfer(
                    [current],
                    [clean_times[source_side]],
                    [clean_times[destination_side]],
                )
            )
        row = {field: value for field, value in zip(fields, site, strict=True)}
        row.update(
            {
                "base_to_donor_ncte": values[0],
                "donor_to_base_ncte": values[1],
                "symmetric_ncte": float(np.mean(values)),
                "directional_asymmetry": abs(values[0] - values[1]),
            }
        )
        output.append(row)
    return output


def _peak_summary(
    rows: list[dict[str, Any]],
    value_key: str,
    validity: dict[str, bool],
) -> dict[str, Any]:
    summary = {}
    for metric in GROUPS:
        if not validity[metric]:
            summary[metric] = None
            continue
        selected = [row for row in rows if row["metric"] == metric]
        positive = max(selected, key=lambda row: row[value_key])
        negative = min(selected, key=lambda row: row[value_key])
        summary[metric] = {
            "maximum": positive,
            "minimum": negative,
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
