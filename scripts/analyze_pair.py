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
from action_chunking.metrics import LIBERO_ACTION_GROUPS

GROUPS = ("all", "translation", "rotation", "gripper")
DIRECTIONS = ("base_to_donor", "donor_to_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-action-contrast", type=float, default=0.01)
    parser.add_argument("--minimum-target-contrast", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.input / "metadata.json").read_text())
    records = [json.loads(line) for line in (args.input / "records.jsonl").read_text().splitlines()]

    contrasts = _endpoint_contrasts(args.input, records)
    validity = {
        metric: contrasts[metric]
        >= (args.minimum_target_contrast if metric == "target_direction" else args.minimum_action_contrast)
        for metric in (*GROUPS, "target_direction")
    }
    flow_rows, commitments, curve_statistics = _flow_summary(records, args.threshold, validity)
    _write_csv(args.output / "flow_retention.csv", flow_rows)
    _plot_flow(flow_rows, validity, args.output / "flow_retention")

    residual_rows = _residual_summary(records)
    intervention_peaks: dict[str, Any] = {}
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
        "endpoint_contrasts": contrasts,
        "minimum_action_contrast": args.minimum_action_contrast,
        "minimum_target_contrast": args.minimum_target_contrast,
        "valid_contrasts": validity,
        "intervention_peaks": intervention_peaks,
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
        suffix = "" if validity[metric] else " (low endpoint contrast)"
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


def _endpoint_contrasts(input_directory: Path, records: list[dict[str, Any]]) -> dict[str, float]:
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
    return contrasts


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
