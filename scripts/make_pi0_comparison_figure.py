#!/usr/bin/env python3
"""Render the preregistered matched pi0-versus-pi0.5 comparison panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from action_chunking.pi0_intervention import COMPARISON_OUTPUT_FILENAMES

PRIMARY_METRIC = "all"
PRIMARY_POSITION_LAYER = 17
FLOW_STEPS = tuple(range(10))
LAYERS = tuple(range(18))
POSITION_FLOW_STEPS = (0, 7, 8, 9)
POSITION_BINS = tuple(range(10))
DIMENSION_GROUPS = ("translation", "rotation", "gripper")
PATCHED_TENSORS = ("x_t", "v_t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_audited_comparison(
    comparison_root: Path,
    final_audit_path: Path,
) -> dict[str, Any]:
    summary_path = comparison_root / "summary.json"
    summary = _read_json(summary_path)
    audit = _read_json(final_audit_path)
    required_summary = {
        "schema_version": 1,
        "analysis_unit": "paired_scene_state",
        "comparison": "pi05_minus_pi0",
        "pi05_action_horizon": 10,
        "pi0_action_horizon": 50,
        "primary_position_window": list(range(10)),
        "normalized_position_bins": 10,
        "bootstrap_replicates": 10_000,
    }
    _require_fields(summary, required_summary, "matched-control summary")
    required_audit = {
        "schema_version": 1,
        "passed": True,
        "comparison_summary_sha256": _digest(summary_path),
        "comparison_source_files": 14,
        "intervention_gpus": 2,
    }
    _require_fields(audit, required_audit, "matched-control final audit")
    if len(summary.get("source_files", {})) != 14:
        raise ValueError("matched-control summary does not bind 14 source files")
    for name, source in summary["source_files"].items():
        path = Path(str(source.get("path", "")))
        if not path.is_file() or source.get("sha256") != _digest(path):
            raise ValueError(f"matched-control source changed after analysis: {name}")
    generated = summary.get("output_files", {})
    if set(generated) != set(COMPARISON_OUTPUT_FILENAMES):
        raise ValueError("matched-control summary does not bind all 10 generated CSV files")
    for name in COMPARISON_OUTPUT_FILENAMES:
        path = comparison_root / name
        if not path.is_file() or generated.get(name) != _digest(path):
            raise ValueError(f"matched-control output changed after analysis: {name}")

    timing = summary.get("timing")
    if not isinstance(timing, dict) or not timing:
        raise ValueError("matched-control summary contains no paired timing metrics")
    timing_fields = (
        "formation_step_difference_pi05_minus_pi0",
        "editability_boundary_difference_pi05_minus_pi0",
    )
    for metric, values in timing.items():
        if not isinstance(values, dict):
            raise ValueError(f"matched-control timing metric is invalid: {metric}")
        for field in timing_fields:
            record = values.get(field)
            if (
                not isinstance(record, dict)
                or int(record.get("eligible_state_clusters", 0)) <= 0
                or record.get("mean_difference") is None
                or record.get("ci95_low") is None
                or record.get("ci95_high") is None
            ):
                raise ValueError(f"matched-control timing interval is incomplete: {metric}/{field}")
            numeric = tuple(
                float(record[key]) for key in ("mean_difference", "ci95_low", "ci95_high")
            )
            if not all(math.isfinite(value) for value in numeric) or not numeric[1] <= numeric[0] <= numeric[2]:
                raise ValueError(f"matched-control timing interval is invalid: {metric}/{field}")

    residual = _read_csv(comparison_root / "paired_residual_cells.csv")
    dimension = _read_csv(comparison_root / "paired_dimension_cells.csv")
    positions = _read_csv(comparison_root / "paired_position_normalized_cells.csv")
    _require_exact_cells(
        residual,
        {(step, layer) for step in FLOW_STEPS for layer in LAYERS},
        ("flow_step", "layer"),
        "residual",
    )
    _require_exact_cells(
        dimension,
        {
            (step, tensor, group)
            for step in FLOW_STEPS
            for tensor in PATCHED_TENSORS
            for group in DIMENSION_GROUPS
        },
        ("flow_step", "patched_tensor", "patched_dimension_group"),
        "action-dimension",
    )
    _require_exact_cells(
        positions,
        {
            (step, layer, position)
            for step in POSITION_FLOW_STEPS
            for layer in (0, 8, 14, 17)
            for position in POSITION_BINS
        },
        ("flow_step", "layer", "normalized_position_bin"),
        "normalized-position",
    )
    return {
        "summary": summary,
        "summary_path": summary_path,
        "audit": audit,
        "audit_path": final_audit_path,
        "residual": [row for row in residual if row["metric"] == PRIMARY_METRIC],
        "dimension": [row for row in dimension if row["metric"] == PRIMARY_METRIC],
        "positions": [row for row in positions if row["metric"] == PRIMARY_METRIC],
    }


def make_matched_control_figure(
    comparison_root: Path,
    final_audit_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"matched-control figure output already exists: {output}")
    data = load_audited_comparison(comparison_root, final_audit_path)
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.35), constrained_layout=True)
    _plot_timing(axes[0], data["summary"]["timing"])
    _plot_residual(figure, axes[1], data["residual"])
    _plot_dimensions(axes[2], data["dimension"])
    _plot_positions(figure, axes[3], data["positions"])
    figure.suptitle("Matched model comparison (pi0.5 minus pi0)", fontsize=10.3)
    stem = output / "fig_pi0_matched_control"
    for suffix in ("pdf", "png"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=240)
    plt.close(figure)
    outputs = ["fig_pi0_matched_control.pdf", "fig_pi0_matched_control.png"]
    manifest = {
        "schema_version": 1,
        "artifact": "pi0_matched_control_figure",
        "comparison_summary_sha256": _digest(data["summary_path"]),
        "final_audit_sha256": _digest(data["audit_path"]),
        "comparison_input_sha256": {
            name: _digest(comparison_root / name)
            for name in (
                "paired_residual_cells.csv",
                "paired_dimension_cells.csv",
                "paired_position_normalized_cells.csv",
            )
        },
        "primary_metric": PRIMARY_METRIC,
        "primary_position_layer": PRIMARY_POSITION_LAYER,
        "panels": [
            "paired_formation_and_editability_timing",
            "all_metric_residual_flow_by_layer",
            "all_metric_action_state_dimension_groups",
            "all_metric_normalized_position_at_layer_17",
        ],
        "outputs": outputs,
        "output_sha256": {name: _digest(output / name) for name in outputs},
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _plot_timing(axis: Any, timing: dict[str, Any]) -> None:
    metrics = sorted(timing)
    y = np.arange(len(metrics), dtype=np.float64)
    specs = (
        ("formation_step_difference_pi05_minus_pi0", -0.10, "#377eb8", "formation"),
        ("editability_boundary_difference_pi05_minus_pi0", 0.10, "#e41a1c", "editability"),
    )
    for field, offset, color, label in specs:
        means = np.asarray([float(timing[metric][field]["mean_difference"]) for metric in metrics])
        lows = np.asarray([float(timing[metric][field]["ci95_low"]) for metric in metrics])
        highs = np.asarray([float(timing[metric][field]["ci95_high"]) for metric in metrics])
        axis.errorbar(
            means,
            y + offset,
            xerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            color=color,
            capsize=2,
            markersize=3.5,
            label=label,
        )
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    axis.set(
        title="a  Paired timing",
        xlabel="step difference (pi0.5 - pi0)",
        yticks=y,
        yticklabels=[metric.replace("_", " ") for metric in metrics],
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_residual(figure: Any, axis: Any, rows: list[dict[str, str]]) -> None:
    lookup = {(int(row["layer"]), int(row["flow_step"])): float(row["mean_difference_pi05_minus_pi0"]) for row in rows}
    matrix = np.asarray([[lookup[(layer, step)] for step in FLOW_STEPS] for layer in LAYERS])
    limit = max(float(np.max(np.abs(matrix))), 1e-9)
    image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set(
        title="b  Residual mediation",
        xlabel="flow update",
        ylabel="action-expert layer",
        xticks=FLOW_STEPS,
        yticks=(0, 4, 8, 12, 17),
    )
    figure.colorbar(image, ax=axis, shrink=0.72, label="mean NCTE difference")


def _plot_dimensions(axis: Any, rows: list[dict[str, str]]) -> None:
    colors = {"translation": "#1b9e77", "rotation": "#7570b3", "gripper": "#d95f02"}
    styles = {"x_t": "-", "v_t": "--"}
    for group in DIMENSION_GROUPS:
        for tensor in PATCHED_TENSORS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["patched_dimension_group"] == group and row["patched_tensor"] == tensor
                ),
                key=lambda row: int(row["flow_step"]),
            )
            axis.plot(
                FLOW_STEPS,
                [float(row["mean_difference_pi05_minus_pi0"]) for row in selected],
                color=colors[group],
                linestyle=styles[tensor],
                marker="o",
                markersize=2.5,
                linewidth=1,
                label=f"{group} {tensor}",
            )
    axis.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
    axis.set(
        title="c  Action-state groups",
        xlabel="flow update",
        ylabel="mean NCTE difference",
        xticks=FLOW_STEPS,
    )
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_positions(figure: Any, axis: Any, rows: list[dict[str, str]]) -> None:
    selected = [row for row in rows if int(row["layer"]) == PRIMARY_POSITION_LAYER]
    lookup = {
        (int(row["flow_step"]), int(row["normalized_position_bin"])): float(
            row["mean_difference_pi05_minus_pi0"]
        )
        for row in selected
    }
    matrix = np.asarray(
        [[lookup[(step, position)] for position in POSITION_BINS] for step in POSITION_FLOW_STEPS]
    )
    limit = max(float(np.max(np.abs(matrix))), 1e-9)
    image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set(
        title="d  Normalized chunk position",
        xlabel="normalized position bin",
        ylabel="flow update",
        xticks=POSITION_BINS,
        yticks=np.arange(len(POSITION_FLOW_STEPS)),
        yticklabels=POSITION_FLOW_STEPS,
    )
    figure.colorbar(image, ax=axis, shrink=0.72, label="mean NCTE difference")


def _require_exact_cells(
    rows: list[dict[str, str]],
    expected: set[tuple[Any, ...]],
    fields: tuple[str, ...],
    label: str,
) -> None:
    selected = [row for row in rows if row.get("metric") == PRIMARY_METRIC]
    cells = {tuple(_cell_value(row[field]) for field in fields) for row in selected}
    if len(selected) != len(expected) or cells != expected:
        raise ValueError(f"matched-control {label} figure grid is incomplete")
    if any(int(row.get("eligible_state_clusters", 0)) <= 0 for row in selected):
        raise ValueError(f"matched-control {label} figure grid has no eligible states")
    if any(
        not math.isfinite(float(row.get("mean_difference_pi05_minus_pi0", math.nan)))
        for row in selected
    ):
        raise ValueError(f"matched-control {label} figure grid has a non-finite estimate")


def _cell_value(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"matched-control figure input is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_fields(value: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatched = {
        key: {"expected": wanted, "actual": value.get(key)}
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if mismatched:
        raise ValueError(f"{label} is incompatible: {mismatched}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    manifest = make_matched_control_figure(
        args.comparison_root,
        args.final_audit,
        args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
