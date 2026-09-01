#!/usr/bin/env python3
"""Build compact manuscript figures from immutable analysis tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "all": "#3b4cc0",
    "translation": "#1b9e77",
    "rotation": "#d95f02",
    "target_direction": "#7570b3",
    "source": "#2166ac",
    "donor": "#b2182b",
}
LABELS = {
    "all": "full action",
    "translation": "translation",
    "rotation": "rotation",
    "target_direction": "target direction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--target-online", type=Path, required=True)
    parser.add_argument("--destination-online", type=Path, required=True)
    parser.add_argument("--closure-position", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "formation_curve": args.primary / "formation_curve.csv",
        "flow_curve": args.primary / "flow_curve.csv",
        "residual_cells": args.primary / "residual_cells.csv",
        "dimension_cells": args.primary / "dimension_cells.csv",
        "target_curve": args.target_online / "curve.csv",
        "destination_curve": args.destination_online / "curve.csv",
        "closure_position_cells": args.closure_position / "position_cells.csv",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    _plot_formation_and_editability(
        _read_csv(inputs["formation_curve"]),
        _read_csv(inputs["flow_curve"]),
        _read_csv(inputs["target_curve"]),
        args.output / "fig1_formation_and_editability",
    )
    _plot_causal_localization(
        _read_csv(inputs["residual_cells"]),
        _read_csv(inputs["dimension_cells"]),
        args.output / "fig2_causal_localization",
    )
    _plot_behavioral_editability(
        _read_csv(inputs["target_curve"]),
        _read_csv(inputs["destination_curve"]),
        args.output / "fig3_behavioral_editability",
    )
    _plot_closure_token_case(
        _read_csv(inputs["closure_position_cells"]),
        args.output / "fig4_closure_token_case",
    )
    manifest = {
        "schema_version": 1,
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in inputs.items()},
        "outputs": [
            f"fig{index}_{name}.{suffix}"
            for index, name in (
                (1, "formation_and_editability"),
                (2, "causal_localization"),
                (3, "behavioral_editability"),
                (4, "closure_token_case"),
            )
            for suffix in ("pdf", "png")
        ],
        "marker_definition": "open circles require BH q < 0.05 and a scene-cluster 95% CI strictly above zero",
        "closure_caveat": "descriptive one-state, one-noise-mode case; no population significance markers",
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


def _plot_formation_and_editability(
    formation_rows: list[dict[str, str]],
    flow_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    stem: Path,
) -> None:
    metrics = ["all", "translation", "rotation", "target_direction"]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    for metric in metrics:
        selected = sorted((row for row in formation_rows if row["metric"] == metric), key=_flow_step)
        x = np.asarray([_flow_step(row) for row in selected])
        y = np.asarray([float(row["mean_contrast_relative_error"]) for row in selected])
        low = np.asarray([float(row["relative_error_ci95_low"]) for row in selected])
        high = np.asarray([float(row["relative_error_ci95_high"]) for row in selected])
        axes[0].plot(x, y, marker="o", ms=3, color=COLORS[metric], label=LABELS[metric])
        axes[0].fill_between(x, low, high, color=COLORS[metric], alpha=0.12, linewidth=0)
    axes[0].axhline(0.2, color="0.35", linestyle="--", linewidth=1, label="formation tolerance")
    axes[0].set(title="a  Clean contrast formation", xlabel="flow step", ylabel="relative error", xticks=range(10))
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    for metric in metrics:
        selected = sorted((row for row in flow_rows if row["metric"] == metric), key=_switch_step)
        x = np.asarray([_switch_step(row) for row in selected])
        y = np.asarray([float(row["isotonic_mean_retention"]) for row in selected])
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        axes[1].plot(x, y, marker="o", ms=3, color=COLORS[metric], label=LABELS[metric])
        axes[1].fill_between(x, low, high, color=COLORS[metric], alpha=0.12, linewidth=0)
    axes[1].axhline(0.8, color="0.35", linestyle="--", linewidth=1)
    axes[1].set(
        title="b  Continuous action editability",
        xlabel="source-conditioned updates before switch",
        ylabel="source retention",
        xticks=range(11),
        ylim=(-0.03, 1.05),
    )

    _plot_categorical_curve(axes[2], target_rows)
    axes[2].set(title="c  Behavioral target editability", ylabel="endpoint probability")
    axes[2].legend(frameon=False, loc="center left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    _save(figure, stem)


def _plot_causal_localization(
    residual_rows: list[dict[str, str]], dimension_rows: list[dict[str, str]], stem: Path
) -> None:
    metrics = ["all", "translation", "rotation"]
    steps = sorted({int(row["flow_step"]) for row in residual_rows})
    layers = sorted({int(row["layer"]) for row in residual_rows})
    limit = max(abs(float(row["mean_symmetric_ncte"])) for row in residual_rows)
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.35), constrained_layout=True)
    image = None
    for axis, metric in zip(axes[:3], metrics, strict=True):
        selected = {(int(row["layer"]), int(row["flow_step"])): row for row in residual_rows if row["metric"] == metric}
        values = np.asarray(
            [[float(selected[(layer, step)]["mean_symmetric_ncte"]) for step in steps] for layer in layers]
        )
        image = axis.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        for y, layer in enumerate(layers):
            for x, step in enumerate(steps):
                row = selected[(layer, step)]
                if float(row["q_bh_within_metric_family"]) < 0.05 and float(row["symmetric_ci95_low"]) > 0:
                    axis.plot(x, y, "o", ms=2.4, markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.6)
        axis.set(
            title=LABELS[metric],
            xlabel="flow step",
            xticks=np.arange(len(steps)),
            xticklabels=steps,
            yticks=np.arange(0, len(layers), 3),
            yticklabels=layers[::3],
        )
    axes[0].set_ylabel("action-expert layer")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes[:3], shrink=0.82)
        colorbar.ax.set_title("residual\nNCTE", fontsize=8)
    axes[0].text(
        0.02,
        0.98,
        "a  residual-stream interchange\n○ positive q<.05 + CI>0",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )

    line_specs = [
        ("translation", "translation", "x_t", "-"),
        ("translation", "translation", "v_t", "--"),
        ("rotation", "rotation", "x_t", "-"),
        ("rotation", "rotation", "v_t", "--"),
    ]
    for metric, group, tensor, linestyle in line_specs:
        selected = sorted(
            (
                row
                for row in dimension_rows
                if row["metric"] == metric
                and row["patched_dimension_group"] == group
                and row["patched_tensor"] == tensor
            ),
            key=_flow_step,
        )
        x = np.asarray([_flow_step(row) for row in selected])
        y = np.asarray([float(row["mean_symmetric_ncte"]) for row in selected])
        low = np.asarray([float(row["symmetric_ci95_low"]) for row in selected])
        high = np.asarray([float(row["symmetric_ci95_high"]) for row in selected])
        label = f"{LABELS[metric]} ← {tensor}"
        axes[3].plot(x, y, linestyle=linestyle, marker="o", ms=2.5, color=COLORS[metric], label=label)
        axes[3].fill_between(x, low, high, color=COLORS[metric], alpha=0.1, linewidth=0)
    axes[3].set(
        title="b  matched action coordinates",
        xlabel="flow step",
        ylabel="mean normalized causal transfer",
        xticks=range(10),
        ylim=(-0.03, 1.0),
    )
    axes[3].legend(frameon=False, loc="upper left")
    axes[3].spines[["top", "right"]].set_visible(False)
    _save(figure, stem)


def _plot_behavioral_editability(
    target_rows: list[dict[str, str]], destination_rows: list[dict[str, str]], stem: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), sharey=True, constrained_layout=True)
    _plot_categorical_curve(axes[0], target_rows)
    axes[0].set(title="a  target identity · 15 states", ylabel="endpoint probability")
    _plot_categorical_curve(axes[1], destination_rows)
    axes[1].set(title="b  post-grasp destination · 2 state blocks")
    axes[0].legend(frameon=False, loc="center left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    _save(figure, stem)


def _plot_categorical_curve(axis: Any, rows: list[dict[str, str]]) -> None:
    selected = sorted(rows, key=_switch_step)
    x = np.asarray([_switch_step(row) for row in selected])
    for prefix, label, color, marker in (
        ("source_retention", "source retained", COLORS["source"], "o"),
        ("donor_transfer", "donor transferred", COLORS["donor"], "s"),
    ):
        y = np.asarray([float(row[prefix]) for row in selected])
        low = np.asarray([float(row[f"{prefix}_ci95_low"]) for row in selected])
        high = np.asarray([float(row[f"{prefix}_ci95_high"]) for row in selected])
        axis.plot(x, y, marker=marker, ms=3.5, color=color, label=label)
        axis.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
    axis.axhline(0.8, color="0.45", linestyle="--", linewidth=0.8)
    axis.set(
        xlabel="source-conditioned updates before switch",
        xticks=range(11),
        ylim=(-0.03, 1.05),
    )


def _plot_closure_token_case(rows: list[dict[str, str]], stem: Path) -> None:
    selected_rows = [row for row in rows if row["metric"] == "gripper_closure_time"]
    if not selected_rows:
        raise ValueError("closure-position table has no gripper_closure_time rows")
    layers = sorted({int(row["layer"]) for row in selected_rows})
    steps = sorted({int(row["flow_step"]) for row in selected_rows})
    positions = sorted({int(row["action_position"]) for row in selected_rows})
    limit = max(abs(float(row["mean_symmetric_ncte"])) for row in selected_rows) or 1.0
    figure, axes = plt.subplots(1, len(layers), figsize=(10.6, 2.75), sharey=True, constrained_layout=True)
    image = None
    for axis, layer in zip(axes, layers, strict=True):
        indexed = {
            (int(row["flow_step"]), int(row["action_position"])): row
            for row in selected_rows
            if int(row["layer"]) == layer
        }
        values = np.asarray(
            [[float(indexed[(step, position)]["mean_symmetric_ncte"]) for position in positions] for step in steps]
        )
        image = axis.imshow(values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set(
            title=f"layer {layer}",
            xlabel="future action position",
            xticks=np.arange(len(positions)),
            xticklabels=positions,
            yticks=np.arange(len(steps)),
            yticklabels=steps,
        )
        axis.axvline(positions.index(9), color="black", linewidth=0.65, linestyle=":")
    axes[0].set_ylabel("flow step")
    if image is not None:
        figure.colorbar(image, ax=axes, label="closure-time normalized transfer", shrink=0.82)
    figure.suptitle(
        "Noise-conditional closure case: only future position 9 transfers closure time\n"
        "descriptive n=1 eligible state/noise mode; no population significance claim",
        fontsize=9.5,
    )
    _save(figure, stem)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _flow_step(row: dict[str, str]) -> int:
    return int(row["flow_step"])


def _switch_step(row: dict[str, str]) -> int:
    return int(row["switch_after_steps"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
