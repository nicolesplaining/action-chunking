#!/usr/bin/env python3
"""Compare pi0.5 and matched pi0 interventions on common clean scene states."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from action_chunking.model_comparison import (
    aggregate_paired_cells,
    normalized_position_rows,
    paired_cell_rows,
    paired_flow_shape_rows,
    paired_flow_shape_summary,
    paired_timing_rows,
    paired_timing_summary,
)
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi05-coarse-analysis", type=Path, required=True)
    parser.add_argument("--pi0-coarse-analysis", type=Path, required=True)
    parser.add_argument("--pi05-position-analysis", type=Path, required=True)
    parser.add_argument("--pi0-position-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--pi05-action-horizon", type=int, default=10)
    parser.add_argument("--pi0-action-horizon", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    _validate_analysis_summary(args.pi05_coarse_analysis)
    _validate_analysis_summary(args.pi0_coarse_analysis)
    _validate_analysis_summary(args.pi05_position_analysis)
    _validate_analysis_summary(args.pi0_position_analysis)

    pi05_units = _read_csv(args.pi05_coarse_analysis / "units.csv")
    pi0_units = _read_csv(args.pi0_coarse_analysis / "units.csv")
    pi05_flow = _read_csv(args.pi05_coarse_analysis / "flow_units.csv")
    pi0_flow = _read_csv(args.pi0_coarse_analysis / "flow_units.csv")
    pi05_residual = _read_csv(args.pi05_coarse_analysis / "residual_units.csv")
    pi0_residual = _read_csv(args.pi0_coarse_analysis / "residual_units.csv")
    pi05_dimension = _read_csv(args.pi05_coarse_analysis / "dimension_units.csv")
    pi0_dimension = _read_csv(args.pi0_coarse_analysis / "dimension_units.csv")
    pi05_positions = _read_csv(args.pi05_position_analysis / "position_units.csv")
    pi0_positions = _read_csv(args.pi0_position_analysis / "position_units.csv")
    _validate_coarse_grid(pi05_flow, pi05_residual, pi05_dimension, "pi0.5")
    _validate_coarse_grid(pi0_flow, pi0_residual, pi0_dimension, "pi0")
    _validate_position_grid(pi05_positions, args.pi05_action_horizon, "pi0.5")
    _validate_position_grid(pi0_positions, args.pi0_action_horizon, "pi0")

    timing = paired_timing_rows(pi05_units, pi0_units)
    timing_summary = paired_timing_summary(
        timing,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    _write_csv(args.output / "paired_timing.csv", timing)
    flow_shapes = paired_flow_shape_rows(
        pi05_flow,
        pi0_flow,
    )
    flow_shape_summary = paired_flow_shape_summary(
        flow_shapes,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 10,
    )
    _write_csv(args.output / "paired_flow_shapes.csv", flow_shapes)

    cell_specs = [
        (
            "residual",
            pi05_residual,
            pi0_residual,
            ("flow_step", "layer"),
        ),
        (
            "dimension",
            pi05_dimension,
            pi0_dimension,
            ("flow_step", "patched_tensor", "patched_dimension_group"),
        ),
    ]
    cell_summaries = {}
    for offset, (name, pi05_rows, pi0_rows, fields) in enumerate(cell_specs, start=1):
        paired, aggregated = _compare_cells(
            pi05_rows,
            pi0_rows,
            fields,
            args.bootstrap_replicates,
            args.bootstrap_seed + offset,
        )
        _write_csv(args.output / f"paired_{name}_units.csv", paired)
        _write_csv(args.output / f"paired_{name}_cells.csv", aggregated)
        cell_summaries[name] = _cell_summary(aggregated)

    pi05_first10 = [row for row in pi05_positions if int(row["action_position"]) < 10]
    pi0_first10 = [row for row in pi0_positions if int(row["action_position"]) < 10]
    first10_paired, first10_cells = _compare_cells(
        pi05_first10,
        pi0_first10,
        ("flow_step", "layer", "action_position"),
        args.bootstrap_replicates,
        args.bootstrap_seed + 3,
    )
    _write_csv(args.output / "paired_position_first10_units.csv", first10_paired)
    _write_csv(args.output / "paired_position_first10_cells.csv", first10_cells)

    normalized_pi05 = normalized_position_rows(pi05_positions, args.pi05_action_horizon)
    normalized_pi0 = normalized_position_rows(pi0_positions, args.pi0_action_horizon)
    normalized_paired, normalized_cells = _compare_cells(
        normalized_pi05,
        normalized_pi0,
        ("flow_step", "layer", "normalized_position_bin"),
        args.bootstrap_replicates,
        args.bootstrap_seed + 4,
    )
    _write_csv(args.output / "paired_position_normalized_units.csv", normalized_paired)
    _write_csv(args.output / "paired_position_normalized_cells.csv", normalized_cells)

    source_paths = {
        "pi05_coarse_summary": args.pi05_coarse_analysis / "summary.json",
        "pi05_coarse_units": args.pi05_coarse_analysis / "units.csv",
        "pi05_coarse_flow_units": args.pi05_coarse_analysis / "flow_units.csv",
        "pi05_coarse_residual_units": args.pi05_coarse_analysis / "residual_units.csv",
        "pi05_coarse_dimension_units": args.pi05_coarse_analysis / "dimension_units.csv",
        "pi0_coarse_summary": args.pi0_coarse_analysis / "summary.json",
        "pi0_coarse_units": args.pi0_coarse_analysis / "units.csv",
        "pi0_coarse_flow_units": args.pi0_coarse_analysis / "flow_units.csv",
        "pi0_coarse_residual_units": args.pi0_coarse_analysis / "residual_units.csv",
        "pi0_coarse_dimension_units": args.pi0_coarse_analysis / "dimension_units.csv",
        "pi05_positions_summary": args.pi05_position_analysis / "summary.json",
        "pi05_positions_units": args.pi05_position_analysis / "position_units.csv",
        "pi0_positions_summary": args.pi0_position_analysis / "summary.json",
        "pi0_positions_units": args.pi0_position_analysis / "position_units.csv",
    }
    summary = {
        "schema_version": 1,
        "analysis_unit": "paired_scene_state",
        "comparison": "pi05_minus_pi0",
        "pi05_action_horizon": args.pi05_action_horizon,
        "pi0_action_horizon": args.pi0_action_horizon,
        "primary_position_window": list(range(10)),
        "normalized_position_bins": 10,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "timing": timing_summary,
        "flow_shape": flow_shape_summary,
        "causal_cells": {
            **cell_summaries,
            "position_first10": _cell_summary(first10_cells),
            "position_normalized": _cell_summary(normalized_cells),
        },
        "source_files": {name: {"path": str(path), "sha256": file_digest(path)} for name, path in source_paths.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _compare_cells(
    pi05: list[dict[str, Any]],
    pi0: list[dict[str, Any]],
    fields: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired = paired_cell_rows(pi05, pi0, fields)
    aggregated = aggregate_paired_cells(paired, fields, bootstrap_replicates=replicates, seed=seed)
    return paired, aggregated


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: {
            "common_cells": sum(row["metric"] == metric for row in rows),
            "bh_significant_cells": sum(
                row["metric"] == metric and float(row["q_bh_within_metric_family"]) < 0.05 for row in rows
            ),
        }
        for metric in sorted({str(row["metric"]) for row in rows})
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _validate_analysis_summary(root: Path) -> None:
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary.get("schema_version") != 1
        or summary.get("noise_seeds") != [0]
        or float(summary.get("commitment_threshold", -1.0)) != 0.8
        or float(summary.get("formation_relative_error_tolerance", -1.0)) != 0.2
        or int(summary.get("jobs", 0)) <= 0
        or int(summary.get("pairs", 0)) <= 0
        or int(summary.get("state_clusters", 0)) < 12
    ):
        raise ValueError(f"incompatible intervention analysis summary: {root}")


def _validate_coarse_grid(
    flow: list[dict[str, Any]],
    residual: list[dict[str, Any]],
    dimension: list[dict[str, Any]],
    model: str,
) -> None:
    _validate_scene_metric_grid(
        flow,
        ("switch_after_steps",),
        {(boundary,) for boundary in range(11)},
        model,
        "flow-switch",
    )
    residual_cells = {(int(row["flow_step"]), int(row["layer"])) for row in residual}
    expected_residual = {(flow_step, layer) for flow_step in range(10) for layer in range(18)}
    if residual_cells != expected_residual:
        raise ValueError(
            f"{model} residual grid is incomplete or contains an unexpected registered cell"
        )
    _validate_scene_metric_grid(
        residual,
        ("flow_step", "layer"),
        expected_residual,
        model,
        "residual",
    )
    dimension_cells = {
        (
            int(row["flow_step"]),
            str(row["patched_tensor"]),
            str(row["patched_dimension_group"]),
        )
        for row in dimension
    }
    expected_dimensions = {
        (flow_step, tensor, group)
        for flow_step in range(10)
        for tensor in ("x_t", "v_t")
        for group in ("translation", "rotation", "gripper")
    }
    if dimension_cells != expected_dimensions:
        raise ValueError(
            f"{model} action-dimension grid is incomplete or contains an unexpected registered cell"
        )
    _validate_scene_metric_grid(
        dimension,
        ("flow_step", "patched_tensor", "patched_dimension_group"),
        expected_dimensions,
        model,
        "action-dimension",
    )


def _validate_position_grid(rows: list[dict[str, Any]], action_horizon: int, model: str) -> None:
    cells = {
        (
            int(row["flow_step"]),
            int(row["layer"]),
            int(row["action_position"]),
        )
        for row in rows
    }
    expected = {
        (flow_step, layer, position)
        for flow_step in (0, 7, 8, 9)
        for layer in (0, 8, 14, 17)
        for position in range(action_horizon)
    }
    if cells != expected:
        raise ValueError(
            f"{model} action-position grid is incomplete or contains an unexpected registered cell"
        )
    _validate_scene_metric_grid(
        rows,
        ("flow_step", "layer", "action_position"),
        expected,
        model,
        "action-position",
    )


def _validate_scene_metric_grid(
    rows: list[dict[str, Any]],
    cell_fields: tuple[str, ...],
    expected_cells: set[tuple[Any, ...]],
    model: str,
    family: str,
) -> None:
    grouped: dict[tuple[str, str, int, str], set[tuple[Any, ...]]] = defaultdict(set)
    for row in rows:
        if not _boolean(row.get("eligible")):
            continue
        key = (
            str(row["pair_id"]),
            str(row["scene_state_sha256"]),
            int(row["noise_seed"]),
            str(row["metric"]),
        )
        cell = tuple(_cell_value(row[field]) for field in cell_fields)
        if cell in grouped[key]:
            raise ValueError(f"{model} {family} grid contains a duplicate scene-metric cell")
        grouped[key].add(cell)
    if not grouped:
        raise ValueError(f"{model} {family} grid has no eligible scene-metric units")
    incomplete = [key for key, cells in grouped.items() if cells != expected_cells]
    if incomplete:
        raise ValueError(
            f"{model} {family} grid is incomplete within {len(incomplete)} eligible intervention units"
        )


def _cell_value(value: Any) -> int | str:
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1", 1}:
        return True
    if value in {"False", "false", "0", 0}:
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"paired model comparison produced no rows for {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
