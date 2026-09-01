#!/usr/bin/env python3
"""Compare pi0.5 and matched pi0 interventions on common clean scene states."""

from __future__ import annotations

import argparse
import csv
import json
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
    pi05_units = _read_csv(args.pi05_coarse_analysis / "units.csv")
    pi0_units = _read_csv(args.pi0_coarse_analysis / "units.csv")
    timing = paired_timing_rows(pi05_units, pi0_units)
    timing_summary = paired_timing_summary(
        timing,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    _write_csv(args.output / "paired_timing.csv", timing)
    flow_shapes = paired_flow_shape_rows(
        _read_csv(args.pi05_coarse_analysis / "flow_units.csv"),
        _read_csv(args.pi0_coarse_analysis / "flow_units.csv"),
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
            args.pi05_coarse_analysis / "residual_units.csv",
            args.pi0_coarse_analysis / "residual_units.csv",
            ("flow_step", "layer"),
        ),
        (
            "dimension",
            args.pi05_coarse_analysis / "dimension_units.csv",
            args.pi0_coarse_analysis / "dimension_units.csv",
            ("flow_step", "patched_tensor", "patched_dimension_group"),
        ),
    ]
    cell_summaries = {}
    for offset, (name, pi05_path, pi0_path, fields) in enumerate(cell_specs, start=1):
        paired, aggregated = _compare_cells(
            _read_csv(pi05_path),
            _read_csv(pi0_path),
            fields,
            args.bootstrap_replicates,
            args.bootstrap_seed + offset,
        )
        _write_csv(args.output / f"paired_{name}_units.csv", paired)
        _write_csv(args.output / f"paired_{name}_cells.csv", aggregated)
        cell_summaries[name] = _cell_summary(aggregated)

    pi05_positions = _read_csv(args.pi05_position_analysis / "position_units.csv")
    pi0_positions = _read_csv(args.pi0_position_analysis / "position_units.csv")
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

    normalized_pi05 = normalized_position_rows(
        pi05_positions, args.pi05_action_horizon
    )
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
        "pi05_coarse": args.pi05_coarse_analysis / "summary.json",
        "pi0_coarse": args.pi0_coarse_analysis / "summary.json",
        "pi05_positions": args.pi05_position_analysis / "summary.json",
        "pi0_positions": args.pi0_position_analysis / "summary.json",
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
        "source_files": {
            name: {"path": str(path), "sha256": file_digest(path)}
            for name, path in source_paths.items()
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
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
    aggregated = aggregate_paired_cells(
        paired, fields, bootstrap_replicates=replicates, seed=seed
    )
    return paired, aggregated


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: {
            "common_cells": sum(row["metric"] == metric for row in rows),
            "bh_significant_cells": sum(
                row["metric"] == metric
                and float(row["q_bh_within_metric_family"]) < 0.05
                for row in rows
            ),
        }
        for metric in sorted({str(row["metric"]) for row in rows})
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"paired model comparison produced no rows for {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
