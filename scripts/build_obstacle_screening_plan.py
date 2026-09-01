#!/usr/bin/env python3
"""Freeze a task-diverse obstacle-screen order from the public retarget plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.source_plan.read_text())
    payload = build_plan(source, args.source_plan)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output.is_file() and args.output.read_text() != serialized:
        raise ValueError("existing obstacle screening plan differs from frozen inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(
        f"froze {payload['candidate_rows']} obstacle source rows across "
        f"{payload['target_pair_families']} target-pair families"
    )
    return 0


def build_plan(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if source.get("selection_uses_intervention_outcomes") is not False:
        raise ValueError("source screening plan must exclude intervention outcomes")
    rows = list(source["rows"])
    if not rows:
        raise ValueError("source screening plan contains no rows")
    if len({row["screen_id"] for row in rows}) != len(rows):
        raise ValueError("source screening ids must be unique")
    pair_rank: dict[tuple[str, ...], int] = {}
    indexed = []
    for source_index, row in enumerate(rows):
        key = _pair_key(row)
        if key not in pair_rank:
            pair_rank[key] = len(pair_rank)
        indexed.append((source_index, pair_rank[key], row))
    ordered = sorted(
        indexed,
        key=lambda item: (int(item[2]["init_index"]), item[1], item[0]),
    )
    return {
        "schema_version": 1,
        "protocol_version": "0.14",
        "selection_uses_intervention_outcomes": False,
        "selection_uses_obstacle_intervention_outcomes": False,
        "source_plan": str(source_path),
        "source_plan_sha256": file_digest(source_path),
        "candidate_rows": len(rows),
        "target_pair_families": len(pair_rank),
        "ordering": [
            "init_index",
            "first_occurrence_target_pair_rank",
            "source_row_index_tiebreak",
        ],
        "placement_fractions": [0.35, 0.50, 0.65],
        "lateral_offsets_m": [0.0, -0.05, 0.05],
        "rows": [
            {
                "obstacle_plan_index": obstacle_index,
                "source_row_index": source_index,
                "target_pair_rank": rank,
                "screen_id": row["screen_id"],
                "init_index": int(row["init_index"]),
            }
            for obstacle_index, (source_index, rank, row) in enumerate(ordered)
        ],
    }


def _pair_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row["suite"]),
        str(row["canonical_scene_sha256"]),
        str(row["base_task"]),
        str(row["donor_task"]),
        str(row["base_target"]),
        str(row["donor_target"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
