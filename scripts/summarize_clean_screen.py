#!/usr/bin/env python3
"""Create auditable candidate tables from clean-only paired-chunk screens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from action_chunking.screening import summarize_clean_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads((args.input / "summary.json").read_text())
    records = [json.loads(line) for line in (args.input / "clean_screen.jsonl").read_text().splitlines()]
    pair_rows, contrast_rows = summarize_clean_screen(
        records,
        metadata["noise_seeds"],
        metadata["screen_definition"],
    )
    passing = [row for row in pair_rows if row["passes_all_seeds"]]
    if [row["pair_id"] for row in passing] != metadata["passing_pair_ids"]:
        raise ValueError("recomputed passing pairs disagree with screen summary")

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "pair_screen.csv", pair_rows)
    _write_csv(args.output / "target_contrasts.csv", contrast_rows)
    selected = {
        "schema_version": 1,
        "selection_source": str(args.input),
        "selection_uses_interventions": False,
        "screen_definition": metadata["screen_definition"],
        "selected_pairs": [
            {
                key: row[key]
                for key in (
                    "pair_id",
                    "manifest",
                    "fixture_sha256",
                    "scene_state_sha256",
                    "init_index",
                    "base_target",
                    "donor_target",
                )
            }
            for row in passing
        ],
    }
    (args.output / "selected_pairs.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "pairs": len(pair_rows),
        "target_contrasts": len(contrast_rows),
        "independent_serialized_states": len({row["scene_state_sha256"] for row in pair_rows}),
        "pairs_passing_all_seeds": len(passing),
        "independent_states_passing_all_seeds": len({row["scene_state_sha256"] for row in passing}),
        "selected_pair_ids": [row["pair_id"] for row in passing],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
