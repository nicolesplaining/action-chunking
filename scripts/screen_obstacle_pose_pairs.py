#!/usr/bin/env python3
"""Freeze the first clean-eligible obstacle placement in preregistered order."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from action_chunking.obstacle_screening import obstacle_screen_row
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--corridor-margin", type=float, default=0.02)
    parser.add_argument("--minimum-clearance-gain", type=float, default=0.015)
    parser.add_argument("--minimum-trajectory-contrast", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("pair_family") != "obstacle_pose":
        raise ValueError("obstacle screen requires an obstacle-pose manifest")
    if manifest.get("registered_difference", {}).get("selection_uses_interventions") is not False:
        raise ValueError("obstacle placement grid must explicitly exclude intervention outcomes")
    validation = json.loads((args.clean_validation / "validation_summary.json").read_text())
    jobs = {job["pair_id"]: job for job in validation["jobs"]}
    rows = []
    selected = None
    for entry in manifest["pairs"]:
        pair_id = entry["pair_id"]
        if pair_id not in jobs:
            raise ValueError(f"obstacle validation is missing pair {pair_id}")
        summary_path = Path(jobs[pair_id]["summary"])
        summary = json.loads(summary_path.read_text())
        trajectories = {
            side: _read_jsonl(summary_path.parent / f"{side}_trajectory_records.jsonl")
            for side in ("base", "donor")
        }
        row = obstacle_screen_row(
            entry,
            summary,
            trajectories,
            execution_horizon=args.execution_horizon,
            corridor_margin_m=args.corridor_margin,
            minimum_clearance_gain_m=args.minimum_clearance_gain,
            minimum_trajectory_contrast_m=args.minimum_trajectory_contrast,
        )
        rows.append(row)
        if selected is None and row["eligible"]:
            selected = entry

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "rows.csv", rows)
    selected_manifest_path = None
    if selected is not None:
        selected_manifest = {
            **manifest,
            "schema_version": 1,
            "parent_manifest": str(args.manifest),
            "parent_manifest_sha256": file_digest(args.manifest),
            "selection": {
                "uses_interventions": False,
                "rule": "first_clean_eligible_in_registered_manifest_order",
                "execution_horizon": args.execution_horizon,
                "corridor_margin_m": args.corridor_margin,
                "minimum_clearance_gain_m": args.minimum_clearance_gain,
                "minimum_trajectory_contrast_m": args.minimum_trajectory_contrast,
            },
            "pairs": [selected],
        }
        selected_manifest_path = args.output / "selected_manifest.json"
        selected_manifest_path.write_text(
            json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n"
        )
    summary = {
        "schema_version": 1,
        "selection_uses_interventions": False,
        "manifest": str(args.manifest),
        "manifest_sha256": file_digest(args.manifest),
        "screened_pairs": len(rows),
        "eligible_pairs": sum(row["eligible"] for row in rows),
        "selected_pair_id": selected["pair_id"] if selected is not None else None,
        "selected_manifest": str(selected_manifest_path) if selected_manifest_path else None,
        "selected_manifest_sha256": (
            file_digest(selected_manifest_path) if selected_manifest_path else None
        ),
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
