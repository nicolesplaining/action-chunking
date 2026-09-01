#!/usr/bin/env python3
"""Screen obstacle placements across source states in frozen manifest order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu < 0 or args.port <= 0 or args.noise_seed < 0:
        raise ValueError("gpu and noise seed must be nonnegative; port must be positive")
    source = json.loads(args.source_manifest.read_text())
    entries = list(source["pairs"])
    _validate_source_order(entries)
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    selected = None
    for plan_index, entry in enumerate(entries):
        row_root = args.output / "states" / f"{plan_index:02d}_init_{int(entry['init_index']):02d}"
        result_path = row_root / "row_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = _run_state(plan_index, entry, row_root, args)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        jobs.append(result)
        if result.get("selected_pair_id"):
            selected = result
        _write_summary(args, entries, jobs, selected)
        if selected is not None:
            break
    return 0


def _run_state(
    plan_index: int,
    source_entry: dict[str, Any],
    row_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    row_root.mkdir(parents=True, exist_ok=True)
    fixtures = row_root / "fixtures"
    manifest_path = fixtures / "manifest.json"
    if not manifest_path.is_file():
        subprocess.run(
            [
                str(repo / "scripts" / "run_obstacle_pose_generation.sh"),
                str(args.source_manifest),
                source_entry["pair_id"],
                str(args.gpu),
                str(fixtures),
            ],
            check=True,
        )
    manifest = json.loads(manifest_path.read_text())
    generated = len(manifest["pairs"])
    base = {
        "plan_index": plan_index,
        "init_index": int(source_entry["init_index"]),
        "source_pair_id": source_entry["pair_id"],
        "obstacle_manifest": str(manifest_path),
        "obstacle_manifest_sha256": file_digest(manifest_path),
        "generated_candidates": generated,
        "geometric_exclusions": len(manifest["exclusions"]),
        "geometry_exhausted": bool(manifest.get("geometry_exhausted")),
        "selection_uses_interventions": False,
    }
    if generated == 0:
        return {
            **base,
            "status": "geometry_exhausted",
            "clean_screened_pairs": 0,
            "eligible_pairs": 0,
            "selected_pair_id": None,
        }

    clean = row_root / "clean"
    validation_path = clean / "validation_summary.json"
    if not validation_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_manifest_pair_validations.py"),
                "--manifest",
                str(manifest_path),
                "--output",
                str(clean),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
            ],
            check=True,
        )
    screen = row_root / "screen"
    screen_summary_path = screen / "summary.json"
    if not screen_summary_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "screen_obstacle_pose_pairs.py"),
                "--manifest",
                str(manifest_path),
                "--clean-validation",
                str(clean),
                "--output",
                str(screen),
            ],
            check=True,
        )
    summary = json.loads(screen_summary_path.read_text())
    if summary.get("selection_uses_interventions") is not False:
        raise ValueError("obstacle clean screen must exclude intervention outcomes")
    return {
        **base,
        "status": "clean_screened",
        "clean_validation": str(validation_path),
        "clean_validation_sha256": file_digest(validation_path),
        "screen_summary": str(screen_summary_path),
        "screen_summary_sha256": file_digest(screen_summary_path),
        "clean_screened_pairs": int(summary["screened_pairs"]),
        "eligible_pairs": int(summary["eligible_pairs"]),
        "selected_pair_id": summary["selected_pair_id"],
        "selected_manifest": summary["selected_manifest"],
        "selected_manifest_sha256": summary["selected_manifest_sha256"],
    }


def _write_summary(
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "protocol_version": "0.13",
        "selection_uses_interventions": False,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": file_digest(args.source_manifest),
        "state_order": "source_manifest_order",
        "planned_source_states": len(entries),
        "processed_source_states": len(jobs),
        "catalog_exhausted": len(jobs) == len(entries) and selected is None,
        "stop_threshold_reached": selected is not None,
        "selected_pair_id": selected.get("selected_pair_id") if selected else None,
        "selected_manifest": selected.get("selected_manifest") if selected else None,
        "selected_manifest_sha256": (
            selected.get("selected_manifest_sha256") if selected else None
        ),
        "total_geometric_exclusions": sum(job["geometric_exclusions"] for job in jobs),
        "total_clean_screened_pairs": sum(job["clean_screened_pairs"] for job in jobs),
        "jobs": jobs,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _validate_source_order(entries: list[dict[str, Any]]) -> None:
    if not entries:
        raise ValueError("source manifest contains no pairs")
    indices = [int(entry["init_index"]) for entry in entries]
    if indices != list(range(len(entries))):
        raise ValueError("source pairs must be ordered by contiguous initialization index")
    if any(entry.get("semantic_role", "manipulated_object") != "manipulated_object" for entry in entries):
        raise ValueError("obstacle state catalog requires manipulated-object source pairs")


if __name__ == "__main__":
    raise SystemExit(main())
