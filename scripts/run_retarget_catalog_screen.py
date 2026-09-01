#!/usr/bin/env python3
"""Execute the frozen public-catalog endpoint screen in exact prefix order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from action_chunking.catalog_progress import summarize_catalog_progress
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu < 0 or args.port <= 0 or args.noise_seed < 0:
        raise ValueError("gpu and noise seed must be nonnegative; port must be positive")
    plan = json.loads(args.plan.read_text())
    if plan.get("selection_uses_intervention_outcomes") is not False:
        raise ValueError("catalog plan must explicitly exclude intervention outcomes")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(plan["rows"]):
        row_root = args.output / "rows" / f"{index:05d}_{row['screen_id'][:12]}"
        result_path = row_root / "row_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = _run_row(index, row, row_root, args)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        jobs.append(result)
        progress = summarize_catalog_progress(plan, jobs)
        _write_summary(args, plan, jobs, progress)
        if progress["stop_threshold_reached"]:
            break
    return 0


def _run_row(
    index: int, row: dict[str, Any], row_root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    row_root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    source = row_root / "source"
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file():
        subprocess.run(
            [
                str(repo / "scripts" / "run_instruction_pair_generation.sh"),
                row["suite"],
                row["base_task"],
                row["donor_task"],
                "1",
                str(args.gpu),
                str(source),
                str(row["init_index"]),
            ],
            check=True,
        )
    manifest = json.loads(source_manifest.read_text())
    entries = manifest.get("pairs", [])
    if len(entries) != 1 or int(entries[0]["init_index"]) != int(row["init_index"]):
        raise ValueError("generated source fixture does not match the frozen plan row")
    entry = entries[0]
    if (
        entry["base_target"] != row["base_target"]
        or entry["donor_target"] != row["donor_target"]
    ):
        raise ValueError("generated source targets do not match the frozen plan row")

    clean = row_root / "clean"
    validation_path = clean / "validation_summary.json"
    if not validation_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_manifest_pair_validations.py"),
                "--manifest",
                str(source_manifest),
                "--output",
                str(clean),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
                "--save-sim-states",
            ],
            check=True,
        )
    validation = json.loads(validation_path.read_text())
    job = validation["jobs"][0]
    base_result = {
        "plan_index": index,
        "screen_id": row["screen_id"],
        "cluster_id": row["cluster_id"],
        "suite": row["suite"],
        "source_pair_id": entry["pair_id"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_digest(source_manifest),
        "clean_exact_dual_success_target_first": bool(
            job["exact_dual_success_target_first"]
        ),
    }
    if not job["exact_dual_success_target_first"]:
        return {
            **base_result,
            "status": "clean_endpoint_ineligible",
            "event_gate_directions": 0,
            "eligible_directions": 0,
            "eligible_pair_ids": [],
        }

    candidate = row_root / "candidate"
    candidate_summary = candidate / "generation_summary.json"
    if not candidate_summary.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "generate_retarget_aligned_candidates.py"),
                "--source-manifest",
                str(source_manifest),
                "--rollout-root",
                str(clean),
                "--output",
                str(candidate),
                "--gpu",
                str(args.gpu),
                "--replan-steps",
                "5",
            ],
            check=True,
        )
    generation = json.loads(candidate_summary.read_text())
    if int(generation["generated_source_pairs"]) != 1:
        raise ValueError("clean-eligible catalog row produced no aligned candidate")

    gate = row_root / "gate"
    gate_summary_path = gate / "summary.json"
    if not gate_summary_path.is_file():
        subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_retarget_eligibility_screen.py"),
                "--candidate-root",
                str(candidate),
                "--output",
                str(gate),
                "--gpu",
                str(args.gpu),
                "--port",
                str(args.port),
                "--noise-seed",
                str(args.noise_seed),
                "--execution-horizon",
                "5",
            ],
            check=True,
        )
    gate_summary = json.loads(gate_summary_path.read_text())
    return {
        **base_result,
        "status": "endpoint_screened",
        "gate_summary": str(gate_summary_path),
        "gate_summary_sha256": file_digest(gate_summary_path),
        "event_gate_directions": int(gate_summary["event_gate_directions"]),
        "eligible_directions": int(gate_summary["eligible_directions"]),
        "eligible_pair_ids": list(gate_summary["eligible_pair_ids"]),
    }


def _write_summary(
    args: argparse.Namespace,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
    progress: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "plan": str(args.plan),
        "plan_sha256": file_digest(args.plan),
        "noise_seed": args.noise_seed,
        **progress,
        "jobs": jobs,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"screened {progress['processed_rows']}/{progress['planned_rows']} rows: "
        f"{progress['eligible_clusters']}/{progress['minimum_eligible_clusters']} "
        "eligible clusters",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
