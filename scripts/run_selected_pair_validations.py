#!/usr/bin/env python3
"""Sequentially validate every clean-selected pair and shared noise seed."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clean-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--noise-seeds", default="0,1,2,3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(value) for value in args.noise_seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("noise seeds must be nonempty and unique")
    selection = json.loads(args.selection.read_text())
    if selection.get("selection_uses_interventions") is not False:
        raise ValueError("pair selection must explicitly exclude intervention outcomes")
    launcher = Path(__file__).with_name("run_pair_validation.sh")
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = []
    for pair in selection["selected_pairs"]:
        for seed in seeds:
            job_output = args.output / pair["pair_id"] / f"noise_{seed}"
            summary_path = job_output / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                _validate_existing_summary(summary, pair["pair_id"], seed)
                status = "existing"
            else:
                command = [
                    str(launcher),
                    pair["manifest"],
                    pair["pair_id"],
                    str(args.gpu),
                    str(args.port),
                    str(seed),
                    str(job_output),
                    str(args.clean_screen),
                ]
                completed = subprocess.run(command, check=False)
                if completed.returncode not in {0, 1} or not summary_path.exists():
                    raise RuntimeError(
                        f"validation failed without a behavioral summary for {pair['pair_id']} seed {seed}"
                    )
                summary = json.loads(summary_path.read_text())
                _validate_existing_summary(summary, pair["pair_id"], seed)
                status = "completed"
            jobs.append(
                {
                    "pair_id": pair["pair_id"],
                    "scene_state_sha256": pair["scene_state_sha256"],
                    "noise_seed": seed,
                    "status": status,
                    "both_successful": summary["both_successful"],
                    "first_chunk_exact": all(
                        result["first_chunk_max_abs_error"] == 0.0 for result in summary["results"]
                    ),
                    "summary": str(summary_path),
                }
            )
            _write_summary(args.output, jobs, len(selection["selected_pairs"]) * len(seeds))
    return 0


def _validate_existing_summary(summary: dict[str, Any], pair_id: str, seed: int) -> None:
    if summary["pair_id"] != pair_id or int(summary["noise_seed"]) != seed:
        raise ValueError("existing rollout summary does not match the requested job")
    if not all(result["first_chunk_max_abs_error"] == 0.0 for result in summary["results"]):
        raise ValueError("closed-loop first chunk does not exactly match the clean screen")


def _write_summary(output: Path, jobs: list[dict[str, Any]], expected_jobs: int) -> None:
    completed_jobs = len(jobs)
    summary = {
        "schema_version": 1,
        "expected_jobs": expected_jobs,
        "completed_jobs": completed_jobs,
        "successful_jobs": sum(job["both_successful"] for job in jobs),
        "all_first_chunks_exact": all(job["first_chunk_exact"] for job in jobs),
        "all_behaviorally_successful": completed_jobs == expected_jobs and all(job["both_successful"] for job in jobs),
        "jobs": jobs,
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"validated {completed_jobs}/{expected_jobs}: "
        f"{summary['successful_jobs']} both-successful, exact={summary['all_first_chunks_exact']}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
