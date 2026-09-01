#!/usr/bin/env python3
"""Convert official OpenPI LIBERO text logs into auditable tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from action_chunking.libero_logs import parse_episode_results, summarize_episode_results, validate_task_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-episodes-per-task", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = parse_episode_results(args.log)
    if args.expected_episodes is not None and len(results) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} completed episodes, found {len(results)}")
    rows, summary = summarize_episode_results(results, args.suite)
    validate_task_counts(
        rows,
        expected_tasks=args.expected_tasks,
        expected_episodes_per_task=args.expected_episodes_per_task,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["source_log"] = str(args.log)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
