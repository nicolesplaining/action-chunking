#!/usr/bin/env python3
"""Tabulate closed-loop behavioral eligibility for clean-screened pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from action_chunking.rollouts import paired_rollout_rows, paired_rollout_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.input.glob("*/noise_*/summary.json"))
    summaries = [json.loads(path.read_text()) for path in paths]
    rows = paired_rollout_rows(summaries)
    summary = paired_rollout_summary(rows)
    if args.expected_jobs is not None and summary["paired_noise_jobs"] != args.expected_jobs:
        raise ValueError(
            f"expected {args.expected_jobs} paired rollout jobs, found {summary['paired_noise_jobs']}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "rollouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
