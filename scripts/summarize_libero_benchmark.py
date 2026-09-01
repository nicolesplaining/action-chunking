#!/usr/bin/env python3
"""Combine complete per-suite LIBERO summaries into an auditable benchmark table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from action_chunking.libero_logs import combine_suite_summaries

EXPECTED_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes-per-suite", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = [path.resolve() for path in args.summary]
    summaries = [json.loads(path.read_text()) for path in source_paths]
    rows, combined = combine_suite_summaries(
        summaries,
        expected_suites=EXPECTED_SUITES,
        expected_episodes_per_suite=args.expected_episodes_per_suite,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "suites.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    combined["source_summaries"] = [str(path) for path in source_paths]
    (args.output / "summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
