#!/usr/bin/env python3
"""Freeze a closed-loop-outcome-blind public-catalog screen and stop rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.catalog_selection import build_retarget_screening_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", action="append", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-states-per-pair", type=int, default=50)
    parser.add_argument("--minimum-eligible-clusters", type=int, default=59)
    parser.add_argument("--minimum-valid-prediction-clusters", type=int, default=59)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_retarget_screening_plan(
        args.catalog,
        args.exclusions,
        initial_states_per_pair=args.initial_states_per_pair,
        minimum_eligible_clusters=args.minimum_eligible_clusters,
        minimum_valid_prediction_clusters=args.minimum_valid_prediction_clusters,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in plan.items() if key not in {"rows", "excluded"}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
