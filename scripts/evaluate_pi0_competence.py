#!/usr/bin/env python3
"""Evaluate and serialize the frozen matched-pi0 competence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.competence import evaluate_pi0_competence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-summary", type=Path, required=True)
    parser.add_argument("--pair-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_pi0_competence(
        json.loads(args.suite_summary.read_text()),
        json.loads(args.pair_summary.read_text()),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "competence_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
