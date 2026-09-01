#!/usr/bin/env python3
"""Independently reconstruct a completed held-out retargeting study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.utility_artifacts import audit_utility_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_utility_study(args.study_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.is_file():
        if args.output.read_text() != rendered:
            raise ValueError("existing utility audit differs from reconstruction")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
