#!/usr/bin/env python3
"""Bind a matched-pi0 intervention run to its exact passed parity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.pi0_intervention import validate_pi0_intervention_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_pi0_intervention_inputs(
        args.parity_summary,
        args.pytorch_checkpoint,
        args.manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
