#!/usr/bin/env python3
"""Independently audit a completed matched-pi0 intervention/control output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.pi0_intervention import audit_pi0_intervention_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parity-summary", type=Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_pi0_intervention_output(
        args.output_root,
        args.parity_summary,
        args.pytorch_checkpoint,
        args.manifest,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
