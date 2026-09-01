#!/usr/bin/env python3
"""Promote a mutually consistent confirmation audit and figure bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.confirmation_artifacts import promote_confirmation_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = promote_confirmation_artifacts(
        args.audit_root,
        args.figure_root,
        args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
