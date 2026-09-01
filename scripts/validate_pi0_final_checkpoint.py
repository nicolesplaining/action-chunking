#!/usr/bin/env python3
"""Validate the frozen final matched-pi0 Orbax checkpoint identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.pi0_checkpoint import validate_pi0_final_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_pi0_final_checkpoint(args.checkpoint), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
