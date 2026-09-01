#!/usr/bin/env python3
"""Fail before conversion unless the preserved bfloat16 parity failure is exact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.conversion import validate_prior_conversion_failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = validate_prior_conversion_failure(
        args.summary,
        args.jax_checkpoint,
        args.manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
