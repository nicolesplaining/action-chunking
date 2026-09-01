#!/usr/bin/env python3
"""Freeze a retarget-utility prediction from offline flow-switch records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_chunking.utility_prediction import predict_last_successful_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-target-contrast", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in (args.input / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    prediction = predict_last_successful_boundary(
        records,
        args.direction,
        threshold=args.threshold,
        minimum_target_contrast=args.minimum_target_contrast,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prediction.json").write_text(
        json.dumps(prediction, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
