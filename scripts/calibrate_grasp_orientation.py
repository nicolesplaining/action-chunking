#!/usr/bin/env python3
"""Freeze grasp-frame separability from clean controls only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.orientation import (
    GRASP_ORIENTATION_WINDOW_STEPS,
    MINIMUM_GRASP_REFERENCE_CONTRAST_RAD,
    contact_aligned_grasp_frame,
    quaternion_geodesic_rad,
)
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    root = Path(selection["clean_validation"])
    rows = [
        _clean_pair_row(root, pair_id, GRASP_ORIENTATION_WINDOW_STEPS)
        for pair_id in selection["model_clean_eligible_pairs"]
    ]
    contrasts = np.asarray([row["reference_contrast_rad"] for row in rows])
    dispersions = np.asarray(
        [value for row in rows for value in row["maximum_window_dispersion_rad_by_side"].values()]
    )
    payload = {
        "schema_version": 1,
        "selection_uses_continuation_outcomes": False,
        "selection": str(args.selection),
        "selection_sha256": file_digest(args.selection),
        "clean_validation": str(root),
        "quaternion_convention": "xyzw",
        "metric": "2*acos(abs(dot(q1,q2)))",
        "contact_window": "inclusive three-step window ending at first registered target contact",
        "window_steps": GRASP_ORIENTATION_WINDOW_STEPS,
        "minimum_reference_contrast_rad": MINIMUM_GRASP_REFERENCE_CONTRAST_RAD,
        "eligible_pairs": len(rows),
        "pairs_passing_contrast": int(
            np.sum(contrasts >= MINIMUM_GRASP_REFERENCE_CONTRAST_RAD)
        ),
        "all_pairs_pass_contrast": bool(
            np.all(contrasts >= MINIMUM_GRASP_REFERENCE_CONTRAST_RAD)
        ),
        "contrast_summary_rad": _summary(contrasts),
        "window_dispersion_summary_rad": _summary(dispersions),
        "rows": rows,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output.is_file() and args.output.read_text() != serialized:
        raise ValueError("existing grasp-orientation calibration differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(
        f"clean grasp-frame contrast: {payload['pairs_passing_contrast']}/{len(rows)} pass; "
        f"minimum={contrasts.min():.6f} rad",
        flush=True,
    )
    return 0


def _clean_pair_row(root: Path, pair_id: str, window_steps: int) -> dict[str, Any]:
    run = root / pair_id / "noise_0"
    summary = json.loads((run / "summary.json").read_text())
    frames = {}
    dispersions = {}
    contact_steps = {}
    targets = {}
    for result in summary["results"]:
        side = result["side"]
        target = result["target"]
        contact_step = int(result["first_contact_step_by_object"][target])
        frame = contact_aligned_grasp_frame(
            _jsonl(run / f"{side}_trajectory_records.jsonl"),
            contact_step,
            window_steps=window_steps,
        )
        frames[side] = frame["quaternion_xyzw"]
        dispersions[side] = frame["maximum_window_dispersion_rad"]
        contact_steps[side] = contact_step
        targets[side] = target
    if set(frames) != {"base", "donor"}:
        raise ValueError(f"clean pair {pair_id} must contain base and donor controls")
    return {
        "pair_id": pair_id,
        "targets": targets,
        "contact_steps": contact_steps,
        "reference_contrast_rad": quaternion_geodesic_rad(frames["base"], frames["donor"]),
        "maximum_window_dispersion_rad_by_side": dispersions,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q3": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
