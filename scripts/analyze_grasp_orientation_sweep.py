#!/usr/bin/env python3
"""Analyze contact-aligned grasp-frame formation in a dynamic-retarget sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_chunking.analysis import commitment_step
from action_chunking.orientation import (
    GRASP_ORIENTATION_WINDOW_STEPS,
    GRASP_SOURCE_RETENTION_THRESHOLD,
    MINIMUM_GRASP_REFERENCE_CONTRAST_RAD,
    contact_aligned_grasp_frame,
    orientation_affinity,
)
from action_chunking.pairs import file_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = analyze_grasp_orientation_sweep(
        args.sweep,
        args.manifest,
        args.pair_id,
        args.calibration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def analyze_grasp_orientation_sweep(
    sweep: Path,
    manifest_path: Path,
    pair_id: str,
    calibration_path: Path,
) -> dict[str, Any]:
    """Return a registered orientation curve without mutating source artifacts."""
    manifest = json.loads(manifest_path.read_text())
    entry = _manifest_entry(manifest, pair_id)
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("selection_uses_continuation_outcomes") is not False:
        raise ValueError("orientation calibration must exclude continuation outcomes")
    if not calibration.get("all_pairs_pass_contrast"):
        raise ValueError("clean grasp-orientation calibration did not pass")
    minimum_contrast = float(calibration["minimum_reference_contrast_rad"])
    window_steps = int(calibration["window_steps"])
    if minimum_contrast != MINIMUM_GRASP_REFERENCE_CONTRAST_RAD:
        raise ValueError("calibration changed the frozen orientation-contrast threshold")
    if window_steps != GRASP_ORIENTATION_WINDOW_STEPS:
        raise ValueError("calibration changed the frozen contact window")
    side = _new_side(entry)
    old_side = entry["origin_side"]
    old_target = entry[f"{old_side}_target"]
    new_target = entry[f"{side}_target"]

    source = _endpoint(sweep / "continue_after_10", side, old_target, window_steps)
    destination = _endpoint(sweep / "restart_after_0", side, new_target, window_steps)
    rows = []
    for boundary in range(11):
        run = sweep / f"continue_after_{boundary}"
        summary = json.loads((run / "summary.json").read_text())
        result = _result(summary, side)
        all_contacts = {
            str(target): int(step)
            for target, step in result["first_contact_step_by_object"].items()
        }
        first_object = min(all_contacts, key=all_contacts.get) if all_contacts else None
        contacts = {
            target: all_contacts[target]
            for target in (old_target, new_target)
            if target in all_contacts
        }
        first_target = min(contacts, key=contacts.get) if contacts else None
        if first_target is None:
            rows.append(
                {
                    "switch_after_steps": boundary,
                    "first_contact_object": first_object,
                    "first_registered_target": None,
                    "correct_target_first": False,
                    "orientation_censored": True,
                    "censor_reason": "no_registered_target_contact",
                }
            )
            continue
        contact_step = contacts[first_target]
        frame = contact_aligned_grasp_frame(
            _jsonl(run / f"{side}_trajectory_records.jsonl"),
            contact_step,
            window_steps=window_steps,
        )
        affinity = orientation_affinity(
            frame["quaternion_xyzw"],
            source["frame"]["quaternion_xyzw"],
            destination["frame"]["quaternion_xyzw"],
            minimum_reference_contrast_rad=minimum_contrast,
        )
        rows.append(
            {
                "switch_after_steps": boundary,
                "first_contact_object": first_object,
                "first_registered_target": first_target,
                "first_contact_step": contact_step,
                "correct_target_first": first_object == new_target,
                "orientation_censored": False,
                "censor_reason": None,
                "frame": frame,
                **affinity,
            }
        )

    complete = all(not row["orientation_censored"] for row in rows)
    boundary = None
    fitted = None
    if complete:
        raw = np.asarray([float(row["source_retention"]) for row in rows])
        boundary, fitted_values = commitment_step(raw, GRASP_SOURCE_RETENTION_THRESHOLD)
        fitted = fitted_values.tolist()
    payload = {
        "schema_version": 1,
        "pair_id": pair_id,
        "side": side,
        "old_target": old_target,
        "new_target": new_target,
        "metric": "contact_aligned_first_registered_target_grasp_frame",
        "interpretation": (
            "target-conditioned grasp geometry; target identity is reported separately and the "
            "curve does not isolate orientation from object choice"
        ),
        "quaternion_convention": "xyzw",
        "window_steps": window_steps,
        "minimum_reference_contrast_rad": minimum_contrast,
        "retention_threshold": GRASP_SOURCE_RETENTION_THRESHOLD,
        "calibration": str(calibration_path),
        "calibration_sha256": file_digest(calibration_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_digest(manifest_path),
        "source_control": source,
        "destination_control": destination,
        "all_boundaries_have_registered_target_contact": complete,
        "orientation_editability_boundary": boundary,
        "predicted_last_orientation_correction_boundary": (
            boundary - 1 if boundary is not None and boundary > 0 else None
        ),
        "isotonic_source_retention": fitted,
        "rows": rows,
    }
    return payload


def _endpoint(run: Path, side: str, expected_target: str, window_steps: int) -> dict[str, Any]:
    summary = json.loads((run / "summary.json").read_text())
    result = _result(summary, side)
    contacts = result["first_contact_step_by_object"]
    if expected_target not in contacts:
        raise ValueError(f"control {run.name} did not contact expected target")
    first = min(contacts, key=contacts.get)
    if first != expected_target:
        raise ValueError(f"control {run.name} contacted {first!r} before expected target")
    step = int(contacts[expected_target])
    return {
        "run": run.name,
        "target": expected_target,
        "frame": contact_aligned_grasp_frame(
            _jsonl(run / f"{side}_trajectory_records.jsonl"), step, window_steps=window_steps
        ),
    }


def _result(summary: dict[str, Any], side: str) -> dict[str, Any]:
    matches = [result for result in summary["results"] if result["side"] == side]
    if len(matches) != 1:
        raise ValueError(f"expected one {side!r} result, found {len(matches)}")
    return matches[0]


def _manifest_entry(manifest: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["pairs"] if entry["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest entry for {pair_id!r}, found {len(matches)}")
    return matches[0]


def _new_side(entry: dict[str, Any]) -> str:
    origin = entry.get("origin_side")
    if origin not in {"base", "donor"}:
        raise ValueError("grasp-orientation sweep requires a pre-contact origin side")
    return "donor" if origin == "base" else "base"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
