from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import pytest

analyze = runpy.run_path("scripts/analyze_grasp_orientation_sweep.py")[
    "analyze_grasp_orientation_sweep"
]


def test_grasp_orientation_curve_preserves_target_identity_and_geometry(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "pair",
                        "origin_side": "base",
                        "base_target": "mug",
                        "donor_target": "bowl",
                    }
                ]
            }
        )
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "selection_uses_continuation_outcomes": False,
                "all_pairs_pass_contrast": True,
                "minimum_reference_contrast_rad": 0.2,
                "window_steps": 3,
            }
        )
    )
    sweep = tmp_path / "sweep"
    _write_run(sweep / "restart_after_0", "bowl", 0.0)
    for boundary in range(11):
        retention = boundary / 10
        target = "bowl" if boundary < 8 else "mug"
        _write_run(sweep / f"continue_after_{boundary}", target, retention * 0.6)

    result = analyze(sweep, manifest, "pair", calibration)

    assert result["all_boundaries_have_registered_target_contact"] is True
    assert result["orientation_editability_boundary"] == 8
    assert result["predicted_last_orientation_correction_boundary"] == 7
    assert [row["correct_target_first"] for row in result["rows"]] == [True] * 8 + [False] * 3
    assert result["rows"][0]["source_retention"] == pytest.approx(0.0)
    assert result["rows"][10]["source_retention"] == pytest.approx(1.0)


def test_grasp_orientation_curve_censors_missing_target_contact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "pair",
                        "origin_side": "base",
                        "base_target": "mug",
                        "donor_target": "bowl",
                    }
                ]
            }
        )
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "selection_uses_continuation_outcomes": False,
                "all_pairs_pass_contrast": True,
                "minimum_reference_contrast_rad": 0.2,
                "window_steps": 3,
            }
        )
    )
    sweep = tmp_path / "sweep"
    _write_run(sweep / "restart_after_0", "bowl", 0.0)
    for boundary in range(11):
        _write_run(
            sweep / f"continue_after_{boundary}",
            None if boundary == 5 else ("bowl" if boundary < 8 else "mug"),
            boundary / 10 * 0.6,
        )

    result = analyze(sweep, manifest, "pair", calibration)

    assert result["all_boundaries_have_registered_target_contact"] is False
    assert result["orientation_editability_boundary"] is None
    assert result["rows"][5]["censor_reason"] == "no_registered_target_contact"


def _write_run(root: Path, target: str | None, angle: float) -> None:
    root.mkdir(parents=True)
    contacts = {} if target is None else {target: 3}
    (root / "summary.json").write_text(
        json.dumps({"results": [{"side": "donor", "first_contact_step_by_object": contacts}]})
    )
    rows = [
        {
            "step_in_episode": step,
            "eef_quat": [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)],
        }
        for step in range(1, 4)
    ]
    (root / "donor_trajectory_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
