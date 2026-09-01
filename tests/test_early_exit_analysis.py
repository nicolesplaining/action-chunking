from __future__ import annotations

import json
import runpy
from pathlib import Path

analyze_early_exit_pilot = runpy.run_path("scripts/analyze_early_exit_pilot.py")[
    "analyze_early_exit_pilot"
]


def test_early_exit_pilot_uses_scene_clusters_and_exact_full_control(tmp_path) -> None:
    manifest = {"pairs": [_entry(index) for index in range(16)]}
    clean = tmp_path / "clean"
    full = tmp_path / "full"
    early = tmp_path / "early"
    _write_catalog(clean, manifest["pairs"], None)
    _write_catalog(full, manifest["pairs"], 10)
    _write_catalog(early, manifest["pairs"], 7, failed_pair="pair-14")

    rows, summary = analyze_early_exit_pilot(manifest, clean, full, early)

    assert len(rows) == 16
    assert summary["eligible_scene_clusters"] == 15
    assert summary["composite_preserved_clusters"] == 14
    assert summary["full_control_exact_clusters"] == 15
    assert summary["velocity_evaluation_savings_fraction"] == 0.3
    assert summary["median_first_replan_latency_savings_fraction"] == 0.3
    assert summary["pilot_positive"] is True


def _entry(index: int) -> dict:
    return {
        "pair_id": f"pair-{index:02d}",
        "scene_state_sha256": f"state-{index:02d}",
        "base_target": "mug",
        "donor_target": "bowl",
    }


def _write_catalog(
    root: Path,
    entries: list[dict],
    after_steps: int | None,
    *,
    failed_pair: str | None = None,
) -> None:
    root.mkdir()
    intervention = (
        None
        if after_steps is None
        else {
            "after_steps": after_steps,
            "family": "early_exit",
            "schema_version": 1,
            "total_flow_steps": 10,
        }
    )
    jobs = []
    for index, entry in enumerate(entries):
        pair_id = entry["pair_id"]
        pair_root = root / pair_id
        pair_root.mkdir()
        eligible = index < 15
        jobs.append(
            {
                "pair_id": pair_id,
                "exact_dual_success_target_first": eligible,
                "early_exit_compute_exact": after_steps is not None,
            }
        )
        failed = pair_id == failed_pair
        pair_summary = {
            "pair_id": pair_id,
            "results": [
                _result("base", "mug", after_steps, failed),
                _result("donor", "bowl", after_steps, False),
            ],
        }
        (pair_root / "summary.json").write_text(json.dumps(pair_summary))
        for side in ("base", "donor"):
            actions = [[[float(index), 0.0]]]
            (pair_root / f"{side}_actions.json").write_text(json.dumps(actions))
    (root / "validation_summary.json").write_text(
        json.dumps(
            {
                "expected_pairs": 16,
                "completed_pairs": 16,
                "intervention": intervention,
                "intervene_replans": "all" if intervention is not None else None,
                "jobs": jobs,
            }
        )
    )


def _result(
    side: str,
    target: str,
    after_steps: int | None,
    failed: bool,
) -> dict:
    diagnostics = []
    if after_steps is not None:
        diagnostics = [
            {
                "replan_index": 0,
                "integration_ms": float(after_steps),
            }
        ]
    return {
        "side": side,
        "success": not failed,
        "first_contact_step_by_object": {target: 1},
        "early_exit_diagnostics": diagnostics,
    }
