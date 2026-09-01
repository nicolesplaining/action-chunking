from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

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
    assert summary["target_first_preserved_clusters"] == 15
    assert summary["eventual_success_preserved_clusters"] == 14
    assert summary["composite_preserved_clusters"] == 14
    assert summary["composite_preservation_rate"] == 14 / 15
    assert summary["composite_preservation_clopper_pearson_ci95"][0] < 14 / 15
    assert summary["full_control_exact_clusters"] == 15
    assert summary["velocity_evaluation_savings_fraction"] == 0.3
    assert summary["median_first_replan_latency_savings_fraction"] == 0.3
    assert summary["median_first_replan_latency_savings_fraction_bootstrap_ci95"] == [
        0.3,
        0.3,
    ]
    assert summary["positive_latency_savings_clusters"] == 15
    assert summary["latency_savings_two_sided_sign_test_p"] < 0.001
    assert summary["pilot_positive"] is True


def test_early_exit_pilot_rejects_duplicate_simulator_state_clusters(tmp_path) -> None:
    manifest = {"pairs": [_entry(index) for index in range(16)]}
    manifest["pairs"][15]["identity_hashes"] = manifest["pairs"][14][
        "identity_hashes"
    ]
    clean = tmp_path / "clean"
    full = tmp_path / "full"
    early = tmp_path / "early"
    _write_catalog(clean, manifest["pairs"], None)
    _write_catalog(full, manifest["pairs"], 10)
    _write_catalog(early, manifest["pairs"], 7)

    with pytest.raises(ValueError, match="unique state hashes"):
        analyze_early_exit_pilot(manifest, clean, full, early)


def _entry(index: int) -> dict:
    return {
        "pair_id": f"pair-{index:02d}",
        "identity_hashes": {"sim_state": f"{index:064x}"},
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
        if after_steps is None:
            pair_root = pair_root / "noise_0"
            pair_root.mkdir()
        eligible = index < 15
        jobs.append(
            {
                "pair_id": pair_id,
                "exact_dual_success_target_first": eligible,
                "early_exit_compute_exact": after_steps is not None,
            }
        )
        failed = pair_id == failed_pair or (after_steps is None and not eligible)
        pair_summary = {
            "pair_id": pair_id,
            "both_successful": not failed,
            "results": [
                _result("base", "mug", after_steps, failed),
                _result("donor", "bowl", after_steps, False),
            ],
        }
        (pair_root / "summary.json").write_text(json.dumps(pair_summary))
        for side in ("base", "donor"):
            actions = [[[float(index), 0.0]]]
            (pair_root / f"{side}_actions.json").write_text(json.dumps(actions))
    if after_steps is None:
        legacy_jobs = [
            {
                "pair_id": entry["pair_id"],
                "noise_seed": 0,
                "status": "completed",
                "simulator_state_exact": True,
                "first_chunk_exact": True,
                "initial_input_modes": ["strict"],
                "scene_state_sha256": entry["identity_hashes"]["sim_state"],
                "both_successful": index < 15,
            }
            for index, entry in enumerate(entries)
        ]
        catalog_summary = {
            "expected_jobs": 16,
            "completed_jobs": 16,
            "all_simulator_states_exact": True,
            "all_first_chunks_exact": True,
            "jobs": legacy_jobs,
        }
    else:
        catalog_summary = {
            "noise_seed": 0,
            "expected_pairs": 16,
            "completed_pairs": 16,
            "all_initial_states_exact": True,
            "intervention": intervention,
            "intervene_replans": "all",
            "jobs": jobs,
        }
    (root / "validation_summary.json").write_text(json.dumps(catalog_summary))


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
