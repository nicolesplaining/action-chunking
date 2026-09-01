from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "prepare_catalog_retarget_study.py")
)
prepare_catalog_handoff = _module["prepare_catalog_handoff"]


def test_catalog_handoff_enriches_rows_and_freezes_candidate_index(tmp_path: Path) -> None:
    row_root = tmp_path / "rows" / "00000_a"
    gate_path = row_root / "gate" / "summary.json"
    manifest_path = row_root / "candidate" / "pair" / "aligned" / "manifest.json"
    gate_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps(_gate()))
    manifest_path.write_text(json.dumps({"pairs": [{"pair_id": "candidate-a"}]}))
    catalog_path = tmp_path / "summary.json"
    catalog = _catalog(gate_path)
    catalog_path.write_text(json.dumps(catalog))

    gate, index = prepare_catalog_handoff(catalog, catalog_path)

    assert gate["eligible_clusters"] == 1
    assert gate["confirmatory_population_complete"] is True
    assert gate["rows"][0]["cluster_id"] == "scene-a"
    assert index["manifest_by_pair"] == {"candidate-a": str(manifest_path)}


def test_catalog_handoff_rejects_incomplete_screen(tmp_path: Path) -> None:
    catalog_path = tmp_path / "summary.json"
    catalog_path.write_text("{}")
    catalog = {
        "selection_uses_continuation_outcomes": False,
        "stop_threshold_reached": False,
        "catalog_exhausted": False,
        "minimum_eligible_clusters": 1,
        "planned_rows": 2,
        "processed_rows": 1,
        "eligible_clusters": 0,
        "eligible_directions": 0,
        "eligible_cluster_ids": [],
        "jobs": [_catalog_job(0, "a", "scene-a", eligible=0)],
    }
    with pytest.raises(ValueError, match="stop rule"):
        prepare_catalog_handoff(catalog, catalog_path)


def test_catalog_handoff_recomputes_counts_and_first_stop_crossing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "summary.json"
    catalog_path.write_text("{}")
    catalog = {
        "selection_uses_continuation_outcomes": False,
        "stop_threshold_reached": True,
        "catalog_exhausted": False,
        "minimum_eligible_clusters": 1,
        "planned_rows": 3,
        "processed_rows": 2,
        "eligible_clusters": 2,
        "eligible_directions": 2,
        "eligible_cluster_ids": ["scene-a", "scene-b"],
        "jobs": [
            _catalog_job(0, "a", "scene-a"),
            _catalog_job(1, "b", "scene-b"),
        ],
    }

    with pytest.raises(ValueError, match="continued after the first"):
        prepare_catalog_handoff(catalog, catalog_path)


def test_catalog_handoff_rejects_pair_id_and_protocol_mismatches(tmp_path: Path) -> None:
    row_root = tmp_path / "rows" / "00000_a"
    gate_path = row_root / "gate" / "summary.json"
    manifest_path = row_root / "candidate" / "pair" / "aligned" / "manifest.json"
    gate_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps(_gate()))
    manifest_path.write_text(json.dumps({"pairs": [{"pair_id": "candidate-a"}]}))
    catalog_path = tmp_path / "summary.json"
    catalog = _catalog(gate_path)
    catalog_path.write_text(json.dumps(catalog))

    catalog["jobs"][0]["eligible_pair_ids"] = ["different"]
    with pytest.raises(ValueError, match="eligible pair ids differ"):
        prepare_catalog_handoff(catalog, catalog_path)

    catalog = _catalog(gate_path)
    changed_gate = _gate()
    changed_gate["rows"][0]["execution_horizon"] = 4
    gate_path.write_text(json.dumps(changed_gate))
    catalog["jobs"][0]["gate_summary_sha256"] = hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="seed zero and horizon five"):
        prepare_catalog_handoff(catalog, catalog_path)


def _catalog(gate_path: Path) -> dict:
    return {
        "selection_uses_continuation_outcomes": False,
        "stop_threshold_reached": True,
        "catalog_exhausted": False,
        "minimum_eligible_clusters": 1,
        "planned_rows": 2,
        "processed_rows": 1,
        "eligible_clusters": 1,
        "eligible_directions": 1,
        "eligible_cluster_ids": ["scene-a"],
        "jobs": [
            {
                "plan_index": 0,
                "screen_id": "a",
                "cluster_id": "scene-a",
                "source_pair_id": "source-a",
                "gate_summary": str(gate_path),
                "gate_summary_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                "eligible_directions": 1,
                "eligible_pair_ids": ["candidate-a"],
            }
        ],
    }


def _catalog_job(
    index: int, screen_id: str, cluster_id: str, *, eligible: int = 1
) -> dict:
    return {
        "plan_index": index,
        "screen_id": screen_id,
        "cluster_id": cluster_id,
        "eligible_directions": eligible,
    }


def _gate() -> dict:
    return {
        "selection_uses_continuation_outcomes": False,
        "rows": [
            {
                "pair_id": "candidate-a",
                "source_pair_id": "source-a",
                "origin_side": "base",
                "new_side": "donor",
                "noise_seed": 0,
                "execution_horizon": 5,
                "event_exact_initial_state": True,
                "controller_replay_exact": True,
                "source_chunk_exact": True,
                "source_input_exact": True,
                "old_event_induced": True,
                "restart_avoids_old_event": True,
                "event_gate_pass": True,
                "competence_exact_initial_state": True,
                "restart_new_target_first": True,
                "clean_tasks_competent": True,
                "eligible": True,
            }
        ],
    }
