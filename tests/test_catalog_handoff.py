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
    assert gate["valid_prediction_clusters"] == 1
    assert gate["rows"][0]["cluster_id"] == "scene-a"
    assert gate["rows"][0]["action_only_prediction"]["valid"] is True
    assert index["manifest_by_pair"] == {"candidate-a": str(manifest_path)}


def test_catalog_handoff_rejects_incomplete_screen(tmp_path: Path) -> None:
    catalog_path = tmp_path / "summary.json"
    catalog_path.write_text("{}")
    catalog = {
        **_execution_fields(
            tmp_path,
            [("a", "scene-a"), ("b", "scene-b")],
        ),
        "selection_uses_continuation_outcomes": False,
        "stop_threshold_reached": False,
        "catalog_exhausted": False,
        "minimum_eligible_clusters": 1,
        "minimum_valid_prediction_clusters": 1,
        "planned_rows": 2,
        "processed_rows": 1,
        "eligible_clusters": 0,
        "eligible_directions": 0,
        "eligible_cluster_ids": [],
        "valid_prediction_clusters": 0,
        "valid_prediction_cluster_ids": [],
        "parallel_workers": [{"gpu": 0, "port": 8003}],
        "selection_uses_only_contiguous_completed_prefix": True,
        "selection_uses_action_only_prediction_validity": True,
        "speculative_endpoint_rows_excluded_from_selection": [],
        "jobs": [_catalog_job(tmp_path, 0, "a", "scene-a", eligible=0)],
    }
    with pytest.raises(ValueError, match="stop rule"):
        prepare_catalog_handoff(catalog, catalog_path)


def test_catalog_handoff_recomputes_counts_and_first_stop_crossing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "summary.json"
    catalog_path.write_text("{}")
    catalog = {
        **_execution_fields(
            tmp_path,
            [("a", "scene-a"), ("b", "scene-b"), ("c", "scene-c")],
        ),
        "selection_uses_continuation_outcomes": False,
        "stop_threshold_reached": True,
        "catalog_exhausted": False,
        "minimum_eligible_clusters": 1,
        "minimum_valid_prediction_clusters": 1,
        "planned_rows": 3,
        "processed_rows": 2,
        "eligible_clusters": 2,
        "eligible_directions": 2,
        "eligible_cluster_ids": ["scene-a", "scene-b"],
        "valid_prediction_clusters": 2,
        "valid_prediction_cluster_ids": ["scene-a", "scene-b"],
        "parallel_workers": [{"gpu": 0, "port": 8003}],
        "selection_uses_only_contiguous_completed_prefix": True,
        "selection_uses_action_only_prediction_validity": True,
        "speculative_endpoint_rows_excluded_from_selection": [],
        "jobs": [
            _catalog_job(tmp_path, 0, "a", "scene-a"),
            _catalog_job(tmp_path, 1, "b", "scene-b"),
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
    _rewrite_job_artifact(catalog["jobs"][0])
    with pytest.raises(ValueError, match="eligible pair ids differ"):
        prepare_catalog_handoff(catalog, catalog_path)

    catalog = _catalog(gate_path)
    changed_gate = _gate()
    changed_gate["rows"][0]["execution_horizon"] = 4
    gate_path.write_text(json.dumps(changed_gate))
    catalog["jobs"][0]["gate_summary_sha256"] = hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()
    _rewrite_job_artifact(catalog["jobs"][0])
    with pytest.raises(ValueError, match="seed zero and horizon five"):
        prepare_catalog_handoff(catalog, catalog_path)


def test_catalog_handoff_excludes_and_audits_parallel_speculation(tmp_path: Path) -> None:
    row_root = tmp_path / "rows" / "00000_a"
    gate_path = row_root / "gate" / "summary.json"
    manifest_path = row_root / "candidate" / "pair" / "aligned" / "manifest.json"
    gate_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps(_gate()))
    manifest_path.write_text(json.dumps({"pairs": [{"pair_id": "candidate-a"}]}))
    catalog_path = tmp_path / "summary.json"
    catalog = _catalog(gate_path)
    catalog["parallel_workers"] = [
        {"gpu": 0, "port": 8003},
        {"gpu": 1, "port": 8004},
    ]
    speculative = _catalog_job(tmp_path, 1, "b", "scene-b", eligible=0)
    catalog["speculative_endpoint_rows_excluded_from_selection"] = [speculative]
    catalog_path.write_text(json.dumps(catalog))

    gate, _index = prepare_catalog_handoff(catalog, catalog_path)

    assert gate["eligible_clusters"] == 1
    assert {row["cluster_id"] for row in gate["rows"]} == {"scene-a"}

    Path(speculative["row_result"]).write_text("{}")
    with pytest.raises(ValueError, match="row result changed"):
        prepare_catalog_handoff(catalog, catalog_path)


def _catalog(gate_path: Path) -> dict:
    job = _catalog_job(gate_path.parents[3], 0, "a", "scene-a")
    job.update(
        {
            "source_pair_id": "source-a",
            "gate_summary": str(gate_path),
            "gate_summary_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "eligible_directions": 1,
            "eligible_pair_ids": ["candidate-a"],
            "action_only_predictions": [
                _prediction(
                    gate_path.parents[3],
                    "candidate-a",
                    "donor",
                    gate_path.parent.parent
                    / "candidate"
                    / "pair"
                    / "aligned"
                    / "manifest.json",
                )
            ],
        }
    )
    _rewrite_job_artifact(job)
    return {
        **_execution_fields(
            gate_path.parents[3],
            [("a", "scene-a"), ("b", "scene-b")],
        ),
        "selection_uses_continuation_outcomes": False,
        "stop_threshold_reached": True,
        "catalog_exhausted": False,
        "minimum_eligible_clusters": 1,
        "minimum_valid_prediction_clusters": 1,
        "planned_rows": 2,
        "processed_rows": 1,
        "eligible_clusters": 1,
        "eligible_directions": 1,
        "eligible_cluster_ids": ["scene-a"],
        "valid_prediction_clusters": 1,
        "valid_prediction_cluster_ids": ["scene-a"],
        "parallel_workers": [{"gpu": 0, "port": 8003}],
        "selection_uses_only_contiguous_completed_prefix": True,
        "selection_uses_action_only_prediction_validity": True,
        "speculative_endpoint_rows_excluded_from_selection": [],
        "jobs": [job],
    }


def _catalog_job(
    root: Path,
    index: int,
    screen_id: str,
    cluster_id: str,
    *,
    eligible: int = 1,
) -> dict:
    row_root = root / "rows" / f"{index:05d}_{screen_id}"
    source = row_root / "source" / "manifest.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({"index": index}))
    raw = {
        "plan_index": index,
        "screen_id": screen_id,
        "cluster_id": cluster_id,
        "source_manifest": str(source),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "eligible_directions": eligible,
        "eligible_pair_ids": [f"candidate-{index}"] if eligible else [],
        "action_only_predictions": (
            [_prediction(root, f"candidate-{index}", "donor")] if eligible else []
        ),
    }
    result = row_root / "row_result.json"
    result.write_text(json.dumps(raw))
    return {
        **raw,
        "row_result": str(result),
        "row_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
    }


def _rewrite_job_artifact(job: dict) -> None:
    path = Path(job["row_result"])
    raw = {
        key: value
        for key, value in job.items()
        if key not in {"row_result", "row_result_sha256"}
    }
    path.write_text(json.dumps(raw))
    job["row_result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(
    root: Path,
    pair_id: str,
    new_side: str,
    manifest: Path | None = None,
) -> dict:
    prediction = root / "predictions" / pair_id / new_side / "prediction.json"
    actions = prediction.with_name("actions.npz")
    manifest = manifest or prediction.with_name("manifest.json")
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text(
        json.dumps(
            {
                "pair_id": pair_id,
                "new_side": new_side,
                "valid": True,
                "predicted_last_successful_boundary": 7,
            }
        )
    )
    actions.write_bytes(b"action-only samples")
    if not manifest.is_file():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"pairs": [{"pair_id": pair_id}]}))
    return {
        "pair_id": pair_id,
        "new_side": new_side,
        "prediction": str(prediction),
        "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "actions": str(actions),
        "actions_sha256": hashlib.sha256(actions.read_bytes()).hexdigest(),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "valid": True,
        "predicted_last_successful_boundary": 7,
    }


def _execution_fields(root: Path, rows: list[tuple[str, str]]) -> dict:
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "selection_uses_continuation_outcomes": False,
                "selection_uses_action_only_prediction_validity": True,
                "stop_rule": {
                    "minimum_eligible_clusters": 1,
                    "minimum_valid_prediction_clusters": 1,
                },
                "rows": [
                    {"screen_id": screen_id, "cluster_id": cluster_id}
                    for screen_id, cluster_id in rows
                ],
            }
        )
    )
    return {
        "code_commit": "a" * 40,
        "plan": str(plan),
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
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
