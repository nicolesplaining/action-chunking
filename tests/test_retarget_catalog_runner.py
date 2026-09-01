from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_module = runpy.run_path("scripts/run_retarget_catalog_screen.py")
WorkerConfig = _module["WorkerConfig"]
parse_workers = _module["_parse_workers"]
run_catalog = _module["run_catalog"]


def test_two_worker_catalog_keeps_exact_prefix_and_records_speculation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "screen_id": f"screen-{index}",
            "cluster_id": f"cluster-{index}",
            "suite": "libero_goal",
        }
        for index in range(3)
    ]
    plan = {
        "selection_uses_intervention_outcomes": True,
        "selection_uses_continuation_outcomes": False,
        "selection_uses_action_only_prediction_validity": True,
        "stop_rule": {
            "minimum_eligible_clusters": 1,
            "minimum_valid_prediction_clusters": 1,
        },
        "rows": rows,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    calls = []

    def fake_run_row(index, row, row_root, worker):
        calls.append((index, worker.gpu, worker.port))
        source = row_root / "source" / "manifest.json"
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({"index": index}))
        eligible = int(index == 0)
        result = {
            "plan_index": index,
            "screen_id": row["screen_id"],
            "cluster_id": row["cluster_id"],
            "suite": row["suite"],
            "source_pair_id": f"pair-{index}",
            "source_manifest": str(source),
            "source_manifest_sha256": _digest(source),
            "clean_exact_dual_success_target_first": True,
            "status": "endpoint_screened",
            "event_gate_directions": eligible,
            "eligible_directions": eligible,
            "eligible_pair_ids": [f"pair-{index}"] if eligible else [],
            "action_only_predictions": [],
        }
        if eligible:
            gate = row_root / "gate" / "summary.json"
            gate.parent.mkdir(parents=True)
            gate.write_text("{}")
            result["gate_summary"] = str(gate)
            result["gate_summary_sha256"] = _digest(gate)
            prediction = row_root / "predictions" / "prediction.json"
            actions = row_root / "predictions" / "actions.npz"
            prediction.parent.mkdir(parents=True)
            prediction.write_text(
                json.dumps(
                    {
                        "pair_id": f"pair-{index}",
                        "new_side": "donor",
                        "valid": True,
                        "predicted_last_successful_boundary": 7,
                    }
                )
            )
            actions.write_bytes(b"actions")
            result["action_only_predictions"] = [
                {
                    "pair_id": f"pair-{index}",
                    "new_side": "donor",
                    "prediction": str(prediction),
                    "prediction_sha256": _digest(prediction),
                    "actions": str(actions),
                    "actions_sha256": _digest(actions),
                    "manifest": str(source),
                    "manifest_sha256": _digest(source),
                    "valid": True,
                    "predicted_last_successful_boundary": 7,
                }
            ]
        return result

    monkeypatch.setitem(run_catalog.__globals__, "_run_row", fake_run_row)
    monkeypatch.setitem(
        run_catalog.__globals__,
        "audit_prediction_artifacts",
        lambda *_args: {
            "valid": True,
            "predicted_last_successful_boundary": 7,
        },
    )
    workers = [WorkerConfig(0, 8003, 0), WorkerConfig(1, 8004, 0)]

    summary = run_catalog(plan, plan_path, tmp_path / "output", workers, "a" * 40)

    assert sorted(calls) == [(0, 0, 8003), (1, 1, 8004)]
    assert summary["processed_rows"] == 1
    assert summary["code_commit"] == "a" * 40
    assert len(summary["jobs"]) == 1
    assert summary["jobs"][0]["plan_index"] == 0
    assert len(summary["speculative_endpoint_rows_excluded_from_selection"]) == 1
    assert summary["speculative_endpoint_rows_excluded_from_selection"][0][
        "plan_index"
    ] == 1
    assert not (tmp_path / "output" / "rows" / "00002_screen-2").exists()


def test_parallel_workers_require_distinct_gpus_and_ports() -> None:
    with pytest.raises(ValueError, match="distinct GPUs"):
        parse_workers("0:8003,0:8004", 0, 8003, 0)
    with pytest.raises(ValueError, match="distinct ports"):
        parse_workers("0:8003,1:8003", 0, 8003, 0)


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
