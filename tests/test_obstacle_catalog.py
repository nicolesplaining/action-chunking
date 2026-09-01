from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_obstacle_catalog_screen.py")
)
_validate_source_order = _module["_validate_source_order"]
_write_summary = _module["_write_summary"]


def test_obstacle_catalog_preserves_source_order_and_denominator(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"pairs": []}))
    output = tmp_path / "output"
    output.mkdir()
    entries = [
        {"pair_id": "state-0", "init_index": 0},
        {"pair_id": "state-1", "init_index": 1},
    ]
    _validate_source_order(entries)
    jobs = [
        {
            "geometric_exclusions": 9,
            "clean_screened_pairs": 0,
            "selected_pair_id": None,
        },
        {
            "geometric_exclusions": 3,
            "clean_screened_pairs": 6,
            "selected_pair_id": "obstacle-1",
            "selected_manifest": "/frozen/selected.json",
            "selected_manifest_sha256": "digest",
        },
    ]
    args = SimpleNamespace(source_manifest=source, output=output)

    _write_summary(args, entries, jobs, jobs[1])
    summary = json.loads((output / "summary.json").read_text())

    assert summary["selection_uses_interventions"] is False
    assert summary["state_order"] == "source_manifest_order"
    assert summary["processed_source_states"] == 2
    assert summary["total_geometric_exclusions"] == 12
    assert summary["total_clean_screened_pairs"] == 6
    assert summary["selected_pair_id"] == "obstacle-1"
    assert summary["stop_threshold_reached"] is True


def test_obstacle_catalog_rejects_reordered_states() -> None:
    with pytest.raises(ValueError, match="ordered"):
        _validate_source_order(
            [
                {"pair_id": "state-1", "init_index": 1},
                {"pair_id": "state-0", "init_index": 0},
            ]
        )
