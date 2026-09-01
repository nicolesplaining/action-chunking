from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

_module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_selected_pair_interventions.py"))
_intersect_pairs = _module["_intersect_pairs"]
_require_minimum_selection = _module["_require_minimum_selection"]
_validation_summary_hashes = _module["_validation_summary_hashes"]


def test_clean_intersection_preserves_model_selection_order() -> None:
    assert _intersect_pairs(["c", "a", "b"], ["b", "c"]) == ["c", "b"]


def test_clean_intersection_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="no eligible pair intersection"):
        _intersect_pairs(["a"], ["b"])


def test_matched_grid_requires_twelve_common_pairs() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        _require_minimum_selection([str(index) for index in range(11)], 12)

    _require_minimum_selection([str(index) for index in range(12)], 12)


def test_clean_validation_hashes_every_pair_summary(tmp_path) -> None:
    expected = {}
    for pair_id, content in (("a", b"one"), ("b", b"two")):
        path = tmp_path / pair_id / "noise_0" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        expected[f"{pair_id}/noise_0/summary.json"] = hashlib.sha256(content).hexdigest()

    assert _validation_summary_hashes(tmp_path) == expected


def test_clean_validation_hashes_rejects_empty_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="no pair summaries"):
        _validation_summary_hashes(tmp_path)
