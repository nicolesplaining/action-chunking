from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_selected_pair_interventions.py")
)
_intersect_pairs = _module["_intersect_pairs"]


def test_clean_intersection_preserves_model_selection_order() -> None:
    assert _intersect_pairs(["c", "a", "b"], ["b", "c"]) == ["c", "b"]


def test_clean_intersection_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="no eligible pair intersection"):
        _intersect_pairs(["a"], ["b"])
