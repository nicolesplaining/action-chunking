from __future__ import annotations

import runpy
from pathlib import Path

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "run_eligible_retarget_study.py")
)
_boolean = _module["_boolean"]


def test_strict_serialized_boolean_parser() -> None:
    assert _boolean("True") is True
    assert _boolean("False") is False
