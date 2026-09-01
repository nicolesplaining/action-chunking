from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from action_chunking.utility_artifacts import audit_utility_study

_figure = runpy.run_path("scripts/make_utility_figure.py")
load = _figure["load_audited_utility"]
make = _figure["make_utility_figure"]
_study = runpy.run_path("tests/test_utility_audit.py")["_study"]


def test_utility_figure_requires_reconstructed_final_audit(tmp_path: Path) -> None:
    root, _rollouts = _study(tmp_path / "study")
    audit_path = tmp_path / "final_audit.json"
    audit_path.write_text(json.dumps(audit_utility_study(root)))

    data = load(root, audit_path)
    manifest = make(root, audit_path, tmp_path / "figure")

    assert len(data["jobs"]) == 1
    assert manifest["bootstrap_replicates"] == 10_000
    assert (tmp_path / "figure" / "fig_retarget_utility.pdf").is_file()
    assert (tmp_path / "figure" / "fig_retarget_utility.png").is_file()


def test_utility_figure_rejects_wrong_final_audit(tmp_path: Path) -> None:
    root, _rollouts = _study(tmp_path / "study")
    audit = audit_utility_study(root)
    audit["utility_summary_sha256"] = "0" * 64
    audit_path = tmp_path / "final_audit.json"
    audit_path.write_text(json.dumps(audit))

    with pytest.raises(ValueError, match="differs from current"):
        load(root, audit_path)
