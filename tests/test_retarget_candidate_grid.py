from __future__ import annotations

import runpy
from pathlib import Path

_module = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "generate_retarget_candidate_grid.py")
)
_dual_success_target_first = _module["_dual_success_target_first"]


def test_candidate_grid_requires_dual_success_and_target_first_contacts() -> None:
    entry = {
        "pair_id": "pair",
        "base_target": "wine",
        "donor_target": "bowl",
    }
    summary = {
        "pair_id": "pair",
        "both_successful": True,
        "results": [
            {"side": "base", "first_contact_step_by_object": {"wine": 10}},
            {"side": "donor", "first_contact_step_by_object": {"bowl": 11}},
        ],
    }

    assert _dual_success_target_first(entry, summary) is True
    summary["results"][1]["first_contact_step_by_object"] = {"wine": 5, "bowl": 11}
    assert _dual_success_target_first(entry, summary) is False
