from __future__ import annotations

import json
import runpy
from pathlib import Path

analyze_recovery_endpoints = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "analyze_recovery_pilot.py")
)["analyze_recovery_endpoints"]


def test_recovery_endpoint_gate_requires_exact_swapped_chunk_and_donor_first_contact(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    clean = tmp_path / "clean.json"
    targets = {"base": "alpha", "donor": "cream"}
    identity_results = [
        _result("base", "alpha", {"alpha": 60}, steps=130),
        _result("donor", "cream", {"cream": 90}, steps=140),
    ]
    donor_results = [
        _result("base", "alpha", {"cream": 55, "alpha": 170}, steps=220),
        _result("donor", "cream", {"alpha": 58, "cream": 205}, steps=255),
    ]
    clean.write_text(json.dumps({"pair_id": "pair", "results": identity_results}))
    for boundary, results in ((0, donor_results), (10, identity_results)):
        root = sweep / f"switch_after_{boundary}"
        root.mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps(_summary(boundary, results)))
    chunks = {"base": [[1.0, 0.0]], "donor": [[2.0, 0.0]]}
    for side, other in (("base", "donor"), ("donor", "base")):
        (sweep / "switch_after_10" / f"{side}_actions.json").write_text(json.dumps([chunks[side]]))
        (sweep / "switch_after_0" / f"{side}_actions.json").write_text(json.dumps([chunks[other]]))

    rows, summary = analyze_recovery_endpoints(sweep, clean)

    assert targets == {row["side"]: row["source_target"] for row in rows}
    assert summary["recovery_eligible_directions"] == 2
    assert summary["eligible_directions_eventually_contact_source_rate"] == 1.0
    assert summary["eligible_directions_eventual_success_rate"] == 1.0
    assert summary["interpretation_allowed"] is True
    assert {row["source_contact_delay_steps"] for row in rows} == {110, 115}


def test_recovery_endpoint_gate_rejects_nonperturbing_direction(tmp_path) -> None:
    sweep = tmp_path / "sweep"
    clean = tmp_path / "clean.json"
    results = [
        _result("base", "alpha", {"alpha": 60}, steps=130),
        _result("donor", "cream", {"cream": 90}, steps=140),
    ]
    clean.write_text(json.dumps({"pair_id": "pair", "results": results}))
    for boundary in (0, 10):
        root = sweep / f"switch_after_{boundary}"
        root.mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps(_summary(boundary, results)))
        for side, chunk in (("base", [[1.0]]), ("donor", [[2.0]])):
            (root / f"{side}_actions.json").write_text(json.dumps([chunk]))

    _, summary = analyze_recovery_endpoints(sweep, clean)

    assert summary["recovery_eligible_directions"] == 0
    assert summary["interpretation_allowed"] is False


def _summary(boundary: int, results: list[dict]) -> dict:
    return {
        "pair_id": "pair",
        "noise_seed": 0,
        "intervention": {"family": "flow_switch", "switch_after_steps": boundary},
        "intervene_replans": "0",
        "stop_after_first_task_contact": False,
        "stop_after_registered_destination": False,
        "results": results,
    }


def _result(side: str, target: str, contacts: dict[str, int], *, steps: int) -> dict:
    return {
        "side": side,
        "target": target,
        "success": True,
        "steps": steps,
        "first_contact_step_by_object": contacts,
        "live_initial_input_diagnostics": {
            "observation/image": {"array_equal": True},
            "observation/wrist_image": {"array_equal": True},
            "observation/state": {"array_equal": True},
        },
        "restored_sim_state_max_abs_error": 0.0,
        "intervention_replans_applied": [0],
    }
