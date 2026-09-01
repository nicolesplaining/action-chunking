#!/usr/bin/env python3
"""Audit a first-replan donor-chunk recovery endpoint pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = analyze_recovery_endpoints(args.sweep, args.clean_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "directions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["source_sweep"] = str(args.sweep.resolve())
    summary["source_clean_summary"] = str(args.clean_summary.resolve())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def analyze_recovery_endpoints(sweep: Path, clean_summary_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donor_summary = _load_boundary(sweep, 0)
    identity_summary = _load_boundary(sweep, 10)
    clean_summary = json.loads(clean_summary_path.read_text())
    pair_id = donor_summary["pair_id"]
    for summary in (identity_summary, clean_summary):
        if summary["pair_id"] != pair_id:
            raise ValueError("recovery endpoint and clean summaries use different pairs")
    for summary, boundary in ((donor_summary, 0), (identity_summary, 10)):
        if summary.get("intervention") != {"family": "flow_switch", "switch_after_steps": boundary}:
            raise ValueError(f"boundary {boundary} does not contain the expected flow switch")
        if summary.get("intervene_replans") != "0":
            raise ValueError("recovery pilot must intervene only at replan zero")
        if summary.get("stop_after_first_task_contact") or summary.get("stop_after_registered_destination"):
            raise ValueError("recovery endpoint must retain the full rollout")
    if int(donor_summary["noise_seed"]) != int(identity_summary["noise_seed"]):
        raise ValueError("recovery endpoints use different noise seeds")

    donor_results = {result["side"]: result for result in donor_summary["results"]}
    identity_results = {result["side"]: result for result in identity_summary["results"]}
    clean_results = {result["side"]: result for result in clean_summary["results"]}
    if set(donor_results) != {"base", "donor"} or set(identity_results) != set(donor_results):
        raise ValueError("recovery endpoints must contain exactly base and donor directions")
    if set(clean_results) != set(donor_results):
        raise ValueError("clean summary directions differ from recovery endpoints")

    first_chunks = {
        boundary: {
            side: _first_chunk(sweep / f"switch_after_{boundary}" / f"{side}_actions.json")
            for side in ("base", "donor")
        }
        for boundary in (0, 10)
    }
    rows = []
    for side in ("base", "donor"):
        other = "donor" if side == "base" else "base"
        donor_result = donor_results[side]
        identity_result = identity_results[side]
        clean_result = clean_results[side]
        source_target = str(identity_result["target"])
        donor_target = str(identity_results[other]["target"])
        identity_matches_clean = _outcome_signature(identity_result) == _outcome_signature(clean_result)
        full_donor_chunk_exact = bool(np.array_equal(first_chunks[0][side], first_chunks[10][other]))
        donor_contacts = {str(key): int(value) for key, value in donor_result["first_contact_step_by_object"].items()}
        identity_contacts = {
            str(key): int(value) for key, value in identity_result["first_contact_step_by_object"].items()
        }
        first_object = min(donor_contacts, key=donor_contacts.get) if donor_contacts else None
        identity_first_object = min(identity_contacts, key=identity_contacts.get) if identity_contacts else None
        donor_first = first_object == donor_target and donor_target != source_target
        identity_source_first = identity_first_object == source_target
        induced_donor_first = donor_first and identity_source_first
        source_contact_step = donor_contacts.get(source_target)
        identity_source_contact_step = identity_contacts.get(source_target)
        rows.append(
            {
                "pair_id": pair_id,
                "noise_seed": int(donor_summary["noise_seed"]),
                "side": side,
                "source_target": source_target,
                "donor_target": donor_target,
                "identity_matches_clean_outcome": identity_matches_clean,
                "full_donor_first_chunk_exact": full_donor_chunk_exact,
                "first_contact_object_identity": identity_first_object,
                "identity_first_contact_is_source": identity_source_first,
                "first_contact_object_after_donor_chunk": first_object,
                "first_contact_is_donor": donor_first,
                "donor_first_contact_induced_by_chunk": induced_donor_first,
                "source_target_eventually_contacted": source_contact_step is not None,
                "source_contact_step_after_donor_chunk": source_contact_step,
                "source_contact_step_identity": identity_source_contact_step,
                "source_contact_delay_steps": (
                    source_contact_step - identity_source_contact_step
                    if source_contact_step is not None and identity_source_contact_step is not None
                    else None
                ),
                "eventual_success_after_donor_chunk": bool(donor_result["success"]),
                "identity_success": bool(identity_result["success"]),
                "steps_after_donor_chunk": int(donor_result["steps"]),
                "identity_steps": int(identity_result["steps"]),
                "completion_delay_steps": int(donor_result["steps"]) - int(identity_result["steps"]),
                "initial_input_exact": all(
                    value["array_equal"] for value in donor_result["live_initial_input_diagnostics"].values()
                ),
                "simulator_state_exact": float(donor_result["restored_sim_state_max_abs_error"]) == 0.0,
                "intervention_applied_only_at_replan_zero": donor_result["intervention_replans_applied"] == [0],
            }
        )
    eligible = [row for row in rows if row["donor_first_contact_induced_by_chunk"]]
    summary = {
        "schema_version": 1,
        "pair_id": pair_id,
        "noise_seed": int(donor_summary["noise_seed"]),
        "directions": len(rows),
        "recovery_eligible_directions": len(eligible),
        "all_identity_outcomes_match_clean": all(row["identity_matches_clean_outcome"] for row in rows),
        "all_full_donor_first_chunks_exact": all(row["full_donor_first_chunk_exact"] for row in rows),
        "all_initial_inputs_exact": all(row["initial_input_exact"] for row in rows),
        "all_simulator_states_exact": all(row["simulator_state_exact"] for row in rows),
        "all_interventions_only_at_replan_zero": all(
            row["intervention_applied_only_at_replan_zero"] for row in rows
        ),
        "eligible_directions_eventually_contact_source_rate": (
            sum(row["source_target_eventually_contacted"] for row in eligible) / len(eligible) if eligible else None
        ),
        "eligible_directions_eventual_success_rate": (
            sum(row["eventual_success_after_donor_chunk"] for row in eligible) / len(eligible) if eligible else None
        ),
        "interpretation_allowed": bool(eligible)
        and all(row["identity_matches_clean_outcome"] for row in rows)
        and all(row["full_donor_first_chunk_exact"] for row in rows)
        and all(row["initial_input_exact"] and row["simulator_state_exact"] for row in rows)
        and all(row["intervention_applied_only_at_replan_zero"] for row in rows),
    }
    return rows, summary


def _load_boundary(sweep: Path, boundary: int) -> dict[str, Any]:
    path = sweep / f"switch_after_{boundary}" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _first_chunk(path: Path) -> np.ndarray:
    chunks = json.loads(path.read_text())
    if not chunks:
        raise ValueError(f"no action chunks in {path}")
    chunk = np.asarray(chunks[0])
    if chunk.ndim != 2:
        raise ValueError(f"invalid first action chunk in {path}")
    return chunk


def _outcome_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(result["success"]),
        int(result["steps"]),
        tuple(sorted((str(key), int(value)) for key, value in result["first_contact_step_by_object"].items())),
    )


if __name__ == "__main__":
    raise SystemExit(main())
