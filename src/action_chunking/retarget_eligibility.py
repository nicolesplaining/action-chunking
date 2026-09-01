"""Clean-endpoint eligibility for behaviorally sensitive retargeting states."""

from __future__ import annotations

from typing import Any


def eligibility_row(
    entry: dict[str, Any],
    event_summary: dict[str, Any],
    execution_horizon: int,
    competence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one pre-contact state without inspecting continuation outcomes."""
    if execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive")
    if event_summary.get("pair_id") != entry.get("pair_id"):
        raise ValueError("rollout summary and pre-contact entry have different pair ids")
    origin = entry.get("origin_side")
    if origin not in {"base", "donor"}:
        raise ValueError("pre-contact entry must identify a base or donor origin")
    new_side = "donor" if origin == "base" else "base"
    by_side = {result["side"]: result for result in event_summary["results"]}
    if set(by_side) != {"base", "donor"}:
        raise ValueError("rollout summary must contain exactly base and donor results")

    old_result = by_side[origin]
    restart_result = by_side[new_side]
    old_target = entry[f"{origin}_target"]
    new_target = entry[f"{new_side}_target"]
    old_event_step = _contact_step(old_result, old_target)
    restart_old_event_step = _contact_step(restart_result, old_target)
    event_exact = all(_result_exact(result) for result in (old_result, restart_result))
    old_event_induced = old_event_step is not None and old_event_step <= execution_horizon
    restart_avoids_old_event = (
        restart_old_event_step is None or restart_old_event_step > execution_horizon
    )
    event_gate_pass = event_exact and old_event_induced and restart_avoids_old_event

    competence_exact = None
    restart_first_contact = None
    restart_new_target_first = None
    clean_tasks_competent = None
    if competence_summary is not None:
        if competence_summary.get("pair_id") != entry.get("pair_id"):
            raise ValueError("competence summary and pre-contact entry have different pair ids")
        competence_by_side = {
            result["side"]: result for result in competence_summary["results"]
        }
        if set(competence_by_side) != {"base", "donor"}:
            raise ValueError("competence summary must contain exactly base and donor results")
        competence_exact = all(_result_exact(result) for result in competence_by_side.values())
        restart_first_contact = _first_contact_object(competence_by_side[new_side])
        restart_new_target_first = restart_first_contact == new_target
        clean_tasks_competent = bool(
            competence_by_side[origin]["success"] and competence_by_side[new_side]["success"]
        )
    eligible = bool(
        event_gate_pass
        and competence_exact
        and restart_new_target_first
        and clean_tasks_competent
    )
    return {
        "pair_id": entry["pair_id"],
        "source_pair_id": entry["source_pair_id"],
        "origin_side": origin,
        "new_side": new_side,
        "snapshot_step": int(entry["snapshot_step"]),
        "precontact_offset_steps": int(entry["precontact_offset_steps"]),
        "noise_seed": int(event_summary["noise_seed"]),
        "execution_horizon": execution_horizon,
        "old_target": old_target,
        "new_target": new_target,
        "old_event_step": old_event_step,
        "restart_old_event_step": restart_old_event_step,
        "restart_first_contact_object": restart_first_contact,
        "event_exact_initial_state": event_exact,
        "old_event_induced": old_event_induced,
        "restart_avoids_old_event": restart_avoids_old_event,
        "event_gate_pass": event_gate_pass,
        "competence_run_completed": competence_summary is not None,
        "competence_exact_initial_state": competence_exact,
        "restart_new_target_first": restart_new_target_first,
        "clean_tasks_competent": clean_tasks_competent,
        "eligible": eligible,
    }


def _contact_step(result: dict[str, Any], target: str) -> int | None:
    value = result.get("first_contact_step_by_object", {}).get(target)
    return int(value) if value is not None else None


def _first_contact_object(result: dict[str, Any]) -> str | None:
    contacts = result.get("first_contact_step_by_object", {})
    return min(contacts, key=contacts.get) if contacts else None


def _result_exact(result: dict[str, Any]) -> bool:
    diagnostics = result.get("live_initial_input_diagnostics", {})
    return (
        bool(diagnostics)
        and all(bool(field.get("array_equal")) for field in diagnostics.values())
        and result.get("restored_sim_state_max_abs_error") == 0.0
    )
