"""Auditable summaries for paired closed-loop rollout validation."""

from __future__ import annotations

from typing import Any

from action_chunking.libero_logs import wilson_interval


def paired_rollout_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten paired summaries and derive first-contact validity fields."""

    rows = []
    seen_jobs = set()
    for summary in summaries:
        job = (summary["pair_id"], int(summary["noise_seed"]))
        if job in seen_jobs:
            raise ValueError(f"duplicate rollout job {job}")
        seen_jobs.add(job)
        if {result["side"] for result in summary["results"]} != {"base", "donor"}:
            raise ValueError(f"rollout job {job} must contain base and donor results")
        for result in summary["results"]:
            contacts = result["first_contact_step_by_object"]
            first_contact_object = min(contacts, key=contacts.get) if contacts else None
            diagnostics = result.get("live_initial_input_diagnostics")
            initial_input_exact = (
                all(value["array_equal"] for value in diagnostics.values()) if diagnostics is not None else None
            )
            state_error = result.get("restored_sim_state_max_abs_error")
            rows.append(
                {
                    "pair_id": summary["pair_id"],
                    "noise_seed": int(summary["noise_seed"]),
                    "side": result["side"],
                    "target": result["target"],
                    "success": bool(result["success"]),
                    "steps": int(result["steps"]),
                    "first_chunk_max_abs_error": result["first_chunk_max_abs_error"],
                    "first_contact_object": first_contact_object,
                    "first_contact_step": contacts.get(first_contact_object) if first_contact_object else None,
                    "target_contact_step": contacts.get(result["target"]),
                    "first_contact_is_target": first_contact_object == result["target"],
                    "both_sides_successful": bool(summary["both_successful"]),
                    "initial_input_mode": result.get("initial_input_mode"),
                    "initial_input_exact": initial_input_exact,
                    "restored_sim_state_max_abs_error": state_error,
                    "simulator_state_exact": state_error == 0.0 if state_error is not None else None,
                }
            )
    return sorted(rows, key=lambda row: (row["pair_id"], row["noise_seed"], row["side"]))


def paired_rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate behavioral validity without treating sides as independent pairs."""

    if not rows:
        raise ValueError("no paired rollout rows")
    jobs = {(row["pair_id"], row["noise_seed"]) for row in rows}
    pairs = {row["pair_id"] for row in rows}
    successful_jobs = sum(_job_all(rows, job, "success") for job in jobs)
    target_contact_jobs = sum(_job_all(rows, job, "first_contact_is_target") for job in jobs)
    eligible_jobs = sum(
        _job_all(rows, job, "success")
        and _job_all(rows, job, "first_contact_is_target")
        and _job_all(rows, job, "initial_input_exact")
        and _job_all(rows, job, "simulator_state_exact")
        and all(
            row["first_chunk_max_abs_error"] == 0.0
            for row in rows
            if (row["pair_id"], row["noise_seed"]) == job
        )
        for job in jobs
    )
    success_ci = wilson_interval(successful_jobs, len(jobs))
    target_contact_ci = wilson_interval(target_contact_jobs, len(jobs))
    return {
        "schema_version": 1,
        "pairs": len(pairs),
        "paired_noise_jobs": len(jobs),
        "paired_noise_jobs_both_successful": successful_jobs,
        "paired_noise_jobs_both_successful_ci95_low": success_ci[0],
        "paired_noise_jobs_both_successful_ci95_high": success_ci[1],
        "paired_noise_jobs_both_first_contacts_target": target_contact_jobs,
        "paired_noise_jobs_both_first_contacts_target_ci95_low": target_contact_ci[0],
        "paired_noise_jobs_both_first_contacts_target_ci95_high": target_contact_ci[1],
        "paired_noise_jobs_strictly_eligible": eligible_jobs,
        "side_rollouts": len(rows),
        "successful_side_rollouts": sum(row["success"] for row in rows),
        "first_contact_target_side_rollouts": sum(row["first_contact_is_target"] for row in rows),
        "initial_input_exact_side_rollouts": sum(row["initial_input_exact"] is True for row in rows),
        "simulator_state_exact_side_rollouts": sum(row["simulator_state_exact"] is True for row in rows),
        "all_first_chunks_exact": all(row["first_chunk_max_abs_error"] == 0.0 for row in rows),
        "pairs_successful_for_all_tested_noise": sum(
            all(row["success"] for row in rows if row["pair_id"] == pair) for pair in pairs
        ),
    }


def _job_all(rows: list[dict[str, Any]], job: tuple[str, int], field: str) -> bool:
    selected = [row[field] for row in rows if (row["pair_id"], row["noise_seed"]) == job]
    return len(selected) == 2 and all(value is True for value in selected)
