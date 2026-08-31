"""Auditable summaries for paired closed-loop rollout validation."""

from __future__ import annotations

from typing import Any


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
                }
            )
    return sorted(rows, key=lambda row: (row["pair_id"], row["noise_seed"], row["side"]))


def paired_rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate behavioral validity without treating sides as independent pairs."""

    if not rows:
        raise ValueError("no paired rollout rows")
    jobs = {(row["pair_id"], row["noise_seed"]) for row in rows}
    pairs = {row["pair_id"] for row in rows}
    return {
        "schema_version": 1,
        "pairs": len(pairs),
        "paired_noise_jobs": len(jobs),
        "paired_noise_jobs_both_successful": sum(
            all(row["success"] for row in rows if (row["pair_id"], row["noise_seed"]) == job)
            for job in jobs
        ),
        "side_rollouts": len(rows),
        "successful_side_rollouts": sum(row["success"] for row in rows),
        "first_contact_target_side_rollouts": sum(row["first_contact_is_target"] for row in rows),
        "all_first_chunks_exact": all(row["first_chunk_max_abs_error"] == 0.0 for row in rows),
        "pairs_successful_for_all_tested_noise": sum(
            all(row["success"] for row in rows if row["pair_id"] == pair) for pair in pairs
        ),
    }
