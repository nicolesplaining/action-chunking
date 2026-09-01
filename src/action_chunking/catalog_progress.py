"""Resume-safe accounting for outcome-blind catalog screens."""

from __future__ import annotations

from typing import Any


def summarize_catalog_progress(plan: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a completed prefix and count independent eligible clusters."""
    rows = plan["rows"]
    if len(jobs) > len(rows):
        raise ValueError("catalog progress contains more jobs than planned rows")
    for index, job in enumerate(jobs):
        if int(job.get("plan_index", -1)) != index:
            raise ValueError("catalog jobs must form a contiguous ordered prefix")
        if job.get("screen_id") != rows[index]["screen_id"]:
            raise ValueError("catalog job does not match its frozen plan row")
        if job.get("cluster_id") != rows[index]["cluster_id"]:
            raise ValueError("catalog job has a different cluster id than the plan")
    eligible_clusters = sorted(
        {
            job["cluster_id"]
            for job in jobs
            if int(job.get("eligible_directions", 0)) > 0
        }
    )
    minimum = int(plan["stop_rule"]["minimum_eligible_clusters"])
    return {
        "planned_rows": len(rows),
        "processed_rows": len(jobs),
        "eligible_directions": sum(int(job.get("eligible_directions", 0)) for job in jobs),
        "eligible_clusters": len(eligible_clusters),
        "eligible_cluster_ids": eligible_clusters,
        "minimum_eligible_clusters": minimum,
        "stop_threshold_reached": len(eligible_clusters) >= minimum,
        "catalog_exhausted": len(jobs) == len(rows),
    }
