"""Resume-safe accounting for closed-loop-outcome-blind catalog screens."""

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
    eligible_clusters = []
    valid_prediction_clusters = []
    seen = set()
    for job in jobs:
        if int(job.get("eligible_directions", 0)) <= 0:
            continue
        cluster_id = str(job["cluster_id"])
        if cluster_id in seen:
            continue
        seen.add(cluster_id)
        eligible_clusters.append(cluster_id)
        predictions = job.get("action_only_predictions")
        if not isinstance(predictions, list) or len(predictions) != int(
            job["eligible_directions"]
        ):
            raise ValueError("eligible catalog job lacks its action-only predictions")
        if predictions and predictions[0].get("valid") is True:
            valid_prediction_clusters.append(cluster_id)
    minimum = int(plan["stop_rule"]["minimum_eligible_clusters"])
    minimum_valid = int(plan["stop_rule"]["minimum_valid_prediction_clusters"])
    stop_reached = (
        len(eligible_clusters) >= minimum
        and len(valid_prediction_clusters) >= minimum_valid
    )
    return {
        "planned_rows": len(rows),
        "processed_rows": len(jobs),
        "eligible_directions": sum(int(job.get("eligible_directions", 0)) for job in jobs),
        "eligible_clusters": len(eligible_clusters),
        "eligible_cluster_ids": sorted(eligible_clusters),
        "minimum_eligible_clusters": minimum,
        "valid_prediction_clusters": len(valid_prediction_clusters),
        "valid_prediction_cluster_ids": sorted(valid_prediction_clusters),
        "minimum_valid_prediction_clusters": minimum_valid,
        "stop_threshold_reached": stop_reached,
        "catalog_exhausted": len(jobs) == len(rows),
    }
