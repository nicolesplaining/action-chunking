from __future__ import annotations

import pytest

from action_chunking.catalog_progress import summarize_catalog_progress


def test_catalog_progress_counts_clusters_not_directions() -> None:
    plan = _plan()
    jobs = [
        _job(0, "a", "cluster-a", 2),
        _job(1, "b", "cluster-a", 1),
        _job(2, "c", "cluster-b", 1),
    ]

    progress = summarize_catalog_progress(plan, jobs)

    assert progress["eligible_directions"] == 4
    assert progress["eligible_clusters"] == 2
    assert progress["stop_threshold_reached"] is True


def test_catalog_progress_rejects_nonprefix_job() -> None:
    with pytest.raises(ValueError, match="contiguous ordered prefix"):
        summarize_catalog_progress(_plan(), [_job(1, "a", "cluster-a", 0)])


def _plan() -> dict:
    return {
        "stop_rule": {"minimum_eligible_clusters": 2},
        "rows": [
            {"screen_id": "a", "cluster_id": "cluster-a"},
            {"screen_id": "b", "cluster_id": "cluster-a"},
            {"screen_id": "c", "cluster_id": "cluster-b"},
        ],
    }


def _job(index: int, screen_id: str, cluster_id: str, eligible: int) -> dict:
    return {
        "plan_index": index,
        "screen_id": screen_id,
        "cluster_id": cluster_id,
        "eligible_directions": eligible,
    }
