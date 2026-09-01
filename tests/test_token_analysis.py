from __future__ import annotations

import pytest

from action_chunking.token_analysis import executed_token_contrast


def test_executed_token_contrast_uses_scene_clusters() -> None:
    rows = []
    for cluster, scale in (("a", 1.0), ("b", 2.0)):
        for position in range(10):
            rows.append(
                {
                    "scene_state_sha256": cluster,
                    "metric": "target_direction",
                    "eligible": True,
                    "flow_step": 9,
                    "layer": 17,
                    "action_position": position,
                    "symmetric_ncte": scale if position < 5 else 0.0,
                }
            )

    result = executed_token_contrast(rows, bootstrap_replicates=100, seed=1)

    target = result["target_direction"]
    assert target["eligible_state_clusters"] == 2
    assert target["mean_executed_minus_deferred_ncte"] == pytest.approx(1.5)
    assert target["positive_state_fraction"] == 1.0


def test_executed_token_contrast_rejects_incomplete_site() -> None:
    rows = [
        {
            "scene_state_sha256": "a",
            "metric": "translation",
            "eligible": True,
            "flow_step": 9,
            "layer": 17,
            "action_position": position,
            "symmetric_ncte": 0.0,
        }
        for position in range(9)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        executed_token_contrast(rows)


def test_executed_token_contrast_accepts_positions_beyond_primary_window() -> None:
    rows = [
        {
            "scene_state_sha256": "a",
            "metric": "translation",
            "eligible": True,
            "flow_step": 9,
            "layer": 17,
            "action_position": position,
            "symmetric_ncte": 1.0 if position < 5 else 0.0,
        }
        for position in range(50)
    ]

    result = executed_token_contrast(rows, bootstrap_replicates=100)

    assert result["translation"]["executed_positions"] == list(range(5))
    assert result["translation"]["deferred_positions"] == list(range(5, 10))
