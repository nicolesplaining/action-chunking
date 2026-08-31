from __future__ import annotations

import pytest

from action_chunking.libero_logs import parse_episode_results, summarize_episode_results, wilson_interval


def test_parse_episode_results_ignores_incomplete_episode(tmp_path) -> None:
    log = tmp_path / "client.log"
    log.write_text(
        "INFO:root:Task: pick alpha\n"
        "INFO:root:Success: True\n"
        "\x1b[31mINFO:root:Task: pick beta\x1b[0m\n"
        "INFO:root:Success: False\n"
        "INFO:root:Task: interrupted\n"
    )
    results = parse_episode_results(log)
    assert [(result.task, result.success) for result in results] == [
        ("pick alpha", True),
        ("pick beta", False),
    ]

    rows, summary = summarize_episode_results(results, "libero_object")
    assert [row["task"] for row in rows] == ["pick alpha", "pick beta"]
    assert summary["episodes"] == 2
    assert summary["success_rate"] == 0.5


def test_summarize_episode_results_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no completed episodes"):
        summarize_episode_results([], "libero_object")


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(491, 500)
    assert low < 491 / 500 < high
    assert wilson_interval(16, 16)[1] == 1.0
    assert wilson_interval(0, 16)[0] == 0.0
