from __future__ import annotations

import pytest

from action_chunking.libero_logs import (
    combine_suite_summaries,
    parse_episode_results,
    summarize_episode_results,
    validate_task_counts,
    wilson_interval,
)


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


def test_validate_task_counts_accepts_complete_balanced_suite() -> None:
    rows = [{"task": f"task {index}", "episodes": 50} for index in range(10)]
    validate_task_counts(rows, expected_tasks=10, expected_episodes_per_task=50)


def test_validate_task_counts_rejects_missing_or_truncated_tasks() -> None:
    rows = [{"task": "complete", "episodes": 50}, {"task": "truncated", "episodes": 49}]
    with pytest.raises(ValueError, match="expected 10 tasks, found 2"):
        validate_task_counts(rows, expected_tasks=10, expected_episodes_per_task=50)
    with pytest.raises(ValueError, match="truncated=49"):
        validate_task_counts(rows, expected_tasks=2, expected_episodes_per_task=50)


def test_combine_suite_summaries_requires_exact_suite_set_and_counts() -> None:
    summaries = [
        {"suite": "suite_a", "episodes": 500, "successes": 490},
        {"suite": "suite_b", "episodes": 500, "successes": 480},
    ]
    rows, combined = combine_suite_summaries(
        summaries,
        expected_suites={"suite_a", "suite_b"},
        expected_episodes_per_suite=500,
    )
    assert [row["suite"] for row in rows] == ["suite_a", "suite_b"]
    assert combined["episodes"] == 1000
    assert combined["successes"] == 970
    assert combined["micro_success_rate"] == 0.97
    assert combined["macro_suite_success_rate"] == 0.97

    with pytest.raises(ValueError, match="suite set mismatch"):
        combine_suite_summaries(summaries, expected_suites={"suite_a", "suite_c"})
    with pytest.raises(ValueError, match="duplicate suite"):
        combine_suite_summaries([summaries[0], summaries[0]])
    with pytest.raises(ValueError, match="expected 499 episodes"):
        combine_suite_summaries(summaries, expected_episodes_per_suite=499)
