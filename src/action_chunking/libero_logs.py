"""Strict parsing for the pinned OpenPI LIBERO evaluator's text logs."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    task: str
    success: bool


def parse_episode_results(path: Path) -> list[EpisodeResult]:
    """Extract one result per completed episode, ignoring partial starts."""

    results = []
    current_task = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line).strip()
        marker = "Task: "
        if marker in line:
            current_task = line.split(marker, 1)[1].strip()
        success_marker = "Success: "
        if success_marker in line:
            if current_task is None:
                raise ValueError("success record appeared before a task record")
            value = line.split(success_marker, 1)[1].strip()
            if value not in {"True", "False"}:
                raise ValueError(f"invalid success value {value!r}")
            results.append(EpisodeResult(task=current_task, success=value == "True"))
            current_task = None
    return results


def summarize_episode_results(results: list[EpisodeResult], suite: str) -> tuple[list[dict], dict]:
    """Return deterministic per-task rows and a suite-level summary."""

    if not results:
        raise ValueError("no completed episodes found")

    grouped: dict[str, list[bool]] = {}
    for result in results:
        grouped.setdefault(result.task, []).append(result.success)
    rows = [
        {
            "suite": suite,
            "task": task,
            "episodes": len(successes),
            "successes": sum(successes),
            "success_rate": sum(successes) / len(successes),
            "success_rate_ci95_low": wilson_interval(sum(successes), len(successes))[0],
            "success_rate_ci95_high": wilson_interval(sum(successes), len(successes))[1],
        }
        for task, successes in sorted(grouped.items())
    ]
    total_successes = sum(result.success for result in results)
    ci_low, ci_high = wilson_interval(total_successes, len(results))
    summary = {
        "schema_version": 1,
        "suite": suite,
        "tasks": len(rows),
        "episodes": len(results),
        "successes": total_successes,
        "success_rate": total_successes / len(results),
        "success_rate_ci95_low": ci_low,
        "success_rate_ci95_high": ci_high,
    }
    return rows, summary


def validate_task_counts(
    rows: list[dict],
    *,
    expected_tasks: int | None = None,
    expected_episodes_per_task: int | None = None,
) -> None:
    """Reject suite summaries with missing, duplicated, or truncated task blocks."""

    if expected_tasks is not None and len(rows) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} tasks, found {len(rows)}")
    if expected_episodes_per_task is None:
        return
    mismatches = {
        str(row["task"]): int(row["episodes"])
        for row in rows
        if int(row["episodes"]) != expected_episodes_per_task
    }
    if mismatches:
        details = ", ".join(f"{task}={episodes}" for task, episodes in sorted(mismatches.items()))
        raise ValueError(f"expected {expected_episodes_per_task} episodes per task; found {details}")


def combine_suite_summaries(
    summaries: list[dict],
    *,
    expected_suites: set[str] | None = None,
    expected_episodes_per_suite: int | None = None,
) -> tuple[list[dict], dict]:
    """Combine independently validated suite summaries into one benchmark result."""

    if not summaries:
        raise ValueError("no suite summaries provided")
    rows = []
    seen = set()
    for summary in summaries:
        suite = str(summary["suite"])
        if suite in seen:
            raise ValueError(f"duplicate suite summary: {suite}")
        seen.add(suite)
        episodes = int(summary["episodes"])
        successes = int(summary["successes"])
        if episodes <= 0 or not 0 <= successes <= episodes:
            raise ValueError(f"invalid counts for {suite}: {successes}/{episodes}")
        if expected_episodes_per_suite is not None and episodes != expected_episodes_per_suite:
            raise ValueError(f"expected {expected_episodes_per_suite} episodes for {suite}, found {episodes}")
        low, high = wilson_interval(successes, episodes)
        rows.append(
            {
                "suite": suite,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
                "success_rate_ci95_low": low,
                "success_rate_ci95_high": high,
            }
        )
    if expected_suites is not None and seen != expected_suites:
        missing = sorted(expected_suites - seen)
        unexpected = sorted(seen - expected_suites)
        raise ValueError(f"suite set mismatch: missing={missing}, unexpected={unexpected}")
    rows.sort(key=lambda row: row["suite"])
    total_episodes = sum(row["episodes"] for row in rows)
    total_successes = sum(row["successes"] for row in rows)
    low, high = wilson_interval(total_successes, total_episodes)
    combined = {
        "schema_version": 1,
        "suites": len(rows),
        "episodes": total_episodes,
        "successes": total_successes,
        "micro_success_rate": total_successes / total_episodes,
        "micro_success_rate_ci95_low": low,
        "micro_success_rate_ci95_high": high,
        "macro_suite_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
    }
    return rows, combined


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials <= 0 or not 0 <= successes <= trials or z <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials and z > 0")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)
