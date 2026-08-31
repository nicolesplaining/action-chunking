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
