"""Strict parsing for the pinned OpenPI LIBERO evaluator's text logs."""

from __future__ import annotations

import dataclasses
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
        }
        for task, successes in sorted(grouped.items())
    ]
    total_successes = sum(result.success for result in results)
    summary = {
        "schema_version": 1,
        "suite": suite,
        "tasks": len(rows),
        "episodes": len(results),
        "successes": total_successes,
        "success_rate": total_successes / len(results),
    }
    return rows, summary
