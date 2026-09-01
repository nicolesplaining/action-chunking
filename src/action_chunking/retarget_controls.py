"""Identity controls shared by dynamic-retarget generation and analysis."""

from __future__ import annotations

from typing import Any

BOUNDARY_ZERO_BEHAVIOR_FIELDS = (
    "first_contact_object",
    "first_contact_step",
    "first_contact_replan_index",
    "new_target_contact_step",
    "old_target_contact_step",
    "new_target_first",
    "old_target_first",
    "first_chunk_new_target_contact",
    "first_chunk_old_event",
    "no_registered_contact_first_chunk",
    "eventual_new_task_success",
    "clean_replanning_rescue",
    "first_chunk_correction_survives",
    "completion_steps",
)


def boundary_zero_behavior_exact(
    rows: list[dict[str, Any]],
    sides: list[str],
) -> bool | None:
    """Check boundary-zero continue/restart behavioral equivalence."""
    by_key = {
        (str(row["strategy"]), int(row["switch_after_steps"]), str(row["side"])): row
        for row in rows
    }
    expected = {
        (strategy, 0, side)
        for strategy in ("restart", "continue")
        for side in sides
    }
    if not expected <= set(by_key):
        return None
    return all(
        by_key[("restart", 0, side)][field]
        == by_key[("continue", 0, side)][field]
        for side in sides
        for field in BOUNDARY_ZERO_BEHAVIOR_FIELDS
    )
