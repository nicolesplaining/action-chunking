"""Frozen accounting for mid-sampling instruction retargeting."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RetargetPlan:
    """Velocity-field evaluations before and after a late instruction update."""

    strategy: str
    switch_after_steps: int
    total_flow_steps: int
    pre_event_velocity_evaluations: int
    post_event_velocity_evaluations: int
    discarded_velocity_evaluations: int

    @property
    def post_event_evaluation_savings(self) -> int:
        return self.total_flow_steps - self.post_event_velocity_evaluations

    @property
    def post_event_evaluation_savings_fraction(self) -> float:
        return self.post_event_evaluation_savings / self.total_flow_steps


def retarget_plan(strategy: str, switch_after_steps: int, total_flow_steps: int) -> RetargetPlan:
    """Validate a retarget strategy and return its non-overlapping compute counts."""

    if total_flow_steps <= 0:
        raise ValueError("total_flow_steps must be positive")
    if not 0 <= switch_after_steps <= total_flow_steps:
        raise ValueError("switch_after_steps must lie within the integration horizon")
    if strategy not in {"continue", "restart"}:
        raise ValueError("retarget strategy must be continue or restart")
    post_event = total_flow_steps - switch_after_steps if strategy == "continue" else total_flow_steps
    discarded = 0 if strategy == "continue" else switch_after_steps
    return RetargetPlan(
        strategy=strategy,
        switch_after_steps=switch_after_steps,
        total_flow_steps=total_flow_steps,
        pre_event_velocity_evaluations=switch_after_steps,
        post_event_velocity_evaluations=post_event,
        discarded_velocity_evaluations=discarded,
    )
