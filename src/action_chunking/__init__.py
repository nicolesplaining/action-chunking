"""Causal instrumentation for action-chunking policies."""

from __future__ import annotations

from typing import Any

__all__ = [
    "PatchSpec",
    "PreparedCondition",
    "ResidualTrace",
    "ResidualTracer",
    "SamplingTrace",
    "prepare_condition",
    "sample_actions",
]


def __getattr__(name: str) -> Any:
    """Lazily expose torch instrumentation without loading it for pair tools."""

    if name in {"PreparedCondition", "SamplingTrace", "prepare_condition", "sample_actions"}:
        from action_chunking import sampling

        return getattr(sampling, name)
    if name in {"PatchSpec", "ResidualTrace", "ResidualTracer"}:
        from action_chunking import tracing

        return getattr(tracing, name)
    raise AttributeError(name)
