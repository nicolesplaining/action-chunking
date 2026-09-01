"""Preregistered state-cluster contrasts for future action-token patches."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def executed_token_contrast(
    rows: list[dict[str, Any]],
    *,
    flow_step: int = 9,
    layer: int = 17,
    execution_horizon: int = 5,
    action_horizon: int = 10,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260831,
) -> dict[str, dict[str, Any]]:
    """Compare causal transfer in executed versus deferred token positions."""
    if not 0 < execution_horizon < action_horizon or bootstrap_replicates <= 0:
        raise ValueError("token horizons and bootstrap count are invalid")
    selected = [
        row
        for row in rows
        if bool(row["eligible"])
        and int(row["flow_step"]) == flow_step
        and int(row["layer"]) == layer
    ]
    grouped: dict[tuple[str, str], dict[int, float]] = {}
    for row in selected:
        key = (str(row["scene_state_sha256"]), str(row["metric"]))
        position = int(row["action_position"])
        values = grouped.setdefault(key, {})
        if position in values:
            raise ValueError("duplicate action-token cell within a scene state")
        values[position] = float(row["symmetric_ncte"])
    expected_positions = set(range(action_horizon))
    by_metric: dict[str, list[float]] = {}
    for (_cluster, metric), values in sorted(grouped.items()):
        if not expected_positions <= set(values):
            raise ValueError("incomplete action-token grid in the primary comparison window")
        executed = np.mean([values[position] for position in range(execution_horizon)])
        deferred = np.mean(
            [values[position] for position in range(execution_horizon, action_horizon)]
        )
        by_metric.setdefault(metric, []).append(float(executed - deferred))
    rng = np.random.default_rng(seed)
    output = {}
    for metric, raw_values in sorted(by_metric.items()):
        values = np.asarray(raw_values, dtype=np.float64)
        indices = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        bootstrap = values[indices].mean(axis=1)
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
        null = (signs * values[None, :]).mean(axis=1)
        observed = float(values.mean())
        output[metric] = {
            "flow_step": flow_step,
            "layer": layer,
            "executed_positions": list(range(execution_horizon)),
            "deferred_positions": list(range(execution_horizon, action_horizon)),
            "eligible_state_clusters": len(values),
            "mean_executed_minus_deferred_ncte": observed,
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "positive_state_fraction": float(np.mean(values > 0.0)),
            "p_two_sided_exact_sign_flip": float(np.mean(np.abs(null) >= abs(observed))),
        }
    return output
