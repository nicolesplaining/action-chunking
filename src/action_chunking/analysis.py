"""Small, dependency-light statistical helpers used by pilot analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def isotonic_increasing(values: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """Least-squares nondecreasing fit using the pool-adjacent-violators algorithm."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("values must be finite")
    if weights is None:
        observation_weights = np.ones_like(observations)
    else:
        observation_weights = np.asarray(weights, dtype=np.float64)
        if observation_weights.shape != observations.shape or np.any(observation_weights <= 0):
            raise ValueError("weights must be positive and match values")

    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(observations, observation_weights, strict=True)):
        blocks.append({"start": index, "end": index + 1, "weight": weight, "mean": value})
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = float(left["weight"]) + float(right["weight"])
            pooled_mean = (
                float(left["mean"]) * float(left["weight"])
                + float(right["mean"]) * float(right["weight"])
            ) / total_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": total_weight,
                    "mean": pooled_mean,
                }
            )

    fitted = np.empty_like(observations)
    for block in blocks:
        fitted[int(block["start"]) : int(block["end"])] = float(block["mean"])
    return fitted


def commitment_step(retention: ArrayLike, threshold: float = 0.8) -> tuple[int | None, NDArray[np.float64]]:
    """Return the first boundary whose isotonic retention stays above threshold."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    fitted = isotonic_increasing(retention)
    qualifying = np.flatnonzero(fitted >= threshold)
    return (int(qualifying[0]) if qualifying.size else None), fitted


def symmetric_mean(first: Sequence[float], second: Sequence[float]) -> NDArray[np.float64]:
    """Average two sign-aligned directional effects after validating their grid."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or first_array.ndim != 1:
        raise ValueError("directional effects must be matching one-dimensional arrays")
    return (first_array + second_array) / 2.0


def benjamini_hochberg(p_values: ArrayLike) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values, preserving the original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a finite one-dimensional array in [0, 1]")
    if values.size == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted
