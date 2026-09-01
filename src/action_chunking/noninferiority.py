"""Exact design calculations for paired retargeting noninferiority."""

from __future__ import annotations

import math


def minimum_zero_loss_clusters(margin: float = 0.05, alpha: float = 0.05) -> int:
    """Minimum clusters whose zero-loss exact upper bound is below ``margin``."""
    if not 0.0 < margin < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("margin and alpha must lie strictly between zero and one")
    estimate = math.floor(math.log(alpha) / math.log(1.0 - margin)) + 1
    while zero_loss_upper_bound(estimate, alpha) >= margin:
        estimate += 1
    return estimate


def zero_loss_upper_bound(clusters: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound after zero losses in ``clusters`` trials."""
    if clusters <= 0:
        raise ValueError("clusters must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return 1.0 - alpha ** (1.0 / clusters)
