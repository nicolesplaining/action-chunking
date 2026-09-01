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


def binomial_upper_bound(losses: int, clusters: int, alpha: float = 0.05) -> float:
    """One-sided Clopper--Pearson upper bound for a paired loss probability.

    The implementation avoids a SciPy dependency and inverts the exact binomial
    CDF with bisection.  ``clusters`` is the independent scene-cluster count,
    not a direction, action token, or noise-seed count.
    """
    if clusters <= 0:
        raise ValueError("clusters must be positive")
    if losses < 0 or losses > clusters:
        raise ValueError("losses must lie in [0, clusters]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if losses == 0:
        return zero_loss_upper_bound(clusters, alpha)
    if losses == clusters:
        return 1.0

    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(losses, clusters, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes == trials else 0.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(value + 1)
        - math.lgamma(trials - value + 1)
        + value * log_probability
        + (trials - value) * log_complement
        for value in range(successes + 1)
    ]
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)
