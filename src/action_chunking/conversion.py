"""Parity checks for JAX-to-PyTorch policy conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


def conversion_parity_summary(
    identifiers: list[str],
    reference: np.ndarray,
    converted: np.ndarray,
    *,
    max_abs_tolerance: float = 0.02,
    minimum_cosine_similarity: float = 0.999,
) -> dict[str, Any]:
    """Evaluate frozen per-case physical-action conversion tolerances."""
    if reference.shape != converted.shape or reference.ndim != 3:
        raise ValueError("conversion arrays must share [case, action, dimension] shape")
    if len(identifiers) != reference.shape[0] or len(set(identifiers)) != len(identifiers):
        raise ValueError("conversion case identifiers must be unique and complete")
    if max_abs_tolerance <= 0.0 or not 0.0 < minimum_cosine_similarity <= 1.0:
        raise ValueError("conversion tolerances are invalid")
    rows = []
    for identifier, source, destination in zip(
        identifiers, reference, converted, strict=True
    ):
        difference = destination - source
        source_flat = source.ravel()
        destination_flat = destination.ravel()
        denominator = np.linalg.norm(source_flat) * np.linalg.norm(destination_flat)
        cosine = float(np.dot(source_flat, destination_flat) / denominator)
        row = {
            "case": identifier,
            "max_abs_error": float(np.max(np.abs(difference))),
            "mean_abs_error": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "cosine_similarity": cosine,
        }
        row["passed"] = bool(
            row["max_abs_error"] <= max_abs_tolerance
            and row["cosine_similarity"] >= minimum_cosine_similarity
        )
        rows.append(row)
    return {
        "schema_version": 1,
        "cases": len(rows),
        "shape_per_case": list(reference.shape[1:]),
        "max_abs_tolerance": max_abs_tolerance,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "maximum_case_abs_error": max(row["max_abs_error"] for row in rows),
        "minimum_case_cosine_similarity": min(row["cosine_similarity"] for row in rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }
