"""Contact-aligned end-effector orientation endpoints for retargeting sweeps."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

GRASP_ORIENTATION_WINDOW_STEPS = 3
MINIMUM_GRASP_REFERENCE_CONTRAST_RAD = 0.2
GRASP_SOURCE_RETENTION_THRESHOLD = 0.8


def normalize_quaternion(quaternion: Iterable[float]) -> np.ndarray:
    """Return a unit quaternion, preserving robosuite's xyzw convention."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be positive")
    return value / norm


def quaternion_geodesic_rad(first: Iterable[float], second: Iterable[float]) -> float:
    """Shortest SO(3) angular distance between two unit quaternions."""
    dot = abs(float(np.dot(normalize_quaternion(first), normalize_quaternion(second))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def mean_quaternion(quaternions: Iterable[Iterable[float]]) -> np.ndarray:
    """Compute the sign-invariant Markley mean of a nonempty quaternion sequence."""
    values = np.asarray([normalize_quaternion(value) for value in quaternions])
    if values.shape[0] == 0:
        raise ValueError("quaternion mean requires at least one value")
    eigenvalues, eigenvectors = np.linalg.eigh(values.T @ values)
    result = normalize_quaternion(eigenvectors[:, int(np.argmax(eigenvalues))])
    if float(np.dot(result, values[-1])) < 0.0:
        result = -result
    return result


def contact_aligned_grasp_frame(
    trajectory_records: list[dict[str, Any]],
    contact_step: int,
    *,
    window_steps: int = GRASP_ORIENTATION_WINDOW_STEPS,
) -> dict[str, Any]:
    """Summarize the inclusive window ending at a registered contact step."""
    if contact_step <= 0:
        raise ValueError("contact step must be positive")
    if window_steps <= 0:
        raise ValueError("window steps must be positive")
    start = max(1, contact_step - window_steps + 1)
    selected = [
        record
        for record in trajectory_records
        if start <= int(record["step_in_episode"]) <= contact_step
    ]
    steps = [int(record["step_in_episode"]) for record in selected]
    expected = list(range(start, contact_step + 1))
    if steps != expected:
        raise ValueError(f"trajectory does not contain exact contact window {expected}")
    frame = mean_quaternion(record["eef_quat"] for record in selected)
    distances = [quaternion_geodesic_rad(record["eef_quat"], frame) for record in selected]
    return {
        "contact_step": contact_step,
        "window_steps": len(selected),
        "trajectory_steps": steps,
        "quaternion_xyzw": frame.tolist(),
        "maximum_window_dispersion_rad": max(distances),
    }


def orientation_affinity(
    frame: Iterable[float],
    source_frame: Iterable[float],
    destination_frame: Iterable[float],
    *,
    minimum_reference_contrast_rad: float = MINIMUM_GRASP_REFERENCE_CONTRAST_RAD,
) -> dict[str, float]:
    """Locate a grasp frame relative to source and destination clean controls.

    Source retention is 1 at the source reference and 0 at the destination
    reference. It is intentionally not clipped: off-segment values diagnose
    geometry that neither clean control explains.
    """
    if minimum_reference_contrast_rad <= 0.0:
        raise ValueError("minimum reference contrast must be positive")
    contrast = quaternion_geodesic_rad(source_frame, destination_frame)
    if contrast < minimum_reference_contrast_rad:
        raise ValueError("grasp-orientation reference contrast is below the frozen threshold")
    source_distance = quaternion_geodesic_rad(frame, source_frame)
    destination_distance = quaternion_geodesic_rad(frame, destination_frame)
    source_retention = (destination_distance - source_distance + contrast) / (2.0 * contrast)
    return {
        "source_distance_rad": source_distance,
        "destination_distance_rad": destination_distance,
        "reference_contrast_rad": contrast,
        "source_retention": source_retention,
        "destination_transfer": 1.0 - source_retention,
    }
