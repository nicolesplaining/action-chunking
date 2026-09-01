"""Deterministic outcome metrics for action-chunk interventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

LIBERO_ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "translation": (0, 1, 2),
    "rotation": (3, 4, 5),
    "gripper": (6,),
    "all": tuple(range(7)),
}


def normalized_causal_transfer(
    patched: ArrayLike,
    base: ArrayLike,
    donor: ArrayLike,
    *,
    minimum_contrast: float = 1e-12,
) -> float:
    """Project a patched outcome onto the clean base-to-donor contrast."""

    patched_array = np.asarray(patched, dtype=np.float64).reshape(-1)
    base_array = np.asarray(base, dtype=np.float64).reshape(-1)
    donor_array = np.asarray(donor, dtype=np.float64).reshape(-1)
    if patched_array.shape != base_array.shape or donor_array.shape != base_array.shape:
        raise ValueError("patched, base, and donor outcomes must have the same shape")
    contrast = donor_array - base_array
    denominator = float(np.dot(contrast, contrast))
    if denominator <= minimum_contrast:
        return float("nan")
    return float(np.dot(patched_array - base_array, contrast) / denominator)


def action_group_transfer(
    patched: ArrayLike,
    base: ArrayLike,
    donor: ArrayLike,
    indices: tuple[int, ...],
) -> float:
    """Compute NCTE for selected physical action dimensions."""

    patched_array, base_array, donor_array = _action_arrays(patched, base, donor)
    return normalized_causal_transfer(
        patched_array[:, indices],
        base_array[:, indices],
        donor_array[:, indices],
    )


def per_position_transfer(patched: ArrayLike, base: ArrayLike, donor: ArrayLike) -> NDArray[np.float64]:
    """Compute an NCTE value independently for every future action position."""

    patched_array, base_array, donor_array = _action_arrays(patched, base, donor)
    return np.asarray(
        [
            normalized_causal_transfer(patched_array[index], base_array[index], donor_array[index])
            for index in range(base_array.shape[0])
        ],
        dtype=np.float64,
    )


def target_direction_affinity(
    actions: ArrayLike,
    end_effector_position: ArrayLike,
    base_target_position: ArrayLike,
    donor_target_position: ArrayLike,
    *,
    executed_horizon: int = 5,
) -> float:
    """Score early translation toward donor versus base target.

    LIBERO's first three controls are Cartesian delta-position commands. This
    score is positive when their cumulative direction favors the donor target
    and negative when it favors the base target.
    """

    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[1] < 3:
        raise ValueError("actions must have shape [horizon, action_dim>=3]")
    if not 1 <= executed_horizon <= action_array.shape[0]:
        raise ValueError("executed_horizon must lie within the action chunk")
    eef = _xyz(end_effector_position, "end_effector_position")
    base_direction = _unit(_xyz(base_target_position, "base_target_position") - eef)
    donor_direction = _unit(_xyz(donor_target_position, "donor_target_position") - eef)
    displacement = action_array[:executed_horizon, :3].sum(axis=0)
    return float(np.dot(displacement, donor_direction) - np.dot(displacement, base_direction))


def gripper_closure_position(actions: ArrayLike, *, threshold: float = 0.0) -> int | None:
    """Return the first future position whose LIBERO gripper command closes."""

    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[1] < 7:
        raise ValueError("actions must have shape [horizon, action_dim>=7]")
    positions = np.flatnonzero(action_array[:, 6] > threshold)
    return int(positions[0]) if len(positions) else None


def gripper_closure_time(actions: ArrayLike, *, threshold: float = 0.0) -> int:
    """Return closure position, encoding right-censoring at the chunk horizon."""

    action_array = np.asarray(actions, dtype=np.float64)
    position = gripper_closure_position(action_array, threshold=threshold)
    return action_array.shape[0] if position is None else position


def summarize_transfer(patched: ArrayLike, base: ArrayLike, donor: ArrayLike) -> dict[str, object]:
    """Return preregistered action-group and temporal transfer summaries."""

    return {
        "ncte": {
            name: action_group_transfer(patched, base, donor, indices)
            for name, indices in LIBERO_ACTION_GROUPS.items()
        },
        "per_position_ncte": per_position_transfer(patched, base, donor).tolist(),
    }


def _action_arrays(*arrays: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    values = tuple(np.asarray(array, dtype=np.float64) for array in arrays)
    if any(value.ndim != 2 for value in values):
        raise ValueError("action chunks must have shape [horizon, action_dim]")
    if len({value.shape for value in values}) != 1:
        raise ValueError("action chunks must have matching shapes")
    return values


def _xyz(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape [3]")
    return array


def _unit(value: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise ValueError("target direction is undefined at zero distance")
    return value / norm
