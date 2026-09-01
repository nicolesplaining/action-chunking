from __future__ import annotations

import math

import numpy as np
import pytest

from action_chunking.orientation import (
    contact_aligned_grasp_frame,
    mean_quaternion,
    normalize_quaternion,
    orientation_affinity,
    quaternion_geodesic_rad,
)


def test_quaternion_geodesic_handles_antipodal_sign() -> None:
    identity = [0.0, 0.0, 0.0, 1.0]
    assert quaternion_geodesic_rad(identity, np.negative(identity)) == pytest.approx(0.0)
    assert quaternion_geodesic_rad(identity, [0.0, 0.0, 1.0, 0.0]) == pytest.approx(math.pi)


def test_markley_mean_handles_mixed_signs() -> None:
    expected = normalize_quaternion([0.1, -0.2, 0.3, 0.9])
    observed = mean_quaternion([expected, -expected, expected])
    assert abs(float(np.dot(observed, expected))) == pytest.approx(1.0)


def test_contact_frame_uses_exact_inclusive_window() -> None:
    records = [
        {"step_in_episode": step, "eef_quat": [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]}
        for step, angle in enumerate([0.0, 0.1, 0.2, 0.3], start=1)
    ]
    frame = contact_aligned_grasp_frame(records, 4, window_steps=3)
    assert frame["trajectory_steps"] == [2, 3, 4]
    assert quaternion_geodesic_rad(frame["quaternion_xyzw"], records[2]["eef_quat"]) < 1e-7


def test_contact_frame_rejects_missing_step() -> None:
    records = [
        {"step_in_episode": 1, "eef_quat": [0.0, 0.0, 0.0, 1.0]},
        {"step_in_episode": 3, "eef_quat": [0.0, 0.0, 0.0, 1.0]},
    ]
    with pytest.raises(ValueError, match="exact contact window"):
        contact_aligned_grasp_frame(records, 3, window_steps=3)


def test_orientation_affinity_maps_clean_endpoints_without_clipping() -> None:
    source = [0.0, 0.0, 0.0, 1.0]
    destination = [0.0, 0.0, math.sin(0.3), math.cos(0.3)]
    source_result = orientation_affinity(source, source, destination)
    destination_result = orientation_affinity(destination, source, destination)
    assert source_result["source_retention"] == pytest.approx(1.0)
    assert destination_result["source_retention"] == pytest.approx(0.0)


def test_orientation_affinity_enforces_frozen_contrast() -> None:
    source = [0.0, 0.0, 0.0, 1.0]
    near = [0.0, 0.0, math.sin(0.01), math.cos(0.01)]
    with pytest.raises(ValueError, match="frozen threshold"):
        orientation_affinity(source, source, near)
