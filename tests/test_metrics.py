from __future__ import annotations

import numpy as np

from action_chunking.metrics import (
    gripper_closure_position,
    normalized_causal_transfer,
    per_position_transfer,
    target_direction_affinity,
)


def test_normalized_causal_transfer_has_interpretable_endpoints() -> None:
    base = np.zeros((2, 3))
    donor = np.ones((2, 3))
    assert normalized_causal_transfer(base, base, donor) == 0.0
    assert normalized_causal_transfer(donor, base, donor) == 1.0
    assert normalized_causal_transfer(2 * donor, base, donor) == 2.0


def test_per_position_transfer_preserves_temporal_localization() -> None:
    base = np.zeros((2, 2))
    donor = np.ones((2, 2))
    patched = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(per_position_transfer(patched, base, donor), [0.0, 1.0])


def test_target_direction_affinity_is_oriented_to_donor() -> None:
    actions = np.zeros((10, 7))
    actions[:5, 0] = 0.1
    score = target_direction_affinity(
        actions,
        end_effector_position=[0.0, 0.0, 0.0],
        base_target_position=[-1.0, 0.0, 0.0],
        donor_target_position=[1.0, 0.0, 0.0],
    )
    assert score == 1.0


def test_gripper_closure_position_handles_right_censoring() -> None:
    actions = np.zeros((4, 7))
    actions[:, 6] = [-1.0, -0.2, 0.1, 1.0]
    assert gripper_closure_position(actions) == 2
    actions[:, 6] = -1.0
    assert gripper_closure_position(actions) is None
