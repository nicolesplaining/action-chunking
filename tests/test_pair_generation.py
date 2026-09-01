from __future__ import annotations

import numpy as np
import pytest

from action_chunking.pairs import (
    advance_action_noise,
    advance_reset_sequence,
    replan_snapshot_step,
)


def test_nonzero_pair_generation_advances_environment_reset_sequence() -> None:
    env = _CountingEnv()

    advance_reset_sequence(env, 16)

    assert env.resets == 16


def test_reset_sequence_rejects_negative_start_index() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        advance_reset_sequence(_CountingEnv(), -1)


def test_replan_snapshot_is_latest_boundary_before_contact() -> None:
    assert replan_snapshot_step(28, 5) == (25, 5)
    assert replan_snapshot_step(30, 5) == (25, 5)
    assert replan_snapshot_step(31, 5) == (30, 6)


def test_action_noise_advances_exact_number_of_draws() -> None:
    advanced = np.random.default_rng(0)
    reference = np.random.default_rng(0)
    advance_action_noise(advanced, 3, (10, 32))
    for _ in range(3):
        reference.standard_normal((10, 32), dtype=np.float32)
    assert (
        advanced.standard_normal((10, 32), dtype=np.float32)
        == reference.standard_normal((10, 32), dtype=np.float32)
    ).all()


class _CountingEnv:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
