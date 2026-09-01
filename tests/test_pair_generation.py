from __future__ import annotations

import pytest

from action_chunking.pairs import advance_reset_sequence


def test_nonzero_pair_generation_advances_environment_reset_sequence() -> None:
    env = _CountingEnv()

    advance_reset_sequence(env, 16)

    assert env.resets == 16


def test_reset_sequence_rejects_negative_start_index() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        advance_reset_sequence(_CountingEnv(), -1)


class _CountingEnv:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
