from __future__ import annotations

import pytest

from action_chunking.noninferiority import minimum_zero_loss_clusters, zero_loss_upper_bound


def test_five_point_zero_loss_design_requires_59_clusters() -> None:
    assert minimum_zero_loss_clusters(0.05, 0.05) == 59
    assert zero_loss_upper_bound(58, 0.05) >= 0.05
    assert zero_loss_upper_bound(59, 0.05) < 0.05


def test_zero_loss_design_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        minimum_zero_loss_clusters(0.0, 0.05)
    with pytest.raises(ValueError):
        zero_loss_upper_bound(0)
