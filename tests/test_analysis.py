from __future__ import annotations

import numpy as np

from action_chunking.analysis import benjamini_hochberg, commitment_step, isotonic_increasing


def test_isotonic_increasing_pools_violations() -> None:
    np.testing.assert_allclose(isotonic_increasing([0.0, 0.8, 0.4, 1.0]), [0.0, 0.6, 0.6, 1.0])


def test_commitment_step_uses_fitted_persistent_crossing() -> None:
    step, fitted = commitment_step([0.0, 0.9, 0.7, 0.95], threshold=0.8)
    assert step == 1
    np.testing.assert_allclose(fitted, [0.0, 0.8, 0.8, 0.95])


def test_benjamini_hochberg_restores_original_order() -> None:
    np.testing.assert_allclose(benjamini_hochberg([0.04, 0.01, 0.03]), [0.04, 0.03, 0.04])
