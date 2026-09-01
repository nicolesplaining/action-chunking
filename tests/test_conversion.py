from __future__ import annotations

import numpy as np

from action_chunking.conversion import conversion_parity_summary


def test_conversion_parity_requires_every_case() -> None:
    reference = np.ones((2, 10, 7), dtype=np.float64)
    converted = reference.copy()
    converted[1, 0, 0] += 0.03

    result = conversion_parity_summary(["a", "b"], reference, converted)

    assert result["passed_cases"] == 1
    assert result["passed"] is False


def test_conversion_parity_accepts_small_error() -> None:
    reference = np.ones((1, 10, 7), dtype=np.float64)
    converted = reference + 0.001

    result = conversion_parity_summary(["a"], reference, converted)

    assert result["passed"] is True
