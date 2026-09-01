from __future__ import annotations

import pytest

from action_chunking.competence import evaluate_pi0_competence


def test_pi0_competence_passes_only_both_frozen_levels() -> None:
    result = evaluate_pi0_competence(_suite(460), _pairs(12))

    assert result["suite"]["passed"] is True
    assert result["heldout_pairs"]["passed"] is True
    assert result["passed"] is True
    assert result["architecture_timing_claim_allowed"] is True


def test_pi0_competence_rejects_pair_level_failure() -> None:
    result = evaluate_pi0_competence(_suite(460), _pairs(11))

    assert result["passed"] is False
    assert result["failure_interpretation"] == "competence_limited_control"


def test_pi0_competence_rejects_wrong_episode_count() -> None:
    suite = _suite(460)
    suite["episodes"] = 499
    with pytest.raises(ValueError, match="exactly 500"):
        evaluate_pi0_competence(suite, _pairs(12))


def _suite(successes: int) -> dict:
    return {"suite": "libero_goal", "episodes": 500, "successes": successes}


def _pairs(passing: int) -> dict:
    return {
        "expected_pairs": 16,
        "completed_pairs": 16,
        "jobs": [
            {"exact_dual_success_target_first": index < passing}
            for index in range(16)
        ],
    }
