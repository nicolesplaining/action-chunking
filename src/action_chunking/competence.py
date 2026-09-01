"""Frozen competence gates for matched model controls."""

from __future__ import annotations

from typing import Any

from action_chunking.libero_logs import wilson_interval

PI0_SUITE_EPISODES = 500
PI0_SUITE_MINIMUM_RATE = 0.90
PI0_SUITE_MINIMUM_WILSON_LOWER = 0.87
PI0_PAIR_EXPECTED = 16
PI0_PAIR_MINIMUM_EXACT_DUAL_SUCCESS_TARGET_FIRST = 12


def evaluate_pi0_competence(
    suite: dict[str, Any], pair_validation: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the preregistered two-level pi0 competence gate."""
    if suite.get("suite") != "libero_goal":
        raise ValueError("pi0 competence requires the libero_goal suite")
    episodes = int(suite.get("episodes", -1))
    successes = int(suite.get("successes", -1))
    if episodes != PI0_SUITE_EPISODES or not 0 <= successes <= episodes:
        raise ValueError("pi0 competence requires exactly 500 valid suite episodes")
    suite_rate = successes / episodes
    suite_wilson_lower, suite_wilson_upper = wilson_interval(successes, episodes)
    suite_passed = bool(
        suite_rate >= PI0_SUITE_MINIMUM_RATE
        and suite_wilson_lower >= PI0_SUITE_MINIMUM_WILSON_LOWER
    )

    if int(pair_validation.get("expected_pairs", -1)) != PI0_PAIR_EXPECTED:
        raise ValueError("pi0 competence requires the frozen 16-pair held-out block")
    if int(pair_validation.get("completed_pairs", -1)) != PI0_PAIR_EXPECTED:
        raise ValueError("pi0 competence pair validation is incomplete")
    jobs = pair_validation.get("jobs", [])
    if len(jobs) != PI0_PAIR_EXPECTED:
        raise ValueError("pi0 competence pair summary has an invalid job count")
    exact_pairs = sum(bool(job.get("exact_dual_success_target_first")) for job in jobs)
    pair_passed = exact_pairs >= PI0_PAIR_MINIMUM_EXACT_DUAL_SUCCESS_TARGET_FIRST
    passed = bool(suite_passed and pair_passed)
    return {
        "schema_version": 1,
        "model": "pi0_libero",
        "checkpoint_selection": "finalized_step_30000_only",
        "suite": {
            "episodes": episodes,
            "successes": successes,
            "success_rate": suite_rate,
            "wilson_95_lower": suite_wilson_lower,
            "wilson_95_upper": suite_wilson_upper,
            "minimum_success_rate": PI0_SUITE_MINIMUM_RATE,
            "minimum_wilson_95_lower": PI0_SUITE_MINIMUM_WILSON_LOWER,
            "passed": suite_passed,
        },
        "heldout_pairs": {
            "expected": PI0_PAIR_EXPECTED,
            "exact_dual_success_target_first": exact_pairs,
            "minimum_required": PI0_PAIR_MINIMUM_EXACT_DUAL_SUCCESS_TARGET_FIRST,
            "passed": pair_passed,
        },
        "passed": passed,
        "architecture_timing_claim_allowed": passed,
        "failure_interpretation": None if passed else "competence_limited_control",
    }
