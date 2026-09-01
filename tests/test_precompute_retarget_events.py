from action_chunking.retarget_eligibility import controller_replay_summary_exact


def test_replay_exact_requires_two_transition_exact_sides() -> None:
    result = {
        "controller_replay_required": True,
        "controller_replay_applied": True,
        "controller_replay_trajectory_max_abs_error": 0.0,
    }

    assert controller_replay_summary_exact({"results": [result, result]}, required=True) is True
    assert controller_replay_summary_exact({"results": [result]}, required=True) is False
    assert (
        controller_replay_summary_exact(
            {"results": [result]}, required=True, expected_results=1
        )
        is True
    )


def test_replay_exact_is_vacuous_when_not_required() -> None:
    assert controller_replay_summary_exact({"results": []}, required=False) is True
