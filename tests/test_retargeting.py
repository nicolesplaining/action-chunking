from action_chunking.retargeting import retarget_plan


def test_continue_reuses_completed_flow_updates() -> None:
    plan = retarget_plan("continue", 7, 10)

    assert plan.pre_event_velocity_evaluations == 7
    assert plan.post_event_velocity_evaluations == 3
    assert plan.discarded_velocity_evaluations == 0
    assert plan.post_event_evaluation_savings == 7
    assert plan.post_event_evaluation_savings_fraction == 0.7


def test_restart_discards_prefix_and_recomputes_full_trajectory() -> None:
    plan = retarget_plan("restart", 7, 10)

    assert plan.pre_event_velocity_evaluations == 7
    assert plan.post_event_velocity_evaluations == 10
    assert plan.discarded_velocity_evaluations == 7
    assert plan.post_event_evaluation_savings == 0


def test_retarget_plan_rejects_invalid_inputs() -> None:
    for strategy, boundary, steps in (("invalid", 1, 10), ("continue", -1, 10), ("continue", 11, 10)):
        try:
            retarget_plan(strategy, boundary, steps)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid retarget plan should fail")
