import dataclasses

import torch

from action_chunking.sampling import (
    PreparedCondition,
    SamplingTrace,
    early_exit_action_estimate,
    sample_actions,
    select_early_exit_actions,
)


@dataclasses.dataclass
class Config:
    action_horizon: int = 2
    action_dim: int = 3


class ConstantVelocityModel:
    config = Config()

    def denoise_step(self, _state, _masks, _cache, x_t, _time):
        return torch.ones_like(x_t)


def test_sampler_integrates_constant_velocity_and_records_clean_estimate():
    model = ConstantVelocityModel()
    noise = torch.zeros(1, 2, 3)
    condition = PreparedCondition(
        state=torch.empty(0), prefix_pad_masks=torch.empty(0), past_key_values=None
    )

    actions, trace = sample_actions(model, noise, lambda _step: condition, num_steps=4)

    torch.testing.assert_close(actions, torch.full_like(noise, -1.0))
    assert trace.times == [1.0, 0.75, 0.5, 0.25]
    assert len(trace.x_t) == len(trace.v_t) == len(trace.clean_action_estimates) == 4
    for estimate in trace.clean_action_estimates:
        torch.testing.assert_close(estimate, torch.full_like(noise, -1.0))


def test_early_exit_uses_latest_clean_estimate_and_full_step_matches_to_roundoff():
    model = ConstantVelocityModel()
    noise = torch.zeros(1, 2, 3)
    condition = PreparedCondition(
        state=torch.empty(0), prefix_pad_masks=torch.empty(0), past_key_values=None
    )

    boundary_state, partial = sample_actions(
        model, noise, lambda _step: condition, num_steps=4, stop_step=2
    )
    full_actions, full = sample_actions(
        model, noise, lambda _step: condition, num_steps=10
    )

    assert not torch.equal(early_exit_action_estimate(partial), boundary_state)
    assert torch.max(torch.abs(early_exit_action_estimate(full) - full_actions)) < 2e-7
    partial_output, partial_error = select_early_exit_actions(boundary_state, partial, 2, 4)
    full_output, full_error = select_early_exit_actions(full_actions, full, 10, 10)
    assert torch.equal(partial_output, early_exit_action_estimate(partial))
    assert partial_error is None
    assert torch.equal(full_output, full_actions)
    assert full_error is not None and 0.0 < full_error < 2e-7


def test_early_exit_rejects_empty_or_inconsistent_trace():
    try:
        early_exit_action_estimate(SamplingTrace())
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty early-exit trace should fail")

    inconsistent = SamplingTrace(times=[1.0])
    try:
        early_exit_action_estimate(inconsistent)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent early-exit trace should fail")

    try:
        select_early_exit_actions(torch.empty(1), SamplingTrace(), 0, 10)
    except ValueError as error:
        assert "1 <= completed" in str(error)
    else:
        raise AssertionError("zero-step early exit should fail")


def test_sampler_applies_state_and_velocity_interventions_at_named_steps():
    model = ConstantVelocityModel()
    noise = torch.zeros(1, 2, 3)
    condition = PreparedCondition(
        state=torch.empty(0), prefix_pad_masks=torch.empty(0), past_key_values=None
    )

    def patch_state(step, state):
        return torch.ones_like(state) if step == 0 else state

    def patch_velocity(step, velocity):
        return torch.full_like(velocity, 2.0) if step == 1 else velocity

    actions, trace = sample_actions(
        model,
        noise,
        lambda _step: condition,
        num_steps=2,
        state_intervention=patch_state,
        velocity_intervention=patch_velocity,
    )

    torch.testing.assert_close(trace.x_t[0], torch.ones_like(noise))
    torch.testing.assert_close(trace.v_t[1], torch.full_like(noise, 2.0))
    torch.testing.assert_close(actions, torch.full_like(noise, -0.5))


def test_resumed_sampler_exactly_matches_one_pass_condition_switch():
    model = ConstantVelocityModel()
    noise = torch.zeros(1, 2, 3)
    source = PreparedCondition(
        state=torch.tensor(1.0), prefix_pad_masks=torch.empty(0), past_key_values=None
    )
    donor = PreparedCondition(
        state=torch.tensor(3.0), prefix_pad_masks=torch.empty(0), past_key_values=None
    )

    def velocity_from_condition(state, _masks, _cache, x_t, _time):
        return torch.ones_like(x_t) * state

    model.denoise_step = velocity_from_condition
    one_pass, one_trace = sample_actions(
        model,
        noise,
        lambda step: source if step < 2 else donor,
        num_steps=4,
    )
    boundary_state, prefix_trace = sample_actions(
        model,
        noise,
        lambda _step: source,
        num_steps=4,
        stop_step=2,
    )
    resumed, suffix_trace = sample_actions(
        model,
        noise,
        lambda _step: donor,
        num_steps=4,
        start_step=2,
        initial_state=boundary_state,
    )

    assert torch.equal(resumed, one_pass)
    assert prefix_trace.times + suffix_trace.times == one_trace.times
    for split, whole in zip(prefix_trace.x_t + suffix_trace.x_t, one_trace.x_t, strict=True):
        assert torch.equal(split, whole)


def test_resumed_sampler_requires_valid_range_and_state():
    model = ConstantVelocityModel()
    noise = torch.zeros(1, 2, 3)
    condition = PreparedCondition(
        state=torch.empty(0), prefix_pad_masks=torch.empty(0), past_key_values=None
    )

    try:
        sample_actions(model, noise, lambda _step: condition, num_steps=4, start_step=2)
    except ValueError as error:
        assert "initial_state" in str(error)
    else:
        raise AssertionError("resumed sampling without state should fail")

    try:
        sample_actions(model, noise, lambda _step: condition, num_steps=4, start_step=3, stop_step=2)
    except ValueError as error:
        assert "start_step" in str(error)
    else:
        raise AssertionError("reversed sampling range should fail")
