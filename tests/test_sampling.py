import dataclasses

import torch

from action_chunking.sampling import PreparedCondition, sample_actions


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
