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
