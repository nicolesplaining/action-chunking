import torch
from torch import nn

from action_chunking.tracing import PatchSpec, ResidualTracer


class AddOneLayer(nn.Module):
    def forward(self, hidden):
        return (hidden + 1, "cache")


def test_capture_is_a_no_op():
    layer = AddOneLayer()
    hidden = torch.zeros(1, 4, 3)

    with ResidualTracer([layer], action_horizon=3) as tracer:
        with tracer.at_step(2):
            output = layer(hidden)

    torch.testing.assert_close(output[0], torch.ones_like(hidden))
    assert output[1] == "cache"
    torch.testing.assert_close(tracer.trace.values[(2, 0)], torch.ones(1, 3, 3))


def test_patch_selected_action_position_only():
    layer = AddOneLayer()
    hidden = torch.zeros(1, 4, 3)
    donor = {(0, 0): torch.full((1, 3, 3), 7.0)}

    with ResidualTracer(
        [layer],
        action_horizon=3,
        patch=PatchSpec(step=0, layer=0, positions=(1,)),
        donor=donor,
    ) as tracer:
        with tracer.at_step(0):
            output = layer(hidden)

    expected = torch.ones_like(hidden)
    expected[:, -2, :] = 7.0
    torch.testing.assert_close(output[0], expected)
    # The captured target is the clean pre-intervention residual.
    torch.testing.assert_close(tracer.trace.values[(0, 0)], torch.ones(1, 3, 3))


def test_patch_all_action_positions_preserves_non_action_suffix():
    layer = AddOneLayer()
    hidden = torch.zeros(1, 4, 2)
    donor = {(1, 0): torch.full((1, 3, 2), -4.0)}

    with ResidualTracer(
        [layer], action_horizon=3, patch=PatchSpec(step=1, layer=0), donor=donor
    ) as tracer:
        with tracer.at_step(1):
            output = layer(hidden)[0]

    torch.testing.assert_close(output[:, :1], torch.ones(1, 1, 2))
    torch.testing.assert_close(output[:, 1:], torch.full((1, 3, 2), -4.0))
