"""Thin, inspectable sampling adapter around the official OpenPI model."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from action_chunking.tracing import ResidualTracer


@dataclasses.dataclass(frozen=True)
class PreparedCondition:
    """Cached image/language condition used by one or more flow steps."""

    state: Tensor
    prefix_pad_masks: Tensor
    past_key_values: Any


@dataclasses.dataclass
class SamplingTrace:
    """Persistent action state and velocity at each flow step."""

    times: list[float] = dataclasses.field(default_factory=list)
    x_t: list[Tensor] = dataclasses.field(default_factory=list)
    v_t: list[Tensor] = dataclasses.field(default_factory=list)
    clean_action_estimates: list[Tensor] = dataclasses.field(default_factory=list)

    def cpu(self) -> SamplingTrace:
        return SamplingTrace(
            times=list(self.times),
            x_t=[value.detach().cpu() for value in self.x_t],
            v_t=[value.detach().cpu() for value in self.v_t],
            clean_action_estimates=[value.detach().cpu() for value in self.clean_action_estimates],
        )


def early_exit_action_estimate(trace: SamplingTrace) -> Tensor:
    """Return the latest clean-action estimate from a nonempty partial trajectory."""
    lengths = {
        len(trace.times),
        len(trace.x_t),
        len(trace.v_t),
        len(trace.clean_action_estimates),
    }
    if len(lengths) != 1:
        raise ValueError("sampling trace fields have inconsistent lengths")
    if not trace.clean_action_estimates:
        raise ValueError("early exit requires at least one velocity-field evaluation")
    return trace.clean_action_estimates[-1].clone()


def select_early_exit_actions(
    integrated_state: Tensor,
    trace: SamplingTrace,
    completed_steps: int,
    total_steps: int,
) -> tuple[Tensor, float | None]:
    """Select a partial clean estimate with an exact integrated full-step control."""
    if not 1 <= completed_steps <= total_steps:
        raise ValueError("early-exit steps must satisfy 1 <= completed <= total")
    if len(trace.clean_action_estimates) != completed_steps:
        raise ValueError("early-exit trace length differs from completed steps")
    estimate = early_exit_action_estimate(trace)
    if integrated_state.shape != estimate.shape:
        raise ValueError("integrated state and clean estimate have different shapes")
    if completed_steps < total_steps:
        return estimate, None
    estimate_error = float(torch.max(torch.abs(estimate - integrated_state)).item())
    return integrated_state.clone(), estimate_error


@torch.no_grad()
def prepare_condition(model: Any, observation: Any) -> PreparedCondition:
    """Run OpenPI's official preprocessing and cache the VLM prefix."""

    images, image_masks, language_tokens, language_masks, state = model._preprocess_observation(
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_attention_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks
    )
    attention_2d = model._prepare_attention_masks_4d(
        _make_attention_masks(prefix_pad_masks, prefix_attention_masks)
    )
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=attention_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return PreparedCondition(state=state, prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values)


@torch.no_grad()
def sample_actions(
    model: Any,
    noise: Tensor,
    condition_for_step: Callable[[int], PreparedCondition],
    *,
    num_steps: int = 10,
    start_step: int = 0,
    stop_step: int | None = None,
    initial_state: Tensor | None = None,
    tracer: ResidualTracer | None = None,
    state_intervention: Callable[[int, Tensor], Tensor] | None = None,
    velocity_intervention: Callable[[int, Tensor], Tensor] | None = None,
) -> tuple[Tensor, SamplingTrace]:
    """Integrate the official OpenPI velocity field with an explicit condition schedule.

    With a constant condition and no tracer this is algebraically identical to
    `PI0Pytorch.sample_actions`, but exposes flow boundaries for recording and
    conditioning-switch interventions.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    stop_step = num_steps if stop_step is None else stop_step
    if not 0 <= start_step <= stop_step <= num_steps:
        raise ValueError("sampling steps must satisfy 0 <= start_step <= stop_step <= num_steps")
    if start_step > 0 and initial_state is None:
        raise ValueError("resumed sampling requires an explicit initial_state")
    expected_tail = (model.config.action_horizon, model.config.action_dim)
    if noise.ndim != 3 or tuple(noise.shape[1:]) != expected_tail:
        raise ValueError(f"noise must have shape [batch, {expected_tail[0]}, {expected_tail[1]}]")

    x_t = noise.clone() if initial_state is None else initial_state.clone()
    _validate_intervention_shape("initial state", x_t, noise)
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
    time = torch.tensor(1.0 - start_step / num_steps, dtype=torch.float32, device=noise.device)
    trace = SamplingTrace()

    for step in range(start_step, stop_step):
        expanded_time = time.expand(noise.shape[0])
        condition = condition_for_step(step)
        if state_intervention is not None:
            x_t = state_intervention(step, x_t)
            _validate_intervention_shape("state", x_t, noise)

        trace.times.append(float(time.item()))
        trace.x_t.append(x_t.detach().clone())
        if tracer is None:
            v_t = model.denoise_step(
                condition.state,
                condition.prefix_pad_masks,
                condition.past_key_values,
                x_t,
                expanded_time,
            )
        else:
            with tracer.at_step(step):
                v_t = model.denoise_step(
                    condition.state,
                    condition.prefix_pad_masks,
                    condition.past_key_values,
                    x_t,
                    expanded_time,
                )

        if velocity_intervention is not None:
            v_t = velocity_intervention(step, v_t)
            _validate_intervention_shape("velocity", v_t, noise)
        trace.v_t.append(v_t.detach().clone())
        clean_time = expanded_time[:, None, None]
        trace.clean_action_estimates.append((x_t - clean_time * v_t).detach().clone())
        x_t = x_t + dt * v_t
        time += dt

    return x_t, trace


def _make_attention_masks(pad_masks: Tensor, attention_masks: Tensor) -> Tensor:
    """Equivalent to OpenPI's public `make_att_2d_masks` helper."""

    cumulative = torch.cumsum(attention_masks, dim=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = pad_masks[:, None, :] * pad_masks[:, :, None]
    return causal & valid


def _validate_intervention_shape(name: str, value: Tensor, reference: Tensor) -> None:
    if not isinstance(value, Tensor) or value.shape != reference.shape:
        shape = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
        raise ValueError(f"{name} intervention must return shape {tuple(reference.shape)}, got {shape}")
