"""Post-layer residual capture and interchange interventions.

This module hooks the public OpenPI PyTorch action expert. It intentionally does
not modify upstream model code. A trace contains action-token residuals only;
state or other suffix tokens are excluded using the configured action horizon.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

Site = tuple[int, int]


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    """One post-layer interchange intervention.

    `positions=None` patches every action-token position. Otherwise positions
    are zero-based within the action chunk, not within the full suffix.
    """

    step: int
    layer: int
    positions: tuple[int, ...] | None = None


@dataclasses.dataclass
class ResidualTrace:
    """Captured clean post-layer action-token residuals."""

    values: dict[Site, Tensor] = dataclasses.field(default_factory=dict)

    def cpu(self) -> ResidualTrace:
        return ResidualTrace({site: value.detach().cpu() for site, value in self.values.items()})


class ResidualTracer:
    """Capture or patch post-layer residuals at explicitly marked flow steps."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        action_horizon: int,
        patch: PatchSpec | None = None,
        donor: Mapping[Site, Tensor] | None = None,
        capture: bool = True,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if patch is not None and donor is None:
            raise ValueError("a donor trace is required when patching")
        if patch is not None and not 0 <= patch.layer < len(layers):
            raise ValueError(f"patch layer {patch.layer} is outside [0, {len(layers)})")

        self.action_horizon = action_horizon
        self.patch = patch
        self.donor = donor
        self.capture = capture
        self.trace = ResidualTrace()
        self._active_step: int | None = None
        self._handles = [layer.register_forward_hook(self._make_hook(index)) for index, layer in enumerate(layers)]

    @contextmanager
    def at_step(self, step: int):
        """Associate all action-expert layer calls in the context with one flow step."""

        if self._active_step is not None:
            raise RuntimeError("flow-step trace contexts cannot be nested")
        self._active_step = step
        try:
            yield
        finally:
            self._active_step = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> ResidualTracer:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _make_hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any):
            if self._active_step is None:
                return None

            hidden, rebuild = _split_output(output)
            if hidden.ndim != 3 or hidden.shape[1] < self.action_horizon:
                raise ValueError(
                    f"expected [batch, suffix, width] with suffix >= {self.action_horizon}, got {tuple(hidden.shape)}"
                )

            site = (self._active_step, layer)
            clean_actions = hidden[:, -self.action_horizon :, :]
            if self.capture:
                self.trace.values[site] = clean_actions.detach().clone()

            if self.patch is None or site != (self.patch.step, self.patch.layer):
                return None

            assert self.donor is not None
            if site not in self.donor:
                raise KeyError(f"donor trace does not contain site {site}")
            donor_actions = self.donor[site].to(device=hidden.device, dtype=hidden.dtype)
            if donor_actions.shape != clean_actions.shape:
                raise ValueError(
                    f"donor shape {tuple(donor_actions.shape)} does not match target {tuple(clean_actions.shape)}"
                )

            patched = hidden.clone()
            target = patched[:, -self.action_horizon :, :]
            if self.patch.positions is None:
                target.copy_(donor_actions)
            else:
                positions = torch.as_tensor(self.patch.positions, device=hidden.device, dtype=torch.long)
                if positions.numel() == 0:
                    raise ValueError("patch positions cannot be empty")
                if torch.any(positions < 0) or torch.any(positions >= self.action_horizon):
                    raise IndexError(f"action-token positions must be in [0, {self.action_horizon})")
                target.index_copy_(1, positions, donor_actions.index_select(1, positions))
            return rebuild(patched)

        return hook


def _split_output(output: Any):
    """Return the hidden tensor and a function that preserves output structure."""

    if isinstance(output, Tensor):
        return output, lambda hidden: hidden
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0], lambda hidden: (hidden, *output[1:])
    if isinstance(output, list) and output and isinstance(output[0], Tensor):
        return output[0], lambda hidden: [hidden, *output[1:]]
    raise TypeError(f"unsupported transformer-layer output type: {type(output)!r}")
