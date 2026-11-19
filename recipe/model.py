"""The network and its recurrent pass over a sequence.

A flat stack of 3x3 convolutions at input resolution predicts, for every
frame, the detail to add to the conventional upscale and a state to carry to
the next frame. Besides the frame's luma it is given its own previous output
and state, both moved along the pre-process's motion onto the current frame,
a plane saying where that motion can be trusted, and the coding information.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from torch import nn

from clearvisionnn.contract import BLOCK_SIZE, CODING_CHANNELS, SR_SCALE
from clearvisionnn.data.batch import Degraded, SequenceBatch
from clearvisionnn.models.blocks import RepConv, conv_relu, stack, zero_init
from clearvisionnn.models.resample import block_field_to_flow, upscale, warp
from recipe.options import Options

# One output sample per position of the 2x2 cell each input sample becomes.
DETAIL_CHANNELS = SR_SCALE * SR_SCALE
# Mean absolute error (0-255) of the motion-compensated previous frame over a
# block: at or below the first the motion is fully trusted, at or above the
# second not at all. The lower bound sits above sensor and coding noise.
TRUSTED_ERROR, UNTRUSTED_ERROR = 3.0, 10.0
MAX_SAMPLE = 255
# Largest magnitude of the signed 8-bit state.
STATE_LEVELS = 127

Trunk = Callable[[torch.Tensor], torch.Tensor]


def motion_flows(view: Degraded, frame: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The motion onto `frame` from the one before, per sample at input and at output resolution."""
    height, width = view.frames.shape[-2:]
    low = block_field_to_flow(view.motion[:, frame], height, width)
    high = F.interpolate(low, scale_factor=SR_SCALE, mode="bilinear", align_corners=False)
    return low, SR_SCALE * high


class Model(nn.Module):
    """The recurrent luma network: `channels` wide, `blocks` deep, at input resolution."""

    def __init__(self, options: Options) -> None:
        super().__init__()
        self.options = options
        inputs = 1 + 1 + options.state_channels + DETAIL_CHANNELS + CODING_CHANNELS
        self.stem = conv_relu(inputs, options.channels)
        self.body = stack(options.channels, options.blocks)
        # Zeroed, so training starts from the conventional upscale and an empty state.
        self.head = zero_init(RepConv(options.channels, DETAIL_CHANNELS + options.state_channels))

    def trunk(self, planes: torch.Tensor) -> torch.Tensor:
        """Detail and next state from one frame's input planes."""
        features = self.stem(planes)
        return self.head(features + self.body(features))

    def forward(self, view: Degraded, trunk: Trunk | None = None) -> torch.Tensor:
        """The upscaled luma of every frame, (batch, frames, 1, 2h, 2w).

        `trunk` stands in for the network's own arithmetic (the 8-bit
        simulation does this); everything around it is unchanged.
        """
        trunk = trunk or self.trunk
        options = self.options
        luma = view.frames[:, :, :1]
        batch, frames, _, height, width = luma.shape
        keep: torch.Tensor | float = 1.0
        if self.training:
            keep = (torch.rand(batch, 1, device=luma.device) >= options.coding_dropout).float()
        outputs: list[torch.Tensor] = []
        state = previous = luma  # replaced on the first frame, which always resets
        # Asked of the GPU once for the whole sequence: asking frame by frame makes it wait every frame.
        cuts = view.reset.all(dim=0).tolist()
        for t in range(frames):
            current = luma[:, t]
            base = upscale(view.frames[:, t])[:, :1]
            # History is dropped only when every sequence of the batch cuts
            # here, which for a single stream is every flagged cut.
            if t == 0 or cuts[t]:
                state = current.new_zeros(batch, options.state_channels, height, width)
                previous, before = base, current
            else:
                low, high = motion_flows(view, t)
                state, previous = warp(state, low), warp(previous, high)
                before = warp(luma[:, t - 1], low)
            coding = (view.coding[:, t] * keep)[:, :, None, None].expand(-1, -1, height, width)
            planes = [
                current,
                _confidence(current, before),
                state,
                F.pixel_unshuffle(previous, SR_SCALE),
                coding.to(current.dtype),
            ]
            out = trunk(torch.cat(planes, 1))
            previous = base + F.pixel_shuffle(out[:, :DETAIL_CHANNELS], SR_SCALE)
            state = torch.tanh(out[:, DETAIL_CHANNELS:])
            if options.round_state:
                # Rounded in the forward pass only: the gradient passes straight through.
                rounded = torch.round(state * STATE_LEVELS) / STATE_LEVELS
                state = state + (rounded - state).detach()
            outputs.append(previous)
        return torch.stack(outputs, 1)


def _confidence(current: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    """How well the motion predicts this frame from the last: 1 where it does, 0 where it fails.

    Judged per analysis block, then interpolated to one value per sample.
    """
    error = F.avg_pool2d((current - before).abs().float(), BLOCK_SIZE, ceil_mode=True)
    error = F.interpolate(error, size=current.shape[-2:], mode="bilinear", align_corners=False)
    low, high = TRUSTED_ERROR / MAX_SAMPLE, UNTRUSTED_ERROR / MAX_SAMPLE
    return (1 - (error - low) / (high - low)).clamp(0, 1).to(current.dtype)


def rollout(model: nn.Module, batch: SequenceBatch) -> torch.Tensor:
    """`model`'s output for `batch` as it is scored: in reduced precision, clamped to [0, 1]."""
    with torch.autocast(batch.target.device.type, dtype=torch.bfloat16):
        return model(batch.view).float().clamp(0, 1)
