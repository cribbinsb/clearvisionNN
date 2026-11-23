"""The network and its recurrent pass over a sequence.

A flat stack of 3x3 convolutions at input resolution predicts, for every
frame, the detail to add to the conventional upscale and a state to carry to
the next frame. Besides the frame's luma it is given its own previous output
and state, both moved along the pre-process's motion onto the current frame,
and a plane saying where that motion can be trusted: 22 input planes with 16
state channels (docs/INTEGER.md, section 1, lists them).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from torch import nn

from clearvisionnn.contract import BLOCK_SIZE, SR_SCALE
from clearvisionnn.data.batch import Degraded, SequenceBatch
from clearvisionnn.models.blocks import RepConv, conv_relu, stack, zero_init
from clearvisionnn.models.resample import block_field_to_flow, upscale
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
        # Luma, confidence, the state, and the previous output folded to input resolution.
        inputs = 1 + 1 + options.state_channels + DETAIL_CHANNELS
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
        # A stream starts at the first frame, and wherever every sequence of the
        # batch cuts (for a single stream, every flagged cut). Asked of the GPU
        # once: asking frame by frame makes it wait every frame.
        starts = view.reset.all(dim=0).tolist()
        starts[0] = True
        # Everything that does not depend on the network's own output is done
        # for all frames in one call each: a step's time is mostly the number
        # of small GPU calls, whatever the network's size.
        low = block_field_to_flow(view.motion.flatten(0, 1), height, width)
        high = SR_SCALE * F.interpolate(
            low, scale_factor=SR_SCALE, mode="bilinear", align_corners=False
        )
        from_low, from_high = _sampling_grid(low), _sampling_grid(high)
        base = upscale(view.frames.flatten(0, 1))[:, :1].unflatten(0, (batch, frames))
        earlier = torch.cat([luma[:, :1], luma[:, :-1]], 1).flatten(0, 1)
        carried = _sample(earlier, from_low).unflatten(0, (batch, frames))
        # A frame that starts a stream is its own prediction: the motion is trusted there.
        starting = torch.tensor(starts, device=luma.device)[None, :, None, None, None]
        before = torch.where(starting, luma, carried).flatten(0, 1)
        confidence = _confidence(luma.flatten(0, 1), before).unflatten(0, (batch, frames))
        from_low = from_low.unflatten(0, (batch, frames))
        from_high = from_high.unflatten(0, (batch, frames))
        outputs: list[torch.Tensor] = []
        state = previous = luma  # replaced on the first frame, which always starts a stream
        for t in range(frames):
            if starts[t]:
                state = luma.new_zeros(batch, options.state_channels, height, width)
                previous = base[:, t]
            else:
                state = _sample(state, from_low[:, t])
                previous = _sample(previous, from_high[:, t])
            planes = [luma[:, t], confidence[:, t], state, F.pixel_unshuffle(previous, SR_SCALE)]
            out = trunk(torch.cat(planes, 1))
            previous = base[:, t] + F.pixel_shuffle(out[:, :DETAIL_CHANNELS], SR_SCALE)
            state = torch.tanh(out[:, DETAIL_CHANNELS:])
            if options.round_state:
                # Rounded in the forward pass only: the gradient passes straight through.
                rounded = torch.round(state * STATE_LEVELS) / STATE_LEVELS
                state = state + (rounded - state).detach()
            outputs.append(previous)
        return torch.stack(outputs, 1)


def _sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    """Where `resample.warp` takes each sample from for `flow`, in the form `grid_sample` reads.

    Made once per sequence, so that warping a frame's history is one call.
    """
    height, width = flow.shape[-2:]
    ys = torch.arange(height, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(width, device=flow.device, dtype=flow.dtype)
    grid_x = (xs[None, None, :] + flow[:, 0] + 0.5) / width * 2 - 1
    grid_y = (ys[None, :, None] + flow[:, 1] + 0.5) / height * 2 - 1
    return torch.stack([grid_x, grid_y], dim=-1)


def _sample(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """`image` brought along the motion `grid` was made from (as `resample.warp` does it)."""
    return F.grid_sample(
        image, grid.to(image.dtype), mode="bilinear", padding_mode="border", align_corners=False
    )


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
