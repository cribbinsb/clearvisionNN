"""The luma-only models: a lean input, optionally with a recurrent state.

The network sees luma alone, without chroma or side-plane channels; chroma is
upscaled conventionally. The recurrent design is the temporal candidate that
gained on noisy camera input: a hidden state and the previous output, both
brought onto the current frame, are inputs like any other.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.config.model import ModelConfig, Temporal
from clearvisionnn.contract import CODING_CHANNELS, SR_SCALE
from clearvisionnn.models.confidence import confidence
from clearvisionnn.models.layouts import LAYOUTS, Layout
from clearvisionnn.models.resample import block_field_to_flow, upscale, warp
from clearvisionnn.models.sequence import FrameInput, History, SequenceModel
from clearvisionnn.models.sizes import get_size
from clearvisionnn.models.state import BoundedState

LUMA_CHANNELS = 1


def input_channels(config: ModelConfig) -> int:
    """Channels of the network input, in the order `LumaSR.step` assembles them."""
    recurrent = config.temporal is Temporal.RECURRENT
    history = config.state_channels + LUMA_CHANNELS * SR_SCALE**2 if recurrent else 0
    coding = CODING_CHANNELS if config.coding_planes else 0
    return LUMA_CHANNELS + int(config.confidence_plane) + history + coding


def state_input_channels(config: ModelConfig) -> range:
    """The input channels holding the recurrent state: the only ones that go negative."""
    first = LUMA_CHANNELS + int(config.confidence_plane)
    recurrent = config.temporal is Temporal.RECURRENT
    return range(first, first + (config.state_channels if recurrent else 0))


class LumaSR(SequenceModel):
    """Detail added to the conventional upscale of luma, from a lean input.

    Input channels, in order: luma; the confidence plane (if configured); the
    hidden state and the previous output folded to input resolution
    (recurrent design); the coding planes (if configured). After a reset the
    conventional upscale and a zero state stand in for history, and confidence
    is 1: there is no prediction to distrust.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._recurrent = config.temporal is Temporal.RECURRENT
        self._align = config.align
        self._coding, self._confidence = config.coding_planes, config.confidence_plane
        self._state = config.state_channels if self._recurrent else 0
        shape = (input_channels(config), LUMA_CHANNELS, self._state, get_size(config.size))
        self.core: Layout = (
            BoundedState(config.layout, *shape)
            if self._recurrent
            else LAYOUTS.get(config.layout)(*shape)
        )

    def step(self, inputs: FrameInput, history: History | None) -> tuple[torch.Tensor, History]:
        """Upscale one frame, using the history of earlier ones if recurrent."""
        frame = inputs.frame
        batch, _, height, width = frame.shape
        luma, base = frame[:, :LUMA_CHANNELS], upscale(frame)
        parts = [luma]
        if self._recurrent:
            previous, hidden, carried = self._aligned_history(inputs, history, base, luma)
            if self._confidence:
                parts.append(confidence(luma, carried))
            parts += [hidden, F.pixel_unshuffle(previous, SR_SCALE)]
        if self._coding:
            planes = inputs.coding.to(luma.dtype)[:, :, None, None]
            parts.append(planes.expand(batch, CODING_CHANNELS, height, width))
        fine, coarse = self.core(torch.cat(parts, dim=1))
        output = torch.cat([base[:, :LUMA_CHANNELS] + fine, base[:, LUMA_CHANNELS:]], dim=1)
        return output, History(output, coarse, luma)

    def _aligned_history(
        self, inputs: FrameInput, history: History | None, base: torch.Tensor, luma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The previous output luma, hidden state and input luma on the current frame's grid."""
        batch, _, height, width = luma.shape
        current = base[:, :LUMA_CHANNELS]
        if history is None or history.luma is None:
            return current, luma.new_zeros(batch, self._state, height, width), luma
        previous, hidden, before = history.output[:, :LUMA_CHANNELS], history.hidden, history.luma
        if self._align:
            flow = block_field_to_flow(inputs.motion, height, width)
            fine_flow = SR_SCALE * F.interpolate(
                flow, scale_factor=SR_SCALE, mode="bilinear", align_corners=False
            )
            previous, hidden, before = (
                warp(previous, fine_flow),
                warp(hidden, flow),
                warp(before, flow),
            )
        valid = (~inputs.reset).to(luma.dtype).view(batch, 1, 1, 1)
        return (
            valid * previous + (1 - valid) * current,
            valid * hidden,
            valid * before + (1 - valid) * luma,
        )
