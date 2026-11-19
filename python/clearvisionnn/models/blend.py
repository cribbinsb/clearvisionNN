"""The recursive blend model (DESIGN.md section 8.1): the main temporal candidate."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.contract import SIDE_CHANNELS, SR_SCALE
from clearvisionnn.models.conditioning import side_planes
from clearvisionnn.models.layouts import LAYOUTS
from clearvisionnn.models.per_frame import YUV_CHANNELS, reconstructed_channels
from clearvisionnn.models.resample import block_field_to_flow, upscale, warp
from clearvisionnn.models.sequence import FrameInput, History, SequenceModel
from clearvisionnn.models.sizes import get_size

BLEND_CHANNELS = 1  # the per-pixel blend weight
VALID_CHANNELS = 1  # tells the network whether history is usable


class BlendSR(SequenceModel):
    """Output = weight x aligned previous output + (1 - weight) x new estimate.

    The previous output and a small hidden state are warped onto the current
    frame outside the network; the network sees them with the current frame
    and the side planes, and predicts the detail of a new estimate, the blend
    weight and the next hidden state. The blend is convex: each output lies
    between the aligned history and the new estimate. That bounds one step,
    not the recursion, because the estimate itself depends on earlier output;
    long-run stability is therefore something to measure (the flicker suite's
    long static and noise-only clips), not something this form guarantees.
    After a reset the weight is forced to zero.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._planes = reconstructed_channels(config.planes)
        self._align, self._side = config.align, config.side_planes
        self._state = config.state_channels
        history_channels = self._planes * SR_SCALE**2 + self._state + VALID_CHANNELS
        self.core = LAYOUTS.get(config.layout)(
            YUV_CHANNELS + SIDE_CHANNELS + history_channels,
            self._planes + BLEND_CHANNELS,
            self._state,
            get_size(config.size),
        )

    def step(self, inputs: FrameInput, history: History | None) -> tuple[torch.Tensor, History]:
        """Upscale one frame using the history of earlier ones."""
        frame = inputs.frame
        batch, _, height, width = frame.shape
        base = upscale(frame)
        previous, hidden, valid = self._aligned_history(inputs, history, base)
        features = torch.cat(
            [
                frame,
                side_planes(inputs.side, height, width, self._side),
                F.pixel_unshuffle(previous, SR_SCALE),
                hidden,
                valid.expand(batch, VALID_CHANNELS, height, width),
            ],
            dim=1,
        )
        fine, coarse = self.core(features)
        estimate = base[:, : self._planes] + fine[:, : self._planes]
        weight = torch.sigmoid(fine[:, self._planes :]) * valid
        blended = weight * previous + (1.0 - weight) * estimate
        output = torch.cat([blended, base[:, self._planes :]], dim=1)
        return output, History(output, torch.tanh(coarse))

    def _aligned_history(
        self, inputs: FrameInput, history: History | None, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The previous reconstructed planes and hidden state on the current frame's grid.

        Where there is no usable history, the conventional upscale and a zero
        state stand in, and `valid` is 0.
        """
        batch, _, height, width = inputs.frame.shape
        current = base[:, : self._planes]
        if history is None:
            zeros = base.new_zeros(batch, self._state, height, width)
            return current, zeros, base.new_zeros(batch, 1, 1, 1)
        previous, hidden = history.output[:, : self._planes], history.hidden
        if self._align:
            flow = block_field_to_flow(inputs.motion, height, width)
            fine_flow = SR_SCALE * F.interpolate(
                flow, scale_factor=SR_SCALE, mode="bilinear", align_corners=False
            )
            previous, hidden = warp(previous, fine_flow), warp(hidden, flow)
        valid = (~inputs.reset).to(base.dtype).view(batch, 1, 1, 1)
        return valid * previous + (1 - valid) * current, valid * hidden, valid
