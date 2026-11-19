"""The per-frame model: the baseline, and the bar the temporal designs must beat."""

from __future__ import annotations

import torch

from clearvisionnn.config.model import ModelConfig, Planes
from clearvisionnn.contract import SIDE_CHANNELS
from clearvisionnn.models.conditioning import side_planes
from clearvisionnn.models.layouts import LAYOUTS
from clearvisionnn.models.resample import upscale
from clearvisionnn.models.sequence import FrameInput, History, SequenceModel
from clearvisionnn.models.sizes import get_size

YUV_CHANNELS = 3


def reconstructed_channels(planes: Planes) -> int:
    """How many leading YUV channels the network reconstructs."""
    return 1 if planes is Planes.LUMA else YUV_CHANNELS


class PerFrameSR(SequenceModel):
    """Detail predicted from the current frame alone, added to a conventional upscale.

    The global residual passes low frequencies, brightness and colour through
    untouched. Planes the network does not reconstruct keep the conventional
    upscale.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._planes = reconstructed_channels(config.planes)
        self._side = config.side_planes
        self.core = LAYOUTS.get(config.layout)(
            YUV_CHANNELS + SIDE_CHANNELS, self._planes, 0, get_size(config.size)
        )

    def step(self, inputs: FrameInput, history: History | None) -> tuple[torch.Tensor, History]:
        """Upscale one frame; history is passed through unused."""
        frame = inputs.frame
        height, width = frame.shape[-2:]
        side = side_planes(inputs.side, height, width, self._side)
        detail = self.core(torch.cat([frame, side], dim=1)).fine
        base = upscale(frame)
        output = torch.cat([base[:, : self._planes] + detail, base[:, self._planes :]], dim=1)
        return output, History(output, frame.new_zeros(frame.shape[0], 0, height, width))
