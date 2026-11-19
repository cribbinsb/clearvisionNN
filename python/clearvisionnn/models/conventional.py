"""The conventional upscaler as a model: the zero point every score is read against."""

from __future__ import annotations

import torch

from clearvisionnn.models.resample import upscale
from clearvisionnn.models.sequence import FrameInput, History, SequenceModel


class ConventionalSR(SequenceModel):
    """Bicubic 2x: what every network's detail is added to, with no detail added."""

    def step(self, inputs: FrameInput, history: History | None) -> tuple[torch.Tensor, History]:
        """Upscale one frame conventionally."""
        output = upscale(inputs.frame)
        batch, _, height, width = inputs.frame.shape
        return output, History(output, output.new_zeros(batch, 0, height, width))
