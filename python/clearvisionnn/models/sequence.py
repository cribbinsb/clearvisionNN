"""The interface every model presents to training and evaluation: one frame at a time."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn


class FrameInput(NamedTuple):
    """What the model is given for one frame (batch axis first)."""

    frame: torch.Tensor  # (batch, 3, h, w) YUV in [0, 1]
    motion: torch.Tensor  # (batch, 2, rows, cols): block motion towards the previous frame
    side: torch.Tensor  # (batch, SIDE_CHANNELS, rows, cols)
    reset: torch.Tensor  # (batch,) bool: history is unusable for this frame


class History(NamedTuple):
    """What a model carries from one frame to the next."""

    output: torch.Tensor  # (batch, 3, 2h, 2w): the previous output frame
    hidden: torch.Tensor  # (batch, state_channels, h, w)
    # (batch, 1, h, w): the previous input luma, kept by models that measure
    # how well motion predicts the next frame.
    luma: torch.Tensor | None = None


class SequenceModel(nn.Module, ABC):
    """A causal super-resolution model, stepped frame by frame."""

    @abstractmethod
    def step(self, inputs: FrameInput, history: History | None) -> tuple[torch.Tensor, History]:
        """The 2x output for one frame, (batch, 3, 2h, 2w), and the history for the next.

        `history` is None for the first frame of a sequence.
        """
