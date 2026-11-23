"""Running a model over a batch of sequences, frame by frame."""

from __future__ import annotations

import torch

from clearvisionnn.data.batch import Degraded
from clearvisionnn.models.sequence import FrameInput, History, SequenceModel


def roll(model: SequenceModel, view: Degraded) -> torch.Tensor:
    """The model's output for every frame of `view`: (batch, frames, 3, 2h, 2w).

    History is carried from frame to frame with gradients, so training sees
    the consequences of each output on the ones that follow.
    """
    history: History | None = None
    outputs = []
    for t in range(view.frames.shape[1]):
        inputs = FrameInput(view.frames[:, t], view.motion[:, t], view.side[:, t], view.reset[:, t])
        output, history = model.step(inputs, history)
        outputs.append(output)
    return torch.stack(outputs, dim=1)
