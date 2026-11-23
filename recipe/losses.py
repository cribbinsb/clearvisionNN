"""The training loss: a pixel term, steadiness along motion, perceptual and texture terms."""

from __future__ import annotations

import torch

from clearvisionnn.data.batch import Degraded, SequenceBatch
from clearvisionnn.losses.perceptual import LumaLpips
from clearvisionnn.models.resample import warp
from recipe.model import motion_flows
from recipe.options import Options
from recipe.texture import texture_term

# Keeps the square root's gradient finite at zero error.
CHARBONNIER_EPSILON = 1e-6
# The original counts as steady along the motion where it changes by less
# than this (of 255): above its own noise, below any visible change.
STEADY_THRESHOLD = 3 / 255
# Frames of a step the perceptual term sees, drawn at random: the feature
# network is far heavier than the one being trained.
PERCEPTUAL_FRAMES = 16


def charbonnier(difference: torch.Tensor) -> torch.Tensor:
    """A smooth absolute value: like L1 away from zero, differentiable at it."""
    return torch.sqrt(difference**2 + CHARBONNIER_EPSILON)


def steadiness(output: torch.Tensor, view: Degraded, target: torch.Tensor) -> torch.Tensor:
    """Change of the output along the motion, where the original itself is steady along it.

    Elsewhere (occlusions, poor vectors, real change) nothing is asked: no
    output could be steady there.
    """
    total = output.new_zeros((), dtype=torch.float32)
    frames = output.shape[1]
    for t in range(1, frames):
        flow = motion_flows(view, t)[1]
        original = (warp(target[:, t - 1, :1], flow) - target[:, t, :1]).abs()
        steady = (original < STEADY_THRESHOLD).float()
        change = charbonnier(warp(output[:, t - 1].float(), flow) - output[:, t].float())
        total = total + (change * steady).mean()
    return total / (frames - 1)


class TrainingLoss:
    """The weighted sum the options ask for, on one step's output."""

    def __init__(self, options: Options, device: torch.device) -> None:
        self._options = options
        self._perceptual = LumaLpips("vgg").to(device) if options.perceptual else None

    def __call__(self, output: torch.Tensor, batch: SequenceBatch) -> torch.Tensor:
        """The loss of (batch, frames, 1, H, W) `output` against the batch's target luma."""
        options = self._options
        target = batch.target[:, :, :1]
        loss = charbonnier(output.float() - target).mean()
        if options.steadiness:
            loss = loss + options.steadiness * steadiness(output, batch.view, batch.target)
        if self._perceptual is not None:
            count = output.shape[0] * output.shape[1]
            pick = torch.randperm(count, device=output.device)[:PERCEPTUAL_FRAMES]
            distance = self._perceptual(
                output.float().flatten(0, 1)[pick], target.flatten(0, 1)[pick]
            )
            loss = loss + options.perceptual * distance
        if options.texture:
            # The first value of a frame's coding record says whether it met a codec at all.
            coded = batch.view.coding[:, :, 0] > 0
            loss = loss + options.texture * texture_term(output, target, coded)
        return loss
