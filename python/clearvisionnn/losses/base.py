"""What a loss is given and the registry losses live in."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from clearvisionnn.registry import Registry

# Smoothing of the Charbonnier penalty: errors well below this are penalised
# quadratically, larger ones linearly.
CHARBONNIER_EPSILON = 1e-3


@dataclass(frozen=True)
class Rollout:
    """A model's output over a batch of sequences, with the target.

    Tensors are (batch, frames, channels, height, width) and hold only the
    planes the network reconstructs.
    """

    output: torch.Tensor
    target: torch.Tensor
    # Output for the batch's alternate view, when it has one.
    alternate: torch.Tensor | None = None


Loss = Callable[[Rollout], torch.Tensor]
# A registered loss is a function, or a class made once per run when it holds
# state (a pretrained feature network).
LOSSES: Registry[Loss | type[Loss]] = Registry("loss")


def charbonnier(difference: torch.Tensor) -> torch.Tensor:
    """Mean Charbonnier penalty of `difference`: a smooth, outlier-tolerant L1."""
    return torch.sqrt(difference**2 + CHARBONNIER_EPSILON**2).mean()
