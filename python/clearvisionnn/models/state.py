"""A layout whose coarse output is hidden state, bounded inside the network."""

from __future__ import annotations

import torch

from clearvisionnn.models.layouts import LAYOUTS, CoreOutput, Layout
from clearvisionnn.models.sizes import NetSize


class BoundedState(Layout):
    """A registered layout with its coarse output squashed to (-1, 1).

    The squash is part of the exported graph, so the runtime stores the state
    as the network returns it instead of squashing millions of values per
    frame itself.
    """

    def __init__(
        self, name: str, in_channels: int, fine_channels: int, coarse_channels: int, size: NetSize
    ) -> None:
        super().__init__(in_channels, fine_channels, coarse_channels, size)
        self.layout = LAYOUTS.get(name)(in_channels, fine_channels, coarse_channels, size)

    def forward(self, x: torch.Tensor) -> CoreOutput:
        """Input-resolution features to the fine output and the bounded state."""
        fine, coarse = self.layout(x)
        return CoreOutput(fine, torch.tanh(coarse))
