"""Perceptual distance: the output should look like the target, not merely average it.

Where the input has lost detail, many sharp pictures fit it, and the output
with the least pixel error is their blurry mean. A distance between the
features of a pretrained image network prefers a sharp, plausible picture.
"""

from __future__ import annotations

import lpips
import torch

from clearvisionnn.losses.base import LOSSES, Rollout

# Limited-range luma levels on the 0-1 scale (16 and 235 of 255).
_BLACK, _WHITE = 16 / 255, 235 / 255
GREY_CHANNELS = 3


class LumaLpips(torch.nn.Module):
    """LPIPS between luma planes shown as grey pictures; lower is closer.

    `net` is the feature network: "vgg" for training (smoother gradients),
    "alex" for scoring, so a run is not scored by the network it was trained
    against.
    """

    def __init__(self, net: str) -> None:
        super().__init__()
        self._network = lpips.LPIPS(net=net, verbose=False).eval().requires_grad_(False)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean distance over (frames, 1, height, width) luma in [0, 1]."""
        return self._network(self._grey(output), self._grey(target)).mean()

    @staticmethod
    def _grey(luma: torch.Tensor) -> torch.Tensor:
        """Full-range grey in [-1, 1], as the feature network expects."""
        full = ((luma - _BLACK) / (_WHITE - _BLACK)).clamp(0.0, 1.0)
        return (full * 2 - 1).expand(-1, GREY_CHANNELS, -1, -1)


@LOSSES.register("lpips")
class LpipsLoss:
    """LPIPS of the luma plane, every frame of the rollout."""

    def __init__(self) -> None:
        self._distance: LumaLpips | None = None

    def __call__(self, rollout: Rollout) -> torch.Tensor:
        """Mean perceptual distance between output and target luma."""
        if self._distance is None:
            self._distance = LumaLpips("vgg").to(rollout.output.device)
        output, target = (x[:, :, :1].flatten(0, 1) for x in (rollout.output, rollout.target))
        return self._distance(output, target)
