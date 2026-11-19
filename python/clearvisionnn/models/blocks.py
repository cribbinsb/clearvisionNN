"""The building block: a 3x3 convolution that trains as several branches and exports as one."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from torch import nn

KERNEL = 3
# Initial scale of the conv branches beside an identity branch. At full scale
# each such layer would double the signal's variance and a deep stack would
# overflow; scaled down, the stack starts close to the identity.
RESIDUAL_INIT_SCALE = 0.1


class RepConv(nn.Module):
    """3x3 conv + 1x1 conv (+ identity when shapes allow), summed.

    The extra branches ease optimisation of a plain network. `fused` folds
    them into a single 3x3 convolution with identical output, which is what is
    exported: the deployed network is plain convs and ReLUs.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, KERNEL, stride, KERNEL // 2)
        self.point = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.identity = in_channels == out_channels and stride == 1
        self._stride = stride
        if self.identity:
            with torch.no_grad():
                self.conv.weight.mul_(RESIDUAL_INIT_SCALE)
                self.point.weight.mul_(RESIDUAL_INIT_SCALE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of the branches."""
        out = self.conv(x) + self.point(x)
        return out + x if self.identity else out

    def fused(self) -> nn.Conv2d:
        """One 3x3 convolution equal to this module."""
        conv = nn.Conv2d(
            self.conv.in_channels, self.conv.out_channels, KERNEL, self._stride, KERNEL // 2
        )
        pad = KERNEL // 2
        weight = self.conv.weight.detach() + F.pad(self.point.weight.detach(), [pad] * 4)
        if self.identity:
            for channel in range(weight.shape[0]):
                weight[channel, channel, pad, pad] += 1.0
        conv.weight.data.copy_(weight)
        conv.bias.data.copy_(self.conv.bias.detach())  # type: ignore[union-attr]
        return conv


def conv_relu(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """A re-parameterisable conv followed by ReLU."""
    return nn.Sequential(RepConv(in_channels, out_channels, stride), nn.ReLU(inplace=True))


def stack(channels: int, blocks: int) -> nn.Sequential:
    """`blocks` conv+ReLU layers at constant width."""
    return nn.Sequential(*(conv_relu(channels, channels) for _ in range(blocks)))


def zero_init(conv: RepConv) -> RepConv:
    """`conv` with all weights zeroed, so the network starts as its fixed residual path."""
    for parameter in conv.parameters():
        nn.init.zeros_(parameter)
    return conv


def fuse(module: nn.Module) -> nn.Module:
    """`module` with every `RepConv` in it replaced by its fused convolution, in place."""
    for name, child in module.named_children():
        if isinstance(child, RepConv):
            setattr(module, name, child.fused())
        else:
            fuse(child)
    return module
