"""Convert a trained model to 8-bit arithmetic by quantisation-aware training.

    python -m recipe.quantise recipe/out/models/base-m-sharp.pt

The network is run through a simulation of the integer runtime (docs/INTEGER.md):
8-bit input planes, 8-bit weights per output channel, 8-bit activations after
every ReLU with a learnt top, and the state rounded to 8 bits. Weights and
tops are then trained briefly so the simulated output follows the float
model's. The result, `<checkpoint name>-int8.pt`, holds float weights that
survive the rounding and the learnt tops; `recipe.export` writes the integers.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - the conventional alias
from torch import nn

from clearvisionnn.data.batch import SequenceBatch
from clearvisionnn.models.blocks import RepConv
from recipe.clips import held_out_set
from recipe.data import Pools
from recipe.losses import TrainingLoss
from recipe.metrics import WARM_UP_FRAMES, psnr
from recipe.model import STATE_LEVELS, Model
from recipe.options import BASE_BATCH, Checkpoint, load_checkpoint, save_checkpoint
from recipe.sources import OUT, Source, add_source_arguments, sources_from

# Largest unsigned 8-bit activation and largest magnitude of a signed 8-bit weight.
ACTIVATION_LEVELS = 255
WEIGHT_LEVELS = 127
# An activation's top starts at this quantile of what the float model
# produces: clipping the rare largest values buys finer steps for the rest.
TOP_QUANTILE = 0.9999
# `torch.quantile` takes a limited number of values; every so-manyth is plenty.
QUANTILE_STRIDE = 97
# Floor of a channel's weight step, so an all-zero channel does not divide by zero.
MIN_WEIGHT_STEP = 1e-8
# Input planes ahead of the state: the frame's luma and the confidence plane.
PLANES_BEFORE_STATE = 2

STEPS = 1500
RATE = 1e-4
# Heavier than in float training, to hold down the flicker that rounding adds.
STEADINESS = 0.4
# Weight on following the float model's frame-to-frame change, beside following its output.
CHANGE_WEIGHT = 4.0
MAX_GRADIENT_NORM = 1.0
# Held-out set whose clips set the activations' starting tops and report progress.
CALIBRATION = Source("sanity", "ncodec")


def round_through(x: torch.Tensor) -> torch.Tensor:
    """`x` rounded, with the gradient of `x` itself (the straight-through estimator)."""
    return x + (torch.round(x) - x).detach()


class Activation(nn.Module):
    """An unsigned 8-bit activation with a learnt top; observes its input until switched on."""

    def __init__(self) -> None:
        super().__init__()
        self.top = nn.Parameter(torch.tensor(1.0))
        self.quantising = False
        self.seen: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` on the 8-bit grid, or `x` itself while observing."""
        if not self.quantising:
            sample = x.detach().float().flatten()[::QUANTILE_STRIDE]
            self.seen.append(float(torch.quantile(sample, TOP_QUANTILE)))
            return x
        step = self.top.abs() / ACTIVATION_LEVELS
        return round_through((x / step).clamp(0, ACTIVATION_LEVELS)) * step


def quantised_weight(block: RepConv) -> torch.Tensor:
    """The block's 3x3 and 1x1 branches as one kernel, rounded to signed 8 bits per output channel.

    The identity branch is left out and applied as an explicit add: folded
    into the kernel, its 1.0 would set the step and swamp the small weights.
    """
    weight = block.conv.weight + F.pad(block.point.weight, [1] * 4)
    step = weight.detach().abs().amax((1, 2, 3), keepdim=True).clamp_min(MIN_WEIGHT_STEP)
    step = step / WEIGHT_LEVELS
    return round_through(weight / step) * step


def _convs(layers: nn.Module) -> list[RepConv]:
    """The convolution blocks inside `layers`, in order."""
    return [module for module in layers.modules() if isinstance(module, RepConv)]


class IntegerTrunk(nn.Module):
    """The model's trunk in simulated integer arithmetic, or in float while observing ranges."""

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self._stem = _convs(model.stem)[0]
        self._body = _convs(model.body)
        # One after the stem, one after each block, one on the sum the head reads.
        self.activations = [Activation() for _ in range(len(self._body) + 2)]
        # Registered as well, so moving the trunk to a device moves their tops.
        self._registered = nn.ModuleList(self.activations)
        self.quantising = False

    def calibrate(self) -> None:
        """Set each top from the ranges observed so far, and switch to integer arithmetic."""
        with torch.no_grad():
            for activation in self.activations:
                activation.top.fill_(sum(activation.seen) / len(activation.seen))
                activation.quantising = True
        self.quantising = True

    def tops(self) -> list[float]:
        """The activations' tops, in network order."""
        return [float(activation.top.detach()) for activation in self.activations]

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        """Detail and next state from one frame's input planes."""
        # The simulated rounding needs more mantissa than reduced precision has.
        with torch.autocast(planes.device.type, enabled=False):
            return self._trunk(planes.float())

    def _trunk(self, planes: torch.Tensor) -> torch.Tensor:
        model = self.model
        if self.quantising:
            # Every input plane is 8-bit: pictures and flags unsigned, the state signed.
            grid = torch.full((planes.shape[1],), float(ACTIVATION_LEVELS), device=planes.device)
            state = slice(PLANES_BEFORE_STATE, PLANES_BEFORE_STATE + model.options.state_channels)
            grid[state] = STATE_LEVELS
            grid = grid[None, :, None, None]
            planes = round_through(planes.clamp(-1, 1) * grid) / grid
        features = self.activations[0](F.relu(self._conv(self._stem, planes)))
        deep = features
        for block, activation in zip(self._body, self.activations[1:-1], strict=True):
            deep = activation(F.relu(self._conv(block, deep)))
        return self._conv(model.head, self.activations[-1](features + deep))

    def _conv(self, block: RepConv, x: torch.Tensor) -> torch.Tensor:
        if not self.quantising:
            return block(x)
        out = F.conv2d(x, quantised_weight(block), block.conv.bias, padding=1)
        return out + x if block.identity else out


@torch.no_grad()
def held_out_psnr(model: Model, trunk: IntegerTrunk, batches: list[SequenceBatch]) -> float:
    """Mean PSNR in dB of the model run through `trunk`, in evaluation mode."""
    device = next(model.parameters()).device
    model.eval()
    scores = []
    for batch in batches:
        batch = batch.to(device)  # noqa: PLW2901 - the same batch, moved
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = model(batch.view, trunk).float().clamp(0, 1)
        scores.append(psnr(output[:, WARM_UP_FRAMES:], batch.target[:, WARM_UP_FRAMES:, :1]))
    model.train()
    return sum(scores) / len(scores)


def teacher_loss(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Distance from the float model's output, and from its frame-to-frame change."""
    change = (output[:, 1:] - output[:, :-1]).float() - (reference[:, 1:] - reference[:, :-1])
    return (output.float() - reference).abs().mean() + CHANGE_WEIGHT * change.abs().mean()


def quantise(
    start: Checkpoint,
    batches: Pools,
    calibration: list[SequenceBatch],
    steps: int,
    device: torch.device,
) -> Checkpoint:
    """Train `start` under the integer simulation for `steps` steps."""
    # The simulation runs in full precision, which costs several times the
    # memory of ordinary training: the batch goes back to the base run's.
    options = replace(start.options, steadiness=STEADINESS, batch=BASE_BATCH, round_state=True)
    model, teacher = Model(options).to(device), Model(options).to(device).eval()
    model.load_state_dict(start.weights)
    teacher.load_state_dict(start.weights)
    trunk = IntegerTrunk(model).to(device)
    # This float pass is also the one in which the activations observe their ranges.
    print(f"float: {held_out_psnr(model, trunk, calibration):.3f} dB", flush=True)
    trunk.calibrate()
    print(f"8-bit, calibrated only: {held_out_psnr(model, trunk, calibration):.3f} dB", flush=True)

    parameters = [*model.parameters(), *(a.top for a in trunk.activations)]
    optimiser = torch.optim.AdamW(parameters, lr=RATE, weight_decay=0)
    loss_of = TrainingLoss(options, device)
    step = 0
    while step < steps:
        for batch in batches:
            output = model(batch.view, trunk)
            with torch.no_grad():
                reference = teacher(batch.view).float()
            loss = loss_of(output, batch) + teacher_loss(output, reference)
            # The rate decays to zero: at a constant one the last weights sit where noise put them.
            for group in optimiser.param_groups:
                group["lr"] = RATE * 0.5 * (1 + math.cos(math.pi * step / steps))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, MAX_GRADIENT_NORM)
            optimiser.step()
            step += 1
            if step >= steps:
                break
    print(f"8-bit, trained: {held_out_psnr(model, trunk, calibration):.3f} dB", flush=True)
    return Checkpoint(options, model.state_dict(), start.step + steps, trunk.tops())


def main() -> None:
    """Convert one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    add_source_arguments(parser)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--models", type=Path, default=OUT / "models", help="where the result goes")
    parser.add_argument("--pools", type=Path, default=OUT / "pools", help="baked pools")
    parser.add_argument("--held-out", type=Path, default=OUT / "held_out", help="held-out sets")
    parser.add_argument("--workers", type=int, default=4, help="processes preparing held-out clips")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    arguments = parser.parse_args()
    start = load_checkpoint(arguments.checkpoint)
    device: torch.device = arguments.device
    frames = start.options.frames
    batches = Pools(arguments.pools, sources_from(arguments), frames, BASE_BATCH, device)
    calibration = held_out_set(arguments.held_out, CALIBRATION, arguments.workers)
    result = quantise(start, batches, calibration, arguments.steps, device)
    save_checkpoint(arguments.models / f"{arguments.checkpoint.stem}-int8.pt", result)


if __name__ == "__main__":
    main()
