"""Train a model: the base run, or the sharpening fine-tune of a base model.

    python -m recipe.train --size m [--resume]
    python -m recipe.train --sharpen recipe/out/models/base-m.pt

The results are `base-<size>.pt` and `<checkpoint name>-sharp.pt` under `--models`.

The base run trains from scratch on live data at a constant rate, cut when
held-out PSNR stops improving, until three cuts have stopped helping
(`--steps` is then only an upper limit). The fine-tune continues a base model
for a fixed number of steps on the baked pools with a perceptual term beside
the pixel loss, which restores the fine texture a pixel loss averages away.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from clearvisionnn.data.batch import SequenceBatch
from clearvisionnn.models.sizes import SIZES
from recipe.clips import held_out_set
from recipe.data import Pools, augment, live_batches
from recipe.losses import TrainingLoss
from recipe.model import Model
from recipe.options import Checkpoint, Options, load_checkpoint, save_checkpoint
from recipe.schedule import Cosine, Plateau, Schedule
from recipe.sources import OUT, add_source_arguments, sources_from

SEED = 0
# The published sizes, as rungs of the package's size ladder (channels x blocks of its flat layout).
PUBLISHED_SIZES = ("s", "m", "l", "xl")
# Upper limit of the base run; the plateau rule ends it sooner.
BASE_STEPS = 45000
SHARPEN_STEPS = 1500
# The fine-tune moves a converged model a little, at half the base rate.
SHARPEN_RATE = 2e-4
SHARPEN_BATCH = 32
# Weight of the perceptual term beside the pixel loss.
SHARPEN_PERCEPTUAL = 0.05
# Gradients are clipped to this norm, so one outlier batch cannot throw the weights.
MAX_GRADIENT_NORM = 1.0
CHECK_EVERY = 1000


@dataclass
class Run:
    """A model being trained, with what trains it."""

    model: Model
    optimiser: torch.optim.Optimizer
    loss_of: TrainingLoss

    @classmethod
    def start(cls, options: Options, device: torch.device) -> Run:
        """A freshly initialised model of `options` on `device`, ready to train."""
        torch.manual_seed(SEED)
        model = Model(options).to(device).to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        optimiser = torch.optim.AdamW(model.parameters(), lr=options.rate, weight_decay=0)
        return cls(model, optimiser, TrainingLoss(options, device))


def train(
    run: Run,
    batches: Iterable[SequenceBatch],
    schedule: Schedule,
    span: tuple[int, int],
    check_every: int,
) -> None:
    """Run steps `span[0]` to `span[1]`, or until the schedule is done."""
    model, optimiser, loss_of = run.model, run.optimiser, run.loss_of
    step, last = span
    device = next(model.parameters()).device
    losses: list[float] = []
    while step < last and not schedule.done:
        for batch in batches:
            batch = augment(batch.to(device))  # noqa: PLW2901 - the same batch, moved and turned
            for group in optimiser.param_groups:
                group["lr"] = schedule.rate(step)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                output = model(batch.view)
            loss = loss_of(output, batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
            optimiser.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % check_every == 0:
                schedule.check(step, losses)
                losses = []
            if step >= last or schedule.done:
                break


def base(arguments: argparse.Namespace) -> None:
    """Train a base model from scratch on live data, under the plateau rule."""
    sources = sources_from(arguments)
    size = SIZES[arguments.size]
    options = Options(channels=size.flat_channels, blocks=size.flat_blocks)
    run = Run.start(options, arguments.device)
    batches = live_batches(sources, options.frames, options.batch, arguments.workers)
    held_out = {s.name: held_out_set(arguments.held_out, s, arguments.workers) for s in sources}
    name = f"base-{arguments.size}"
    resume_file = arguments.models / f"{name}.resume.pt"
    plateau = Plateau(run.model, run.optimiser, options.rate, held_out, resume_file)
    first = plateau.resume() if arguments.resume else 0
    train(run, batches, plateau, (first, arguments.steps or BASE_STEPS), arguments.check_every)
    if plateau.weights is not None:
        run.model.load_state_dict(plateau.weights)
    result = Checkpoint(options, run.model.state_dict(), plateau.best_step)
    save_checkpoint(arguments.models / f"{name}.pt", result)


def sharpen(arguments: argparse.Namespace) -> None:
    """Fine-tune a base model on the baked pools with the perceptual term added."""
    start = load_checkpoint(arguments.sharpen)
    options = replace(
        start.options, rate=SHARPEN_RATE, batch=SHARPEN_BATCH, perceptual=SHARPEN_PERCEPTUAL
    )
    run = Run.start(options, arguments.device)
    run.model.load_state_dict(start.weights)
    sources = sources_from(arguments)
    batches = Pools(arguments.pools, sources, options.frames, options.batch, arguments.device)
    steps = arguments.steps or SHARPEN_STEPS
    train(run, batches, Cosine(options.rate, steps), (0, steps), arguments.check_every)
    result = Checkpoint(options, run.model.state_dict(), start.step + steps)
    save_checkpoint(arguments.models / f"{arguments.sharpen.stem}-sharp.pt", result)


def main() -> None:
    """Run the base training or the fine-tune the arguments ask for."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--size", choices=PUBLISHED_SIZES, help="train a base model of this size")
    mode.add_argument("--sharpen", type=Path, metavar="CHECKPOINT", help="fine-tune this model")
    add_source_arguments(parser)
    parser.add_argument("--steps", type=int, help="steps to run (base: the upper limit)")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted base run")
    parser.add_argument("--models", type=Path, default=OUT / "models", help="where models go")
    parser.add_argument("--pools", type=Path, default=OUT / "pools", help="baked pools")
    parser.add_argument("--held-out", type=Path, default=OUT / "held_out", help="held-out sets")
    parser.add_argument("--workers", type=int, default=8, help="loader processes in all")
    parser.add_argument("--check-every", type=int, default=CHECK_EVERY, help="steps between checks")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    arguments = parser.parse_args()
    # Input sizes are fixed within a run, so the fastest convolution routines can be picked once.
    torch.backends.cudnn.benchmark = True
    if arguments.sharpen:
        sharpen(arguments)
    else:
        base(arguments)


if __name__ == "__main__":
    main()
