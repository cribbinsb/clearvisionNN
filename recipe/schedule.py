"""Learning-rate schedules: the plateau rule of base training and the cosine of the fine-tune."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from clearvisionnn.data.batch import SequenceBatch
from recipe.metrics import WARM_UP_FRAMES, psnr
from recipe.model import rollout

# Steps over which the rate rises linearly from zero.
WARM_UP_STEPS = 100
# Checks without a gain of MIN_GAIN dB before the rate is cut to CUT of
# itself; training ends when a stall follows the last of CUTS cuts. A gain
# smaller than MIN_GAIN is within the noise of the held-out mean.
PATIENCE, MIN_GAIN, CUT, CUTS = 4, 0.01, 0.3, 3
# The cosine ends at this share of the starting rate rather than at zero.
COSINE_FLOOR = 0.0025
_RESUMED = ("best", "best_step", "stall", "cuts", "done", "weights")


class Schedule(Protocol):
    """A learning rate per step, with a periodic look at how training is going."""

    done: bool

    def rate(self, step: int) -> float:
        """The learning rate of `step`."""

    def check(self, step: int, losses: list[float]) -> None:
        """Called every so many steps with the losses since the last call."""


def _warm_up(step: int) -> float:
    return min(1.0, (step + 1) / WARM_UP_STEPS)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


class Cosine:
    """Warm-up, then a cosine from the starting rate to its floor over a fixed number of steps."""

    def __init__(self, rate: float, steps: int) -> None:
        self._rate, self._steps = rate, steps
        self.done = False

    def rate(self, step: int) -> float:
        """The learning rate of `step`."""
        progress = min(step, self._steps) / self._steps
        decay = COSINE_FLOOR + (1 - COSINE_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))
        return self._rate * _warm_up(step) * decay

    def check(self, step: int, losses: list[float]) -> None:
        """Report the training loss."""
        print(f"step={step} lr={self.rate(step):.3g} train_loss={_mean(losses):.5f}", flush=True)


class Plateau:
    """A constant rate after warm-up, cut when the held-out mean stalls.

    Finished when a stall follows the last cut. Keeps the best weights seen,
    and after every check a file from which the run can be resumed.
    """

    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        rate: float,
        held_out: dict[str, list[SequenceBatch]],
        path: Path,
    ) -> None:
        self._model, self._optimiser, self._rate = model, optimiser, rate
        self._held_out, self._path = held_out, path
        self.best, self.best_step, self.stall, self.cuts, self.done = -math.inf, 0, 0, 0, False
        self.weights: dict[str, torch.Tensor] | None = None

    def rate(self, step: int) -> float:
        """The learning rate of `step`."""
        return self._rate * _warm_up(step) * CUT**self.cuts

    def check(self, step: int, losses: list[float]) -> None:
        """Score the held-out sets, apply the rule and save the run's state."""
        scores = {name: self._score(batches) for name, batches in self._held_out.items()}
        mean = _mean(list(scores.values()))
        self.stall = 0 if mean > self.best + MIN_GAIN else self.stall + 1
        if mean > self.best:
            self.best, self.best_step = mean, step
            self.weights = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
        if self.stall >= PATIENCE:
            self.stall = 0
            self.cuts += 1
            self.done = self.cuts > CUTS
        each = " ".join(f"{name}={score:.3f}" for name, score in scores.items())
        print(
            f"step={step} lr={self.rate(step):.3g} cuts={self.cuts} train_loss={_mean(losses):.5f} "
            f"held_out={mean:.4f} best={self.best:.4f} best_step={self.best_step} {each}",
            flush=True,
        )
        state: dict[str, Any] = {name: getattr(self, name) for name in _RESUMED}
        saved = {
            "model": self._model.state_dict(),
            "optimiser": self._optimiser.state_dict(),
            "step": step,
            "state": state,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, self._path)

    def resume(self) -> int:
        """Restore the run saved at the last check, if there is one; the step it had reached."""
        if not self._path.exists():
            return 0
        saved = torch.load(self._path)
        self._model.load_state_dict(saved["model"])
        self._optimiser.load_state_dict(saved["optimiser"])
        for name, value in saved["state"].items():
            setattr(self, name, value)
        return int(saved["step"])

    @torch.no_grad()
    def _score(self, batches: list[SequenceBatch]) -> float:
        """Mean PSNR over one held-out set, in dB."""
        device = next(self._model.parameters()).device
        self._model.eval()
        scores = []
        for batch in batches:
            batch = batch.to(device)  # noqa: PLW2901 - the same batch, moved
            output = rollout(self._model, batch)
            scores.append(psnr(output[:, WARM_UP_FRAMES:], batch.target[:, WARM_UP_FRAMES:, :1]))
        self._model.train()
        return _mean(scores)
