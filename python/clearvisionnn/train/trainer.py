"""The optimisation loop."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader

from clearvisionnn.config.run import RunConfig
from clearvisionnn.data.batch import SequenceBatch
from clearvisionnn.losses import CompositeLoss, Rollout
from clearvisionnn.losses.perceptual import LumaLpips
from clearvisionnn.models.conventional import ConventionalSR
from clearvisionnn.models.per_frame import reconstructed_channels
from clearvisionnn.train import checkpoint
from clearvisionnn.train.checkpoint import TrainState
from clearvisionnn.train.log import RunLog
from clearvisionnn.train.rollout import roll
from clearvisionnn.train.stability import STILL, STILL_ORIGINAL
from clearvisionnn.train.validate import HIGHER_IS_BETTER, validate


def _endless(loader: DataLoader[SequenceBatch]) -> Iterator[SequenceBatch]:
    while True:
        yield from loader


class Trainer:
    """Trains one model for the configured number of steps."""

    def __init__(
        self,
        config: RunConfig,
        run: str,
        state: TrainState,
        training: DataLoader[SequenceBatch],
        validation: DataLoader[SequenceBatch],
        device: torch.device,
    ) -> None:
        self._config, self._state, self._device = config, state, device
        self._training, self._validation = training, validation
        self._run_dir = config.run_dir(run)
        self._log = RunLog(self._run_dir)
        self._loss = CompositeLoss(config.train.losses)
        self._planes = reconstructed_channels(config.model.planes)
        # Scored with a different feature network from the one the loss uses.
        self._perceptual = LumaLpips("alex").to(device)

    def fit(self) -> None:
        """Run from the state's current step to the configured last step."""
        train, state = self._config.train, self._state
        if not state.conventional:
            state.conventional = validate(
                ConventionalSR(), self._validation, self._device, self._perceptual
            )
            self._log.write("validation_conventional", state.step, state.conventional)
        state.model.train()
        batches = _endless(self._training)
        while state.step < train.steps:
            parts = self._step(next(batches))
            state.step += 1
            last = state.step == train.steps
            if state.step % train.log_every == 0:
                parts["lr"] = state.scheduler.get_last_lr()[0]
                self._log.write("train", state.step, parts)
            if state.step % train.validate_every == 0 or last:
                self._validate()
            if state.step % train.checkpoint_every == 0 or last:
                checkpoint.save(state, self._config, self._run_dir / checkpoint.LATEST, {})

    def _step(self, batch: SequenceBatch) -> dict[str, float]:
        train, state = self._config.train, self._state
        batch = batch.to(self._device)
        with torch.autocast(self._device.type, dtype=torch.bfloat16, enabled=train.amp):
            output = roll(state.model, batch.view)[:, :, : self._planes]
            alternate = None
            if self._loss.needs_alternate and batch.alternate is not None:
                alternate = roll(state.model, batch.alternate)[:, :, : self._planes]
            rollout = Rollout(
                output.float(),
                batch.target[:, :, : self._planes],
                None if alternate is None else alternate.float(),
            )
            total, parts = self._loss(rollout)
        state.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), train.grad_clip)
        state.optimizer.step()
        state.scheduler.step()
        parts["loss"] = float(total.detach())
        return parts

    def _validate(self) -> None:
        """Score the model, keep it as a candidate, and promote it to best if it qualifies.

        Best is the best `train.best_by` score among checkpoints that are steady
        where the scene is still.
        Every validated checkpoint is kept, so a different choice across
        metric families can still be made afterwards.
        """
        state = self._state
        scores = validate(state.model, self._validation, self._device, self._perceptual)
        # Steady enough: where the scene is still, the output changes no more
        # than the original footage does, within the tolerance.
        limit = scores[STILL_ORIGINAL] * (1 + self._config.train.temporal_tolerance)
        steady = scores[STILL] <= limit
        self._log.write("validation", state.step, {**scores, "temporal_guard_met": steady})
        candidate = self._run_dir / checkpoint.CANDIDATES / f"step-{state.step:07d}.pt"
        checkpoint.save(state, self._config, candidate, scores)
        chosen_by = self._config.train.best_by
        # Stored so that larger is always better, whichever score is chosen.
        score = scores[chosen_by] if HIGHER_IS_BETTER[chosen_by] else -scores[chosen_by]
        if steady and score > state.best_score:
            state.best_score = score
            checkpoint.save(state, self._config, self._run_dir / checkpoint.BEST, scores)
