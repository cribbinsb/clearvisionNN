"""Saving and restoring a training run.

A checkpoint carries the configuration and provenance of the run that made
it. Whatever loads one builds the model from that stored configuration, and
refuses a requested configuration that differs in anything that changes what
the weights mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.models.sequence import SequenceModel
from clearvisionnn.train.provenance import Provenance, collect

LATEST = "latest.pt"
BEST = "best.pt"
CANDIDATES = "candidates"


@dataclass
class TrainState:
    """Everything that changes during training."""

    model: SequenceModel
    optimizer: torch.optim.Optimizer
    scheduler: Any
    step: int = 0
    best_score: float = float("-inf")
    # Validation scores of the conventional upscale: the reference the
    # temporal guard on `best.pt` is measured against.
    conventional: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Saved:
    """What a checkpoint says about itself."""

    config: RunConfig
    provenance: Provenance
    step: int
    scores: dict[str, float]


def save(state: TrainState, config: RunConfig, path: Path, scores: dict[str, float]) -> None:
    """Write a checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "step": state.step,
        "best_score": state.best_score,
        "conventional": state.conventional,
        "scores": scores,
        "torch_rng": torch.get_rng_state(),
        "config": config.model_dump(mode="json"),
        "provenance": collect(config).model_dump(),
    }
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return payload


def describe(path: Path) -> Saved:
    """The configuration, provenance, step and scores stored in a checkpoint."""
    payload = _load(path)
    return Saved(
        config=RunConfig.model_validate(payload["config"]),
        provenance=Provenance.model_validate(payload["provenance"]),
        step=int(payload["step"]),
        scores=dict(payload["scores"]),
    )


def require_same_model(requested: ModelConfig, saved: ModelConfig, path: Path) -> None:
    """Refuse a requested model that differs from the one the checkpoint was trained as.

    Shapes alone cannot catch this: `align` and `side_planes` change behaviour
    without changing any weight's shape.
    """
    if requested != saved:
        asked, stored = requested.model_dump(mode="json"), saved.model_dump(mode="json")
        changed = {key: (stored[key], asked[key]) for key in stored if stored[key] != asked[key]}
        raise ValueError(
            f"{path} was trained with a different model (stored, requested): {changed}"
        )


def restore(state: TrainState, config: RunConfig, path: Path) -> None:
    """Continue the run stored at `path`; its model and data must be the ones configured.

    The sampling position of the data loader is not restored: a resumed run
    sees different samples from an uninterrupted one from this step on.
    """
    payload = _load(path)
    saved = RunConfig.model_validate(payload["config"])
    require_same_model(config.model, saved.model, path)
    if config.pipeline.degrade != saved.pipeline.degrade:
        raise ValueError(f"{path} was trained on a different degradation chain")
    state.model.load_state_dict(payload["model"])
    state.optimizer.load_state_dict(payload["optimizer"])
    state.scheduler.load_state_dict(payload["scheduler"])
    state.step, state.best_score = payload["step"], payload["best_score"]
    state.conventional = dict(payload["conventional"])
    torch.set_rng_state(payload["torch_rng"])


def load_weights(model: SequenceModel, path: Path) -> None:
    """Load only the model weights from a checkpoint."""
    model.load_state_dict(_load(path)["model"])
