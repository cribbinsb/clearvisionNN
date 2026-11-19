"""Loading a trained run for evaluation or export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clearvisionnn.config.run import RunConfig
from clearvisionnn.models.factory import build_model
from clearvisionnn.models.sequence import SequenceModel
from clearvisionnn.train import checkpoint
from clearvisionnn.train.provenance import file_hash


@dataclass(frozen=True)
class Trained:
    """A run's best model with everything known about where it came from."""

    model: SequenceModel
    saved: checkpoint.Saved
    path: Path
    weights_hash: str


def load_trained(config: RunConfig, run: str) -> Trained:
    """The run's best checkpoint, built as it was trained.

    The model is built from the configuration stored with the weights; the
    requested configuration must agree with it.
    """
    path = config.run_dir(run) / checkpoint.BEST
    saved = checkpoint.describe(path)
    checkpoint.require_same_model(config.model, saved.config.model, path)
    model = build_model(saved.config.model).eval()
    checkpoint.load_weights(model, path)
    return Trained(model, saved, path, file_hash(path))
