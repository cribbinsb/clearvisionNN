"""The top-level configuration of a run."""

from __future__ import annotations

from pathlib import Path

from clearvisionnn.config.base import Config
from clearvisionnn.config.evaluate import EvalConfig
from clearvisionnn.config.model import ModelConfig
from clearvisionnn.config.pipeline import PipelineConfig
from clearvisionnn.config.sources import SourceConfig
from clearvisionnn.config.train import TrainConfig


class RunConfig(Config):
    """Everything one run depends on."""

    name: str
    # Derived data and training outputs go here; never under a source root.
    work_dir: Path
    sources: list[SourceConfig]
    pipeline: PipelineConfig
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    evaluate: EvalConfig = EvalConfig()

    @property
    def data_dir(self) -> Path:
        """Where the pipeline's artefacts live."""
        return self.work_dir / "data"

    def run_dir(self, run: str) -> Path:
        """Where the training run called `run` keeps checkpoints, logs and results."""
        return self.work_dir / "runs" / run
