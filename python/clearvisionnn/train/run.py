"""Setting up and starting a training run from its configuration."""

from __future__ import annotations

import torch

from clearvisionnn.config.run import RunConfig
from clearvisionnn.data.loaders import training_loader, validation_loader
from clearvisionnn.models.factory import build_model, count_parameters
from clearvisionnn.train import checkpoint
from clearvisionnn.train.checkpoint import TrainState
from clearvisionnn.train.optim import build_optimizer, build_scheduler
from clearvisionnn.train.trainer import Trainer


def train_run(config: RunConfig, run: str, resume: bool = False) -> None:
    """Train the configured model; outputs go to the run's directory."""
    torch.manual_seed(config.train.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model).to(device)
    # Channels-last is the fast convolution layout under mixed precision.
    model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    if config.train.init_from_run and not resume:
        source = config.run_dir(config.train.init_from_run) / checkpoint.BEST
        checkpoint.require_same_model(
            config.model, checkpoint.describe(source).config.model, source
        )
        checkpoint.load_weights(model, source)
    optimizer = build_optimizer(model.parameters(), config.train)
    state = TrainState(model, optimizer, build_scheduler(optimizer, config.train))
    latest = config.run_dir(run) / checkpoint.LATEST
    if resume and latest.exists():
        checkpoint.restore(state, config, latest)
    print(
        f"model: {count_parameters(model) / 1e6:.2f}M parameters, from step {state.step}",
        flush=True,
    )
    trainer = Trainer(
        config, run, state, training_loader(config), validation_loader(config), device
    )
    trainer.fit()
