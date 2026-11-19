"""Optimiser and learning-rate schedule."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from clearvisionnn.config.train import TrainConfig


def build_optimizer(parameters, config: TrainConfig) -> torch.optim.Optimizer:  # type: ignore[no-untyped-def]
    """AdamW over `parameters`."""
    return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)


def learning_rate_factor(step: int, config: TrainConfig) -> float:
    """The schedule: linear warm-up, then a cosine from `lr` down to `lr_min`."""
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    floor = config.lr_min / config.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> LambdaLR:
    """The schedule of `learning_rate_factor`, stepped once per optimiser step."""
    return LambdaLR(optimizer, lambda step: learning_rate_factor(step, config))
