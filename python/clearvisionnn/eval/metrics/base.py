"""The metric interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import torch

from clearvisionnn.registry import Registry


class Family(str, Enum):
    """What a metric is sensitive to. No single family decides anything."""

    FIDELITY = "fidelity"
    STRUCTURE = "structure"
    PERCEPTUAL = "perceptual"
    TEMPORAL = "temporal"


class Metric(ABC):
    """Scores a clip's output against its target, one value per frame (or frame pair)."""

    family: ClassVar[Family]
    higher_is_better: ClassVar[bool] = True

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame scores for (frames, 3, H, W) YUV tensors in [0, 1]."""


METRICS: Registry[type[Metric]] = Registry("metric")
