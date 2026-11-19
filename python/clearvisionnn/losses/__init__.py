"""Training losses (DESIGN.md section 8.6). Importing this package registers the built-in ones."""

from clearvisionnn.losses import consistency, fidelity, perceptual, temporal
from clearvisionnn.losses.base import LOSSES, Rollout
from clearvisionnn.losses.composite import CompositeLoss

__all__ = [
    "LOSSES",
    "CompositeLoss",
    "Rollout",
    "consistency",
    "fidelity",
    "perceptual",
    "temporal",
]
