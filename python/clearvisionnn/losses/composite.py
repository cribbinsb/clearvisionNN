"""A weighted sum of registered losses."""

from __future__ import annotations

import torch

from clearvisionnn.losses.base import LOSSES, Loss, Rollout


def _make(name: str) -> Loss:
    """The loss registered as `name`; a registered class is instantiated for this run."""
    registered = LOSSES.get(name)
    return registered() if isinstance(registered, type) else registered


class CompositeLoss:
    """The run's loss: each named term times its weight."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._terms = [(name, _make(name), weight) for name, weight in weights.items()]

    @property
    def needs_alternate(self) -> bool:
        """Whether any term compares the two views of a batch."""
        return any(name == "consistency" for name, _, _ in self._terms)

    def __call__(self, rollout: Rollout) -> tuple[torch.Tensor, dict[str, float]]:
        """The total, and each term's unweighted value for logging."""
        total = rollout.output.new_zeros(())
        parts: dict[str, float] = {}
        for name, loss, weight in self._terms:
            value = loss(rollout)
            total = total + weight * value
            parts[name] = float(value.detach())
        return total, parts
