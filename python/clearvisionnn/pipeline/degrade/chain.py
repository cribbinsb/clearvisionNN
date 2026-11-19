"""A configured sequence of degradation modules, split into stored and live parts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from clearvisionnn.config.pipeline import ModuleSpec
from clearvisionnn.pipeline.degrade.base import MODULES, DegradeModule, Params
from clearvisionnn.video.clip import Clip

ChainParams = list[Params | None]


class DegradeChain:
    """Modules applied in order, from clean target towards network input."""

    def __init__(self, modules: Sequence[DegradeModule]) -> None:
        self._modules = list(modules)

    @classmethod
    def from_specs(cls, specs: Sequence[ModuleSpec]) -> DegradeChain:
        """Build each module from its registered name and parameters."""
        return cls([MODULES.get(spec.type)(**_tuples(spec.params)) for spec in specs])

    @property
    def scale(self) -> int:
        """Output size is the input size divided by this."""
        return int(np.prod([m.scale for m in self._modules], dtype=np.int64)) if self else 1

    def __len__(self) -> int:
        return len(self._modules)

    def split(self) -> tuple[DegradeChain, DegradeChain]:
        """The chain up to its last offline module, and the rest.

        The first part is run once by the pipeline and stored; the second
        runs during training with fresh draws every epoch.
        """
        offline = [i for i, m in enumerate(self._modules) if m.offline]
        cut = offline[-1] + 1 if offline else 0
        return DegradeChain(self._modules[:cut]), DegradeChain(self._modules[cut:])

    def sample(self, rng: np.random.Generator) -> ChainParams:
        """Per-clip parameters of every module."""
        return [module.sample(rng) for module in self._modules]

    def apply(self, clip: Clip, params: ChainParams, rng: np.random.Generator) -> Clip:
        """Run the chain on `clip`; the result is uint8."""
        result = clip.as_float()
        for module, module_params in zip(self._modules, params, strict=True):
            if module_params is not None:
                result = module.apply(result, module_params, rng)
        return result.as_uint8()


def _tuples(params: dict) -> dict:
    """YAML lists as tuples, the form module constructors declare."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
