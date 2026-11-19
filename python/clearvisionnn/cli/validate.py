"""Checks a configuration against the registered components before anything runs.

Field ranges are checked when the file is loaded (`clearvisionnn.config`); this
checks what needs the components themselves: that every name is registered,
that the degradation chain can be built and scales by the network's factor.
"""

from __future__ import annotations

from clearvisionnn.config.run import RunConfig
from clearvisionnn.contract import SR_SCALE
from clearvisionnn.eval.baselines import BASELINES
from clearvisionnn.eval.metrics import METRICS
from clearvisionnn.losses import LOSSES
from clearvisionnn.models.layouts import LAYOUTS
from clearvisionnn.models.sizes import get_size
from clearvisionnn.pipeline.degrade import DegradeChain


def validate_run(config: RunConfig) -> None:
    """Raise with a clear message if `config` names or combines things that cannot work."""
    chain = DegradeChain.from_specs(config.pipeline.degrade.chain)
    if chain.scale != SR_SCALE:
        raise ValueError(f"the degradation chain scales by {chain.scale}, not {SR_SCALE}")
    LAYOUTS.get(config.model.layout)
    get_size(config.model.size)
    for name in config.train.losses:
        LOSSES.get(name)
    for name in config.evaluate.metrics:
        METRICS.get(name)
    for name in [*config.evaluate.baselines, config.evaluate.reference_baseline]:
        BASELINES.get(name)
    if config.evaluate.reference_baseline not in config.evaluate.baselines:
        raise ValueError("evaluate.reference_baseline must be one of evaluate.baselines")
