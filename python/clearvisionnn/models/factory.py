"""Building a model from its configuration."""

from __future__ import annotations

from clearvisionnn.config.model import ModelConfig, Temporal
from clearvisionnn.models.blend import BlendSR
from clearvisionnn.models.layouts import Layout
from clearvisionnn.models.luma import LumaSR
from clearvisionnn.models.per_frame import PerFrameSR
from clearvisionnn.models.sequence import SequenceModel

_MODELS: dict[Temporal, type[SequenceModel]] = {
    Temporal.NONE: PerFrameSR,
    Temporal.BLEND: BlendSR,
}


def build_model(config: ModelConfig) -> SequenceModel:
    """The model `config` describes."""
    if config.luma_only:
        return LumaSR(config)
    return _MODELS[config.temporal](config)


def count_parameters(model: SequenceModel) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trained_core(model: SequenceModel) -> Layout:
    """The network inside `model`: the part that is exported."""
    core = getattr(model, "core", None)
    if not isinstance(core, Layout):
        raise TypeError(f"{type(model).__name__} has no trained network")
    return core
