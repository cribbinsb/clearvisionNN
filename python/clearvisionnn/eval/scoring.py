"""Scoring every method on every clip with every metric."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from clearvisionnn.config.evaluate import EvalConfig, Precision
from clearvisionnn.data.batch import Degraded
from clearvisionnn.eval.baselines import BASELINES
from clearvisionnn.eval.metrics import METRICS, Metric
from clearvisionnn.eval.summary import ClipScore, summarise
from clearvisionnn.models.sequence import SequenceModel
from clearvisionnn.train.rollout import roll

# results[method][metric][clip_id]
Results = dict[str, dict[str, dict[str, ClipScore]]]
MODEL = "model"
# A method: the degraded view on the device in, (frames, 3, H, W) out.
Method = Callable[[Degraded], torch.Tensor]


def build_metrics(names: Iterable[str], device: torch.device) -> dict[str, Metric]:
    """Instantiate the named metrics on `device`."""
    return {name: METRICS.get(name)(device) for name in names}


def model_method(model: SequenceModel, precision: Precision) -> Method:
    """The trained model as a method, at the stated numerical precision."""

    def run(view: Degraded) -> torch.Tensor:
        batched = Degraded(
            *(t[None] for t in (view.frames, view.motion, view.side, view.reset, view.coding))
        )
        device = view.frames.device.type
        with torch.autocast(device, dtype=torch.bfloat16, enabled=precision is Precision.BF16):
            return roll(model, batched)[0].float().clamp(0.0, 1.0)

    return run


def baseline_method(name: str) -> Method:
    """A registered conventional upscaler as a method."""
    upscaler = BASELINES.get(name)
    return lambda view: upscaler(view.frames)


def build_methods(model: SequenceModel, config: EvalConfig) -> dict[str, Method]:
    """The baselines and the model, in report order."""
    methods = {name: baseline_method(name) for name in config.baselines}
    methods[MODEL] = model_method(model, config.precision)
    return methods


@torch.no_grad()
def score_clip(
    results: Results,
    clip_id: str,
    view: Degraded,
    target: torch.Tensor,
    methods: dict[str, Method],
    metrics: dict[str, Metric],
    config: EvalConfig,
) -> None:
    """Add one clip's scores for every method and metric to `results`.

    Methods run one after another, so only one output is held at a time.
    """
    for method, run in methods.items():
        output = run(view)
        for name, metric in metrics.items():
            per_frame = metric(output, target).float().cpu().numpy()
            score = summarise(per_frame, config.window, metric.higher_is_better)
            results.setdefault(method, {}).setdefault(name, {})[clip_id] = score
        del output
