"""Evaluating a trained run on the sealed test clips."""

from __future__ import annotations

import torch

from clearvisionnn.config.run import RunConfig
from clearvisionnn.data.evaluation import EvaluationDataset
from clearvisionnn.data.index import ClipIndex
from clearvisionnn.data.views import ViewMaker
from clearvisionnn.eval.report import write_report
from clearvisionnn.eval.scoring import Results, build_methods, build_metrics, score_clip
from clearvisionnn.pipeline.degrade.chain import DegradeChain
from clearvisionnn.pipeline.records import Split
from clearvisionnn.pipeline.workdir import Workdir
from clearvisionnn.train.load import load_trained
from clearvisionnn.train.provenance import data_fingerprint


def evaluate_run(config: RunConfig, run: str) -> None:
    """Score the run's best checkpoint and the baselines; write the report beside it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = load_trained(config, run)
    model = trained.model.to(device)

    workdir = Workdir(config.data_dir, config.pipeline.disk_budget_gb)
    index = ClipIndex(workdir, Split.TEST, config.pipeline.degrade)
    views = ViewMaker(index, DegradeChain.from_specs(config.pipeline.degrade.chain))
    clips = index.clips[: config.evaluate.max_clips]
    dataset = EvaluationDataset(clips, views)
    metrics = build_metrics(config.evaluate.metrics, device)
    methods = build_methods(model, config.evaluate)

    results: Results = {}
    for item in range(len(dataset)):
        clip_id, sample = dataset[item]
        on_device = sample.to(device)
        score_clip(
            results, clip_id, on_device.view, on_device.target, methods, metrics, config.evaluate
        )
        print(f"[evaluate] {item + 1}/{len(dataset)} {clip_id}", flush=True)
    inputs = {
        "checkpoint": f"{trained.path} ({trained.weights_hash}), step {trained.saved.step}",
        "precision": config.evaluate.precision.value,
        "evaluated on data": data_fingerprint(config),
        **{f"trained with {k}": v for k, v in trained.saved.provenance.model_dump().items()},
    }
    groups = {clip.record.clip_id: clip.record.file_id for clip in clips}
    report = write_report(
        results,
        config.evaluate,
        config.run_dir(run) / "eval",
        f"{config.name} / {run}",
        groups,
        inputs,
    )
    print(report.read_text())
