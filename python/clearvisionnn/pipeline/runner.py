"""Runs the pipeline's stages in order. Each stage skips work that is already current."""

from __future__ import annotations

from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import clean, ingest, segment, targets, variants, verify
from clearvisionnn.pipeline.workdir import Workdir

STAGES = (ingest, segment, targets, clean, variants, verify)


def run_pipeline(config: RunConfig) -> Workdir:
    """Bring every artefact under the run's data directory up to date."""
    workdir = Workdir(config.data_dir, config.pipeline.disk_budget_gb)
    for stage in STAGES:
        print(f"== {stage.STAGE}", flush=True)
        stage.run(config, workdir)
    return workdir
