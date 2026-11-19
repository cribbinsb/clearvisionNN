"""Stage 6 (stored half): run the offline part of the degradation chain on every clip.

Each clip gets several variants with different parameter draws. The live part
of the chain, if any, runs later, during training.
"""

from __future__ import annotations

import zlib

import numpy as np

from clearvisionnn.config.pipeline import DegradeConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import clean
from clearvisionnn.pipeline.degrade.chain import DegradeChain
from clearvisionnn.pipeline.parallel import map_items
from clearvisionnn.pipeline.records import ClipRecord, VariantRecord
from clearvisionnn.pipeline.workdir import Workdir, fingerprint
from clearvisionnn.video.ffmpeg import LOSSLESS_ARGS, decode, encode

STAGE = "variants"


# Part of every chain's tag: raise it whenever the way variants are made or
# stored changes, so old media is not reused.
VARIANT_VERSION = 4


def chain_tag(degrade: DegradeConfig) -> str:
    """Identifies a degradation chain, so several can share one data directory."""
    return fingerprint(degrade.chain, degrade.seed, VARIANT_VERSION)


def records_name(degrade: DegradeConfig) -> str:
    """The record file listing the stored variants of this chain."""
    return f"variants-{chain_tag(degrade)}.jsonl"


def variant_seed(seed: int, clip_id: str, index: int) -> int:
    """A reproducible seed for one variant of one clip."""
    return zlib.crc32(f"{seed}/{clip_id}/{index}".encode())


def _variant_path(tag: str, clip: ClipRecord, index: int) -> str:
    """Where a variant is stored: named by its chain and by its target's fingerprint."""
    return f"variants/{tag}/{clip.clip_id}-{clip.fingerprint}-{index}.mkv"


def _make_variant(
    clip: ClipRecord, index: int, chain: DegradeChain, tag: str, seed: int, workdir: Workdir
) -> VariantRecord:
    rng = np.random.default_rng(variant_seed(seed, clip.clip_id, index))
    params = chain.sample(rng)
    relative = _variant_path(tag, clip, index)
    coding = relative.removesuffix(".mkv") + ".coding.npy"
    if not workdir.path(relative).exists():
        target = decode(workdir.path(clip.target), clip.width, clip.height)
        degraded = chain.apply(target, params, rng)
        # Written before the pictures, whose presence marks the variant as made.
        if degraded.coding is not None:
            workdir.path(coding).parent.mkdir(parents=True, exist_ok=True)
            np.save(workdir.path(coding), degraded.coding)
        # The stored variant is the decoded picture, kept losslessly, so that
        # training reads exactly what a decoder would output.
        encode(degraded, workdir.path(relative), clip.fps, LOSSLESS_ARGS)
    return VariantRecord(
        clip_id=clip.clip_id,
        index=index,
        path=relative,
        coding=coding if workdir.path(coding).exists() else None,
        params=params,
        width=clip.width // chain.scale,
        height=clip.height // chain.scale,
    )


def run(config: RunConfig, workdir: Workdir) -> None:
    """Write the variants and their record file. A chain with no offline part writes none.

    Every variant must be made: a clip without its stored variants would reach
    training with the wrong input. Any failure therefore stops the stage.
    """
    degrade = config.pipeline.degrade
    offline, _ = DegradeChain.from_specs(degrade.chain).split()
    clips = workdir.read_records(clean.RECORDS, ClipRecord)
    tag = chain_tag(degrade)
    jobs = [(c, i) for c in clips for i in range(degrade.variants_per_clip)] if len(offline) else []
    outcome = map_items(
        STAGE,
        lambda job: _make_variant(*job, offline, tag, degrade.seed, workdir),
        jobs,
        config.pipeline.workers,
        describe=lambda job: f"{job[0].clip_id}-{job[1]}",
        checkpoint=workdir.check_disk_budget,
    )
    workdir.write_records(records_name(degrade), outcome.results)
