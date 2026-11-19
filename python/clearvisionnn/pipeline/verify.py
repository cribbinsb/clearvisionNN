"""Final stage: check that stored inputs show the same picture as their targets.

A pairing fault (wrong frames, a transposed picture, a stale variant) leaves
tensors the right shape and training running, learning nothing. This stage
samples clips from every source, upscales a frame of each stored variant
conventionally and compares it with the target. Degradation costs a few grey
levels; a pairing fault costs tens.
"""

from __future__ import annotations

import cv2
import numpy as np

from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import clean, variants
from clearvisionnn.pipeline.parallel import map_items
from clearvisionnn.pipeline.records import ClipRecord, VariantRecord
from clearvisionnn.pipeline.workdir import Workdir
from clearvisionnn.video.reader import read_frames

STAGE = "verify"


def pairing_error(clip: ClipRecord, variant: VariantRecord, workdir: Workdir) -> float:
    """Mean absolute luma difference (0-255) between a target and its stored input, mid-clip."""
    frame = clip.frames // 2
    target = read_frames(workdir.path(clip.target), frame, 1).y[0]
    stored = read_frames(workdir.path(variant.path), frame, 1).y[0]
    restored = cv2.resize(stored, (clip.width, clip.height), interpolation=cv2.INTER_CUBIC)
    return float(np.abs(restored.astype(np.float32) - target).mean())


def _sample(records: list[VariantRecord], clips: dict[str, ClipRecord], per_source: int,
            seed: int) -> list[VariantRecord]:  # fmt: skip
    """Up to `per_source` variants of each source, chosen reproducibly."""
    by_source: dict[str, list[VariantRecord]] = {}
    for record in records:
        by_source.setdefault(clips[record.clip_id].source, []).append(record)
    rng = np.random.default_rng(seed)
    chosen: list[VariantRecord] = []
    for group in by_source.values():
        picks = rng.choice(len(group), size=min(per_source, len(group)), replace=False)
        chosen += [group[int(i)] for i in picks]
    return chosen


def run(config: RunConfig, workdir: Workdir) -> None:
    """Raise if any sampled variant does not show its target."""
    settings = config.pipeline.verify
    clips = {c.clip_id: c for c in workdir.read_records(clean.RECORDS, ClipRecord)}
    name = variants.records_name(config.pipeline.degrade)
    stored = workdir.read_records(name, VariantRecord)
    sample = _sample(stored, clips, settings.clips_per_source, config.pipeline.degrade.seed)
    outcome = map_items(
        STAGE,
        lambda v: (v.clip_id, pairing_error(clips[v.clip_id], v, workdir)),
        sample,
        config.pipeline.workers,
        describe=lambda v: v.clip_id,
    )
    faults = [(clip, error) for clip, error in outcome.results if error > settings.max_error]
    if faults:
        worst = max(faults, key=lambda fault: fault[1])
        raise RuntimeError(
            f"{len(faults)} of {len(sample)} sampled variants do not show their target "
            f"(worst: {worst[0]}, error {worst[1]:.1f} > {settings.max_error})"
        )
    print(f"[{STAGE}] {len(sample)} sampled variants match their targets", flush=True)
