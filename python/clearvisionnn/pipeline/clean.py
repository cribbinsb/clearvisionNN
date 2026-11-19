"""Stage 2 (filtering half): measure each target and reject unusable clips.

Runs on the extracted targets, where decoding is cheap: clips with a cut,
frozen clips, clips with no detail, and clips that came out short or with
irregular timestamps.
"""

from __future__ import annotations

import numpy as np

from clearvisionnn.config.pipeline import CleanConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import segment, targets
from clearvisionnn.pipeline.parallel import map_items
from clearvisionnn.pipeline.records import ClipRecord, Segment
from clearvisionnn.pipeline.workdir import Workdir, fingerprint
from clearvisionnn.video.ffmpeg import decode
from clearvisionnn.video.reader import has_regular_timing

STAGE = "clean"
RECORDS = "clips.jsonl"
# Part of the stage's fingerprint: raise it whenever the rejection rules change,
# so that existing work directories are re-inspected.
RULES_VERSION = 2

# Frame differences are measured on luma decimated by this, which is enough to
# see a cut and averages sensor noise away.
_DECIMATION = 8


def frame_changes(luma: np.ndarray) -> np.ndarray:
    """Mean absolute change between consecutive frames, at low resolution."""
    small = luma[:, ::_DECIMATION, ::_DECIMATION].astype(np.float32)
    return np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))


def sharpness(luma: np.ndarray) -> float:
    """Mean gradient magnitude of the middle frame."""
    frame = luma[len(luma) // 2].astype(np.float32)
    return float(np.abs(np.diff(frame, axis=0)).mean() + np.abs(np.diff(frame, axis=1)).mean())


def _inspect(planned: Segment, workdir: Workdir, config: CleanConfig) -> ClipRecord | None:
    relative = targets.target_path(planned.clip_id, planned.fingerprint)
    path = workdir.path(relative)
    if not path.exists() or not has_regular_timing(path, planned.frames):
        return None
    luma = decode(path, planned.width, planned.height).y
    changes = frame_changes(luma)
    sharp = sharpness(luma)
    rejected = (
        changes.max() > config.cut_threshold
        or changes.mean() < config.frozen_threshold
        or sharp < config.min_sharpness
    )
    if rejected:
        return None
    return ClipRecord(
        clip_id=planned.clip_id,
        file_id=planned.file_id,
        fingerprint=planned.fingerprint,
        source=planned.source,
        split=planned.split,
        target=relative,
        frames=planned.frames,
        fps=planned.fps,
        width=planned.width,
        height=planned.height,
        sharpness=sharp,
        motion=float(changes.mean()),
    )


def run(config: RunConfig, workdir: Workdir) -> None:
    """Write `clips.jsonl`: the targets that passed."""
    planned = workdir.read_records(segment.RECORDS, Segment)
    present = [
        s.clip_id
        for s in planned
        if workdir.path(targets.target_path(s.clip_id, s.fingerprint)).exists()
    ]
    # Segments carry the fingerprint of everything their target depends on, so
    # this key changes exactly when an inspected result could.
    key = fingerprint(planned, present, config.pipeline.clean, RULES_VERSION)
    if workdir.is_current(STAGE, key) and workdir.path(RECORDS).exists():
        return
    outcome = map_items(
        STAGE,
        lambda s: _inspect(s, workdir, config.pipeline.clean),
        planned,
        config.pipeline.workers,
        describe=lambda s: s.clip_id,
    )
    kept = [record for record in outcome.results if record is not None]
    workdir.write_records(RECORDS, kept)
    workdir.mark_current(STAGE, key)
    print(f"[{STAGE}] kept {len(kept)} of {len(planned)} clips", flush=True)
