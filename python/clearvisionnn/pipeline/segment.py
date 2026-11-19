"""Stages 2-3 (planning half): grade each file and plan the clips to cut from it.

A file's compression is judged from its bits per pixel, which sets how far it
must be downscaled before it can serve as ground truth; files too compressed
or too small for any allowed factor yield no clips. Planning is cheap and
runs every time.
"""

from __future__ import annotations

import math
from collections import Counter

from clearvisionnn.config.pipeline import GradeConfig, SegmentConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import ingest
from clearvisionnn.pipeline.records import Segment, SourceFile
from clearvisionnn.pipeline.workdir import Workdir, fingerprint

STAGE = "segment"
RECORDS = "segments.jsonl"
# Part of every target's fingerprint: raise it whenever the way targets are
# cut or converted changes (clearvisionnn.pipeline.targets), so old media is not reused.
TARGET_VERSION = 2


def choose_downscale(bits_per_pixel: float, config: GradeConfig) -> int | None:
    """The smallest allowed downscale for a source this compressed (None: unusable)."""
    for minimum, factor in config.downscale_by_bpp:
        if bits_per_pixel >= minimum:
            return factor
    return None


def target_size(width: int, height: int, downscale: int, align: int) -> tuple[int, int]:
    """The target's size: the source over `downscale`, cropped to a multiple of `align`."""
    return (width // downscale // align * align, height // downscale // align * align)


def plan_file(file: SourceFile, segment: SegmentConfig, grade: GradeConfig) -> list[Segment]:
    """The clips to cut from one file."""
    info = file.info
    downscale = choose_downscale(info.bits_per_pixel, grade)
    if downscale is None:
        return []
    width, height = target_size(info.width, info.height, downscale, grade.align)
    if min(width, height) < grade.min_target_side:
        return []
    # Whole-frame decimation that never leaves the rate above the ceiling.
    step = max(1, math.ceil(info.fps / segment.max_fps - 1e-6))
    span = segment.frames * step  # source frames one clip covers
    count = min(segment.max_per_file, info.frames // span)
    stride = info.frames // max(count, 1)
    planned = []
    for index in range(count):
        unsigned = Segment(
            clip_id=f"{file.file_id}-{index:02d}",
            file_id=file.file_id,
            fingerprint="",
            source=file.source,
            split=file.split,
            path=file.path,
            input_args=file.input_args,
            start_seconds=index * stride / info.fps,
            frames=segment.frames,
            fps=info.fps / step,
            step=step,
            downscale=downscale,
            width=width,
            height=height,
        )
        signature = fingerprint(unsigned, file.revision, TARGET_VERSION)
        planned.append(unsigned.model_copy(update={"fingerprint": signature}))
    return planned


def run(config: RunConfig, workdir: Workdir) -> None:
    """Write `segments.jsonl` from `files.jsonl`."""
    files = workdir.read_records(ingest.RECORDS, SourceFile)
    segments = [
        s for f in files for s in plan_file(f, config.pipeline.segment, config.pipeline.grade)
    ]
    repeated = [clip for clip, count in Counter(s.clip_id for s in segments).items() if count > 1]
    if repeated:
        raise ValueError(f"{len(repeated)} clip identifiers are not unique, e.g. {repeated[0]}")
    workdir.write_records(RECORDS, segments)
    print(f"[{STAGE}] {len(segments)} clips planned from {len(files)} files", flush=True)
