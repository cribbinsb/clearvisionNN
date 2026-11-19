"""Stage 4: cut each planned clip and downscale it into a clean target.

One ffmpeg run per clip: seek, decimate, area-downscale, centre-crop to the
planned size, convert to 8-bit 4:2:0 BT.709 limited range, store losslessly.
A target's file name carries its plan's fingerprint, so a file that exists is
the right one.
"""

from __future__ import annotations

from pathlib import Path

from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import segment
from clearvisionnn.pipeline.parallel import map_items
from clearvisionnn.pipeline.records import Segment
from clearvisionnn.pipeline.workdir import Workdir
from clearvisionnn.video.ffmpeg import LOSSLESS_ARGS, STORED_PIX_FMT, FfmpegError, transcode

STAGE = "targets"
DIRECTORY = "targets"


def target_path(clip_id: str, fingerprint: str) -> str:
    """Where a clip's target is stored, relative to the data directory."""
    return f"{DIRECTORY}/{clip_id}-{fingerprint}.mkv"


def _filters(planned: Segment) -> str:
    chain = []
    if planned.step > 1:
        chain.append(f"fps={planned.fps:.6f}")
    chain += [
        f"scale=iw/{planned.downscale}:ih/{planned.downscale}:flags=area"
        ":out_color_matrix=bt709:out_range=tv",
        f"crop={planned.width}:{planned.height}",
        f"format={STORED_PIX_FMT}",
        # Number the frames afresh at the planned rate. Seeking and
        # variable-frame-rate sources leave irregular timestamps, and readers
        # find a frame by its timestamp.
        f"setpts=N/({planned.fps:.6f}*TB)",
    ]
    return ",".join(chain)


def _make_target(planned: Segment, workdir: Workdir) -> None:
    target = workdir.path(target_path(planned.clip_id, planned.fingerprint))
    if target.exists():
        return
    transcode(
        Path(planned.path),
        target,
        # No auto-rotation: the planned size comes from the stored geometry, and
        # sideways content trains as well as upright. Readers of the stored
        # clip ignore any rotation tag for the same reason (`video.ffmpeg.decode`).
        input_args=[*planned.input_args, "-noautorotate", "-ss", f"{planned.start_seconds:.4f}"],
        output_args=["-an", "-vf", _filters(planned), "-frames:v", str(planned.frames),
                     "-r", f"{planned.fps:.6f}", *LOSSLESS_ARGS],
    )  # fmt: skip


def run(config: RunConfig, workdir: Workdir) -> None:
    """Make the target of every planned clip that does not have one yet."""
    planned = workdir.read_records(segment.RECORDS, Segment)
    outcome = map_items(
        STAGE,
        lambda s: _make_target(s, workdir),
        planned,
        config.pipeline.workers,
        describe=lambda s: s.clip_id,
        tolerated=(FfmpegError,),  # a source file ffmpeg cannot cut is dropped, on record
        checkpoint=workdir.check_disk_budget,
    )
    workdir.write_failures(STAGE, outcome.failures)
