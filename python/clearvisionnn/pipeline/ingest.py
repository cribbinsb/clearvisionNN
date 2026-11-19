"""Stage 1: find and probe every source file, and assign its split.

Discovery runs every time, so a file added to or changed in a source root is
always noticed. The split is decided here, once, by source file, and no later
stage can move a clip between sets (DESIGN.md section 9).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from clearvisionnn.config.pipeline import SplitConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.config.sources import SourceConfig, SourceKind
from clearvisionnn.pipeline.parallel import map_items
from clearvisionnn.pipeline.records import SourceFile, Split
from clearvisionnn.pipeline.workdir import Workdir
from clearvisionnn.video.probe import ProbeError, probe, raw_info

STAGE = "ingest"
RECORDS = "files.jsonl"

_HASH_RESOLUTION = 1_000_000
_ID_HASH_LENGTH = 8


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def file_id(source: str, relative_path: str) -> str:
    """A readable identifier unique to one file of one source.

    The stem alone is not unique: datasets repeat file names across
    directories. The hash covers the whole relative path.
    """
    stem = Path(relative_path).stem
    return f"{source}-{stem}-{_digest(f'{source}/{relative_path}').hex()[:_ID_HASH_LENGTH]}"


def assign_split(source: str, relative_path: str, config: SplitConfig) -> Split:
    """The split of a file: a fixed pseudo-random draw from its name."""
    digest = _digest(f"{config.seed}/{source}/{relative_path}")
    draw = int.from_bytes(digest[:8], "big") % _HASH_RESOLUTION / _HASH_RESOLUTION
    if draw < config.test:
        return Split.TEST
    if draw < config.test + config.validation:
        return Split.VALIDATION
    return Split.TRAIN


def _ingest_file(source: SourceConfig, path: Path, split: SplitConfig) -> SourceFile:
    if source.raw is None:
        info, input_args = probe(path), []
    else:
        raw = source.raw
        info = raw_info(path, raw.width, raw.height, raw.pix_fmt, raw.fps)
        size = f"{raw.width}x{raw.height}"
        input_args = ["-f", "rawvideo", "-pix_fmt", raw.pix_fmt, "-s", size, "-r", f"{raw.fps}"]
    relative = str(path.relative_to(source.root))
    stat = path.stat()
    return SourceFile(
        file_id=file_id(source.name, relative),
        revision=f"{stat.st_size}-{stat.st_mtime_ns}",
        source=source.name,
        path=str(path),
        info=info,
        split=assign_split(source.name, relative, split),
        input_args=input_args,
    )


def run(config: RunConfig, workdir: Workdir) -> None:
    """Write `files.jsonl`: one record per usable reference file."""
    jobs: list[tuple[SourceConfig, Path]] = []
    for source in config.sources:
        if source.kind is not SourceKind.REFERENCE:
            raise NotImplementedError(f"source kind '{source.kind.value}' is not built yet")
        files = sorted(source.root.glob(source.glob))[: source.max_files]
        jobs += [(source, path) for path in files]
    outcome = map_items(
        STAGE,
        lambda job: _ingest_file(*job, config.pipeline.split),
        jobs,
        config.pipeline.workers,
        describe=lambda job: str(job[1]),
        tolerated=(ProbeError,),
    )
    workdir.write_records(RECORDS, outcome.results)
    workdir.write_failures(STAGE, outcome.failures)
