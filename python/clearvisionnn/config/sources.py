"""The source registry: where video comes from and on what terms (DESIGN.md section 9)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from clearvisionnn.config.base import Config, Positive, PositiveFloat


class SourceKind(str, Enum):
    """How a source is used."""

    REFERENCE = "reference"  # good video; the pipeline makes its own degraded version
    PAIRED = "paired"  # a reference with a real degraded counterpart
    DEGRADED = "degraded"  # real target-domain footage with no reference


class RawFormat(Config):
    """Geometry of headerless raw video, which cannot be probed."""

    width: Positive
    height: Positive
    pix_fmt: str
    fps: PositiveFloat


class SourceConfig(Config):
    """One dataset."""

    name: str
    root: Path
    glob: str = "**/*.mp4"
    kind: SourceKind = SourceKind.REFERENCE
    licence: str = "unknown"
    commercial_use: bool = False
    raw: RawFormat | None = None
    # Cap on files taken from this source (None: all), for quick runs.
    max_files: Positive | None = None
    # For PAIRED sources: the directory holding each file's degraded counterpart
    # under the same relative path.
    degraded_root: Path | None = None
