"""The artefacts stages hand to each other, one JSON line per record."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from clearvisionnn.video.probe import VideoInfo


class Split(str, Enum):
    """Which set a source file, and every clip cut from it, belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SourceFile(BaseModel):
    """One ingested video file."""

    # Unique across the run: source name, file stem and a hash of the relative path.
    file_id: str
    # Changes whenever the file's content may have (size and modification time).
    revision: str
    source: str
    path: str
    info: VideoInfo
    split: Split
    # ffmpeg arguments that must precede the input (raw geometry), if any.
    input_args: list[str] = []


class Segment(BaseModel):
    """A planned clip: which frames of which file, and the target geometry."""

    clip_id: str
    file_id: str
    # Identifies everything the target depends on; part of every derived path,
    # so a changed plan or source can never reuse old media.
    fingerprint: str
    source: str
    split: Split
    path: str
    input_args: list[str]
    start_seconds: float
    frames: int
    fps: float
    # Take every `step`-th source frame (decimation of high-frame-rate sources).
    step: int
    downscale: int
    width: int
    height: int


class ClipRecord(BaseModel):
    """A clean target clip that passed the quality gate."""

    clip_id: str
    file_id: str
    fingerprint: str  # of the target this record describes
    source: str
    split: Split
    target: str  # path relative to the data directory
    frames: int
    fps: float
    width: int
    height: int
    sharpness: float
    motion: float


class Failure(BaseModel):
    """An item a stage could not process, kept so that the loss is auditable."""

    item: str
    error: str


class VariantRecord(BaseModel):
    """One stored degraded version of a clip (the offline part of the chain)."""

    clip_id: str
    index: int
    path: str  # relative to the data directory
    width: int
    height: int
    # Per-frame coding information beside the pictures, when a codec made them.
    coding: str | None = None
    # Per-module parameters drawn for this variant; None where a module was skipped.
    params: list[dict[str, Any] | None]
