"""Which clips and stored variants exist for a split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clearvisionnn.config.pipeline import DegradeConfig
from clearvisionnn.pipeline import clean, variants
from clearvisionnn.pipeline.records import ClipRecord, Split, VariantRecord
from clearvisionnn.pipeline.workdir import Workdir


@dataclass(frozen=True)
class IndexedClip:
    """A clip with its stored variants (none when the whole chain runs live)."""

    record: ClipRecord
    variants: tuple[VariantRecord, ...]


class ClipIndex:
    """The clips of one split, with absolute paths resolved against the data directory."""

    def __init__(self, workdir: Workdir, split: Split, degrade: DegradeConfig) -> None:
        self._workdir = workdir
        by_clip: dict[str, list[VariantRecord]] = {}
        for variant in workdir.read_records(variants.records_name(degrade), VariantRecord):
            by_clip.setdefault(variant.clip_id, []).append(variant)
        self.clips = [
            IndexedClip(record, tuple(by_clip.get(record.clip_id, ())))
            for record in workdir.read_records(clean.RECORDS, ClipRecord)
            if record.split is split
        ]
        if not self.clips:
            raise ValueError(f"no {split.value} clips under {workdir.root}")

    def __len__(self) -> int:
        return len(self.clips)

    def resolve(self, relative: str) -> Path:
        """The absolute path of a record's file."""
        return self._workdir.path(relative)
