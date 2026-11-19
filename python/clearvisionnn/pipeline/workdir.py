"""The pipeline's work directory: record files, stage stamps and the disk budget."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

R = TypeVar("R", bound=BaseModel)

_BYTES_PER_GB = 1 << 30


def fingerprint(*parts: Any) -> str:
    """A short stable hash of JSON-serialisable values and pydantic models."""

    def plain(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    text = json.dumps([plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Workdir:
    """Where derived data lives. Nothing is ever written outside `root`."""

    def __init__(self, root: Path, disk_budget_gb: float) -> None:
        self.root = root
        self._budget_bytes = disk_budget_gb * _BYTES_PER_GB
        root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        """The absolute path of an artefact named relative to the root."""
        return self.root / relative

    def write_records(self, name: str, records: Iterable[BaseModel]) -> None:
        """Replace the record file `name` atomically."""
        temporary = self.path(name + ".partial")
        with temporary.open("w") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
        temporary.replace(self.path(name))

    def read_records(self, name: str, model: type[R]) -> list[R]:
        """Every record of the file `name`."""
        with self.path(name).open() as handle:
            return [model.model_validate_json(line) for line in handle if line.strip()]

    def write_failures(self, stage: str, failures: Iterable[BaseModel]) -> None:
        """Record the items `stage` could not process (an empty file when there were none)."""
        self.write_records(f"{stage}.failures.jsonl", failures)

    def is_current(self, stage: str, key: str) -> bool:
        """Whether `stage` last completed with exactly this input fingerprint."""
        stamp = self.path(f"{stage}.stamp")
        return stamp.exists() and stamp.read_text() == key

    def mark_current(self, stage: str, key: str) -> None:
        """Record that `stage` completed for the inputs fingerprinted as `key`."""
        self.path(f"{stage}.stamp").write_text(key)

    def check_disk_budget(self) -> None:
        """Refuse to continue once the work directory exceeds its budget or the disk is full."""
        used = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        if used > self._budget_bytes:
            raise RuntimeError(
                f"{self.root} holds {used / _BYTES_PER_GB:.1f} GB, over its budget of "
                f"{self._budget_bytes / _BYTES_PER_GB:.0f} GB (pipeline.disk_budget_gb)"
            )
        if shutil.disk_usage(self.root).free < _BYTES_PER_GB:
            raise RuntimeError(f"less than 1 GB free on the disk holding {self.root}")
