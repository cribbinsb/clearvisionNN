"""What a result was made from: code, data and library versions.

Stored in every checkpoint and carried into evaluation reports and exported
contracts, so that a number or a model file can be traced to its inputs.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import torch
from pydantic import BaseModel

from clearvisionnn.config.run import RunConfig
from clearvisionnn.pipeline import clean, variants

_REPOSITORY = Path(__file__).resolve().parents[3]
_HASH_LENGTH = 16


class Provenance(BaseModel):
    """The inputs of a training run."""

    code_revision: str  # git commit, or "unknown" outside a repository
    code_dirty: bool  # uncommitted changes were present
    data_fingerprint: str  # of the clip and variant records trained on
    torch_version: str


def _git(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(_REPOSITORY), *arguments], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def file_hash(path: Path) -> str:
    """A short content hash of `path`."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:_HASH_LENGTH]


def data_fingerprint(config: RunConfig) -> str:
    """Identifies the dataset: the records of its clips and of this chain's variants."""
    records = [clean.RECORDS, variants.records_name(config.pipeline.degrade)]
    return "-".join(file_hash(config.data_dir / name) for name in records)


def describe(data: str) -> Provenance:
    """The provenance of something made now from the data identified by `data`."""
    status = _git("status", "--porcelain")
    return Provenance(
        code_revision=_git("rev-parse", "HEAD") or "unknown",
        code_dirty=bool(status) or status is None,
        data_fingerprint=data,
        torch_version=torch.__version__,
    )


def collect(config: RunConfig) -> Provenance:
    """The provenance of a run started now with `config`."""
    return describe(data_fingerprint(config))
