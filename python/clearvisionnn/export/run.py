"""Exporting a trained model: its network file and its contract, into one directory."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from clearvisionnn.config.model import ModelConfig
from clearvisionnn.config.run import RunConfig
from clearvisionnn.export.contract import (
    CONTRACT_FILE,
    INTEGER_FILE,
    ONNX_FILE,
    Contract,
    IntegerContract,
    Origin,
    contract_for,
)
from clearvisionnn.export.integer import quantise_model
from clearvisionnn.export.onnx import export_core
from clearvisionnn.models.factory import trained_core
from clearvisionnn.models.sequence import SequenceModel
from clearvisionnn.train.load import load_trained


def _publish(directory: Path, write: Callable[[Path], Contract]) -> Contract:
    """Fill a new `directory` through `write`, which returns the contract to store with the files.

    Everything is produced in a temporary directory and moved into place
    together, so a reader never sees a network beside another network's contract.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".export-") as staging:
        staged = Path(staging) / "model"
        staged.mkdir()
        contract = write(staged)
        (staged / CONTRACT_FILE).write_text(contract.model_dump_json(indent=1) + "\n")
        if directory.exists():
            shutil.rmtree(directory)
        staged.replace(directory)
    print(f"exported {contract.name} (step {contract.step}) to {directory}", flush=True)
    return contract


def export_model(
    name: str,
    config: ModelConfig,
    model: SequenceModel,
    origin: Origin,
    directory: Path,
    half: bool = False,
) -> Contract:
    """Write `model.onnx` and `model.json` for `model`, built from `config`, under `directory`."""
    layout = trained_core(model)

    def write(staged: Path) -> Contract:
        export_core(layout, staged / ONNX_FILE, half)
        digest = hashlib.sha256((staged / ONNX_FILE).read_bytes()).hexdigest()
        return contract_for(name, config, layout, origin, onnx_sha256=digest, half=half)

    return _publish(directory, write)


def export_integer(
    name: str,
    config: ModelConfig,
    model: SequenceModel,
    origin: Origin,
    directory: Path,
    tops: Sequence[float],
) -> Contract:
    """Write `model.i8` and `model.json`: `model` in 8-bit integer form (docs/INTEGER.md).

    `tops` are the activation tops learnt in quantisation-aware training.
    """
    integer = quantise_model(config, model, tops)
    words = integer.words().tobytes()

    def write(staged: Path) -> Contract:
        (staged / INTEGER_FILE).write_bytes(words)
        described = IntegerContract(
            sha256=hashlib.sha256(words).hexdigest(), halo=integer.halo, tile=integer.tile
        )
        return contract_for(name, config, trained_core(model), origin, integer=described)

    return _publish(directory, write)


def export_run(config: RunConfig, run: str, directory: Path, half: bool = False) -> None:
    """Export the run's best checkpoint, built as it was trained."""
    trained = load_trained(config, run)
    origin = Origin(
        step=trained.saved.step,
        checkpoint_hash=trained.weights_hash,
        provenance=trained.saved.provenance,
    )
    name = f"{config.name}/{run}"
    export_model(name, trained.saved.config.model, trained.model, origin, directory, half)
