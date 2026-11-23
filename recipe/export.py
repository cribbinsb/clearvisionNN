"""Export a checkpoint for the demo page: its network file and its contract.

    python -m recipe.export CHECKPOINT DIRECTORY [--float] [--list]

Builds the matching model of `clearvisionnn.models`, loads the checkpoint's weights
into it and writes `model.json` beside the network. A checkpoint of the 8-bit
conversion (it carries the learnt activation tops) is written in integer form
as `model.i8`, unless `--float` asks for the float graph of its weights; any
other checkpoint is written as `model.onnx`. `--list` also adds the directory
to the model list (`manifest.json`) of the directory that holds it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvisionnn.config.model import ModelConfig, Temporal
from clearvisionnn.export.contract import Origin
from clearvisionnn.export.run import export_integer, export_model
from clearvisionnn.models.factory import build_model, trained_core
from clearvisionnn.models.sizes import SIZES
from clearvisionnn.models.state import BoundedState
from clearvisionnn.train.provenance import describe, file_hash
from recipe.options import Options, load_checkpoint

MANIFEST = "manifest.json"
# How the exported contract names where the weights came from.
ORIGIN = "recipe"


def model_config(options: Options) -> ModelConfig:
    """The package's configuration of the network `recipe.model` builds from `options`."""
    shape = (options.blocks, options.channels)
    sizes = [n for n, size in SIZES.items() if (size.flat_blocks, size.flat_channels) == shape]
    if not sizes:
        raise ValueError(f"no rung of the size ladder is {shape[1]} channels x {shape[0]} blocks")
    return ModelConfig(
        layout="flat",
        size=sizes[0],
        temporal=Temporal.RECURRENT,
        align=True,
        state_channels=options.state_channels,
        side_planes=False,
        luma_only=True,
        confidence_plane=True,
    )


def add_to_list(directory: Path) -> None:
    """Add `directory` to the model list beside it, if it is not there."""
    manifest = directory.parent / MANIFEST
    listed = json.loads(manifest.read_text())
    if directory.name not in listed["models"]:
        listed["models"].append(directory.name)
        manifest.write_text(json.dumps(listed) + "\n")


def main() -> None:
    """Export one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--float", action="store_true", help="export the float graph regardless")
    parser.add_argument("--list", action="store_true", help="add to the directory's model list")
    parser.add_argument("--step", type=int, help="training steps to record, if the file has none")
    arguments = parser.parse_args()
    saved = load_checkpoint(arguments.checkpoint)
    config = model_config(saved.options)
    model = build_model(config).eval()
    core = trained_core(model)
    # The package squashes the state in a wrapper; the weights belong to the layout inside it.
    layout = core.layout if isinstance(core, BoundedState) else core
    layout.load_state_dict(saved.weights)
    stem = arguments.checkpoint.stem
    origin = Origin(
        step=saved.step if arguments.step is None else arguments.step,
        checkpoint_hash=file_hash(arguments.checkpoint),
        provenance=describe(f"{ORIGIN}:{stem}"),
    )
    name = f"{ORIGIN}/{stem}"
    if saved.tops is not None and not arguments.float:
        export_integer(name, config, model, origin, arguments.directory, saved.tops)
    else:
        export_model(name, config, model, origin, arguments.directory)
    if arguments.list:
        add_to_list(arguments.directory)


if __name__ == "__main__":
    main()
