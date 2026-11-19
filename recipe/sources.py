"""Which data a step uses: datasets crossed with degradation chains, and each pair's files.

A dataset under a chain is one run configuration, `configs/<dataset>_<chain>_blend.yaml`:
it names the pipeline work directory holding the clips and the degradation applied to them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

CONFIGS = Path("configs")
# The four training datasets and the two degradation chains (camera output
# after the ISP; video after a real-time encode and decode) of the published models.
DATASETS = ("vcd_flat_s", "sanity", "vcd2", "vc1k")
CHAINS = ("camera", "ncodec")
# Everything the recipe writes goes under this directory, which git ignores.
OUT = Path("recipe/out")


@dataclass(frozen=True)
class Source:
    """One dataset under one degradation chain."""

    dataset: str
    chain: str

    @property
    def name(self) -> str:
        """What the pair's derived files are called."""
        return f"{self.dataset}-{self.chain}"

    @property
    def config(self) -> Path:
        """The run configuration that describes the pair."""
        return CONFIGS / f"{self.dataset}_{self.chain}_blend.yaml"


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags that choose datasets and chains, defaulting to the published recipe's."""
    parser.add_argument(
        "--datasets", default=",".join(DATASETS), help="comma-separated dataset names"
    )
    parser.add_argument("--chains", default=",".join(CHAINS), help="comma-separated chain names")


def sources_from(arguments: argparse.Namespace) -> list[Source]:
    """Every chosen dataset under every chosen chain, chain by chain."""
    return [
        Source(dataset, chain)
        for chain in arguments.chains.split(",")
        for dataset in arguments.datasets.split(",")
    ]
