"""Entry point: `clearvisionnn <command> <config.yaml> [options]`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from clearvisionnn.config import RunConfig, load_config


def _pipeline(config: RunConfig, _: argparse.Namespace) -> None:
    from clearvisionnn.pipeline.runner import run_pipeline

    run_pipeline(config)


def _train(config: RunConfig, args: argparse.Namespace) -> None:
    from clearvisionnn.train.run import train_run

    train_run(config, args.run, resume=args.resume)


def _evaluate(config: RunConfig, args: argparse.Namespace) -> None:
    from clearvisionnn.eval.run import evaluate_run

    evaluate_run(config, args.run)


def _export(config: RunConfig, args: argparse.Namespace) -> None:
    from clearvisionnn.export.run import export_run

    export_run(config, args.run, Path(args.out), args.half)


COMMANDS: dict[str, Callable[[RunConfig, argparse.Namespace], None]] = {
    "pipeline": _pipeline,
    "train": _train,
    "evaluate": _evaluate,
    "export": _export,
}


def main() -> None:
    """Parse the command line and run the chosen command."""
    parser = argparse.ArgumentParser(prog="clearvisionnn", description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("config", help="run configuration (YAML)")
    parser.add_argument("--run", default="default", help="name of the training run")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    parser.add_argument("--out", default="web/models/default", help="export directory")
    parser.add_argument("--half", action="store_true", help="export 16-bit float arithmetic")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command != "pipeline":
        from clearvisionnn.cli.validate import validate_run

        validate_run(config)
    COMMANDS[args.command](config, args)


if __name__ == "__main__":
    main()
