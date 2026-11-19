"""Typed run configuration: one YAML file drives the pipeline, training and evaluation."""

from clearvisionnn.config.load import load_config
from clearvisionnn.config.run import RunConfig

__all__ = ["RunConfig", "load_config"]
