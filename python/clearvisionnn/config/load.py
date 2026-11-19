"""Reading a run configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from clearvisionnn.config.run import RunConfig

BASE_KEY = "base"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` laid over `base`; nested mappings merge, everything else replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    """The file's mapping, laid over the file its `base` key names (if any)."""
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    base = data.pop(BASE_KEY, None)
    if base is None:
        return data
    return _merge(_read(path.parent / base), data)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate the configuration at `path`."""
    return RunConfig.model_validate(_read(Path(path)))
