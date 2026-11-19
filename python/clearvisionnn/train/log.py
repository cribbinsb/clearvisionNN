"""The training log: one JSON object per line, and a short line on the console."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLog:
    """Appends records to `<run>/log.jsonl`."""

    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "log.jsonl"
        self._start = time.monotonic()

    def write(self, kind: str, step: int, values: dict[str, Any]) -> None:
        """Log `values` for `step` under the record type `kind`."""
        record = {"kind": kind, "step": step, "seconds": round(time.monotonic() - self._start, 1)}
        record.update(values)
        with self._path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        shown = " ".join(
            f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items()
        )
        print(f"[{kind}] step {step} {shown}", flush=True)
