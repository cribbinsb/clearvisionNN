#!/usr/bin/env python3
"""Check that no source file has grown into a catch-all (CLAUDE.md, "Rules").

A file over the limit is doing more than one job and must be split. Run from
the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lines per file, by suffix. Tests may run longer: they are lists of cases.
LIMITS = {".py": 250, ".rs": 300}
TEST_LIMIT = 400
ROOTS = ("python", "crates", "tools")
SKIPPED_PARTS = {"target", ".venv", "__pycache__"}


def limit_for(path: Path) -> int:
    """The line limit that applies to `path`."""
    is_test = "tests" in path.parts or path.name.startswith("test_")
    return TEST_LIMIT if is_test else LIMITS[path.suffix]


def main() -> int:
    """Print every file over its limit; exit non-zero if there is one."""
    too_long = []
    for root in ROOTS:
        for path in sorted(Path(root).rglob("*")):
            if path.suffix not in LIMITS or SKIPPED_PARTS & set(path.parts):
                continue
            lines = len(path.read_text().splitlines())
            if lines > limit_for(path):
                too_long.append(f"{path}: {lines} lines (limit {limit_for(path)})")
    print("\n".join(too_long) or "file size: ok")
    return 1 if too_long else 0


if __name__ == "__main__":
    sys.exit(main())
