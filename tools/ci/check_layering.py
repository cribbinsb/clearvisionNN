#!/usr/bin/env python3
"""Check that Python imports only go down the layer order (CLAUDE.md, "Rules").

Each top-level subpackage of `clearvisionnn` sits in one layer and may import only
from its own layer's package or from lower layers. Run from the repository root.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = "clearvisionnn"
ROOT = Path("python") / PACKAGE

# Lowest first. Packages in one tuple are peers and may not import each other.
LAYERS: tuple[tuple[str, ...], ...] = (
    ("registry", "contract", "config", "video"),
    ("pipeline",),
    ("data",),
    ("models", "losses"),
    ("train",),
    ("eval",),
    ("export",),
    ("cli",),
)
# Pairs of peers allowed to import each other (none at present).
PEER_EXCEPTIONS: set[tuple[str, str]] = set()


def layer_of(package: str) -> int:
    """The index of the layer holding `package`."""
    for index, members in enumerate(LAYERS):
        if package in members:
            return index
    raise KeyError(f"'{package}' is not assigned to a layer in {__file__}")


def imported_packages(source: str, own_parts: tuple[str, ...]) -> set[str]:
    """The `clearvisionnn` subpackages imported by a module at `own_parts` inside the package.

    Relative imports are resolved against the module's own position, so they
    are checked like absolute ones.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(own_parts[: len(own_parts) - node.level])
                module = node.module.split(".") if node.module else []
                targets = [base + module] if base + module else [[a.name] for a in node.names]
                names = [".".join([PACKAGE, *target]) for target in targets]
            elif node.module:
                names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == PACKAGE and len(parts) > 1:
                found.add(parts[1])
    return found


def violations_in(source: str, own_parts: tuple[str, ...]) -> list[str]:
    """The forbidden packages a module at `own_parts` imports."""
    own = own_parts[0].removesuffix(".py")
    return sorted(
        other
        for other in imported_packages(source, own_parts) - {own}
        if not (layer_of(other) < layer_of(own) or (own, other) in PEER_EXCEPTIONS)
    )


def self_test() -> None:
    """The checker on known allowed and forbidden imports."""
    assert violations_in("from clearvisionnn.config import x", ("pipeline", "a.py")) == []
    assert violations_in("from clearvisionnn.train import x", ("models", "a.py")) == ["train"]
    assert violations_in("from ..eval import x", ("models", "a.py")) == ["eval"]
    assert violations_in("from .. import cli", ("data", "a.py")) == ["cli"]
    assert violations_in("from . import blocks", ("models", "a.py")) == []
    assert violations_in("import clearvisionnn.eval.run", ("pipeline", "sub", "a.py")) == ["eval"]


def main() -> int:
    """Print every violation; exit non-zero if there is one."""
    self_test()
    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if len(relative.parts) == 1 and relative.name == "__init__.py":
            continue
        own = relative.parts[0].removesuffix(".py")
        for other in violations_in(path.read_text(), relative.parts):
            violations.append(f"{path}: '{own}' may not import '{other}'")
    print("\n".join(violations) or "layering: ok")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
