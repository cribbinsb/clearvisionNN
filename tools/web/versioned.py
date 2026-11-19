#!/usr/bin/env python3
"""Give every file of the page a URL that changes whenever the page changes.

A browser may go on using a file it fetched earlier; when only some files of a
new version are fetched afresh, old and new code run together and fail. So the
page's files are addressed under `v/<build id>/`, the id being a digest of
everything under `web/`: a changed page is a set of URLs no browser has seen,
and only what names the id must never be reused. The development server sends
that with caching forbidden; a published site cannot, so there the index is a
small loader that never changes and asks for the current build's page afresh.

The development server (serve.py) does this on the fly. For publishing, this
script writes the same layout as files:

    python3 tools/web/versioned.py <output directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
INDEX = "index.html"
# The build's page as the loader fetches it, and the file that names the current build's.
PAGE, CURRENT = "page.html", "current.json"
LOADER = Path(__file__).resolve().parent / "loader.html"
VERSIONS = "v"
# Enough of the digest to tell builds apart.
ID_LENGTH = 12
_HEAD = "<head>"


def build_id(root: Path = WEB_ROOT) -> str:
    """A digest of the name, size and modification time of every file under `root`."""
    digest = hashlib.sha1(usedforsecurity=False)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:ID_LENGTH]


def versioned_index(root: Path = WEB_ROOT) -> str:
    """`index.html` with every relative URL of the page resolved under the build's own path."""
    html = (root / INDEX).read_text(encoding="utf-8")
    if _HEAD not in html:
        raise ValueError(f"{INDEX} has no {_HEAD}")
    base = f'{_HEAD}\n  <base href="{VERSIONS}/{build_id(root)}/">'
    return html.replace(_HEAD, base, 1)


def publish(output: Path, root: Path = WEB_ROOT) -> None:
    """Write the page to `output`: its files under their build's path, and what leads there.

    The index is a loader that never changes (loader.html); `current.json` names the build's
    page, which sits under the build's path with everything else.
    """
    build = build_id(root)
    page = versioned_index(root)
    shutil.copytree(root, output / VERSIONS / build)
    (output / VERSIONS / build / PAGE).write_text(page, encoding="utf-8")
    (output / CURRENT).write_text(
        json.dumps({"page": f"{VERSIONS}/{build}/{PAGE}"}) + "\n", encoding="utf-8"
    )
    shutil.copyfile(LOADER, output / INDEX)


def main() -> None:
    """Publish `web/` to the directory given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory to write, which must not hold a build")
    publish(parser.parse_args().output)


if __name__ == "__main__":
    main()
