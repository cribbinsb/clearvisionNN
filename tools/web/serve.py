#!/usr/bin/env python3
"""Serve the demo page for development so that an edit is always what runs.

The page's files are addressed under a path that changes with every edit
(versioned.py), and nothing may be cached: a browser can then never run an
old copy of one file beside new copies of the others.

    python3 tools/web/serve.py [--port 8000]
"""

from __future__ import annotations

import argparse
import functools
import http.server
from http import HTTPStatus

from versioned import INDEX, VERSIONS, WEB_ROOT, versioned_index

DEFAULT_PORT = 8000
_VERSIONED = f"/{VERSIONS}/"


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the index with the current build's path, and every build's path from `web/`."""

    def do_GET(self) -> None:
        """Answer the page itself from the files as they are now; anything else as a file."""
        if self.path.split("?")[0] not in ("/", f"/{INDEX}"):
            super().do_GET()
            return
        body = versioned_index().encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def translate_path(self, path: str) -> str:
        """Map `/v/<any build>/<file>` to `<file>`: there is only the current build on disk."""
        if path.startswith(_VERSIONED):
            path = "/" + path[len(_VERSIONED) :].partition("/")[2]
        return super().translate_path(path)

    def end_headers(self) -> None:
        """Add the header that forbids reuse, then finish the headers."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    """Serve `web/` until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    port = parser.parse_args().port
    handler = functools.partial(_Handler, directory=str(WEB_ROOT))
    with http.server.ThreadingHTTPServer(("", port), handler) as server:
        print(f"http://localhost:{port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
