#!/usr/bin/env python3
"""Check the page's 8-bit WebGPU network against the Rust reference, in headless Chrome.

    python3 tools/web/int8_check.py <model directory under web/models> [--size 320x180] ...

Real frames of a bundled clip go through the page's own session; each frame's
packed input is run by the WGSL shaders, whole-frame and tiled, with every
grouping of tiles the page may choose and with the spelt-out arithmetic used
where the packed dot product is missing, and each result is compared word for
word with the reference network compiled to WebAssembly. A recurrent model's
state is fed back, so the run covers what later frames see. Prints the run's
facts, then a JSON line per variant with its differing words (which must be 0)
and its time per frame as 10th, 50th and 90th percentiles; exits non-zero on
any difference. Needs a built module (tools/web/build.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from smoke import CHROME_FLAGS, serve

SCRIPT = Path(__file__).with_suffix(".js")
PAGE_TIMEOUT_MS = 600_000
# A decoded video only advances in a page that may play it without a click.
AUTOPLAY = "--autoplay-policy=no-user-gesture-required"


def main() -> int:
    """Print the report; exit non-zero if any variant differs from the reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="model directory under web/models, with an 8-bit export")
    parser.add_argument("--size", default="320x180", help="size of the network's input")
    parser.add_argument("--clip", default="tos-bridge")
    parser.add_argument("--frames", type=int, default=6, help="frames compared, state fed back")
    parser.add_argument("--tiles", default="0,128,64", help="tile sides to check; 0: whole frame")
    parser.add_argument("--timed-frames", type=int, default=40, help="timed runs per variant")
    parser.add_argument("--channel", default="chrome", help="Playwright browser channel")
    args = parser.parse_args()
    width, height = (int(side) for side in args.size.split("x"))
    request = {
        "directory": args.directory, "clip": args.clip, "frames": args.frames, "width": width,
        "height": height, "tiles": [int(tile) for tile in args.tiles.split(",")],
        "timedFrames": args.timed_frames,
    }  # fmt: skip
    server = serve()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, args=[*CHROME_FLAGS, AUTOPLAY])
        page = browser.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        # Any document of the site will do; the page itself would start a run of its own.
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/models/manifest.json")
        report = page.evaluate(SCRIPT.read_text(), request)
        browser.close()
    server.shutdown()
    variants = report.pop("variants")
    print(json.dumps(report))
    for variant in variants:  # one line each: there are many
        print(json.dumps(variant))
    exact = all(variant["differingWords"] == 0 for variant in variants)
    return 0 if exact and not report["errors"] and report["distinctOutputWords"] > 1 else 1


if __name__ == "__main__":
    sys.exit(main())
