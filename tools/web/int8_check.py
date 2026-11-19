#!/usr/bin/env python3
"""Check the page's 8-bit WebGPU network against the Rust reference, in headless Chrome.

    python3 tools/web/int8_check.py <model directory under web/models> [--size 320x180] ...

Real frames of a bundled clip go through the page's own session; each frame's
packed input is run by the WGSL shaders, whole-frame and tiled, in every form
and shape the shaders can take (`web/js/int8-forms.js`, `int8-shaders.js`) and
with every grouping of tiles the page may choose, and each result is compared
with the reference network compiled to WebAssembly. A recurrent model's state
is fed back, so the run covers what later frames see.

An exact form must give the reference's words. The approximate form (16-bit
float, checked where the GPU has it) is measured instead: at the output, and
layer by layer on the whole frame, each layer both given the exact layer's
input (`alone`) and run on from its own (`through`); `tolerance.py` holds the
bounds. Prints the run's facts, then a JSON line per variant and per layer:
differing words, the largest difference in 8-bit steps, the share of samples
that differ, and PSNR against the exact result (null: identical), with
`outside` naming whatever broke its bound. Exits non-zero if anything did.
Needs a built module (tools/web/build.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from smoke import CHROME_FLAGS, serve
from tolerance import layer_failures, output_failures

SCRIPT = Path(__file__).with_suffix(".js")
PAGE_TIMEOUT_MS = 600_000
# A decoded video only advances in a page that may play it without a click.
AUTOPLAY = "--autoplay-policy=no-user-gesture-required"


def main() -> int:
    """Print the report; exit non-zero if any variant is outside what its form allows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="model directory under web/models, with an 8-bit export")
    parser.add_argument("--size", default="320x180", help="size of the network's input")
    parser.add_argument("--clip", default="tos-bridge")
    parser.add_argument("--frames", type=int, default=6, help="frames compared, state fed back")
    parser.add_argument("--tiles", default="0,128,64", help="tile sides to check; 0: whole frame")
    parser.add_argument("--channel", default="chrome", help="Playwright browser channel")
    args = parser.parse_args()
    width, height = (int(side) for side in args.size.split("x"))
    request = {
        "directory": args.directory, "clip": args.clip, "frames": args.frames, "width": width,
        "height": height, "tiles": [int(tile) for tile in args.tiles.split(",")],
        "differences": SCRIPT.with_name("differences.js").read_text(),
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
    layers = report.pop("layers")
    print(json.dumps(report))
    failed = bool(report["errors"]) or report["distinctOutputWords"] <= 1
    for variant in variants:  # one line each: there are many
        variant["outside"] = output_failures(variant["form"], variant)
        print(json.dumps(variant))
        failed = failed or bool(variant["outside"])
    for layer in layers:
        layer["outside"] = layer_failures(layer["alone"])
        print(json.dumps(layer))
        failed = failed or bool(layer["outside"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
