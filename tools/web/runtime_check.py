#!/usr/bin/env python3
"""Check the page's all-GPU upscaler against the integer reference, in headless Chrome.

    python3 tools/web/runtime_check.py <model directory under web/models> [--size 320x180] ...

Real frames of a bundled clip are upscaled by the page's own GPU upscaler
(`web/js/gpu-upscaler.js`) and, from the same planes and motion field, by
`cvnn_runtime::fixed::IntegerRuntime` compiled to WebAssembly. After every frame
the network's input, its outputs, the conventional upscale and the output
picture are read back from the GPU and compared word for word. The run covers
what a stream meets: history carried from frame to frame, coding told and not
told, a restart, a lost GPU device, and (over enough frames) every grouping
of tiles the page's tuner tries. Prints a JSON report; exits non-zero on any
difference. Needs a built module (tools/web/build.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from smoke import CHROME_FLAGS, serve

SCRIPT = Path(__file__).with_suffix(".js")
PAGE_TIMEOUT_MS = 1_200_000
# A decoded video only advances in a page that may play it without a click.
AUTOPLAY = "--autoplay-policy=no-user-gesture-required"
QUANTISER = 32.0  # the quantiser the model is told on the frames it is told one


def main() -> int:
    """Print the report; exit non-zero if anything differs from the reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="an 8-bit model's directory under web/models")
    parser.add_argument("--size", default="320x180", help="size of the upscaler's input")
    parser.add_argument("--clip", default="tos-bridge")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--tiles", default="128,0", help="tile sides to check; 0: whole frame")
    parser.add_argument("--restart-at", type=int, default=12, help="frame before which to restart")
    parser.add_argument("--lose-device-at", type=int, default=20, help="frame to lose the GPU at")
    parser.add_argument("--channel", default="chrome", help="Playwright browser channel")
    args = parser.parse_args()
    width, height = (int(side) for side in args.size.split("x"))
    request = {
        "directory": args.directory, "clip": args.clip, "frames": args.frames, "width": width,
        "height": height, "tiles": [int(tile) for tile in args.tiles.split(",")],
        "restartAt": args.restart_at, "loseDeviceAt": args.lose_device_at, "quantiser": QUANTISER,
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
    print(json.dumps(report, indent=1))
    differing = sum(sum(variant["differingWords"].values()) for variant in report["variants"])
    recovered = all(used == 2 for used in report["devicesUsed"])  # each lost one and went on
    return 0 if differing == 0 and recovered and report["largestVectorInSteps"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
