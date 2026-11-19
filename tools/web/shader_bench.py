#!/usr/bin/env python3
"""Time ways of writing the 8-bit network's shaders on this machine's GPU, in headless Chrome.

    python3 tools/web/shader_bench.py <model directory under web/models> '<variants as JSON>' ...

Each variant is a JSON object of options for `web/js/int8-network.js`
(`form`, `span`, `split`, `edgeTests`, `uniformWeights`, `packedDot`, `tile`);
`{}` is the integer form, a pixel to an invocation. For example

    python3 tools/web/shader_bench.py <model> '[{}, {"span": 2}, {"form": "float", "span": 4}]'

Prints a JSON line per variant with the GPU's own milliseconds per frame of the
network (timestamp queries): the least, the median and the 90th percentile
over the rounds. The variants take turns within every round, so a busy GPU
slows all alike; compare the least times, and read the spread between least
and median as the noise. A diagnostic, not a check: it has no pass or fail.
Needs a built module (tools/web/build.sh).
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


def main() -> int:
    """Print the timings; exit non-zero if the GPU reported an error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="model directory under web/models, with an 8-bit export")
    parser.add_argument("variants", help="JSON list of option objects for int8-network.js")
    parser.add_argument("--size", default="640x360", help="size of the network's input")
    parser.add_argument("--burst", type=int, default=8, help="frames timed together")
    parser.add_argument("--rounds", type=int, default=15, help="timings kept per variant")
    parser.add_argument("--channel", default="chrome", help="Playwright browser channel")
    args = parser.parse_args()
    width, height = (int(side) for side in args.size.split("x"))
    request = {
        "directory": args.directory, "width": width, "height": height,
        "variants": json.loads(args.variants), "burst": args.burst, "rounds": args.rounds,
    }  # fmt: skip
    server = serve()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, args=CHROME_FLAGS)
        page = browser.new_page()
        page.set_default_timeout(PAGE_TIMEOUT_MS)
        # Any document of the site will do; the page itself would start a run of its own.
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/models/manifest.json")
        report = page.evaluate(SCRIPT.read_text(), request)
        browser.close()
    server.shutdown()
    print(json.dumps({"gpu": report.get("gpu"), "errors": report["errors"]}))
    for result in report["results"]:
        print(json.dumps(result))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
