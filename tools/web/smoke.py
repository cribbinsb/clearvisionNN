#!/usr/bin/env python3
"""Run the demo page unattended in headless Chrome and print its report.

The page's `?auto=<backend>` mode upscales the test pattern for a number of
frames and writes frames per second and per-step timings into `#report`.
`--size` is the size of the original; the upscaler's input is half of it.

    python3 tools/web/smoke.py [--backend wasm] [--size 640x360] [--model 0] [--frames 20]
        [--view magnify] [--noise 3] [--codec avc] [--rate bpp:0.08] [--screenshot view.png]
        [--switches 'form=half&steps=1']

Needs Playwright for Python and a built module (tools/web/build.sh).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
# "default" is whichever backend the page picks for the model: the 8-bit one
# for an 8-bit model, the best float one otherwise.
BACKENDS = (
    "default",
    "webgpu-int8",
    "webgpu-int8-network",
    "webgpu",
    "webnn-gpu",
    "webnn-npu",
    "webnn-cpu",
    "wasm",
)
VIEWS = ("original", "input", "output", "split", "magnify")
# WebGPU and WebNN are behind flags in headless Chrome on Linux.
CHROME_FLAGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebMachineLearningNeuralNetwork",
    "--use-angle=vulkan",
    "--ignore-gpu-blocklist",
    # Chrome withholds 16-bit float shaders ("shader-f16") on NVIDIA Vulkan
    # drivers older than 615.71; without them a half-precision model cannot
    # run on WebGPU and the page refuses it. Harmless on other drivers.
    "--enable-dawn-features=vulkan_enable_f16_on_nvidia",
]
PAGE_TIMEOUT_MS = 180_000


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the page without logging each request."""

    def log_message(self, format: str, *args: object) -> None:
        """Discard the access log."""


def serve() -> http.server.ThreadingHTTPServer:
    """Serve the page on a free local port, in a background thread."""
    handler = functools.partial(_QuietHandler, directory=str(WEB_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(args: argparse.Namespace) -> dict[str, object]:
    """Load the page in unattended mode and return its report."""
    server = serve()
    query = (
        f"auto={args.backend}&size={args.size}&model={args.model}"
        f"&frames={args.frames}&source={args.source}&view={args.view}&zoom={args.zoom}"
        f"&noise={args.noise}&codec={args.codec}&rate={args.rate}&keyint={args.keyint}&tell={args.tell}"
        f"&blur={args.blur}&denoise={args.denoise}&kernel={args.kernel}"
    )
    if args.switches:
        query += f"&{args.switches}"
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html?{query}"
    messages: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, args=CHROME_FLAGS)
        page = browser.new_page()
        page.on("console", lambda message: messages.append(f"{message.type}: {message.text}"))
        page.on("pageerror", lambda error: messages.append(f"pageerror: {error}"))
        page.goto(url)
        try:
            page.wait_for_selector("#report[data-done]", state="attached", timeout=PAGE_TIMEOUT_MS)
            report = json.loads(page.inner_text("#report") or page.text_content("#report") or "{}")
            if args.screenshot:
                page.locator("#view").screenshot(path=args.screenshot)
        except Exception as error:
            report = {"ok": False, "error": str(error), "status": page.text_content("#status")}
        browser.close()
    server.shutdown()
    report["console"] = [m for m in messages if not m.startswith("log")][-8:]
    return report


def succeeded(report: dict[str, object], frames: int) -> bool:
    """Whether the page processed every frame asked for and reported finite timings.

    Functional success only: the timings themselves mean nothing on a busy
    or headless machine.
    """
    timings = [report.get(key) for key in ("prepare", "network", "finish")]
    finite = all(isinstance(t, (int, float)) and math.isfinite(t) for t in timings)
    return bool(report.get("ok")) and report.get("frames") == frames and finite


def main() -> int:
    """Print the report; exit non-zero if the page failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="default", choices=BACKENDS)
    parser.add_argument("--size", default="640x360", help="size of the original; the input is half")
    parser.add_argument("--model", type=int, default=0, help="index in web/models/manifest.json")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--source", default="pattern", help="'pattern' or 'clip:<id>'")
    parser.add_argument("--view", default="split", choices=VIEWS)
    parser.add_argument("--zoom", type=int, default=4, help="magnification of the magnify view")
    parser.add_argument("--noise", type=float, default=0.0, help="sensor noise, grey levels")
    parser.add_argument("--blur", type=float, default=0.0, help="lens blur, pixels of the original")
    parser.add_argument("--denoise", type=int, default=0, help="camera denoise strength; 0 is off")
    parser.add_argument("--kernel", default="bicubic", help="area, bilinear, bicubic or lanczos")
    parser.add_argument(
        "--codec",
        default="none",
        help="none; avc, hevc, vp9, av1 (the browser's); nanocodec-h264, -hevc",
    )
    parser.add_argument(
        "--tell", type=int, default=1, help="0 withholds the quantiser from the model"
    )
    parser.add_argument("--rate", default="bpp:0.08", help="bpp:<bits per pixel> or qp:<quantiser>")
    parser.add_argument("--keyint", type=int, default=0, help="key-frame interval; 0: first only")
    parser.add_argument("--screenshot", help="write a picture of the view here when the run ends")
    parser.add_argument(
        "--switches",
        default="",
        help="diagnostic switches for the page's address (web/js/tuning.js): form=half&steps=1",
    )
    parser.add_argument("--channel", default="chrome", help="Playwright browser channel")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=1))
    return 0 if succeeded(report, args.frames) else 1


if __name__ == "__main__":
    sys.exit(main())
