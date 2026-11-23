# Procedures

## Set-up

```
python3 -m venv --system-site-packages .venv      # reuses the base environment's torch
. .venv/bin/activate
pip install --no-deps -e .
pip install maturin ruff mypy types-PyYAML playwright   # the `dev` extra, less what the base has
(cd crates/cvnn-py && env -u CONDA_PREFIX maturin develop --release)
```

- nanocodec must be checked out beside this repository (`../nanocodec`), at
  the revision in `tools/nanocodec.rev` (update the file when moving to a
  newer one).
- A clean environment instead installs `.[evaluate,export,dev]` with pip.
- Install with `--no-deps`: letting pip resolve torch's dependencies into the
  virtualenv pulled in a second cuDNN, which fails at the first convolution
  with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
- Rebuild `cvnn-py` after any change under `crates/`.

## Checks

`tools/ci/check.sh` runs everything that must pass.

## Running

```
clearvisionnn pipeline configs/codec.yaml               # build or update the data
clearvisionnn train configs/codec.yaml --run <name>     # add --resume to continue
clearvisionnn evaluate configs/codec.yaml --run <name>  # scores on the sealed test clips
```

- `configs/smoke.yaml` runs the whole path on a few files in a few minutes.
- Outputs: data under `<work_dir>/data`, runs under `<work_dir>/runs/<name>`
  (`log.jsonl`, `latest.pt`, `best.pt`, `eval/results.md`).
- Long jobs are launched in the background with output to a log file.

## Browser demo

```
clearvisionnn export configs/<config>.yaml --run <name> --out web/models/<dir>   # then list <dir> in web/models/manifest.json
python recipe/export.py <checkpoint.pt> web/models/<dir> --list   # a checkpoint of the recipe, straight into the page's model list
tools/web/build.sh                              # builds web/pkg (needs wasm-bindgen-cli 0.2.105)
tools/web/build-nanocodec.sh                    # builds web/nanocodec from ../nanocodec: the demo's own encoder, optional
python3 tools/web/serve.py                      # open http://localhost:8000; an edit is always what runs
python3 tools/web/smoke.py                      # unattended run in headless Chrome; prints timings
python3 tools/web/smoke.py --model 2 --backend webgpu    # a float model (index in manifest.json) on a named backend
python3 tools/web/int8_check.py recurrent-s-8bit      # the 8-bit network's shaders against the Rust reference: exact forms bit for bit, the 16-bit float form within its tolerance
python3 tools/web/runtime_check.py recurrent-s-8bit   # the whole all-GPU frame against the integer runtime, likewise
python3 tools/web/shader_bench.py <model> '[{}, {"form": "float", "span": 4}]'   # the GPU's own time for ways of writing the network's shaders (a diagnostic)
python3 tools/web/smoke.py --backend webgpu --size 1280x720 --view magnify --noise 3 --codec avc --rate bpp:0.08 --screenshot work/demo-screenshots/x.png
python3 tools/web/smoke.py --size 1280x720 --preset webcam --switches steps=1   # a ready-made input, with its grain profile on the output; `gpu finish` in the report is the pass the grain is computed in
cargo run --release -p clearvisionnn-runtime --example timing [width height]   # time each runtime step natively
```

- The page's model list (`web/models/manifest.json`) leads with the 8-bit
  models, and the page starts on the first model the browser can run: an
  8-bit one on "WebGPU 8-bit" where WebGPU works, else a float one. So
  `smoke.py` without arguments runs model 0, 8-bit, on its own backend
  (`--backend default`); a float model needs `--model <index>` and may name
  any ONNX backend (`webgpu`, `webnn-gpu`, `webnn-cpu`, `wasm`). A backend
  that cannot run the model is an error, not a fallback.
- The two browser checks are the documented proof that the page's shaders
  compute what `cvnn-runtime` defines (CLAUDE.md, "Rules"): run both after any
  change to `web/js/int8-*.js`, `runtime-shaders.js`, `gpu-runtime.js` or
  `crates/cvnn-runtime/src/{integer,fixed,grain}`. They need Chrome with a GPU, so
  `tools/ci/check.sh` cannot run them; it runs the tuner's unit test
  (`tools/web/schedule.test.mjs`). `runtime_check.py` also covers a restart, a
  lost device, grain on the output picture (every profile taken in turn frame
  by frame, `--grain`; the network's input on the following frames shows that
  none of it reached the history), and, over its default 30 frames, every grouping of tiles the
  tuner tries: a grouping that changed the output would show as a difference.
  Both check every form of the shaders: the exact ones must give the
  reference's words; the 16-bit float form, which the checks run wherever the
  GPU has `shader-f16`, must stay within `tools/web/tolerance.py` (section 8
  of docs/INTEGER.md says where the bounds come from). The reference runs in
  WebAssembly and is slow for the larger models: the medium one takes about
  ten minutes for 30 frames at 320x180, so follow a long stream at a smaller
  size (`--size 160x88 --frames 160`).
- The page chooses the form of the shaders from the GPU's maker
  (`web/js/int8-plan.js`) and says which it runs in its diagnostics line.
  Switches in the address override it, to compare forms on a device one
  reload at a time: `?form=integer|float|half`, `?span=<n>`, `?split=<n>`,
  `?words=looped|written`, `?edges=tested|bordered`, `?weights=storage`, `?packed=0`, `?tile=<n>`, and
  `?steps=1`, which times each step of a frame on the GPU apart
  (`web/js/tuning.js` says what each does). `tools/web/shader_bench.py` times
  any of them on this machine's GPU without the page.
- "WebGPU 8-bit" runs the whole frame on the GPU; "WebGPU 8-bit network, the
  rest in WebAssembly" (`--backend webgpu-int8-network`) keeps the float
  session around the integer network, for comparison.
- `recipe/export.py` writes a checkpoint of the quantisation-aware
  fine-tune (one with `tops`) as an 8-bit model, `model.i8`; `--float`
  exports the float graph of the same weights instead.
- `--size` of `tools/web/smoke.py` is the size of the original; the upscaler's
  input is half of it. `--view`, `--zoom`, `--preset`, `--noise`, `--codec`, `--rate`
  (`bpp:<bits per pixel>` or `qp:<quantiser>`) and `--keyint` set the page's
  view and "degrade the input" controls: the ready-made input first, then any
  step named by itself; `--screenshot` saves the view. There is no grain
  option: the ready-made input decides the profile (`clean` is `produced`,
  the others `webcam`).
- `tools/web/build-nanocodec.sh` builds nanocodec's browser module (read
  from the sibling checkout, or `NANOCODEC=<path>`) so the page can offer
  "nanocodec H.264/HEVC" as the codec of its simulated sender; those report
  the quantiser and type of every frame, which the page shows.
  Without it the page simply does not list them.
- `tools/web/build.sh` needs a toolchain with the `wasm32-unknown-unknown`
  target (`~/.cargo/bin` on this machine) and `WASM_BINDGEN` pointing at
  wasm-bindgen 0.2.105 if it is not on the path.
- `tools/web/publish.sh` builds both modules and publishes the page to
  GitHub Pages as the single commit of the `gh-pages` branch, every file but
  `index.html` under a path named for the build (`tools/web/versioned.py`).
  The build published before is kept beside the new one, so an open page
  keeps working until it is reloaded.
  Models under `web/models` are committed.
- `clearvisionnn export ... --half` writes a half-precision (fp16) graph. On WebGPU
  it needs the `shader-f16` feature, which Chrome withholds on NVIDIA Vulkan
  drivers older than 615.71; the page then refuses the model with a message.
  `tools/web/smoke.py` passes `--enable-dawn-features=vulkan_enable_f16_on_nvidia`
  so such models can be tested on this machine; a normal Chrome needs the same
  flag (or a newer driver) to run them.
- **Do not publish a model trained on data whose licence has not been checked.**

## Adding things

| To add | Do |
|---|---|
| A dataset | An entry in `configs/sources.yaml` with its licence and commercial-use flag |
| A degradation | A module in `pipeline/degrade/` registered with `@MODULES.register`, imported in its `__init__`, then named in a config's chain |
| A loss, metric or baseline | A registered function or class in `losses/`, `eval/metrics/` or `eval/baselines.py`, then named in the config |
| A network layout | A `Layout` subclass in `models/layouts.py` registered with `@LAYOUTS.register` |
| A network size | An entry in `models/sizes.py` |
