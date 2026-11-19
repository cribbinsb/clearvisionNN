# Architecture

The design and its reasons are in `docs/DESIGN.md`. This file says how the code
is arranged to deliver it.

## 1. Data flow

```
source video ─► pipeline ─► targets, variants, records ─► data ─► batches
                                                                     │
                 clearvisionnn_native (Rust pre-process) ◄────────────────┤
                                                                     ▼
                                              models ◄─ losses ◄─ train ─► checkpoints
                                                                     │
                                                                   eval ─► report
```

## 2. Rust crates

| Crate | Job |
|---|---|
| `cvnn-common` | `BlockGrid`, `FrameFeatures`, `MotionVector`, `Error` |
| `cvnn-preproc` | `Preprocessor`: luma frames in, `FrameFeatures` out. Motion search, stillness and noise are nanocodec's `nc-analysis`; this crate adds sub-sample refinement (`subpel`) and the reset signal (`reset`) |
| `cvnn-runtime` | `Session`: the two halves around the network call. `prepare` runs the pre-process, the conventional upscale and the history warp, and returns the network's input; `finish` blends the network's outputs into the picture and stores history |
| `cvnn-py` | `clearvisionnn_native`: `analyze_sequence` (the pre-process over a sequence, for data loading) and `RuntimeSession` (for the parity test) |
| `cvnn-web` | `Upscaler`: the session for the page, RGBA in and out |

Motion convention everywhere: the content of block `p` in the current frame
is found at `p + vector` in the previous frame, in samples of the analysed
(input-resolution) frame.

## 3. Pipeline (`python/clearvisionnn/pipeline`)

| Stage | Module | Reads | Writes |
|---|---|---|---|
| Ingest | `ingest.py` | source roots | `files.jsonl` (probe result, split) |
| Grade and plan | `segment.py` | `files.jsonl` | `segments.jsonl` (clip positions, downscale, target size) |
| Targets | `targets.py` | `segments.jsonl` | `targets/<clip>.mkv` (lossless 8-bit 4:2:0) |
| Clean | `clean.py` | targets | `clips.jsonl` (clips that passed, with measurements) |
| Variants | `variants.py` | `clips.jsonl` | `variants/<chain>/<clip>-<fingerprint>-<n>.mkv`, `variants-<chain>.jsonl` |
| Verify | `verify.py` | clips and variants | nothing; fails the run if sampled inputs do not show their targets |

- **Identity.** A file's ID is its source name, stem and a hash of its
  relative path; a clip's ID adds the clip index. Each planned clip carries a
  fingerprint of its source file's revision, its plan and the target code's
  version. Target and variant file names include it, so an existing file is
  by construction the right one, and a changed source, plan or chain makes
  new files.
- **Re-running.** Ingest and planning run every time (a new or changed source
  file is always seen). Media stages skip files that exist. Clean re-inspects
  when the planned set, the targets present or its rules change.
- **Failures.** `parallel.map_items` returns results and failures. Each stage
  names the error types it tolerates and records them in
  `<stage>.failures.jsonl`; any other error stops the stage. Variants
  tolerate none, and the data layer refuses a clip without its stored variants.
- `docs/DESIGN.md` lists cleaning before grading. Here the rejection half of
  cleaning runs on the extracted targets, where decoding is cheap.
- **Degradation chain** (`degrade/`): modules applied in order. The chain is
  split at its last offline module (an encoder round trip): the part up to it
  is stored as variants, the rest runs during training with fresh draws. A
  chain with no offline module runs entirely during training.
- Several chains share one data directory; each keeps its own variants.
- Not built yet: paired and degraded-only sources, the align stage, profiling.

## 4. Data (`python/clearvisionnn/data`)

- `ClipIndex`: the clips and variants of one split.
- `ViewMaker`: target and degraded input for a space-time window.
- `TrainingDataset`: random windows, fresh degradation, optional scene cut
  inside the sample, side-plane dropout, optional second view for the
  consistency loss.
- `features.analyse`: calls the Rust pre-process and scales its output into
  the network's side planes. The scales are part of the model's contract.
- Frames reach the network as 4:4:4 YUV in [0, 1].

## 5. Models (`python/clearvisionnn/models`)

- `blocks.RepConv`: trains as 3x3 + 1x1 + identity, exports as one 3x3 (`fuse`).
- `trunk.Trunk`: the half- and quarter-resolution body, shared by all layouts.
- `layouts`: `unet` (light input-resolution stage and skip) and `packed`
  (nothing at input resolution). Both return `fine` (2x) and `coarse` (1x) outputs.
- `per_frame.PerFrameSR`, `blend.BlendSR`: the temporal variants; both
  implement `SequenceModel.step`.
- `luma.LumaSR`: the luma-only family (`model.luma_only`): a lean input of
  luma, optional coding planes (`coding_planes`: known, QP, key frame) and,
  for `temporal: recurrent`, a hidden state, the previous output and an
  optional motion-prediction `confidence` plane (`confidence.py`). Its state
  is bounded inside the exported graph (`state.BoundedState`). The runtime
  mirrors it in `cvnn-runtime`'s `session/luma.rs`, `confidence.rs` and
  `coding.rs`; the shared constants are read through `clearvisionnn.contract`.
- `resample`: the conventional upscale and the motion warp, outside the network.
- `sizes`: the S to XXL ladder.

## 6. Training and evaluation

- `train/rollout.roll` steps a model over a sequence with gradients through history.
- `losses/`: registered terms combined by `CompositeLoss`.
- `eval/`: registered metrics (each in a family) and baselines; per-clip mean
  and worst window; bootstrap interval over clips of the gain against a
  reference baseline.

## 7. Export and the browser demo

- `export/`: `clearvisionnn export` writes the fused network core as `model.onnx`
  (plain convs, ReLU, depth-to-space; dynamic height and width) and its
  contract `model.json`, and checks ONNX Runtime against PyTorch.
- Only the core is in the graph. The upscale, warp and blend stay in
  `cvnn-runtime`, so the graph needs nothing a backend might lack.
- A model fine-tuned for 8-bit arithmetic is exported in integer form instead
  (`export/integer.py`, `export_integer`): `model.i8` and a contract with
  `network: "integer"`. docs/INTEGER.md specifies the arithmetic;
  `cvnn_runtime::integer` is its reference implementation (and the session's
  codec between float tensors and packed 8-bit planes), which the tests check
  against an independent NumPy calculation and the page's WGSL shaders are
  checked against bit for bit (`tools/web/int8_check.py`).
- For an integer model the page does not use the float session at all: the
  whole frame is integers on the GPU (`cvnn_runtime::fixed` is the reference,
  section 7 of docs/INTEGER.md the definition, `tools/web/runtime_check.py`
  the bit-for-bit check). Planes and the block motion field go up; the
  conventional upscale, the warp of the state and of the previous output, the
  confidence plane, the network, the residual add and colour conversion are
  shader passes; the state and the previous output stay on the GPU, and the
  pictures reach the view as bitmaps the GPU still holds. Only the motion
  search (`cvnn-preproc`, in WebAssembly) and the simulated sender stay on the
  processor. Float models, and browsers without WebGPU, use the session in
  WebAssembly as before.
- A contract's `network` field names the kind of network file, and with it
  what can run the model: the page offers the 8-bit backend for an integer
  model and the ONNX Runtime backends for a graph.
- `recipe/export.py` exports a checkpoint of the training recipe
  (`recipe/`) through the same path, by building the matching
  `LumaSR` and loading the recipe's weights into it.
- `web/`: a static page built around three pictures of each frame: the
  original, the input and the output. Per frame: capture the original as RGBA
  -> `Sender.send` (WebAssembly: optional sensor noise, then the 2x reduction
  the training data was made with, `cvnn_runtime::reduce`, to 4:2:0) -> an
  optional real encode and decode through WebCodecs -> `Upscaler.prepare`
  (WebAssembly) -> the network through ONNX Runtime Web on WebGPU, WebNN or
  WebAssembly -> `Upscaler.finish` -> canvas; or, for an 8-bit model, motion
  search (WebAssembly) -> everything else on the GPU -> canvas. The decoded frame is paired
  with its own original by timestamp, so the pictures compared are always of
  one moment.

| Module | Job |
|---|---|
| `js/app.js` | Controls and the frame loop: restarts on a change of source, size, model, backend or degradation, never on a change of view |
| `js/unattended.js` | The unattended mode used by the smoke test |
| `js/wasm.js` | The WebAssembly module, initialised once |
| `js/models.js` | The model list and contracts |
| `js/ort.js` | The ONNX Runtime Web build every backend shares |
| `js/backends.js` | The backends, each a whole upscaler or a network, each naming the kind of network file it runs; which this browser can use; which can run a model |
| `js/onnx-network.js` | The float graph through ONNX Runtime Web on one execution provider |
| `js/tensors.js` | How session and network exchange a frame's tensors: float arrays for a graph, packed 8-bit planes for the integer form |
| `js/gpu-device.js` | A WebGPU device for the page's own shaders; whether the browser has one |
| `js/gpu-upscaler.js` | The 8-bit upscaler entirely on the GPU: motion search in WebAssembly, the rest in the page's shaders; recovers from a lost device |
| `js/gpu-runtime.js` | A whole frame on the GPU: buffers, the passes around the network, the pictures; nothing read back unless asked |
| `js/runtime-shaders.js` | Everything around the integer network in WGSL: upscale, flow, warp, confidence, input assembly, finish, colour |
| `js/int8-backend.js` | The integer network alone on WebGPU, for the WebAssembly session to work around |
| `js/int8-model.js` | An integer model file as the shaders need it |
| `js/int8-shaders.js` | The integer network in WGSL: one shader per kind of layer, for every tiling and grouping |
| `js/int8-network.js` | Buffers, pipelines and the dispatches of one frame |
| `js/int8-schedule.js` | Choosing how many tiles go through the network together, by timing on the device |
| `js/webgpu-io.js` | WebGPU tensor traffic: input kept in a GPU buffer, only the changing channels uploaded, outputs read back through reused staging buffers; refuses a half-precision model where the GPU lacks `shader-f16` |
| `js/source.js` | Sources of the original, as ImageBitmaps with their capture times: sample clips, test pattern, camera, video file |
| `js/clip-player.js`, `js/clip-frames.js`, `js/mp4.js` | A clip or file played frame by frame, every frame in order, more slowly when the page cannot keep up; the frame-exact decoder and MP4 reader are nanocodec's |
| `js/degrade.js`, `js/degrade-presets.js` | The "degrade the input" controls as settings: a list of ready-made inputs, and each step by itself; which codecs the browser offers |
| `js/sender.js` | The simulated sender and channel: the camera (blur, noise, denoise, reduction), optional codec; pairs decoded frames with their originals |
| `js/codec.js`, `js/planar.js` | The WebCodecs encode-decode round trip; reading decoded frames as 4:2:0 |
| `js/upscaler.js` | The upscaler: session, network, session |
| `js/frame-pipeline.js` | One frame through sender and upscaler; two frames may be inside at once, the sender's work on one overlapping the GPU's on the one before |
| `js/pipeline-worker.js`, `js/remote-pipeline.js` | The worker the pipeline runs in, and the page's side of it: the page's own thread only fetches frames (as ImageBitmaps) and shows results |
| `js/view.js`, `js/magnify.js`, `js/zoom.js` | What the canvas shows: original, input (bicubic), output, input beside output, or the magnifier's four quadrants; zoom and pan of all but the magnifier |
| `js/stats.js` | Timings, kept apart for the upscaler and for the simulated sender |
