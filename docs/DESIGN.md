# clearvisionNN: fast video super-resolution

Design thoughts and literature review, revised 2026-11-17 after a second research pass and a look at `../nanocodec`. Nothing here has been built or measured yet.

## 1. Summary

**Recommendation.** Build a fixed-2x, causal, convolutional U-Net that fuses the current frame with motion-compensated history through a learned per-pixel blend. Feed it the side information we can get almost for free from nanocodec: motion vectors, match quality, static and noise maps, and (after a decode) QP and block modes. Build the large version first, natively on the RTX 5090, and cut it down to browser size afterwards.

The main points behind that:

- **Design the top model first; cut down only where hardware demands it.** The top model is sized by where quality stops improving, not by the weakest device: about 5M parameters to start, larger if quality is still rising. It is used to settle data, losses and flicker behaviour. Smaller models for weaker hardware are distilled and narrowed from it afterwards. The only constraints it carries from day one are being causal and using operators that every target can run, so that cutting down is a shrink and not a redesign.
- **Motion and pre-analysis are cheap, and nanocodec already has them.** Its WebAssembly decoders run in about 2 ms per 720p frame on one thread and can emit per-block motion vectors, QP and prediction modes. Its source analysis gives an 8x8 motion field with match errors, camera motion, sensor-noise level and static maps. That removes the two constraints the first draft of this document assumed: "no alignment" and "blind to codec metadata".
- **Learned blending with reprojected history is the proven route to temporal stability.** It is what shipped game upscalers do (Arm NSS, FSR 4, the 2026 PSSR) and what recursive video denoisers do (EMVD). Sony reports that predicting the output directly only held up for a few frames regardless of model size, and that an explicit learned blend fixed it.
- **The degradation model is still most of the work.** Quality in both cases depends on how well training data matches the real camera and the real encoder. Nanocodec can be the encoder in the training pipeline, so training and deployment see the same streams and the same side information.
- **Plain convolutions remain the right building block at this size.** At matched latency, light transformers and state-space models are 20-200x slower for 0.5-1 dB; shipping video and game upscalers are all small conv U-Nets.
- **Compute is a secondary question.** Our own experience (a 50M-parameter YOLO at 640x640 and 30 fps in a phone browser) puts the browser at about 80 GMACs per frame; the few published measurements come out lower (20-45). It decides how far the cut-down has to go, not what the top model looks like.
- **Licensing is manageable.** Enough permissively licensed code and CC BY or CDLA data exists; most standard SR training sets are research-only.

## 2. Requirements as understood

| Item | Assumption |
|---|---|
| Scale | Fixed 2x |
| Case A | Live camera video after the ISP; a crop of an HD/4K frame is enlarged. Soft from ISP denoising and plastic-lens optics |
| Case B | Video after a live H.264/H.265 encode and decode at moderate bitrate; compression artefacts present. Any encoder and any decoder, not nanocodec only |
| Generality | Start with a general, non-specific solution. Data sources and degradation models are swappable, so a specific camera or encoder can be substituted later |
| Structure | A pre-processing stage in code feeding a neural network. Rust natively, the same Rust as WebAssembly in the browser; Python for model development |
| Browser targets | 360p to 720p and 540p to 1080p, on a very modern integrated GPU or high-end phone, via WebGPU |
| PC targets | Same network family at wider settings |
| Latency | Causal: no future frames |
| Temporal | Flicker is a first-class acceptance criterion |
| Models | Separate weights per case; same architecture |
| Commercial use | Development may use non-commercial data. A shippable model must be retrainable on commercially usable data alone, so every data source carries a licence tag |
| Training | Local, RTX 5090 |
| Code hygiene | No reading source of projects that are not permissively licensed. Nanocodec (ours, BSD-3-Clause) is freely reusable |

## 3. What nanocodec gives us

This is from reading `../nanocodec` docs and the public interfaces of `nc-analysis`, `nc-common::stream_stats` and `nc-web`. I did not run anything.

### 3.1 Decoder side (case B)

All six decoders compile to WebAssembly and, in the `analysis` build, push a `BlockRecord` per coding, prediction and transform block:

| Field | Use for super-resolution |
|---|---|
| Motion vectors and reference indices (quarter-sample or finer) | Align history to the current frame |
| Prediction type: intra, inter, skip, merge | Skip blocks are copies of the previous frame; intra blocks are fresh. A strong prior for the blend weight |
| QP and QP delta per block | Local compression strength, for conditioning |
| Non-zero coefficient count and bits per block | Where the encoder spent bits; where residual was dropped |
| Block rectangles and transform sizes | Where blocking edges can occur |
| Picture type | I-frame pulsing |

Measured decode cost in the nanocodec results, one thread, 720p at 600 kbps on a Ryzen 5950X: H.264 1.7-1.9 ms and HEVC 2.2-2.3 ms per frame in WebAssembly. With analysis compiled in but switched off it costs one branch per block; the cost with it switched on I did not find a figure for.

Case B is general, so these records are an optional extra, not a dependency. The nanocodec decoders decode other encoders' streams, so wherever software decode is acceptable the records are available whatever produced the stream. With a hardware decoder (WebCodecs, a plain `<video>`, a native hardware path), only the picture type and perhaps an average QP are available. The baseline design therefore relies on pixel-domain analysis of the decoded frames (3.2), which works everywhere, and treats decoder records as additional inputs that may be absent.

### 3.2 Source analysis (both cases)

`nc-analysis` computes, from pictures alone, once per frame:

| Output | Use for super-resolution |
|---|---|
| 8x8 motion field, whole-sample vectors, with the SAD at each vector | Alignment, plus a per-block confidence for rejecting bad matches |
| 16/32/64 aggregated fields with whole-versus-split SADs | Whether an area moves as one |
| Global camera motion (translation and zoom) | Fallback vector; pan detection |
| Per-8x8 mean, variance, change against the previous frame, static flag | Where history can be trusted fully |
| Persistent stillness map | Long-term accumulation on fixed cameras |
| Running sensor-noise sigma | Strength control; separating noise from texture |
| Frame mean luma | Lighting jumps, for state reset |

Static detection scales with the measured noise, so sensor noise on a still scene does not read as motion. That is exactly the discrimination a temporal super-resolver needs.

Cost: the whole encoder, of which this analysis is a part, runs 720p at 109-235 fps on one Zen 5 core, and pre-analysis is about 7% of the fastest H.264 preset's instructions. I did not find a separate WebAssembly timing for the analysis alone; it needs one measurement.

### 3.3 Training

- **Encoder in the loop.** Nanocodec encodes low-delay CBR in six formats far faster than real time, so it can generate the case B training data, with exactly the settings the deployed stream uses.
- **Same side information in training and deployment.** The records and analysis fields are produced by the same code in both places, so there is no simulation gap for the conditioning inputs.
- **Other encoders are needed too.** Case B is general, so training also uses x264, x265 and hardware encoders, with nanocodec as one encoder among them.

### 3.4 Limits to keep in mind

- **Bitstream vectors are not true motion.** The encoder picks them to save bits. In one published ablation (CDA-VSR), bitstream vectors alone were 0.17 dB behind vectors plus a small learned refinement. Running the source analysis on decoded frames, or a small learned refinement, covers this.
- **Analysis vectors are whole-sample on an 8x8 grid.** At 2x that is a two-pixel step in the output. The network has to absorb sub-sample error, or a refinement stage is added.
- **The camera case has no bitstream,** so it uses 3.2 only.

## 4. System structure

```
frames ──► pre-process (Rust: native or WebAssembly, CPU) ──► motion field + side planes ─┐
   │                                                                                      ▼
   └────────────────────────────────────────────────► GPU: warp history ──► network ──► blend ──► output
```

- **One Rust crate for the pre-process,** built on `nc-analysis` and nanocodec's SIMD kernels, compiled natively (AVX2, AVX-512, NEON) and to WebAssembly (SIMD128). Nanocodec already builds this way, so this is known to work.
- **Python bindings to that same crate** for training. The network must see exactly the same motion fields and side planes in training as in deployment; binding the real code, instead of re-implementing it in Python, guarantees that.
- **Python and PyTorch for the model** and for trying out new pre-process ideas. An idea that proves useful is ported into the Rust crate before any model that depends on it is treated as final.
- **GPU stages** (warp, network, blend) run in WebGPU in the browser. Natively they can run through ONNX Runtime plus a small amount of GPU code, or through the same WGSL shaders using Rust's `wgpu`, which would give one codebase for both targets. To be decided in phase 0.
- **CPU-side frames.** The pre-process needs luma on the CPU. With software decode or camera capture it already is; with hardware decode in the browser it costs one small copy per frame (about 0.5 MB at 540p).

## 5. Repository structure and working rules

Modelled on nanocodec's rules (`../nanocodec/CLAUDE.md`, `docs/ARCHITECTURE.md`), adapted for a project that is half Rust and half Python. Clean structure is a requirement, not a preference: when a quick hack conflicts with it, the structure wins.

### 5.1 Layout

```
crates/
  cvnn-common/     frame and side-plane types, parameter registry, errors
  cvnn-preproc/    the pre-process: motion field and side planes, built on nc-analysis
  cvnn-runtime/    per-frame state, reset logic, warp and blend, the model-runner trait
  cvnn-py/         Python bindings to cvnn-preproc and cvnn-runtime
  cvnn-web/        browser bindings (WebAssembly)
  cvnn-cli/        native command: upscale a file or stream, benchmarks
python/clearvisionnn/
  config/        typed run configuration, the source registry
  pipeline/      ingest, clean, align, grade, targets, profile, degrade: one module per stage
  data/          dataset and sampling over the pipeline's outputs
  models/        the network definitions (single source for every export)
  losses/  train/  eval/
  export/        ONNX and shader generation, parity checks
configs/         run configurations: data sets, degradation profiles, model sizes
web/             demo page and browser tests
tools/ci/        structure checks    tools/dev/   developer scripts
docs/            ARCHITECTURE, STATUS, PROCEDURES, handoff/, analysis/, results/
CLAUDE.md        the working rules below
```

Nanocodec's crates (`nc-common`, `nc-dsp`, `nc-analysis`, and a decoder where wanted) are dependencies, not copies. Anything missing from them is added there, not forked here.

### 5.2 Rules

Structure:

- **One-way dependencies, checked by a script in CI.** Rust: front ends → `cvnn-runtime` → `cvnn-preproc` → `cvnn-common` → nanocodec crates. Python: `train` and `eval` → `models`, `losses`, `data` → `pipeline` → `config`. Nothing imports upward or sideways.
- **Pipeline stages talk only through their outputs.** Each stage reads and writes artefacts with a declared schema and a manifest. No stage calls another's internals, so any stage can be replaced.
- **Everything swappable is a registered component:** data sources, degradation modules, model variants, losses, metrics. A configuration names them; code never hard-wires one.
- **No global mutable state.** State lives in explicit context structs and objects; configuration is passed in, typed, never read from globals or the environment.
- **`unsafe` is forbidden in every clearvisionNN crate.** SIMD lives in nanocodec's `nc-dsp`.

One source of truth:

- **One implementation of anything the network consumes.** The pre-process exists only in Rust. Python calls it through bindings and never re-implements it. A test checks native, WebAssembly and Python-bound outputs are bit-identical on fixed clips.
- **One model definition.** The PyTorch module is the source; ONNX files and shaders are generated from it, and a test checks all three agree within a stated tolerance.
- **One runtime path.** The command-line tool, the browser demo and the evaluation harness run frames through the same `cvnn-runtime` logic.

Code:

- **Small modules and functions, one job each.** Every file starts with a doc comment saying what it does and where it fits. Comments explain why, not what.
- **No magic numbers.** Every threshold and heuristic is a named, registered parameter with a default, a range and a stated purpose.
- **No dead code, no commented-out code,** no second "legacy" or test-only path for something that has a real one.
- **Tooling:** `cargo fmt`, `cargo clippy -D warnings`, `cargo test`; `ruff`, a type checker and `pytest` for Python. All must pass before a change lands.
- **Borrowed code** only from permissively licensed projects, reimplemented and credited in `THIRD_PARTY.md`. Ideas from papers are always fine.

Evidence:

- **Every run is reproducible:** it records its configuration, code commit, data manifest and seed.
- **Never overfit.** A sealed test set is used only at milestones. Design decisions are made on held-out clips with an interval over clips, not on a hand-picked few.
- **Correctness and evaluation are separate.** Parity and unit tests gate landing; quality measurements never do double duty as correctness tests.

Housekeeping:

- **Never write to `datasets/`.** All derived data goes to a separate work directory with a size budget that the pipeline enforces.
- **Docs stay current:** `PROCEDURES.md` (how to build, run and measure), `handoff/` notes per area.

## 6. Literature and project review

Two research passes were run, reading papers, READMEs and licence files only. Numbers are as reported by the cited sources. Items not confirmed from a primary source are marked (unverified).

### 6.1 Building blocks at matched latency

The first pass looked at tiny networks; the second re-asked the question at 0.5-5M parameters.

- **Plain convolutions hold up.** The NTIRE 2026 efficient-SR winner is a pure CNN, and the report says the bottleneck is "memory bandwidth rather than arithmetic complexity". An entry with about half the winner's FLOPs ran about 3.5x slower.
- **Light transformers and state-space models are not competitive on speed.** At under 1M parameters, 2x to 720p output on an RTX 4090: SwinIR-light 1410 ms, ATD-light 734 ms, MambaIRv2-light 580 ms, HiT-SRF 268 ms, ESC-light 73 ms (ESC paper, ICCV 2025).
- **Re-parameterisation is universal.** Multi-branch training blocks folded to single 3x3 convs win the efficiency challenges through 2026, including the Mobile AI 2026 NPU track.
- **Depthwise designs are slower than their parameter counts suggest** on GPUs. In Mobile AI 2026 the one depthwise entry ran 46.5 ms against 4.33 ms at similar size (cause not isolated).

One independent latency benchmark (traiNNer-redux; RTX 4090, 640x480 input, 2x; PSNR on Urban100 from mixed training sets, so indicative only):

| Architecture | Type | Params | FPS | PSNR |
|---|---|---|---|---|
| compact (SRVGG) | Plain conv, full resolution | 0.60M | 246 | 31.72 |
| ArtCNN r8f64 | Plain conv | 0.93M | 160 | 32.10 |
| SPAN | Conv + parameter-free attention | 2.6M | 81 | 32.24 |
| PLKSR tiny | Partial large kernel | 2.35M | 54 | 32.58 |
| SAFMN | Depthwise multi-scale | 0.23M | 44 | 31.84 |
| ArtCNN r16f96 | Plain conv | 4.1M | 31 | 32.81 |
| ESRGAN with input unshuffle | Dense conv at half resolution | 16.7M | 25 | 33.08 |
| ESRGAN without unshuffle | Dense conv at full resolution | 16.7M | 5.2 | 33.41 |
| OmniSR / HiT-SRF | Attention | 0.78M / 0.85M | 7.7 / 4.3 | 33.30 / 33.13 |
| ATD-light / SwinIR-S | Transformer | 0.80M / 0.91M | 1.0 / 1.0 | 33.27 / 32.76 |

The two ESRGAN rows are the direct evidence for a half-resolution trunk: moving the trunk to half resolution costs about 0.3 dB at equal parameters and gains nearly 5x in speed.

### 6.2 What shipping upscalers do

| Product | Publicly known | Confidence |
|---|---|---|
| NVIDIA RTX Video Super Resolution (v2.1, Feb 2026) | U-Net CNN, single RGB frame in and out, four variants, 0.61-2.59 ms at 1080p input on an RTX 4090 | Primary (NGC model card) |
| Arm NSS (2025) | Four-level U-Net, 540p to 1080p. Inputs include reprojected hidden features and a disocclusion mask. Outputs per pixel: a filter kernel, accumulation and rectification weights, and a hidden state. Trained on about 100-frame sequences with a spatio-temporal loss | Primary (Arm blog, model card) |
| AMD FSR 4 | About 100k parameters, U-Net with 16/32/64 channels; four reprojected recurrent state channels in and out; outputs filter parameters and a blend factor; 8-bit arithmetic | Unverified (third-party reading of leaked code; not to be read by us) |
| Sony PSSR, 2026 version | Replaced direct colour prediction with predicted blend weights and an explicit blend. About 90% of parameters at 540p or below. Luma-only input to the weight predictor "caused significant ghosting"; a chroma-distance input was added | Primary (SIGGRAPH 2026 talk) |
| NVIDIA DLSS 4 / 4.5 | Vision transformer with layers fused into custom kernels; sizes not published | Vendor |
| Microsoft Edge VSR (2023) | Trained on about 1M clips with real H.264 at CRF 15-35; about 1 MB model at about 20 ms per 360p frame on an RTX 2060, plus a 0.1 MB model | Primary (blog) |
| Meta, phones | Luma-only, pruned, 8-bit, under 10 ms per frame at 540p to 1080p | Conference talk |
| Microsoft Teams (2025) | 360p to about 720p on the NPU | Primary (blog) |

The common pattern: a small conv U-Net, three or four levels, most parameters at low resolution. The temporal ones reproject history with motion vectors and have the network predict a blend weight and a few recurrent channels, not the output directly.

### 6.3 Online video SR and alignment

All figures 4x, 180p input, on H.264-compressed REDS4, retrained under the same settings by the CDA-VSR authors (CVPR 2026):

| Method | Alignment | Params | GMACs | PSNR at CRF 18 / 28 |
|---|---|---|---|---|
| RRN | None | 3.4M | 193 | 27.10 / 24.96 |
| SSL-uni | Flow + warp | 2.2M | 92 | 27.54 / 25.09 |
| KSNet-uni | Multi-flow deformable | 3.0M | 148 | 27.58 / 25.12 |
| TMP | Propagated offsets | 3.1M | 176 | 27.68 / 25.17 |
| BasicVSR-uni | Flow + warp | 4.0M | 254 | 27.63 / 25.13 |
| CDA-VSR | Bitstream vectors + deformable refinement | 3.3M | 78 | 27.76 / 25.30 |

- **Alignment buys 0.2-0.5 dB** over unaligned recurrence at light compression, shrinking to about 0.1-0.3 dB at heavy compression. The comparison is confounded by RRN's age.
- **Bitstream vectors give the best result at the lowest cost** in this table. CDA-VSR's ablation: flow-only 27.73 dB at 15.5 ms; vectors-only 27.59; learned offsets only 27.35; vectors plus refinement 27.76 at 10.8 ms.
- **Evidence gap.** None of these is at 2x, on real camera footage, or compared with a per-frame network of equal cost.
- **A warning.** In Microsoft's ICME 2025 conferencing challenge (H.265, causal, ranked by viewing tests), per-frame models won both tracks; the only recurrent entry ranked below bilinear, and the organisers report flicker or error propagation beyond about 200 frames for some approaches. Recurrence done badly is worse than none.

### 6.4 Recursive blending and long-sequence stability

| Finding | Source |
|---|---|
| Fusion as "a recursive convex combination" of the current frame and the previous fused frame, with network-predicted weights, runs on mobile chips. Feeding back the fused image worked better than feeding back the final output | EMVD (CVPR 2021) |
| Adding motion compensation to that scheme improves it | Improved EMVD (2022) |
| With an explicit weighted sum, "the gradient to history color is a single multiply. This gating structure is a GRU"; a blend "cannot exceed the convex hull of the input samples" | PSSR talk (2026) |
| Recurrent VSR trained on short clips diverges on long, low-motion sequences; a Lipschitz constraint on the recurrent path fixes it | Chiche et al. (MRVSR) |
| The divergence is a train-versus-inference gap in hidden states; initialising training clips from stored states closes it | Mori & Ukita (2024) |
| Training clip length matters more than batch size for recurrent real-world VSR | RealBasicVSR |
| Cleaning the hidden state before reuse removes propagated artefacts | FastRealVSR |
| A spatial-only GAN gives no temporal supervision, so hallucinated detail flickers | TecoGAN |

Named failure modes of blending: ghosting, disocclusion artefacts, and lag on slow changes such as lighting. The convex bound prevents divergence but not these.

### 6.5 Compressed video

- **Train on real codec output** at mixed quality, with some uncompressed samples (COMISR and everything since).
- **Do not deblock first.** A separate artefact-removal stage before SR lowered quality in COMISR's test. One network should do both, as RTX Video does.
- **Quality pulses with the GOP** (MFQE, NTIRE 2021). Frame type and QP conditioning is the published remedy (CAVSR); one 2025 paper reports slice type substitutes for per-frame quality level with under 0.03 dB loss.
- **Real stream pairs beat synthetic ones.** StreamSR (ICLR 2026) found training on real platform encodes beat bicubic-synthetic data, and that fine-tuning data moved viewing scores as much as architecture did.
- **Browser metadata without nanocodec:** frame type through WebCodecs and WebRTC; average QP through WebRTC statistics; motion vectors nowhere.

### 6.6 Degradation modelling

- **Standard blind pipelines:** BSRGAN and Real-ESRGAN (blur, resize, noise, JPEG, ringing), extended to video by RealBasicVSR.
- **Calibration beats breadth.** A CVPR 2026 paper finds that calibrating blur and noise for a specific device beats large pools of arbitrary degradations. We have one known camera.
- **Gap in the literature.** Neither pass found a paper modelling ISP spatial or temporal noise reduction, or sharpening halos, as an SR degradation. Section 9.2 is a proposal.
- **Lens correction** for arbitrary lenses is an active area (OmniLens, UniCAC). With one known lens, a measured PSF is simpler.
- **No new paired real video SR datasets** from 2024-2026 were found.

### 6.7 Perceptual quality

- **No credible route to GAN-like texture at about 1M parameters.** The AIM 2025 efficient perceptual SR winners are about 20x our compute, and the organisers note the winner visibly damaged a real crop while scoring top. One-step diffusion students are 100M+ parameters.
- **A modest route exists.** Charbonnier plus a small VGG term plus a Sobel edge term gave a 0.37M per-frame network the best viewing score in StreamSR, ahead of NVIDIA's VSR as measured there. Its temporal behaviour was not measured.

### 6.8 Evaluation practice

- **No flicker metric validated against human judgement of SR output was found.** Recent challenges rank by viewing tests.
- **Closest candidates,** both full-reference and from rendering research: ColorVideoVDP (MIT) and CGVQM (Apache-2.0).
- **VMAF rewards sharpening.** In AIM 2024 the best-VMAF entry had the lowest PSNR of the top three.

## 7. Compute budget

Units: one MAC is one multiply-add. Ultralytics "GFLOPs" count each multiply-add as two, so a YOLOv8l (43.7M parameters, 165 GFLOPs at 640x640) is about 83 GMACs per frame. An earlier draft of this document got that wrong.

| Evidence | Implied budget per frame at 30 fps |
|---|---|
| Our own experience: a 50M-parameter YOLO at 640x640 runs at 30 fps in the browser on a phone | about 80 GMACs |
| Published: one 128-channel 3x3 conv at 512x512 on Intel Lunar Lake, ONNX Runtime WebGPU, 42.8 ms | about 30 GMACs |
| Published: a large segmentation network and small upscalers on Apple M4 in the browser | 23-45 GMACs |
| Published, phones | No compute-bound browser measurement exists |

The published points are few, on older runtime versions or single kernels, and do not include any YOLO of this size, so our own measurement takes precedence. Two things to check when carrying it over to super-resolution: activation maps here are as large as a YOLO's largest only if the trunk stays at half resolution or below, and one published tablet test throttled from 57 ms to 185 ms per frame after about 10 s of continuous inference.

Working position: the top model is not sized to this budget. It is sized by quality, and should run in real time on a good discrete GPU. The figures here only say which cut-down each class of device gets: about 80 GMACs per frame at 540p input in a phone or integrated-GPU browser, confirmed or adjusted by measurement in phase 0.

Network sizes (3x3 convs; MACs are parameters times pixels at each scale, plus a small head):

| Size | Half-res stage | Quarter-res stage | Params | 360p in | 540p in | 1080p in |
|---|---|---|---|---|---|---|
| XXL (if quality is still rising at XL) | 16 x 128 ch | 16 x 256 ch | ~12M | 285 G | 640 G | 2560 G |
| XL (top model, first build) | 12 x 96 ch | 12 x 192 ch | ~5M | 120 G | 270 G | 1080 G |
| L | 8 x 64 ch | 8 x 128 ch | ~1.5M | 40 G | 90 G | 360 G |
| M | 12 x 48 ch | 8 x 96 ch | ~0.9M | 27 G | 60 G | 240 G |
| S | 8 x 32 ch | 8 x 64 ch | ~0.4M | 11 G | 25 G | 100 G |

**MACs are not the whole cost.** On some accelerators time follows the number of pixels in thin layers, not the arithmetic: a layer with one to four channels at full resolution is padded to the hardware's vector width, and a pixel-shuffle of a thin tensor can cost as much as the body of a small network. One measured case ran the same compute about four times faster once every layer worked on wide tensors at half resolution. Phase 0 therefore times thin full-resolution layers and the pixel-shuffle separately, not just whole networks.

XL at 540p input is real-time territory for a high-end discrete GPU (the 4K challenge timings imply about 250 GMACs per frame at 30 fps on an RTX 4090 for tiny networks, which is probably pessimistic for this shape). On an 80 GMAC budget, L is the cut-down for phones and integrated GPUs at 540p, with M and S below it; at 360p input there is room for something between L and XL. Motion estimation and analysis run on the CPU in WebAssembly and do not come out of this budget.

## 8. Proposed design

### 8.1 Per-frame data flow

```
previous output, previous state ──► warp by motion field ──┐
                                                           ▼
current LR frame ──────────────────────────────────────► network ──► detail estimate, blend weight, new state
side planes (match error, static, noise, QP, block type) ──┘              │
                                                                          ▼
conventional 2x upscale of current frame ──► + detail ──► blend with warped history ──► output
```

1. **Motion field.** Case B with nanocodec decode: bitstream vectors. Otherwise: `nc-analysis` block search on the incoming frames. A small learned refinement at quarter resolution is an optional extra.
2. **Warp.** The previous output and a small state are warped to the current frame. This runs outside the network, as a small GPU pass, so the exported model needs no sampling operator.
3. **Network.** Sees the current frame, the warped history and the side planes.
4. **Blend.** The output is a per-pixel convex combination of the warped history and the new estimate, with the weight predicted by the network. Each output lies between the aligned history and the new estimate. That bounds one step, not the recursion, since the estimate depends on earlier output; long-run stability is measured on the flicker suite, not assumed.

### 8.2 Network

- **Three-level conv U-Net.** Input resolution (light), half and quarter resolution (most of the capacity). A third reduced scale is worth testing; shipping designs use three or four levels.
- **Downsampling by stride-2 convolution,** not a pixel-unshuffle operator, which is missing or slow in ONNX Runtime's WebGPU backend.
- **Skip from an input-resolution encoder layer into the head,** so fine detail does not depend on the pixel-shuffle alone.
- **Plain 3x3 convs with ReLU,** re-parameterised in training. Alternatives (large-kernel or NAFNet-style blocks) are worth a test at the lowest scale only, where they are cheap. One ablation found tanh better than ReLU in a small network; cheap to test.
- **Pixel-shuffle tail, global residual** over a conventional upscale. Low frequencies, brightness and colour pass through untouched.
- **Packed variant, to compare.** Nothing runs at input resolution: the pre-process packs each 2x2 block of input pixels into channels, the network works at half resolution and below, and its last layer emits the 4x4 output pixels of each block as channels (16 per colour plane). The final reorder to pixels is done in the pre-process code or the display pass, where it is nearly free. This removes the thin full-resolution layers of section 7 at the price of the input-resolution skip; which wins is a measurement of both speed and quality.
- **Enough depth to see context.** A short stack of plain convs at one scale sees only a few tens of pixels, and added width then buys little. The quarter-resolution stage is what gives this design its reach, and the size ladder is run early to confirm that quality does rise with size on our data.
- **No ReLU on narrow layers near the output.** A ReLU on a layer of only a few channels can die and silently reduce the network to its global residual. The last activation sits on a full-width layer, and training checks that the final-stage channels are alive.
- **Luma or RGB output: test both.** The cheap baseline is luma through the network with chroma upscaled conventionally. Luma-only halves the tail and avoids chroma flicker, and Meta ships it. Against: it cannot repair chroma blocking in case B, RTX Video is RGB, and Sony found luma-only input to the blend predictor caused ghosting. Whatever the output, the blend predictor gets chroma.
- **4:2:0 input.** Nanocodec decoders output I420, and luma rearranged to half resolution plus the two chroma planes is a natural six-channel input to the half-resolution stage. Worth testing against RGB input.
- **Small state.** Four to eight channels; FSR 4 reportedly uses four.
- **Operator whitelist.** Conv (including stride 2), Relu, Add, Mul, Sigmoid, Concat, Clip, DepthToSpace. Avoided: PRelu, ConvTranspose, SpaceToDepth, GridSample, deformable conv, LSTM/GRU, 3D conv, normalisation layers.

### 8.3 Side planes

All are cheap, low-resolution maps upsampled to the stage where they enter.

| Plane | Source | Purpose |
|---|---|---|
| Match error at the chosen vector, raw and divided by the noise level | Analysis SAD, or decoder residual energy | Reject bad history. In noise units, the plane says directly whether a mismatch is more than noise can explain |
| Static / persistent-still flags | Analysis | Allow long accumulation |
| Noise sigma | Analysis | Strength; tell noise from texture |
| QP | Decoder records | Compression strength, per block |
| Prediction type, coded-residual flag | Decoder records | Skip blocks are copies; intra blocks are new |
| Picture type | Decoder or transport | I-frame pulsing |

The network must also work with planes missing (hardware decode, unknown encoder), so training drops them at random.

**What became of the frame's quantiser and type (November 2026).** The luma-only recurrent models that were released took them as three whole-frame planes (known, quantiser, key frame). They were removed. Told each frame's quantiser, a model used it as a switch for how much to smooth: an encoder's frame quantiser follows the moving foreground, so a still background that had been coded well earlier and was merely being skipped was smoothed and sharpened again as the number moved, and the picture flickered. The replacement models were trained with the three planes always zero, which cured the flicker and left dead inputs in the network and in every runtime, so the inputs were taken out: the network has 22 input planes (docs/INTEGER.md, section 1). Per-block records from a decoder remain an open idea; a per-frame number does not.

### 8.4 What will be compared

| Variant | Purpose |
|---|---|
| Per-frame | Baseline, and the bar to beat; per-frame models won the 2025 conferencing challenge |
| Stack of the current and two previous input frames, no alignment | The simplest temporal model; reported to gain 0.5-1.5 dB and reduce flicker over per-frame in small networks, so it is the temporal bar to beat |
| The same stack with the previous inputs aligned | Separates the value of alignment from the value of recursion |
| Blend, no alignment | Isolates the value of motion |
| Blend with alignment (main candidate) | Section 8.1 |
| Free hidden state with alignment | Tests whether the convex blend costs quality |
| With and without side planes | Measures what nanocodec's information is worth |

All at equal compute, judged on the flicker suite in section 10.

### 8.5 Training for stability

- **Long unrolls:** about 100-frame sequences, as Arm does, with state carried across batches.
- **Stored-state initialisation** (Mori & Ukita), so training starts from states the network really produces.
- **Random state resets** in training, so a reset is always safe at scene cuts and stream restarts. A fixed share of samples (about 15%) start with no usable history; for the stacked variants the current frame is repeated in place of the previous ones.
- **Static scenes, scene cuts and lighting changes** in the training clips.
- **Reset signal** from the analysis: frame mean-luma jumps and match failure across the frame.

### 8.6 Losses

- **Fidelity:** Charbonnier on the output.
- **Temporal difference:** match the output's frame-to-frame change to the ground truth's. Flow-free; has published precedent (Dai et al., CVPR 2022).
- **Degradation consistency:** the same clean clip degraded twice (different noise, different encoder settings) should give the same output. Targets noise amplification and GOP pulsing. The first form to build is the simplest: two passes over identical frames that differ only in the noise draw, with the difference between the outputs penalised. It costs nothing at run time and has been reported to cut flicker markedly while slightly improving fidelity.
- **Warping loss:** Lai et al., short and long term.
- **To test:** a small VGG term and a Sobel edge term, per StreamSR. No GAN in the first version.

## 9. Training data and degradation pipelines

The pipeline is general first and specific later. Any real deployment is expected to train on bespoke data from the actual device, so the lasting products of the general phase are the pipeline, the architecture and the training recipe, more than the general weights themselves. That makes per-device adaptation a designed-in step:

- **Capture kit:** a defined set of recordings from a new device (static scene, pans, moving subjects, low light, test charts) that is enough to calibrate the degradation modules and to validate against.
- **Adaptation run:** swap in the calibrated modules or real pairs, then fine-tune or retrain, and re-run the flicker suite on the device's own footage.
- **Licence note:** fine-tuning general weights that were trained on non-commercial data does not make them clean. A release model for a device is trained from scratch on commercially usable data plus the device data.

The general pipeline itself:

- **A source registry.** Each dataset is an entry with its location, format, a licence tag and a commercial-use flag. A training run names which entries it uses, so a commercially clean model is a configuration choice and not a rebuild.
- **Pluggable degradation stages.** Each stage (lens, noise, ISP denoise, sharpen, scale, encode) is a module with randomised parameters. The general model uses wide ranges; a specific camera or encoder is handled later by swapping in a calibrated module or real captured data for that stage.
- **Ground truth from downscaling.** Clean targets are made by downscaling sources by at least 2x, so they are sharp at the pixel level and most of any source compression is averaged away.
- **Mild, realistic degradation by default.** Training on heavily degraded examples makes a model over-sharpen real footage and amplify its noise. Each run therefore includes samples that are only lightly degraded, samples pre-sharpened the way a camera would (about a third), and samples left almost unchanged (about a third), with strengths skewed toward the mild end, so the network learns to do nothing when nothing is needed. A distilled teacher is checked for the same habit before use.
- **Per-clip and per-frame parameters.** Each degradation module declares which of its parameters are fixed for a clip (optics, ISP tuning, encoder settings) and which are drawn fresh per frame (sensor noise). Only what really varies in time varies in training, which is what lets a temporal model learn to average it away.
- **Degradation is applied during training, not baked in.** Each epoch sees fresh draws, so the pre-process (stage 7) runs on freshly degraded samples inside the data-loading workers, through the Python bindings to the one Rust implementation. Encoder round trips, which are slower, are cached in several variants per clip.
- **The pipeline owns the test split.** Clips are assigned to train, validation or sealed test by source at ingest, and no later stage can move them. Splitting later, by frame or by crop, leaks test clips into training.
- **Quality gate at ingest.** Sources are ranked by bitrate per pixel and a blockiness and sharpness check, and weak ones are downscaled further or used only as test material.

**The whole pipeline, from unlabelled raw video to a trained model.** One configuration file drives it; retraining for a new domain or device means pointing it at a different folder of video.

| Stage | What it does |
|---|---|
| 1. Ingest | Reads any container, codec or raw format. Normalises colour range and matrix, bit depth, frame rate and interlacing |
| 2. Clean | Splits at scene cuts; drops frozen and duplicate frames, borders, overlays and unusable clips |
| 2a. Align (paired sources only) | Registers the degraded clip to its reference: time offset, sub-pixel shift, scale and crop, and a low-frequency colour and tone match |
| 3. Grade and select | Measures each clip's bitrate per pixel, blockiness, sharpness, noise, motion and texture. Sets how far each source must be downscaled to serve as ground truth, and picks a balanced set of clips |
| 4. Targets | Downscales to make clean high-resolution clips |
| 5. Profile (optional) | Estimates the device's own blur, noise and sharpening from the same footage, to set the degradation modules |
| 6. Degrade | Applies the configured modules (lens, noise, ISP, scale, encode and decode) to make the low-resolution inputs |
| 7. Pre-process | Runs the Rust analysis to produce motion fields and side planes, exactly as deployment does |
| 8. Train and evaluate | Trains, then scores on held-out clips from the same set, the flicker suite and the fixed benchmarks |

Stages cache their results by content and configuration, so changing the data or one module re-runs only what depends on it. Nanocodec's clip tools (pool building, grading, selection) cover part of stages 2 and 3.

**Three kinds of source, mixable in one run with per-source weights:**

| Kind | What it is | How it is used |
|---|---|---|
| Reference only | Good-quality video with no degraded counterpart | The pipeline makes its own degraded version (stages 4-6) |
| Paired | A reference and a real degraded version of the same content | Used directly as training pairs after alignment; also used to fit the degradation modules, and as the most trustworthy test set |
| Degraded only | Real footage from the target device or stream, no reference | Profiling (stage 5), viewing tests and no-reference checks |

No labels or paired captures are needed, but pairs are used when they exist. How far unlabelled video goes differs by case:

- **Case B: unlabelled video is sufficient.** The degradation is a real encoder run with known settings, so pairs are exact. New content or a new encoder is a configuration change.
- **Case A: unlabelled video is sufficient to train, but the degradation has to be modelled.** Footage from the device is already soft, so it is the input domain, not ground truth. Downscaling it gives sharp targets, but the pipeline still has to know what blur and denoising the device applies in order to reproduce it. There are three levels: wide random ranges (the general model); automatic estimation from the footage in stage 5, which is feasible but the least proven part of the pipeline; or a few chart shots included in the footage, which makes the estimate much more reliable. None of these is labelling.

### 9.1 Paired camera data: raw sensor frames with the camera's output

If the raw sensor data can be captured alongside the camera's normal output, processing the raw offline with a slow, high-quality ISP gives a reference, and the pair goes in as a paired source. Three points decide how much that is worth:

- **Resolution of the reference.** A reference made from the same sensor frame is sharper than the camera output but at the same resolution. Such pairs teach restoration at 1x and measure the camera's degradation exactly; they do not by themselves contain 2x detail. The 2x training pairs then come from applying that measured degradation to downscaled references. This assumes the ISP and lens act the same per pixel at either scale, which is the standard assumption and can be checked on the pairs.
- **A true 2x reference** is possible if the offline processing combines several raw frames (static scenes, or small camera shifts). Those pairs can train 2x directly, and are the best validation set either way.
- **Tone and colour must match.** An offline ISP will not reproduce the camera's colour and tone exactly. The align stage matches low frequencies so the network learns detail, not a colour change; the global-residual design helps here too.

Even a small paired set is valuable: it replaces the least proven part of the pipeline (blind profiling) with a direct measurement.

### 9.2 Case A: camera after ISP

A proposal, from clean HR to simulated camera LR:

1. **Lens PSF and downscale.** For the general model, a random family of blur kernels with corner falloff and slight chromatic difference; for a specific camera, the measured PSF.
2. **Sensor noise** in a linearised domain, before denoising.
3. **Spatial noise reduction** with an edge-preserving filter at random strength. Gaussian blur does not reproduce texture flattening.
4. **Temporal noise reduction** as a recursive, motion-adaptive blend across the clip.
5. **Sharpening** with halos and clipping.
6. **Chroma subsampling** to 4:2:0 with the real siting.

Each of these stages is one module. For a specific camera later: real footage to check the simulation against, ISP controls or raw capture to measure the ISP's effect directly, and a reference capture with a much better lens for validation.

### 9.3 Case B: after encode and decode

1. Optionally apply the case A degradation, if the source is that camera.
2. Downscale 2x with a random choice of real scaler kernels.
3. Encode with a mix of real encoders in low-delay configuration: x264, x265, NVENC and nanocodec, across bitrates, presets and rate-control modes.
4. Decode, and run the same Rust pre-process the deployed system uses. Where the decode is nanocodec's analysis build, keep the block records as optional extra inputs.

Randomise bitrate, GOP length and the GOP phase of each clip. Keep some uncompressed clips. Match colour range and matrix exactly between training and deployment. If the transport adapts resolution, the runtime needs to handle size changes and state resets.

### 9.4 Datasets

**Local data** (sampled briefly, not audited). It is mostly surveillance, body-worn, street and traffic video, which suits the intended use. Most of it is already H.264 at moderate bitrate, so its value as ground truth depends on how far it is downscaled:

| Dataset | Sampled format | Use |
|---|---|---|
| uvg_vcm | 3840x2160, 60 fps, raw 4:4:4 16-bit | Best ground truth; no compression |
| personpath22 | 1080p at about 35 Mbit/s | Good ground truth after 2x downscale |
| CHAD | 1080p at about 20 Mbit/s | Good ground truth after 2x downscale |
| JAAD, otw | 1080p at about 5 Mbit/s | Ground truth after 3-4x downscale |
| MEVA, CrossView | 1080p at about 2 Mbit/s; several hundred GB | Too compressed for ground truth at 2x; useful after 4x downscale, and as real compressed test material |
| chirla | 1080x720 MPEG-4 at about 4 Mbit/s | As above |
| UCF_Crimes, XD-Violence, nextgqa | 240p-480p | Real low-quality test material only |
| BMD-45, UVH-26, StreetScene, VACaps | No video files found by extension; format not checked | To inspect |

Licences for these have not been checked. Several are research datasets and some will carry non-commercial terms (nanocodec's own notes record MOTChallenge as CC BY-NC-SA). That does not block development: all of it can be used now, and the registry's flags are filled in before a model is trained for release.

Two rules keep the later retrain clean. The release retrain covers the whole chain, including any teacher model used for distillation, since a student distilled from a teacher trained on non-commercial data inherits the problem. And the commercially usable subset is kept large enough to train on by itself, which is worth checking early by training one model on it alone.

**Public data:**

| Use | Datasets | Terms |
|---|---|---|
| Commercial-safe | REDS | CC BY 4.0 |
| Commercial-safe | Microsoft VSR-Challenge (real H.265 conferencing pairs; 3x and 4x); PD12M (public-domain images, needs quality filtering) | CDLA-Permissive-2.0 |
| Commercial-safe, to verify | Microsoft VCD (160 talking-head clips, 1080p, 10 s each) | Reported as CDLA-Permissive-2.0 |
| To verify | Netflix Open Content. The research pass reported CC BY 4.0 from the site; nanocodec's legal notes record the Netflix sequences it uses as CC BY-NC-ND 4.0. These may be different sets | Conflicting |
| Check per item | LDV 1-3; YouTube-UGC; StreamSR (real H.264, VP9 and AV1 artefacts; reported as CC BY 4.0); Xiph sequences; MCL-JCV | CC-sourced or per-sequence terms |
| Benchmark only | Vimeo-90K, DIV2K, LSDIR, UVG, Inter4K, BVI-DVC, BVI-AOM, OpenVid | Research-only or non-commercial |

## 10. Evaluation

**No single metric decides anything.** As in nanocodec, decisions use a combined score over several metric families, plus guards. PSNR alone rewards blur, VMAF alone rewards over-sharpening, and perceptual metrics alone reward hallucination.

| Family | Metrics | Catches |
|---|---|---|
| Fidelity | PSNR-YUV, XPSNR | Wrong pixels, colour errors |
| Structure | MS-SSIM | Lost or distorted structure |
| Perceptual | VMAF-NEG, LPIPS, DISTS | Softness that PSNR forgives |
| Temporal | Frame-difference error, warping error, tOF/tLP, ColorVideoVDP, CGVQM | Flicker, shimmer, ghosting |

- **Combined score.** The fidelity, structure and perceptual families count equally. Plain VMAF is reported but not scored, because it rewards sharpening.
- **A common unit.** For case B the natural one is nanocodec's: bitrate saved at equal quality, comparing a low-resolution stream plus this upscaler against the same stream with a conventional upscaler, per metric. For case A the output is usually encoded afterwards, so the same unit applies one step later: each candidate's output goes through a realistic low-delay encode at call-like bitrates and is scored against the uncompressed ground truth, giving bits at equal quality after the encoder. This also shows whether the gain survives the encoder, and penalises sharpening, which costs bits (one test found classic sharpening cost about 20% more bits while a learned upscaler that removed noise cost about 10% fewer). Where there is no encoder downstream, each metric is expressed as the share of the gap closed between the conventional upscaler and the large offline reference model. Both are proposals to be tried.
- **Worst case, not just the mean.** Following nanocodec's real-use objective, each clip also reports its worst short window of frames and its quality pumping over time (the change in windowed quality from one window to the next), since a few bad frames or a pulse are what viewers notice.
- **Guards, not part of the score.** Temporal metrics must not regress; an over-sharpening check (edge overshoot, output contrast against the ground truth's, and the gap between plain VMAF and VMAF-NEG); a do-no-harm check on clean, already-sharp input.
- **Decided on an interval over clips.** A change is adopted on a bootstrap interval of the per-clip score over held-out clips, not on an average over a few favourites.
- **Reuse.** Nanocodec's evaluation code already has these metrics, the combined score, the interval logic and a flicker reference, and is the starting point. Otherwise use the official LPIPS and DISTS repos, piq or torchmetrics, and libvmaf; the pyiqa toolbox is now non-commercial.

**Temporal metrics** are listed above. Blurry output scores well on warping error, so they are read alongside quality. Frame-to-frame change is always reported against the ground truth's own, which is the floor set by real motion.

**Flicker test suite:**

| Test | What it exposes |
|---|---|
| Static tripod scene, 5,000+ frames, quality over time | Drift and accumulation errors |
| Static scene with sensor noise only; output temporal variance against input | Noise amplification |
| Slow pan over fine texture | Shimmer; sub-sample alignment error |
| Moving objects over detailed background | Ghosting and disocclusion |
| The same clips at 2x, 4x and 8x motion (frame skipping), against the per-frame model | Whether temporal input ever makes fast motion worse |
| Lighting change on a static scene | Blend lag |
| Scene cuts and stream restarts | State contamination |
| Quality against GOP position (case B) | I-frame pulsing |
| Small pairwise viewing test | Everything the metrics miss |

**Baselines:** the upscaler the target product really ships, where known (it may be softer than bicubic, so bicubic understates the gain), bicubic and Lanczos, Lanczos plus sharpening, a motion-gated three-frame denoise followed by Lanczos plus mild sharpening (the best option with no network), FSR 1, the per-frame variant of our own network, and a large offline model (BasicVSR++ class, retrained on our degradations) as an upper bound.

## 11. Deployment

### 11.1 Export

- One PyTorch definition, fused to plain convs, exported with static shapes per input resolution, opset 17-20.
- Inputs: current frame, warped history, warped state, side planes. Outputs: frame, new state. The warp and the motion search stay outside the model.
- fp16 and fp32 variants. int8 is slower than fp16 on WebGPU and only pays on NPUs.
- For NPUs the packed variant of section 8.2 is the likely export: wide tensors only, with packing and the final pixel reorder outside the graph. An int8 output tensor carries its own scale and zero point, which the consumer must apply.

### 11.2 PC

ONNX Runtime with TensorRT for RTX or CUDA on NVIDIA, OpenVINO on Intel, Windows ML elsewhere on Windows. Nanocodec natively for decode and analysis.

### 11.3 Browser

WebGPU. Two ways to run the network, to be chosen by measurement:

| | ONNX Runtime Web | Generated WGSL shaders |
|---|---|---|
| Effort | Low | A code generator from the layer list; the network is only convs and a pixel-shuffle |
| Data path | Storage buffers; needs glue passes from and to textures | Textures throughout; the warp, blend and display passes are the same kind of code |
| Download | About 29 MB of runtime | Small |
| Known problems | In 1.31.1, GPU-buffer inputs fail on a page-supplied device (1.30.0 works). An older report, closed unfixed, of wrong results when a graph input has more than three channels. Reported crashes and memory growth on iOS Safari 26 | Ours to maintain |

This design already needs custom GPU passes for the warp and blend, and feeds the network many-channel inputs, so the generated-shader route is more attractive here than it would be for a plain per-frame model. ONNX remains the reference and the PC format either way.

Per-frame pipeline:

1. Decode in a worker: nanocodec WebAssembly (with block records), or WebCodecs.
2. Motion field and side planes: from the records, or `nc-analysis` in WebAssembly on the decoded or captured frame.
3. Upload the frame and the small side-plane textures; warp history on the GPU.
4. Run the network, blend, and render to a WebGPU canvas. No readback.

WebGPU ships in Chrome and Edge (desktop and Android), Safari 26, and Firefox on Windows and Apple Silicon. WebNN is still behind a flag.

## 12. Code and licence policy

Safe to read (permissive):

| Project | Licence | Useful for |
|---|---|---|
| nanocodec | BSD-3-Clause, ours | Decode, records, analysis, training encoder |
| BasicSR, Real-ESRGAN, traiNNer-redux | Apache-2.0, BSD-3 | Training framework, degradations, reference architectures |
| ECBSR, SPAN, RLFN (ByteDance repo), SAFMN, RT4KSR, DIPNet | Apache-2.0 / MIT | Re-parameterised blocks |
| RealBasicVSR, BasicVSR++, TecoGAN, TMP, NanoVSR | Apache-2.0 / MIT | Video degradation, temporal losses, teachers |
| Arm Neural Graphics Model Gym | Apache-2.0 (code only; the NSS weights have a separate licence) | Recurrent blend training reference |
| EfRLFN / StreamSR | MIT | Streaming loss recipe |
| WebSR, Anime4K-WebGPU | MIT | Browser shader implementation |
| ColorVideoVDP, CGVQM, LPIPS, DISTS, libvmaf | MIT / Apache-2.0 / BSD | Metrics |

Papers only: FSR 4 source (an accidental release, since withdrawn), Bicubic++ and RLSP (CC BY-NC-SA), VRT/RVRT (CC BY-NC), anything without a licence file (RRN, CDA-VSR, FastRealVSR and others), and copyleft tools.

## 13. Proposed phases

| Phase | Work | Output |
|---|---|---|
| 0 | Set up the repository skeleton, `CLAUDE.md` rules and CI checks of section 5. Measure: random-weight networks at sizes S to XL in the browser on real target devices (ONNX Runtime Web and generated shaders) and natively, in both the U-Net and packed layouts, with thin full-resolution layers and the pixel-shuffle timed separately; `nc-analysis` and analysis-build decode time in WebAssembly | A measured browser budget; runtime choice |
| 1 | The configurable pipeline of section 9 (profiling can follow), run on the local data; the Rust pre-process crate with Python bindings; evaluation and flicker suite; baselines | Raw video in, training set and scores out |
| 2 | XL per-frame model for each case, natively | Quality ceiling without temporal input |
| 3 | XL temporal variants (section 8.4) | The temporal design, and the value of each side plane |
| 4 | Cut down as hardware requires: distil and narrow to L, M and S | Quality against cost curve |
| 5 | Export; browser and PC demos on real video | End-to-end system |
| 6 | Optional: perceptual loss tuning, int8 for NPUs | Refinements |

## 14. Open questions

1. **Licences of the local datasets.** Needs a pass before a release model is trained, not before development.
2. **PC targets.** Resolution, frame rate, minimum GPU.
3. **Native GPU path.** ONNX Runtime, or the same WGSL shaders through `wgpu`.
4. **Sample clips** of the two real cases, when available, to check the general degradation models against.

## 15. References

Efficient SR and benchmarks
- NTIRE 2024 / 2025 / 2026 efficient SR: https://arxiv.org/abs/2404.10343 , https://arxiv.org/abs/2504.10686 , https://arxiv.org/abs/2604.03198
- Mobile AI 2026: https://arxiv.org/html/2611.00963
- ESC (transformer latency table): https://arxiv.org/html/2503.06671v2
- traiNNer-redux benchmarks: https://trainner-redux.readthedocs.io/en/latest/benchmarks.html
- AIS 2024 real-time 4K SR: https://arxiv.org/abs/2404.16484
- AIM 2024 efficient VSR for AV1: https://arxiv.org/abs/2409.17256
- AIM 2025 efficient perceptual SR: https://arxiv.org/html/2510.12765v1
- AIM 2025 efficient deblurring: https://arxiv.org/html/2510.12788
- ECBSR: https://github.com/xindongzhang/ECBSR
- SPAN: https://arxiv.org/abs/2311.12770
- Bicubic++: https://arxiv.org/abs/2305.02126
- StreamSR / EfRLFN: https://arxiv.org/html/2602.11339 , https://github.com/EvgeneyBogatyrev/EfRLFN

Shipping upscalers
- NVIDIA RTX Video model card: https://catalog.ngc.nvidia.com/orgs/nvidia/multimedia/models/dlpp/-
- Arm NSS: https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/how-arm-neural-super-sampling-works , https://github.com/arm/neural-graphics-model-gym
- PSSR talk (SIGGRAPH 2026): https://advances.realtimerendering.com/s2026/content/siggraph2026_advances_dcraig_v1_0_publish.pdf
- DLSS 4: https://research.nvidia.com/labs/adlr/DLSS4
- Microsoft Edge VSR: https://blogs.windows.com/msedgedev/2023/03/08/video-super-resolution-in-microsoft-edge/
- Windows VSR API: https://learn.microsoft.com/en-us/windows/ai/apis/video-super-resolution
- AMD FSR 1: https://github.com/GPUOpen-Effects/FidelityFX-FSR

Video SR and alignment
- CDA-VSR: https://arxiv.org/abs/2603.07694
- TS-Mamba: https://arxiv.org/html/2508.10453v1
- TMP: https://arxiv.org/abs/2312.09909
- RRN: https://arxiv.org/abs/2008.05765
- RLSP: https://arxiv.org/abs/1909.08080
- FRVSR: https://arxiv.org/abs/1801.04590
- BasicVSR / BasicVSR++: https://arxiv.org/abs/2012.02181 , https://github.com/ckkelvinchan/BasicVSR_PlusPlus
- RealBasicVSR: https://arxiv.org/abs/2111.12704
- FastRealVSR: https://arxiv.org/abs/2212.07339
- NanoVSR: https://arxiv.org/abs/2607.10495
- ICME 2025 conferencing VSR challenge: https://arxiv.org/html/2506.12269v2 , https://github.com/microsoft/VSR-Challenge

Blending and stability
- EMVD: https://arxiv.org/abs/2103.05407
- Improved EMVD: https://gretsi.fr/data/colloque/pdf/2022_zheng966.pdf
- GRU-VD: https://arxiv.org/abs/2210.09135
- Chiche et al.: https://arxiv.org/abs/2112.08950
- Mori & Ukita: https://arxiv.org/html/2403.15832
- Lai et al.: https://arxiv.org/abs/1808.00449
- TecoGAN: https://arxiv.org/abs/1811.09393
- Dai et al., relation-based temporal loss: https://arxiv.org/abs/2204.02957

Compressed video
- MFQE 2.0: https://arxiv.org/abs/1902.09707
- COMISR: https://arxiv.org/abs/2105.01237
- CAVSR: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Compression-Aware_Video_Super-Resolution_CVPR_2023_paper.html
- CIAF: https://arxiv.org/abs/2210.08229
- PnP-VCVE: https://arxiv.org/html/2504.15380v1
- NTIRE 2021 / 2022 compressed video: https://arxiv.org/abs/2104.10781 , https://arxiv.org/abs/2204.09314
- WebCodecs frame metadata registry: https://w3c.github.io/webcodecs/video_frame_metadata_registry.html

Degradation
- BSRGAN: https://arxiv.org/abs/2103.14006
- Real-ESRGAN: https://arxiv.org/abs/2107.10833
- RAW-domain device calibration: https://arxiv.org/abs/2603.12493
- OmniLens: https://arxiv.org/pdf/2409.05809

Deployment
- ONNX Runtime Web WebGPU guide: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- WebGPU operator table: https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md
- Conv im2col measurement on Lunar Lake: https://github.com/microsoft/onnxruntime/pull/26603
- GPU-buffer input bug: https://github.com/microsoft/onnxruntime/issues/33642
- Multi-channel input bug: https://github.com/microsoft/onnxruntime/issues/29651
- iPad throttling report: https://github.com/microsoft/onnxruntime/discussions/33453
- Ultralytics YOLOv8 figures: https://docs.ultralytics.com/models/yolov8/
- WebGPU implementation status: https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
- WebSR: https://github.com/sb2702/websr

Datasets and metrics
- REDS: https://seungjunnah.github.io/Datasets/reds.html
- Netflix Open Content: https://opencontent.netflix.com/
- PD12M: https://huggingface.co/datasets/Spawning/PD12M
- LDV: https://github.com/RenYang-home/LDV_dataset
- ColorVideoVDP: https://github.com/gfxdisp/ColorVideoVDP
- CGVQM: https://github.com/IntelLabs/CGVQM
- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- DISTS: https://github.com/dingkeyan93/DISTS
- VMAF: https://github.com/Netflix/vmaf
