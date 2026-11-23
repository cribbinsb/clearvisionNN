# clearvisionNN

[![crates.io](https://img.shields.io/crates/v/clearvisionnn.svg)](https://crates.io/crates/clearvisionnn) [![docs.rs](https://img.shields.io/docsrs/clearvisionnn)](https://docs.rs/clearvisionnn) [![Code: BSD-3-Clause](https://img.shields.io/badge/code-BSD--3--Clause-blue.svg)](LICENSE) [![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-orange.svg)](web/models/LICENSE.md)

Fast, causal 2× video super-resolution that runs in a browser tab. It is built
for two kinds of input: camera video straight from the camera's own
processing, and video that has been through a real-time H.264/H.265 encode and
decode, as on a video call. It keeps the picture steady from frame to frame,
which matters as much as sharpness once the video moves.

**[Try the browser demo](https://cribbinsb.github.io/clearvisionNN/)** (needs WebGPU for the 8-bit models) ·
**[Rust crate](https://crates.io/crates/clearvisionnn)** · [API docs](https://docs.rs/clearvisionnn) ·
[Training recipe](recipe/README.md) · [Architecture](docs/ARCHITECTURE.md) · [8-bit arithmetic](docs/INTEGER.md)

## What it does

Each frame is upscaled as it arrives, using only that frame and what the
upscaler remembers of earlier ones: no look-ahead, so no added delay. The aim
is not to reproduce the original pixel for pixel, which the input no longer
determines, but a picture that looks like it and that reduces back to the
input it was given.

| Model | Network | Parameters | Compute per input pixel |
|---|---|---|---|
| N | 16 channels × 6 layers | 23 k | about 17 k multiply-adds |
| S | 24 channels × 10 layers | 69 k | about 65 k |
| M | 32 channels × 15 layers | 169 k | about 160 k |
| L | 40 channels × 15 layers | 259 k | about 250 k |
| XL | 64 channels × 17 layers | 726 k | about 720 k |

Two releases of the models are in the demo: V2 (November 2026, the default)
and V1, the first release. Measured on Xiph test sequences that neither was
trained on (23 sequences) and on 100 held-out phone video-call files (VC-1K),
640×360 input to 1280×720, against the same input enlarged with a bicubic
filter. "Camera" is clean camera input; "codec" is the same after H.264/H.265
at 0.04 to 0.3 bits per pixel. The figures are for the 8-bit models the demo
runs. VMAF is taken at the clips' own size, so it compares these pictures
with each other, not with VMAF published elsewhere.

Xiph:

| | Camera: VMAF | PSNR (dB) | LPIPS | Codec: VMAF | PSNR (dB) | LPIPS |
|---|---|---|---|---|---|---|
| Bicubic | 87.08 | 32.26 | 0.2305 | 55.19 | 28.51 | 0.3495 |
| N V1 | 91.87 | 33.68 | 0.1255 | 57.73 | 28.68 | 0.2846 |
| N V2 | 92.52 | 33.58 | 0.1169 | 58.19 | 28.62 | 0.2782 |
| S V1 | 92.61 | 34.04 | 0.1166 | 57.95 | 28.73 | 0.2815 |
| S V2 | 93.29 | 33.88 | 0.1031 | 58.72 | 28.74 | 0.2731 |
| M V1 | 92.85 | 34.16 | 0.1102 | 58.13 | 28.83 | 0.2823 |
| M V2 | 93.38 | 33.97 | 0.1103 | 59.04 | 28.76 | 0.2759 |
| L V1 | 93.03 | 34.26 | 0.1082 | 58.36 | 28.87 | 0.2838 |
| L V2 | 93.59 | 34.22 | 0.0945 | 59.58 | 28.80 | 0.2771 |
| XL V1 | 93.30 | 34.41 | 0.0989 | 59.27 | 28.99 | 0.2929 |
| XL V2 | 93.66 | 34.27 | 0.0875 | 59.67 | 28.72 | 0.2706 |

Phone video calls (VC-1K):

| | Camera: VMAF | PSNR (dB) | LPIPS | Codec: VMAF | PSNR (dB) | LPIPS |
|---|---|---|---|---|---|---|
| Bicubic | 89.49 | 35.92 | 0.1721 | 67.19 | 32.40 | 0.2855 |
| N V1 | 93.93 | 37.74 | 0.0952 | 69.83 | 32.87 | 0.2345 |
| N V2 | 94.41 | 37.64 | 0.0897 | 70.23 | 32.84 | 0.2279 |
| S V1 | 94.51 | 38.06 | 0.0909 | 69.99 | 32.92 | 0.2302 |
| S V2 | 95.03 | 37.98 | 0.0809 | 70.92 | 33.01 | 0.2135 |
| M V1 | 94.84 | 38.31 | 0.0871 | 70.30 | 33.07 | 0.2214 |
| M V2 | 95.09 | 38.08 | 0.0832 | 71.26 | 33.05 | 0.2114 |
| L V1 | 94.93 | 38.36 | 0.0863 | 70.52 | 33.11 | 0.2201 |
| L V2 | 95.23 | 38.35 | 0.0739 | 71.93 | 33.15 | 0.2118 |
| XL V1 | 95.14 | 38.60 | 0.0788 | 71.74 | 33.31 | 0.2215 |
| XL V2 | 95.33 | 38.47 | 0.0687 | 72.21 | 33.12 | 0.2051 |

V2 has the higher VMAF at every size on both kinds of input (0.2 to 0.7 on
camera input, 0.3 to 1.4 after a codec) and the better LPIPS almost
everywhere, for a little PSNR (up to 0.23 dB on camera input, up to 0.28 dB
after a codec). After a low-bitrate codec the gain over bicubic is small for
every model: the detail is no longer in the input to recover. Each model is
a single training run.

## How it works

**1. Pre-process (Rust, `crates/cvnn-preproc`).** A block motion search finds,
for every 8×8 block of the new frame, where it came from in the previous one.
It also detects scene cuts, at which the upscaler forgets what it remembers.
The search is built on [nanocodec](https://github.com/cribbinsb/nanocodec)'s
encoder analysis. There is one implementation, used by training and by every
runtime.

**2. What the network is given.** All at the input's resolution:

- the new frame's luma;
- the previous output, moved along the motion to where it belongs now;
- 16 channels of the network's own memory from the previous frame, moved the
  same way;
- a confidence plane: how well the motion predicted this frame from the last,
  block by block, so the network knows where its memory is not to be trusted.

That is 22 planes, and nothing about how the frame was coded: a model told
each frame's quantiser used it as a switch for how much to smooth, which
flickered, so one model serves coded and uncoded input alike without it.

**3. The network (`python/clearvisionnn/models`).** A plain stack of 3×3
convolutions with ReLU at the input's resolution, with one long skip
connection, ending in a pixel shuffle to twice the size. It predicts a
correction to an ordinary bicubic upscale of the luma, so a network that
outputs nothing is still a correct upscaler. Colour is upscaled
conventionally. Beside the picture it writes its 16 memory channels for the
next frame.

**4. Around the network (Rust, `crates/cvnn-runtime`).** The conventional
upscale, the warping of memory along the motion, colour conversion and the
assembly of the network's input are specified once, in Rust. Training checks
its own model against that code frame for frame.

**5. 8-bit integer form (`docs/INTEGER.md`).** The released models run in
integer arithmetic: 8-bit weights and activations, the memory stored in 8
bits. The reference is in Rust. In the browser the same arithmetic is written
as WebGPU shaders, which a documented check holds to the reference bit for
bit on real frames (`tools/web/int8_check.py`, `tools/web/runtime_check.py`).
The frame is processed in tiles, so the working set stays in the GPU's cache.

**6. The page (`web/`).** A static page: the camera simulation, the reduction
and the motion search are the Rust code compiled to WebAssembly, in a worker;
the network and everything round it are on the GPU. Nothing leaves the
device.

## How the models were trained

The full recipe, with the commands, is in [`recipe/`](recipe/README.md). In
outline:

1. **Data.** Clean video is cut into short clips; inputs are made from them by
   degradation chains that imitate a sender. The camera chain adds sensor
   noise, applies a camera's edge-preserving denoise and reduces by two. The
   codec chain adds noise, reduces, and encodes with x264 or x265 in a
   low-delay configuration at 0.04 to 0.3 bits per pixel. One model is
   trained on both.
2. **Base training.** Sequences of six frames, unrolled through the recurrent
   network, with a Charbonnier pixel loss and a small term that penalises
   frame-to-frame change where the original does not change. Mirror and
   transpose augmentation, including the motion vectors. The learning rate is
   cut when the held-out score stops improving, and training ends after
   three cuts.
3. **Sharpening.** A short fine-tune adds a perceptual term (LPIPS, weight
   0.2; 0.05 in V1) and, from V2, a texture-amplitude term beside the pixel loss. It gives up about 0.2 dB of PSNR for a clearly
   better LPIPS, more detail and a higher VMAF.
4. **Conversion to 8 bits.** Quantisation-aware training in which the 8-bit
   model follows the float model's output and its frame-to-frame change, at a
   learning rate that decays to zero. This keeps the 8-bit model's flicker
   within a few per cent of the float model's.

## History: what changed from V1 to V2

V1 was the first release; V2 (November 2026) replaced it as the default and
V1 stays in the demo's model list for comparison. V2 keeps the same networks,
sizes and speed. What changed:

- **Texture.** V1 smoothed fine texture, most visibly on a background that
  had been still for a while after an encode ("waxy"). V2's sharpening stage
  has a term that asks for the original's amount of texture, on still and
  moving parts of the picture, and a stronger perceptual term (LPIPS weight
  0.2, was 0.05). This is a deliberate trade: more texture and better
  perceptual scores for about 0.1 to 0.3 dB of PSNR on clean camera input.
- **No coding inputs.** V1 took three planes saying how each frame was coded
  (quantiser, key frame). Told the quantiser, the model used it as a switch
  for how much to smooth, which flickered as an encoder's quantiser moved.
  V2 takes 22 input planes and nothing about the coding. The V1 models in the
  demo are the released ones with those inputs removed, which computes exactly
  what they did when given zeros.
- **More varied training data.** About 4,500 training clips, up from about
  1,500: more webcam models, user-generated video, street and hand-held
  footage, nature, night scenes and screen recordings. Webcam and phone
  material is still 65 % of every batch. The clear gain is on screen content;
  on webcam material the extra data made little difference.
- **A longer base schedule** (40,000 steps; M uses 20,000).
- **A clean test split.** Eleven scenes that were both in the training data
  and among the test clips were taken out of training, two of them Xiph
  sequences (crowd-run and park-joy), which the tables at the top leave out.
- **Grain on the output.** The runtime adds a little fixed-level grain to the
  displayed luma (never fed back to the network): 0.35 grey levels for
  produced content, 0.75 for webcam and video-call input. An encoder removes a
  camera's noise and the network cannot know how much there was; without it
  the picture looks cleaner than the original ever did.

Texture, screen content and flicker, which the tables at the top do not
show: the 8-bit models on validation and other held-out clips, V1 run without
its coding inputs. LPIPS: lower is better. "Still" and
"moving" are crops of long-still background and of moving texture after an
encode at 0.08 bits per pixel; "camera" and "codec" are whole held-out clips;
"screen" is held-out screen recordings as camera input; "flicker" is the
frame-to-frame change, in grey levels, where the input does not change.

| Size | Model | Still LPIPS | Moving LPIPS | Camera PSNR / LPIPS | Codec PSNR / LPIPS | Screen PSNR | Flicker |
|---|---|---|---|---|---|---|---|
| N | V1 | 0.0693 | 0.1179 | 37.60 / 0.0922 | 33.11 / 0.2016 | 27.10 | 0.180 |
| N | V2 | 0.0737 | 0.1111 | 37.41 / 0.0879 | 33.06 / 0.1963 | 27.30 | 0.166 |
| S | V1 | 0.0745 | 0.1172 | 37.86 / 0.0884 | 33.15 / 0.1986 | 27.30 | 0.185 |
| S | V2 | 0.0660 | 0.1107 | 37.67 / 0.0810 | 33.28 / 0.1864 | 27.73 | 0.188 |
| M | V1 | 0.0733 | 0.1200 | 38.03 / 0.0867 | 33.37 / 0.1937 | 27.40 | 0.179 |
| M | V2 | 0.0714 | 0.1068 | 37.75 / 0.0807 | 33.30 / 0.1860 | 27.93 | 0.198 |
| L | V1 | 0.0721 | 0.1171 | 38.09 / 0.0860 | 33.41 / 0.1942 | 27.48 | 0.182 |
| L | V2 | 0.0702 | 0.1035 | 37.98 / 0.0706 | 33.37 / 0.1858 | 28.14 | 0.191 |
| XL | V1 | 0.0785 | 0.1201 | 38.27 / 0.0777 | 33.57 / 0.1963 | 27.81 | 0.167 |
| XL | V2 | 0.0652 | 0.0979 | 38.04 / 0.0659 | 33.33 / 0.1801 | 28.30 | 0.177 |

V2 is better at every size on moving texture (6 to 18 %), on camera and codec
LPIPS, and on screen content (0.2 to 0.7 dB). It is worse on camera PSNR at
every size (0.11 to 0.28 dB), which is the texture trade; N's still-crop LPIPS
is 6 % worse; flicker is a little higher at S, M, L and XL; XL loses 0.24 dB on
coded input. Each V2 model is a single training run. V1 trained on some of
the scenes the still and moving crops come from and V2 did not, so those two
columns favour V1.

What V2 does not fix: faint texture, under about four grey levels, is still
lost after an encode. The encoder has already removed most of it before the
network sees the frame, and three attempts to train it back failed.

## Build and run

Start with [`docs/PROCEDURES.md`](docs/PROCEDURES.md) for building, running
and checking, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the
code is laid out. [`docs/DESIGN.md`](docs/DESIGN.md) is the original design
document, with the literature it draws on.

```
tools/web/build.sh              # the WebAssembly module, into web/pkg
python3 tools/web/serve.py      # the demo at http://localhost:8000
tools/ci/check.sh               # every check the project requires
tools/web/publish.sh            # build the page and publish it to the gh-pages branch
```

The crates depend on nanocodec by path, as a sibling checkout at the revision
in `tools/nanocodec.rev`.

## Use from Rust

`cargo add clearvisionnn --features models`
([crates.io](https://crates.io/crates/clearvisionnn), [docs](https://docs.rs/clearvisionnn)), or in `Cargo.toml`:

```toml
clearvisionnn = { version = "0.1", features = ["models"] }
```

```rust
use clearvisionnn::models::Size;
use clearvisionnn::runtime::integer::IntegerModel;

// The medium model, built in by the `models` feature.
let model = IntegerModel::from_words(&Size::M.words())?;
```

One crate gathers the rest:

| Crate | What it holds | Licence |
|---|---|---|
| `clearvisionnn` | Re-exports the three below; `models` feature adds the weights | BSD-3-Clause |
| `clearvisionnn-common` | Shared types | BSD-3-Clause |
| `clearvisionnn-preproc` | The motion search and side planes in front of the network | BSD-3-Clause |
| `clearvisionnn-runtime` | Everything around the network for a stream, and the 8-bit network itself | BSD-3-Clause |
| `clearvisionnn-models` | The released 8-bit models, N to XL, built in | **CC BY-NC 4.0** |
| `clearvisionnn-web` | The browser bindings the demo page uses | BSD-3-Clause |

The weights are a crate of their own because their licence differs from the
code's: some of the video they were trained on is for non-commercial use
only. The `models` feature is therefore off by default, so that nobody takes
on non-commercial data without asking for it. With it on,
`clearvisionnn::models::Size::M.words()` gives a model ready for
`clearvisionnn::runtime::integer::IntegerModel::from_words`; without it, load
a model file of your own the same way (the files under `web/models`, or ones
made with the recipe).

## Licence

The code is under the BSD 3-Clause licence ([`LICENSE`](LICENSE)).

**The model weights under `web/models` are under CC BY-NC 4.0**, for
non-commercial use, because some of the video they were trained on is
licensed that way: see [`web/models/LICENSE.md`](web/models/LICENSE.md).

The sample clips on the demo page belong to their rights holders:
[`web/clips/ATTRIBUTION.md`](web/clips/ATTRIBUTION.md).

This project was written by AI coding agents under one person's direction.
