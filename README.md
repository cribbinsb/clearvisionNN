# clearvisionNN

Fast, causal 2× video super-resolution that runs in a browser tab. It is built
for two kinds of input: camera video straight from the camera's own
processing, and video that has been through a real-time H.264/H.265 encode and
decode, as on a video call. It keeps the picture steady from frame to frame,
which matters as much as sharpness once the video moves.

**Live demo:** <https://cribbinsb.github.io/clearvisionNN/> (needs a browser
with WebGPU for the 8-bit models).

## What it does

Each frame is upscaled as it arrives, using only that frame and what the
upscaler remembers of earlier ones: no look-ahead, so no added delay. The aim
is not to reproduce the original pixel for pixel, which the input no longer
determines, but a picture that looks like it and that reduces back to the
input it was given.

| Model | Network | Parameters | Compute per input pixel |
|---|---|---|---|
| N | 16 channels × 6 layers | 20 k | about 17 k multiply-adds |
| S | 24 channels × 10 layers | 69 k | about 65 k |
| M | 32 channels × 15 layers | 169 k | about 160 k |
| L | 40 channels × 15 layers | 259 k | about 250 k |
| XL | 64 channels × 17 layers | 726 k | about 720 k |

Measured on 25 Xiph test sequences the models never saw, 640×360 input to
1280×720, against the same input enlarged with a bicubic filter. "Camera" is
clean camera input; "codec" is the same after H.264/H.265 at 0.04 to 0.3 bits
per pixel. The figures are for the float models; the 8-bit versions the demo
runs are within 0.01 dB of them on the validation set.

| | Camera: PSNR (dB) | LPIPS | VMAF | Codec: PSNR (dB) | LPIPS | VMAF |
|---|---|---|---|---|---|---|
| Bicubic | 31.75 | 0.241 | 87.1 | 28.03 | 0.363 | 54.3 |
| S | 33.57 | 0.120 | 93.1 | 28.46 | 0.324 | 58.2 |
| M | 33.73 | 0.113 | 93.5 | 28.52 | 0.321 | 58.8 |
| L | 33.83 | 0.110 | 93.6 | 28.53 | 0.316 | 58.8 |
| XL | 33.99 | 0.100 | 93.9 | 28.56 | 0.309 | 59.1 |

After a low-bitrate codec the gain is small for every model: the detail is no
longer in the input to recover.

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
  block by block, so the network knows where its memory is not to be trusted;
- three coding planes: the frame's quantiser and whether it is a key frame,
  when the decoder knows them. They are zero when it does not, or when there
  is no codec, so one model serves both kinds of input.

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
   low-delay configuration at 0.04 to 0.3 bits per pixel, keeping each
   frame's quantiser and type. One model is trained on both.
2. **Base training.** Sequences of six frames, unrolled through the recurrent
   network, with a Charbonnier pixel loss and a small term that penalises
   frame-to-frame change where the original does not change. Mirror and
   transpose augmentation, including the motion vectors. The learning rate is
   cut when the held-out score stops improving, and training ends after
   three cuts.
3. **Sharpening.** A short fine-tune adds a perceptual term (LPIPS, weight
   0.05) beside the pixel loss. It gives up about 0.2 dB of PSNR for a clearly
   better LPIPS, more detail and a higher VMAF.
4. **Conversion to 8 bits.** Quantisation-aware training in which the 8-bit
   model follows the float model's output and its frame-to-frame change, at a
   learning rate that decays to zero. This keeps the 8-bit model's flicker
   within a few per cent of the float model's.

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

## Licence

The code is under the BSD 3-Clause licence ([`LICENSE`](LICENSE)).

**The model weights under `web/models` are under CC BY-NC 4.0**, for
non-commercial use, because some of the video they were trained on is
licensed that way: see [`web/models/LICENSE.md`](web/models/LICENSE.md).

The sample clips on the demo page belong to their rights holders:
[`web/clips/ATTRIBUTION.md`](web/clips/ATTRIBUTION.md).

This project was written by AI coding agents under one person's direction.
