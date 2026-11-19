# The integer network and runtime

Sections 1 to 6 are the network; section 7 is everything around it. Together
they are a whole frame in integers, which a GPU computes without anything
crossing back to the processor.

A flat luma model (stem, a stack of blocks, a long skip, a head) run entirely
in 8-bit planes and 32-bit integer accumulators. This file is the
specification. `cvnn_runtime::integer` (the network) and `cvnn_runtime::fixed`
(the runtime) are its reference implementation; the page's WGSL shaders
(`web/js/int8-shaders.js`, `web/js/runtime-shaders.js`) and the tests' NumPy
version of the network (`python/tests/integer_reference.py`) must agree with
it bit for bit.

It realises what the recipe's quantisation-aware fine-tune simulates
(`recipe/quantise.py`), so trained weights transfer: section 6 says where the
two cannot agree exactly and by how much.

## 1. Planes

Every plane is 8 bits per sample.

| Plane | Values | Real value of one step |
|---|---|---|
| Unsigned input (luma, confidence, previous output, coding planes) | 0..255 | 1/255 |
| Signed input (the recurrent state) | -127..127 | 1/127 |
| Activation after a ReLU, layer `i` | 0..255 | `top_i`/255, `top_i` learnt in training |
| State output | -127..127 | 1/127 |
| Picture output (the residual) | 16-bit signed | 1/32 of an 8-bit level |

A float input `x` becomes `clamp(round(255 x), 0, 255)` or
`clamp(round(127 x), -127, 127)`, halves to even.

Storage: four channels to a 32-bit word, a byte each, lowest byte first. An
unsigned value `u` is stored as the signed byte `u - 128`, so that one signed
packed dot product serves every plane. The state's width is a field of the
model (`state_bits`, 8 today): a 16-bit state would double the state's words
in the input and output and need two dot products per state word in the stem.

## 2. Accumulation

All arithmetic is 32-bit two's complement. Weights are signed 8-bit, the
convolution and its 1x1 branch summed and rounded per output channel:
`w = round(kernel / s_w)`, `s_w = max|kernel| / 127`. The identity branch is
not folded in.

- **Stem.** Inputs are regrouped: unsigned planes first, then signed, each
  group padded with zero planes to whole words. To put both on the one scale
  1/(255·127): `acc = bias + 127·Σ w·u + 255·Σ w·s`.
- **Block and head.** `acc = bias + Σ w·u`.

Padding at the frame edge is the value zero. `bias = round(b / unit)`, where
`unit` is the real value of one accumulator step (`s_w/(255·127)` in the stem,
`s_in·s_w` elsewhere). The exporter refuses a model whose accumulator could
leave 32 bits.

## 3. Requantisation

A real scale `m` is held per output channel as a multiplier `M < 2^14` and a
shift `sh`, `m ≈ M / 2^(16+sh)`:

```
scaled(acc)   = (acc >> 16)·M + (((acc & 0xFFFF)·M) >> 16)        exact floor of acc·M / 2^16
rounded(v)    = (v + ((1 << sh) >> 1)) >> sh                       halves towards +infinity
```

Shifts are arithmetic. No product leaves 32 bits.

| Layer | Output |
|---|---|
| Stem | `clamp(rounded(scaled(acc)), 0, 255)` |
| Block | `clamp(rounded(scaled(acc) + I·u_own), 0, 255)`; `I = round(2^sh · s_in/s_out)` is the identity branch |
| Last block | block output `g`, then with the stem's output `f`: `clamp((g·M_g + f·M_f + ((1 << sh_L) >> 1)) >> sh_L, 0, 255)` |
| Head, picture outputs | `clamp(rounded(scaled(acc)), -32768, 32767)`, in 1/32 level |
| Head, state outputs | `v = rounded(scaled(acc))` in 1/1024; output `sign(v) · table[min(|v|, len-1)]` |

`table[k] = round(127·tanh(k/1024))`, 4096 entries, stored in the model file so
that no implementation evaluates tanh. The clamp to 0..255 is the ReLU.

The four picture outputs of an input pixel are its 2x2 output pixels, row by
row. The output per pixel is two words of picture outputs (16 bits each),
then the state, four channels to a word.

## 4. Tiles

A model is deployed with a tile side and a halo, both in its file, because
they are part of its arithmetic: every device computes the same picture.

A tile is given `halo` pixels of real neighbouring input. Layer `i` of `L`
(stem 0, head `L-1`) computes the tile grown by `min(halo, L-1-i)`, clipped to
the frame; it reads zeros beyond what layer `i-1` computed, and the stem reads
the frame itself. With `halo >= L-1` this equals the untiled network; the
default (halo 6, tile 128) is an approximation whose seams measured under a
quarter of a level on the float network. Tile side 0 means the whole frame as
one tile, with no halo.

How tiles are grouped into dispatches, or ordered, never changes the result.

## 5. The file (`model.i8`)

32-bit little-endian words.

| Words | Content |
|---|---|
| 16 | Header: magic `FCI8`, schema version, float input channels, unsigned input words, signed input words, trunk channels, blocks, picture outputs (4), state outputs, halo, tile, tanh table length, state bits; the rest zero |
| 4 per input word | Input slots: the float input channel each input byte holds, or -1 for padding |
| per layer | Weights packed `[tap][input word][output]` (taps row by row), then per output: bias, multiplier, shift; a block adds its identity multipliers |
| 3 | The long skip: `M_f`, `M_g`, `sh_L` (between the last block and the head) |
| table length / 4 | The tanh table, a byte per entry |

Layers are in order: stem, blocks, (skip), head. The contract beside it
(`model.json`, `network: "integer"`) repeats the schema version, tile, halo and
state bits and carries the file's hash.

## 6. Against the training simulation

Measured on validation batches, the integer network scores the same as the
simulation; sample by sample they differ slightly, because:

- real scales are 14-bit multipliers here (the larger part: with 20-bit
  multipliers the difference halves), biases are rounded to accumulator steps,
  and rounding is halves-up rather than halves-to-even;
- the picture outputs are rounded to 1/32 level;
- the state is rounded to 8 bits where it is produced, here; the simulation
  keeps it in float until it is next an input. This is the largest difference
  for a recurrent model and the one the simulation should adopt;
- unsigned inputs are clamped at 0 here, at -1 in the simulation (no sample in
  the measured frames was affected).

## 7. The runtime

Everything around the network for a luma-only model, in fixed point
(`cvnn_runtime::fixed`; the float definitions it stands in for are
`cvnn-runtime`'s session and `models/`). Shifts are arithmetic;
`round(v, n) = (v + 2^(n-1)) >> n`.

**Scales.** Picture samples that keep more than 8 bits (the conventional
upscale, the output, the previous output) count 1/32 of a level, 0..8160: the
network's residual step, so the two add exactly. Block motion vectors count
1/32 of an input sample. Per-sample flow counts 2^-13 of a sample.

**Conventional upscale.** The 2x kernels are exact in integers. Output sample
`i` along an axis takes taps starting at source `i/2 + i%2 - taps/2`, indices
clamped to the plane; bicubic weights (4 taps, 256ths) are -9, 67, 225, -27
for even `i` and the reverse for odd; bilinear (2 taps, quarters) 1, 3 and 3, 1.
Luma: `clamp(round(bicubic sum, 11), 0, 8160)`. Chroma: bilinear to luma size
(kept as the exact sum, 16ths), then `clamp(round(bicubic sum, 15), 0, 8160)`.

**Flow.** A sample at index `i` along an axis lies between block centres at
`p = 2i - 7` sixteenths: cells `p >> 4` and the next (clamped), weight of the
second `p - 16 (p >> 4)`. The flow at input size is the exact bilinear sum of
the block vectors (weights in 256ths). At output size it is the bilinear 2x
of that, `(sum + 4) >> 3`: sixteenths removed and doubled, an output sample
being half an input one.

**Warp.** A sample at index `i` displaced by flow `d` along an axis of `n`
samples reads at `a = clamp(i 2^13 + d, 0, (n-1) 2^13)`: first sample
`min(a >> 13, n-2)`, weight of the second `(a - first 2^13 + 64) >> 7`, in
64ths. A warped sample is `round(bilinear sum, 12)`. The state (signed 8-bit)
is warped at input size, the previous output (picture steps) at output size
and then folded 2x2 into four input planes, each `clamp(round(v, 5), 0, 255)`.

**Confidence.** The previous input luma is warped at input size and kept as
the unrounded sum (2^-12 level). Each 8x8 block's error is the sum of
`|luma 2^12 - warped|` over the samples it has, divided (truncating) by their
count. Spread between block centres as the flow is, `e = round(sum, 8)`, and
`confidence = min((255 max(10 2^12 - e, 0) + 14336) / 28672, 255)`: full at 3
levels of error, none at 10.

**Without usable history** (the first frame, a reset, a lost device, a lost
frame): confidence 255, state 0, and the conventional upscale stands in for
the previous output.

**Coding planes.** Known, quantiser / 51 and key frame, each
`round(255 x)` (halves to even); all zero when nothing is known.

**Finish.** Output luma is `clamp(conventional + residual, 0, 8160)` and is
what the next frame warps. RGBA from Y, U, V in picture steps, BT.709 limited
range, coefficients in 2^-14: with `L = 19077 (Y - 512)`, `b = U - 4096`,
`r = V - 4096`: red `L + 29372 r`, green `L - 8731 r - 3494 b`, blue
`L + 34610 b`, each `clamp(round(x, 19), 0, 255)`.

**Against the float runtime.** The conventional upscale differs only by its
rounding to 1/32 level. The rest differs by what fixed point drops: vectors
rounded to 1/32 sample, warp weights to 1/64, the previous output rounded to
1/32 level where it is produced instead of kept in float, the state warped
after its rounding to 8 bits instead of before.
