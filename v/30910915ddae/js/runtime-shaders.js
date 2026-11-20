// Everything around the integer network in WGSL: the conventional upscale,
// the motion field as per-sample flow, warping the previous output and the
// state onto the current frame, the confidence plane, assembling the
// network's input, adding its detail, and colour conversion. All of it is
// integer arithmetic defined in docs/INTEGER.md ("The runtime"); the
// reference is cvnn_runtime::fixed, which these shaders must match bit for bit.

import { packingHelpers } from './int8-forms.js';

export const GROUP_SIDE = 8; // workgroup side, in samples
const WORKGROUP = `@compute @workgroup_size(${GROUP_SIDE}, ${GROUP_SIDE})`;

// What every pass knows of the frame: the input size, whether history is
// usable, and the coding planes' values.
const FRAME = `
struct Frame { size: vec2<i32>, valid: i32, unused: i32, coding: vec4<i32> }
@group(0) @binding(0) var<uniform> f: Frame;`;

// 2x kernels: taps of an output sample along one axis, as crate::fixed::upscale.
const KERNELS = `
const BICUBIC = array<vec4<i32>, 2>(vec4<i32>(-9, 67, 225, -27), vec4<i32>(-27, 225, 67, -9));
const BILINEAR = array<vec2<i32>, 2>(vec2<i32>(1, 3), vec2<i32>(3, 1));
fn firstTap(index: i32, taps: i32) -> i32 { return index / 2 + index % 2 - taps / 2; }`;

// Picture steps (1/32 level) to an RGBA word: BT.709 limited range, coefficients in 2^-14.
const COLOUR = `
fn rgba(y: i32, u: i32, v: i32) -> u32 {
  let luma = 19077 * (y - 512);
  let cb = u - 4096;
  let cr = v - 4096;
  let sums = vec3<i32>(luma + 29372 * cr, luma - 8731 * cr - 3494 * cb, luma + 34610 * cb);
  let c = vec3<u32>(clamp((sums + vec3<i32>(1 << 18u)) >> vec3<u32>(19u), vec3<i32>(0), vec3<i32>(255)));
  return c.x | (c.y << 8u) | (c.z << 16u) | 0xff000000u;
}`;

// One axis of a warp: the first of two samples and the second's weight in 64ths.
const WARP = `
fn axis(position: i32, flow: i32, len: i32) -> vec2<i32> {
  let last = len - 1;
  let at = clamp((position << 13u) + flow, 0, last << 13u);
  let first = min(at >> 13u, last - 1);
  return vec2<i32>(first, (at - (first << 13u) + 64) >> 7u);
}`;

// A sample between block centres: the two cells and the second's weight in 16ths.
const SPREAD = `
fn spreadTap(index: i32, cells: i32) -> vec3<i32> {
  let position = 2 * index - 7;
  let base = position >> 4u;
  return vec3<i32>(clamp(base, 0, cells - 1), clamp(base + 1, 0, cells - 1), position - (base << 4u));
}
fn gridSize() -> vec2<i32> { return (f.size + 7) / 8; }`;

const bytesOf = (name) => `fn ${name}At(i: i32) -> i32 { return i32((${name}[i >> 2u] >> (u32(i & 3) * 8u)) & 255u); }`;

/** Chroma brought to luma size bilinearly: both planes of a sample in one word, in 16ths of a level. */
export const WIDE_CHROMA = `${FRAME}${KERNELS}
@group(0) @binding(1) var<storage, read> chroma: array<u32>;
@group(0) @binding(2) var<storage, read_write> wide: array<u32>;
${bytesOf('chroma')}
${WORKGROUP}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = vec2<i32>(gid.xy);
  if (any(p >= f.size)) { return; }
  let half = f.size / 2;
  let first = vec2<i32>(firstTap(p.x, 2), firstTap(p.y, 2));
  var sums = vec2<i32>(0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let s = clamp(first + vec2<i32>(i, j), vec2<i32>(0), half - 1);
      let at = s.y * half.x + s.x;
      let weight = BILINEAR[p.x % 2][i] * BILINEAR[p.y % 2][j];
      sums += weight * vec2<i32>(chromaAt(at), chromaAt(half.x * half.y + at));
    }
  }
  wide[p.y * f.size.x + p.x] = u32(sums.x) | (u32(sums.y) << 16u);
}`;

/** The conventional upscale at output size: luma and chroma in picture steps, and as RGBA. */
export const CONVENTIONAL = `${FRAME}${KERNELS}${COLOUR}
@group(0) @binding(1) var<storage, read> luma: array<u32>;
@group(0) @binding(2) var<storage, read> wide: array<u32>;
@group(0) @binding(3) var<storage, read_write> base: array<i32>;
@group(0) @binding(4) var<storage, read_write> colour: array<u32>;
@group(0) @binding(5) var<storage, read_write> picture: array<u32>;
${bytesOf('luma')}
${WORKGROUP}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = vec2<i32>(gid.xy);
  if (any(p >= 2 * f.size)) { return; }
  let first = vec2<i32>(firstTap(p.x, 4), firstTap(p.y, 4));
  var y = 0;
  var uv = vec2<i32>(0);
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      let s = clamp(first + vec2<i32>(i, j), vec2<i32>(0), f.size - 1);
      let at = s.y * f.size.x + s.x;
      let weight = BICUBIC[p.x % 2][i] * BICUBIC[p.y % 2][j];
      y += weight * lumaAt(at);
      uv += weight * vec2<i32>(i32(wide[at] & 0xffffu), i32(wide[at] >> 16u));
    }
  }
  // Sums carry 2^16 per level (luma) and 2^20 (chroma); picture steps are 2^5.
  let steps = clamp(vec3<i32>((y + 1024) >> 11u, (uv.x + 16384) >> 15u, (uv.y + 16384) >> 15u), vec3<i32>(0), vec3<i32>(8160));
  let o = p.y * 2 * f.size.x + p.x;
  base[o] = steps.x;
  colour[o] = u32(steps.y) | (u32(steps.z) << 16u);
  picture[o] = rgba(steps.x, steps.y, steps.z);
}`;

/** The block motion field as a flow per input sample, in 2^-13 of a sample. */
export const FLOW = `${FRAME}${SPREAD}
@group(0) @binding(1) var<storage, read> vectors: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> flow: array<vec2<i32>>;
${WORKGROUP}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = vec2<i32>(gid.xy);
  if (any(p >= f.size)) { return; }
  let grid = gridSize();
  let across = spreadTap(p.x, grid.x);
  let down = spreadTap(p.y, grid.y);
  let top = vectors[down.x * grid.x + across.x] * (16 - across.z) + vectors[down.x * grid.x + across.y] * across.z;
  let bottom = vectors[down.y * grid.x + across.x] * (16 - across.z) + vectors[down.y * grid.x + across.y] * across.z;
  flow[p.y * f.size.x + p.x] = top * (16 - down.z) + bottom * down.z;
}`;

/** Each block's mean error of the previous luma, warped onto the frame, against the current one. */
export const BLOCK_ERRORS = `${FRAME}${WARP}${SPREAD}
@group(0) @binding(1) var<storage, read> luma: array<u32>;
@group(0) @binding(2) var<storage, read> before: array<u32>;
@group(0) @binding(3) var<storage, read> flow: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read_write> errors: array<i32>;
${bytesOf('luma')}
${bytesOf('before')}
${WORKGROUP}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let block = vec2<i32>(gid.xy);
  let grid = gridSize();
  if (any(block >= grid)) { return; }
  var sum = 0;
  var count = 0;
  for (var j = 0; j < 8; j++) {
    for (var i = 0; i < 8; i++) {
      let p = 8 * block + vec2<i32>(i, j);
      if (any(p >= f.size)) { continue; }
      let at = p.y * f.size.x + p.x;
      let x = axis(p.x, flow[at].x, f.size.x);
      let y = axis(p.y, flow[at].y, f.size.y);
      let a = y.x * f.size.x + x.x;
      let above = beforeAt(a) * (64 - x.y) + beforeAt(a + 1) * x.y;
      let below = beforeAt(a + f.size.x) * (64 - x.y) + beforeAt(a + f.size.x + 1) * x.y;
      sum += abs((lumaAt(at) << 12u) - (above * (64 - y.y) + below * y.y));
      count += 1;
    }
  }
  errors[block.y * grid.x + block.x] = sum / count;
}`;

/**
 * The network's input: each input word assembled from the planes its slots
 * name. `model` gives the words (int8-model.js); `contract` says which
 * channel is what: luma; confidence; the state; the previous output folded
 * to input size; the coding planes.
 */
export function assembleShader(model, contract, packedDot) {
  const recurrent = contract.temporal === 'recurrent';
  const confidence = contract.confidence_plane ? 1 : 0;
  const state = recurrent ? contract.state_channels : 0;
  const folded = recurrent ? 4 : 0;
  const firstState = 1 + confidence;
  const firstFolded = firstState + state;
  const firstCoding = firstFolded + folded;
  const value = (channel) => {
    if (channel === 0) return 'lumaAt(at)';
    if (channel < firstState) return 'trust';
    if (channel < firstFolded) return `s${(channel - firstState) >> 2}[${(channel - firstState) & 3}]`;
    if (channel < firstCoding) return `carried[${channel - firstFolded}]`;
    return `f.coding[${channel - firstCoding}]`;
  };
  const words = [...Array(model.inputWords).keys()].map((word) => {
    const unsigned = word < model.unsignedWords;
    const lanes = model.slots.slice(4 * word, 4 * word + 4)
      .map((channel) => (channel < 0 ? '0' : value(channel)));
    // Unsigned planes are stored 128 below their value.
    return `input[at * ${model.inputWords} + ${word}] = pack8(vec4<i32>(${lanes.join(', ')})${unsigned ? ' - 128' : ''});`;
  });
  const stateWords = [...Array(state / 4).keys()];
  const stored = (offset) => stateWords.map((w) => `unpack8(state[(${offset}) * ${model.outputWords} + ${2 + w}])`);
  const warpedState = stateWords.map((w) => {
    const [a, b, c, d] = ['a', 'a + 1', 'a + f.size.x', 'a + f.size.x + 1'].map((offset) => stored(offset)[w]);
    return `s${w} = ((${a} * (64 - x.y) + ${b} * x.y) * (64 - y.y) + (${c} * (64 - x.y) + ${d} * x.y) * y.y + 2048) >> vec4<u32>(12u);`;
  });
  return `${FRAME}${KERNELS}${WARP}${SPREAD}
@group(0) @binding(1) var<storage, read> luma: array<u32>;
@group(0) @binding(2) var<storage, read> base: array<i32>;
@group(0) @binding(3) var<storage, read> flow: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read> errors: array<i32>;
@group(0) @binding(5) var<storage, read> previous: array<i32>;
@group(0) @binding(6) var<storage, read> state: array<u32>;
@group(0) @binding(7) var<storage, read_write> input: array<u32>;
${bytesOf('luma')}
${packingHelpers(packedDot)}
// Confidence of a sample from its block errors, spread between block centres.
fn confidenceAt(p: vec2<i32>) -> i32 {
  let grid = gridSize();
  let across = spreadTap(p.x, grid.x);
  let down = spreadTap(p.y, grid.y);
  let top = errors[down.x * grid.x + across.x] * (16 - across.z) + errors[down.x * grid.x + across.y] * across.z;
  let bottom = errors[down.y * grid.x + across.x] * (16 - across.z) + errors[down.y * grid.x + across.y] * across.z;
  let error = (top * (16 - down.z) + bottom * down.z + 128) >> 8u;
  return min((255 * max((10 << 12u) - error, 0) + 14336) / 28672, 255);
}
// The flow of an output sample: the input flow upscaled bilinearly, in output samples.
fn fineFlow(q: vec2<i32>) -> vec2<i32> {
  let first = vec2<i32>(firstTap(q.x, 2), firstTap(q.y, 2));
  var sum = vec2<i32>(0);
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      let s = clamp(first + vec2<i32>(i, j), vec2<i32>(0), f.size - 1);
      sum += BILINEAR[q.x % 2][i] * BILINEAR[q.y % 2][j] * flow[s.y * f.size.x + s.x];
    }
  }
  return (sum + 4) >> vec2<u32>(3u);
}
// The previous output brought onto output sample q, in picture steps.
fn carriedAt(q: vec2<i32>) -> i32 {
  let fine = 2 * f.size;
  let motion = fineFlow(q);
  let x = axis(q.x, motion.x, fine.x);
  let y = axis(q.y, motion.y, fine.y);
  let a = y.x * fine.x + x.x;
  let above = previous[a] * (64 - x.y) + previous[a + 1] * x.y;
  let below = previous[a + fine.x] * (64 - x.y) + previous[a + fine.x + 1] * x.y;
  return (above * (64 - y.y) + below * y.y + 2048) >> 12u;
}
${WORKGROUP}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = vec2<i32>(gid.xy);
  if (any(p >= f.size)) { return; }
  let at = p.y * f.size.x + p.x;
  // Without usable history the conventional upscale stands in, the state is
  // empty, and there is no prediction to distrust.
  var trust = 255;
  var carried = vec4<i32>(0);
  ${stateWords.map((w) => `var s${w} = vec4<i32>(0);`).join(' ')}
  ${recurrent ? `for (var k = 0; k < 4; k++) {
    let q = 2 * p + vec2<i32>(k % 2, k / 2);
    var steps = base[q.y * 2 * f.size.x + q.x];
    if (f.valid != 0) { steps = carriedAt(q); }
    carried[k] = clamp((steps + 16) >> 5u, 0, 255);
  }
  if (f.valid != 0) {
    ${confidence ? 'trust = confidenceAt(p);' : ''}
    let x = axis(p.x, flow[at].x, f.size.x);
    let y = axis(p.y, flow[at].y, f.size.y);
    let a = y.x * f.size.x + x.x;
    ${warpedState.join('\n    ')}
  }` : ''}
  ${words.join('\n  ')}
}`;
}

/** The network's detail added to the conventional upscale: the output, kept for the next frame, and as RGBA. */
export const finishShader = (outputWords) => `${FRAME}${COLOUR}
@group(0) @binding(1) var<storage, read> base: array<i32>;
@group(0) @binding(2) var<storage, read> colour: array<u32>;
@group(0) @binding(3) var<storage, read> outputs: array<u32>;
@group(0) @binding(4) var<storage, read_write> previous: array<i32>;
@group(0) @binding(5) var<storage, read_write> picture: array<u32>;
${WORKGROUP}
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = vec2<i32>(gid.xy);
  if (any(p >= f.size)) { return; }
  let at = (p.y * f.size.x + p.x) * ${outputWords};
  // Four 16-bit residuals, for the 2x2 output samples row by row.
  let pair = vec2<i32>(bitcast<i32>(outputs[at]), bitcast<i32>(outputs[at + 1]));
  let residual = vec4<i32>((pair.x << 16u) >> 16u, pair.x >> 16u, (pair.y << 16u) >> 16u, pair.y >> 16u);
  for (var k = 0; k < 4; k++) {
    let q = 2 * p + vec2<i32>(k % 2, k / 2);
    let o = q.y * 2 * f.size.x + q.x;
    let sample = clamp(base[o] + residual[k], 0, 8160);
    previous[o] = sample;
    picture[o] = rgba(sample, i32(colour[o] & 0xffffu), i32(colour[o] >> 16u));
  }
}`;

/** A picture's RGBA words drawn to a canvas of its own size. */
export const PRESENT = `
@group(0) @binding(0) var<storage, read> picture: array<u32>;
@group(0) @binding(1) var<uniform> width: vec4<u32>;
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(f32(i32(index & 1u) * 4 - 1), f32(i32(index >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let at = vec2<u32>(position.xy);
  return unpack4x8unorm(picture[at.y * width.x + at.x]);
}`;
