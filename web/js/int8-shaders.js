// The integer network in WGSL (docs/INTEGER.md is the arithmetic; the Rust
// reference in cvnn_runtime::integer is what these shaders are checked against).
//
// Every plane is four channels to a u32, a signed byte each; unsigned values
// are stored 128 below their value so that one signed packed dot product
// serves all of them. One shader per kind of layer serves every tiling and
// every grouping of tiles: an invocation finds its tile from its z
// coordinate, and a tile's activations sit together in the scratch buffers.
//
// Nearly every read a layer makes is of a weight, so how the weights are
// bound decides the speed: as a uniform array they run the extra-large model
// in under a third of the time a storage array takes on an NVIDIA GPU
// (measured; uniform reads are far cheaper there). The four outputs' weights
// of a word are one vec4 either way, so one read fetches them together.
//
// The products themselves can be formed two ways. Packed 8-bit dot products
// are what GPUs with hardware for them (NVIDIA, AMD, Intel) run fastest. An
// Apple GPU has none and runs integer arithmetic slowly: there the products
// are formed in 32-bit float, about sixteen times faster (measured on an M5),
// and exactly: a layer's sum of products stays below 2^24, up to which every
// whole number is a float (`FLOAT_EXACT_TERMS`). The weights are then a 4x4
// float matrix per input word and group of four outputs.

export const GROUP_SIDE = 8; // workgroup side, in pixels
const TAPS = 9;
const PAD_UNSIGNED = '0x80808080u'; // four stored zeros of an unsigned plane
const PAD_SIGNED = '0u';

// Chrome has the packed built-ins; where they are missing the same arithmetic is spelt out.
const PACKED = `
fn dot8(a: u32, b: u32) -> i32 { return dot4I8Packed(a, b); }
fn pack8(v: vec4<i32>) -> u32 { return pack4xI8Clamp(v); }
fn unpack8(a: u32) -> vec4<i32> { return unpack4xI8(a); }`;
const SPELT_OUT = `
fn unpack8(a: u32) -> vec4<i32> { return vec4<i32>(vec4<u32>(a) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u); }
fn dot8(a: u32, b: u32) -> i32 { return dot(unpack8(a), unpack8(b)); }
fn pack8(v: vec4<i32>) -> u32 {
  let c = vec4<u32>(clamp(v, vec4<i32>(-128), vec4<i32>(127)) & vec4<i32>(255));
  return c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u);
}`;

/** WGSL for dot8, pack8 and unpack8: the packed built-ins, or the same arithmetic spelt out. */
export const packingHelpers = (packedDot) => (packedDot ? PACKED : SPELT_OUT);

const COMMON = `
struct Params { frame: vec2<i32>, grid: vec2<i32>, tile: vec2<i32>, window: vec2<i32>, halo: i32, reach: i32, reachBelow: i32, firstTile: i32 }
@group(1) @binding(0) var<uniform> p: Params;
// accumulator * multiplier / 2^16, rounded down, without leaving 32 bits.
fn scaled(a: vec4<i32>, m: vec4<i32>) -> vec4<i32> {
  return (a >> vec4<u32>(16u)) * m + (((a & vec4<i32>(65535)) * m) >> vec4<u32>(16u));
}
// value / 2^shift, halves towards positive infinity.
fn rounded(v: vec4<i32>, shift: vec4<i32>) -> vec4<i32> {
  let s = vec4<u32>(shift);
  return (v + vec4<i32>((vec4<u32>(1u) << s) >> vec4<u32>(1u))) >> s;
}
// An input word against the weights of four outputs.
fn dots(x: u32, w: vec4<u32>) -> vec4<i32> {
  return vec4<i32>(dot8(x, w.x), dot8(x, w.y), dot8(x, w.z), dot8(x, w.w));
}`;

const range = (count) => [...Array(count).keys()];

// The most products a float accumulator may sum and stay exact: each is at most 128 * 128, and
// whole numbers are exact in a 32-bit float up to 2^24.
export const FLOAT_EXACT_TERMS = Math.floor(2 ** 24 / (128 * 128));

/**
 * The weights of a layer with `vectors` groups of them (a vec4 of packed words, or a 4x4 float
 * matrix): a uniform array where that fits, else a storage one.
 */
function weightsBinding(vectors, uniform, float) {
  const element = float ? 'mat4x4<f32>' : 'vec4<u32>';
  return uniform
    ? `@group(0) @binding(2) var<uniform> wts: array<${element}, ${vectors}>;`
    : `@group(0) @binding(2) var<storage, read> wts: array<${element}>;`;
}

/** How a tap's products are added up: in packed integers, or (`float`) in exact 32-bit float. */
function products(float) {
  return float
    ? { zero: 'vec4<f32>(0.0)', word: (read) => `vec4<f32>(unpack8(${read}))`, add: (sum, k) => `${sum}${k} += wts[w + ${k}] * x;`, whole: (name) => `vec4<i32>(${name})` }
    : { zero: 'vec4<i32>(0)', word: (read) => read, add: (sum, k) => `${sum}${k} += dots(x, wts[w + ${k}]);`, whole: (name) => name };
}

/** The taps of the stem: input words read from the frame, unsigned and signed planes summed apart. */
function stemTaps(shape, float) {
  const outputs = range(shape.trunkWords);
  const form = products(float);
  const words = range(shape.inputWords).map((word) => {
    const unsigned = word < shape.unsignedWords;
    const sum = unsigned ? 'u' : 's';
    return `{ let x = ${form.word(`select(${unsigned ? PAD_UNSIGNED : PAD_SIGNED}, src[b + ${word}], ok)`)}; let w = (t * ${shape.inputWords} + ${word}) * ${shape.trunkWords};
      ${outputs.map((k) => form.add(sum, k)).join('\n      ')} }`;
  });
  return {
    declare: outputs.map((k) => `var u${k} = ${form.zero}; var s${k} = ${form.zero};`).join(' '),
    index: `(gg.y * p.frame.x + gg.x) * ${shape.inputWords}`,
    valid: 'all(gg >= vec2<i32>(0)) && all(gg < p.frame)',
    body: words.join('\n    '),
    accumulators: outputs.map((k) => `let a${k} = par[${k}] + ${shape.stemGains[0]} * ${form.whole(`u${k}`)} + ${shape.stemGains[1]} * ${form.whole(`s${k}`)};`).join('\n  '),
  };
}

/** The taps of a layer reading a tile's activations; beyond the region the layer below computed lie zeros. */
function trunkTaps(shape, outputWords, float) {
  const outputs = range(outputWords);
  const form = products(float);
  const words = range(shape.trunkWords).map((word) => `{ let x = ${form.word(`select(${PAD_UNSIGNED}, src[b + ${word}], ok)`)}; let w = (t * ${shape.trunkWords} + ${word}) * ${outputWords};
      ${outputs.map((k) => form.add('q', k)).join('\n      ')} }`);
  return {
    declare: outputs.map((k) => `var q${k} = ${form.zero};`).join(' '),
    index: `(base + (l.y + d.y) * p.window.x + l.x + d.x) * ${shape.trunkWords}`,
    valid: 'all(gg >= vec2<i32>(0)) && all(gg < p.frame) && all(c + d >= vec2<i32>(-p.reachBelow)) && all(c + d < p.tile + p.reachBelow)',
    body: words.join('\n    '),
    // The bias joins the sum of products afterwards: it may be larger than a float holds exactly.
    accumulators: outputs.map((k) => `let a${k} = par[${k}] + ${form.whole(`q${k}`)};`).join('\n  '),
  };
}

/** What each kind of layer writes, given its accumulators a0, a1, ... */
function writes(kind, shape) {
  const words = shape.trunkWords;
  const at = `let o = (base + l.y * p.window.x + l.x) * ${words};`;
  const requantised = (k, extra = '') => `rounded(scaled(a${k}, par[${words + k}])${extra}, par[${2 * words + k}])`;
  if (kind === 'stem') return `${at}\n  ${range(words).map((k) => `dst[o + ${k}] = pack8(${requantised(k)} - 128);`).join('\n  ')}`;
  const own = (k) => ` + par[${3 * words + k}] * (unpack8(src[o + ${k}]) + 128)`;
  if (kind === 'body') return `${at}\n  ${range(words).map((k) => `dst[o + ${k}] = pack8(${requantised(k, own(k))} - 128);`).join('\n  ')}`;
  if (kind === 'last') {
    // The last block's output and the stem's, brought onto the head's input scale.
    return `${at} let skipBy = par[${4 * words}];\n  ${range(words).map((k) => `{ let g = clamp(${requantised(k, own(k))}, vec4<i32>(0), vec4<i32>(255));
    let f = unpack8(stem[o + ${k}]) + 128;
    dst[o + ${k}] = pack8(rounded(g * skipBy.y + f * skipBy.x, vec4<i32>(skipBy.z)) - 128); }`).join('\n  ')}`;
  }
  const headWords = 1 + shape.stateWords;
  const head = (k) => `rounded(scaled(a${k}, par[${headWords + k}]), par[${2 * headWords + k}])`;
  const state = range(shape.stateWords).map((k) => `{ let v = ${head(1 + k)}; let m = min(abs(v), vec4<i32>(par[${3 * headWords}].x));
    dst[o + ${2 + k}] = pack8(sign(v) * vec4<i32>(bounded(m.x), bounded(m.y), bounded(m.z), bounded(m.w))); }`);
  return `let o = (g.y * p.frame.x + g.x) * ${shape.outputWords};
  let r = vec4<u32>(clamp(${head(0)}, vec4<i32>(-32768), vec4<i32>(32767)) & vec4<i32>(65535));
  dst[o] = r.x | (r.y << 16u); dst[o + 1] = r.z | (r.w << 16u);
  ${state.join('\n  ')}`;
}

/**
 * The compute shader of one kind of layer: 'stem', 'body', 'last' (the final
 * block, which also adds the long skip) or 'head'. `shape` is the model's
 * word counts (int8-model.js); `packedDot` says whether the packed built-ins
 * exist, `uniformWeights` whether the weights are bound as a uniform buffer,
 * `floatProducts` whether the products are formed in float (the weights are
 * then `floatWeights` of the model's).
 */
export function layerShader(kind, shape, packedDot, uniformWeights, floatProducts = false) {
  const inputWords = kind === 'stem' ? shape.inputWords : shape.trunkWords;
  const outputWords = kind === 'head' ? 1 + shape.stateWords : shape.trunkWords;
  const taps = kind === 'stem' ? stemTaps(shape, floatProducts) : trunkTaps(shape, outputWords, floatProducts);
  const bounded = kind === 'head' && shape.stateWords
    ? `@group(0) @binding(5) var<storage, read> tanhTable: array<u32>;
fn bounded(magnitude: i32) -> i32 { return i32((tanhTable[magnitude >> 2u] >> (u32(magnitude & 3) * 8u)) & 255u); }` : '';
  return `
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
${weightsBinding(TAPS * inputWords * outputWords, uniformWeights, floatProducts)}
@group(0) @binding(3) var<storage, read> par: array<vec4<i32>>;
${kind === 'last' ? '@group(0) @binding(4) var<storage, read> stem: array<u32>;' : ''}
${bounded}
${packingHelpers(packedDot)}
${COMMON}
@compute @workgroup_size(${GROUP_SIDE}, ${GROUP_SIDE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = vec2<i32>(gid.xy);
  let slot = i32(gid.z);
  let index = p.firstTile + slot;
  if (index >= p.grid.x * p.grid.y) { return; }
  let c = l - p.halo;
  let g = vec2<i32>(index % p.grid.x, index / p.grid.x) * p.tile + c;
  if (any(c < vec2<i32>(-p.reach)) || any(c >= p.tile + p.reach) || any(g < vec2<i32>(0)) || any(g >= p.frame)) { return; }
  let base = slot * p.window.x * p.window.y;
  ${taps.declare}
  for (var t = 0; t < ${TAPS}; t++) {
    let d = vec2<i32>(t % 3 - 1, t / 3 - 1);
    let gg = g + d;
    let ok = ${taps.valid};
    let b = select(0, ${taps.index}, ok);
    ${taps.body}
  }
  ${taps.accumulators}
  ${writes(kind, shape)}
}`;
}

const BYTES_PER_WORD = 4;
const MATRIX_ELEMENTS = 16;

/**
 * A layer's packed weights (four words to a group: one per output, a signed byte per input
 * channel) as the 4x4 float matrices the float shaders multiply an input word by: column j,
 * row i is the weight of input channel j for output i.
 */
export function floatWeights(packed) {
  const words = new Uint32Array(packed.buffer, packed.byteOffset, packed.byteLength / BYTES_PER_WORD);
  const matrices = new Float32Array((words.length / BYTES_PER_WORD) * MATRIX_ELEMENTS);
  for (let group = 0; group < words.length / BYTES_PER_WORD; group += 1) {
    for (let output = 0; output < BYTES_PER_WORD; output += 1) {
      const word = words[group * BYTES_PER_WORD + output];
      for (let input = 0; input < BYTES_PER_WORD; input += 1) {
        // A signed byte: shifted to the top of the word and back down.
        matrices[group * MATRIX_ELEMENTS + input * BYTES_PER_WORD + output] = (word << (24 - 8 * input)) >> 24;
      }
    }
  }
  return matrices;
}
