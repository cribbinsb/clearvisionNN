// The integer network in WGSL (docs/INTEGER.md is the arithmetic; the Rust
// reference in cvnn_runtime::integer is what these shaders are checked against).
//
// Every plane is four channels to a u32, a signed byte each; unsigned values
// are stored 128 below their value so that one signed packed dot product
// serves all of them. One shader per kind of layer serves every tiling and
// every grouping of tiles: an invocation finds its tile from its z
// coordinate, and a tile's activations sit together in the scratch buffers.
//
// The same arithmetic can be written several ways, and which is quick differs
// between GPUs by orders of magnitude, so the ways are options and the page
// times them (gpu-upscaler.js). Three things vary. How the weights are bound.
// How the products are formed: by the packed dot-product built-in, by integer
// multiplies on weights unpacked in the shader, or in floating point on weights
// unpacked beforehand (exact: see FLOAT_EXACT). And how many pixels of a row
// one invocation computes (`span`): every weight it reads then serves that
// many pixels, which matters where reading a buffer costs more than
// arithmetic; the price is that many times the accumulators held at once.
//
// Nearly every read a layer makes is of a weight, so how the weights are
// bound decides the speed: as a uniform array they run the extra-large model
// in under a third of the time a storage array takes on an NVIDIA GPU
// (measured; uniform reads are far cheaper there). The four outputs' weights
// of a word are one vec4 either way, so one read fetches them together.

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
}
// The weights of one input channel for four outputs: the byte "shift" bits below the top of each word.
fn column(w: vec4<u32>, shift: u32) -> vec4<i32> { return vec4<i32>(w << vec4<u32>(shift)) >> vec4<u32>(24u); }`;

// A sum of products of two signed bytes is exact in 32-bit floating point, in
// whatever order it is formed, while it cannot pass 2^24; a layer's can reach
// taps * input channels * 128 * 128.
const FLOAT_EXACT = 2 ** 24;
const LARGEST_PRODUCT = 128 * 128;

const range = (count) => [...Array(count).keys()];

/**
 * A layer's packed weights as the floating-point shaders read them: for each
 * vec4 of words (four outputs' weights of one input word), a 4x4 matrix whose
 * column i holds the four outputs' weights of input channel i.
 */
export function floatWeights(packed) {
  const floats = new Float32Array(packed.length * 4);
  packed.forEach((word, index) => {
    const output = index % 4;
    for (let channel = 0; channel < 4; channel += 1) {
      floats[(index - output) * 4 + channel * 4 + output] = (word << (24 - 8 * channel)) >> 24;
    }
  });
  return floats;
}

// In 16-bit float the bytes are scaled to within one, so the sums stay in range; the sum is
// brought back to whole numbers by this, the product of the two scales.
const HALF_SCALE = 128;

/**
 * A layer's weights as the 16-bit float shaders read them: `floatWeights`, each divided by
 * 128 and written as the bits of a half float (exact: a signed byte over 128 has few digits).
 */
export function halfWeights(packed) {
  const floats = floatWeights(packed);
  const halves = new Uint16Array(floats.length);
  floats.forEach((weight, index) => {
    const magnitude = Math.abs(weight);
    if (magnitude === 0) return;
    // weight / 128 = 1.m * 2^(top - 7), `top` being the magnitude's highest set bit.
    const top = 31 - Math.clz32(magnitude);
    halves[index] = (weight < 0 ? 0x8000 : 0) | ((top - 7 + 15) << 10) | ((magnitude << (10 - top)) & 0x3ff);
  });
  return halves;
}

/** How each way of forming the products declares the weights, holds an input word and its sums, and multiplies. */
const PRODUCTS = {
  packed: { weights: 'vec4<u32>', sum: 'vec4<i32>', input: (word) => word, whole: (sum) => sum, columns: '', product: (x) => `dots(${x}, m)` },
  plain: {
    weights: 'vec4<u32>', sum: 'vec4<i32>', input: (word) => `unpack8(${word})`, whole: (sum) => sum,
    columns: ' let w0 = column(m, 24u); let w1 = column(m, 16u); let w2 = column(m, 8u); let w3 = column(m, 0u);',
    product: (x) => `w0 * ${x}.x + w1 * ${x}.y + w2 * ${x}.z + w3 * ${x}.w`,
  },
  // Not exact: a 16-bit float holds eleven binary digits, which moves a layer's output by at most
  // one step of its 8-bit scale. It is what an Apple GPU runs fastest by far.
  half: {
    weights: 'mat4x4<f16>', sum: 'vec4<f16>', input: (word) => `(vec4<f16>(unpack8(${word})) * ${1 / HALF_SCALE}h)`,
    whole: (sum) => `vec4<i32>(round(vec4<f32>(${sum}) * ${HALF_SCALE * HALF_SCALE}.0))`, columns: '', product: (x) => `m * ${x}`,
  },
  float: { weights: 'mat4x4<f32>', sum: 'vec4<f32>', input: (word) => `vec4<f32>(unpack8(${word}))`, whole: (sum) => `vec4<i32>(${sum})`, columns: '', product: (x) => `m * ${x}` },
};

/** The weights of a layer with `count` elements of `type`: a uniform array where that fits, else a storage one. */
const weightsBinding = (type, count, uniform) => (uniform
  ? `@group(0) @binding(2) var<uniform> wts: array<${type}, ${count}>;`
  : `@group(0) @binding(2) var<storage, read> wts: array<${type}>;`);

/**
 * What differs between the stem, which reads the frame's input words (unsigned
 * and signed planes summed apart), and a layer reading a tile's activations,
 * beyond whose region the layer below computed lie zeros. `n` is the pixel of
 * the invocation's span.
 */
function reading(kind, shape) {
  const inFrame = (n) => `all(gg${n} >= vec2<i32>(0)) && all(gg${n} < p.frame)`;
  // The same tests and addresses for a sample named by where it lies: `l` in the tile's window,
  // `c` in the tile, `g` in the frame (the float forms read each sample once for all its uses).
  const placed = kind === 'stem'
    ? { valid: 'all(g >= vec2<i32>(0)) && all(g < p.frame)', index: `(g.y * p.frame.x + g.x) * ${shape.inputWords}` }
    : {
      valid: 'all(g >= vec2<i32>(0)) && all(g < p.frame) && all(c >= vec2<i32>(-p.reachBelow)) && all(c < p.tile + p.reachBelow)',
      index: `(base + l.y * p.window.x + l.x) * ${shape.trunkWords}`,
    };
  if (kind === 'stem') {
    return {
      placed,
      words: shape.inputWords,
      kinds: ['u', 's'],
      sums: (word) => (word < shape.unsignedWords ? 'u' : 's'),
      pad: (word) => (word < shape.unsignedWords ? PAD_UNSIGNED : PAD_SIGNED),
      index: (n) => `(gg${n}.y * p.frame.x + gg${n}.x) * ${shape.inputWords}`,
      valid: inFrame,
      accumulator: (whole, n, k) => `par[${k}] + ${shape.stemGains[0]} * ${whole(`u${n}_${k}`)} + ${shape.stemGains[1]} * ${whole(`s${n}_${k}`)}`,
    };
  }
  return {
    placed,
    words: shape.trunkWords,
    kinds: ['s'],
    sums: () => 's',
    pad: () => PAD_UNSIGNED,
    index: (n) => `(base + (l${n}.y + d.y) * p.window.x + l${n}.x + d.x) * ${shape.trunkWords}`,
    valid: (n) => `${inFrame(n)} && all(c${n} + d >= vec2<i32>(-p.reachBelow)) && all(c${n} + d < p.tile + p.reachBelow)`,
    accumulator: (whole, n, k) => `par[${k}] + ${whole(`s${n}_${k}`)}`,
  };
}

/** What each kind of layer writes for the pixel at `l` in its tile's window (`g` in the frame), given its accumulators a0, a1, ... */
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
 * word counts (int8-model.js). `packedDot` says whether the packed built-ins
 * exist, `uniformWeights` whether the weights are bound as a uniform buffer,
 * `floatProducts` whether the products are formed in floating point (the
 * weights are then `floatWeights`), `halfProducts` whether in 16-bit floating
 * point (`halfWeights`; not exact), and `span` how many pixels of a row an
 * invocation computes. Throws if floating point would not be exact.
 */
export function layerShader(kind, shape, { packedDot, uniformWeights, floatProducts = false, halfProducts = false, span = 1 }) {
  const outputWords = kind === 'head' ? 1 + shape.stateWords : shape.trunkWords;
  const read = reading(kind, shape);
  if (floatProducts && TAPS * 4 * read.words * LARGEST_PRODUCT > FLOAT_EXACT) throw new Error('A layer this wide is not exact in floating point.');
  const products = PRODUCTS[halfProducts ? 'half' : floatProducts ? 'float' : packedDot ? 'packed' : 'plain'];
  const pixels = range(span);
  const outputs = range(outputWords);
  const bounded = kind === 'head' && shape.stateWords
    ? `@group(0) @binding(5) var<storage, read> tanhTable: array<u32>;
fn bounded(magnitude: i32) -> i32 { return i32((tanhTable[magnitude >> 2u] >> (u32(magnitude & 3) * 8u)) & 255u); }` : '';
  const words = range(read.words).map((word) => `{ let w = (t * ${read.words} + ${word}) * ${outputWords};
      ${pixels.map((n) => `let x${n} = ${products.input(`select(${read.pad(word)}, src[b${n} + ${word}], ok${n})`)};`).join(' ')}
      ${outputs.map((k) => `{ let m = wts[w + ${k}];${products.columns} ${pixels.map((n) => `${read.sums(word)}${n}_${k} += ${products.product(`x${n}`)};`).join(' ')} }`).join('\n      ')} }`);
  // The float forms go row by row: a row's samples (the span's pixels and one either side) are
  // read and converted once, then each weight is read once and used for every pixel it serves.
  const floating = floatProducts || halfProducts;
  const columns = range(span + 2);
  const rows = `for (var dy = -1; dy <= 1; dy++) {
    ${columns.map((j) => `var ok${j}: bool; var b${j}: i32; { let l = first + vec2<i32>(${j - 1}, dy); let c = l - p.halo; let g = corner + c;
      ok${j} = ${read.placed.valid}; b${j} = select(0, ${read.placed.index}, ok${j}); }`).join('\n    ')}
    ${range(read.words).map((word) => `{ ${columns.map((j) => `let v${j} = ${products.input(`select(${read.pad(word)}, src[b${j} + ${word}], ok${j})`)};`).join(' ')}
      ${range(3).map((dx) => `{ let w = (((dy + 1) * 3 + ${dx}) * ${read.words} + ${word}) * ${outputWords};
        ${outputs.map((k) => `{ let m = wts[w + ${k}]; ${pixels.map((n) => `${read.sums(word)}${n}_${k} += m * v${n + dx};`).join(' ')} }`).join('\n        ')} }`).join('\n      ')} }`).join('\n    ')}
  }`;
  return `${halfProducts ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
${weightsBinding(products.weights, TAPS * read.words * outputWords, uniformWeights)}
@group(0) @binding(3) var<storage, read> par: array<vec4<i32>>;
${kind === 'last' ? '@group(0) @binding(4) var<storage, read> stem: array<u32>;' : ''}
${bounded}
${packingHelpers(packedDot)}
${COMMON}
fn finish(l: vec2<i32>, g: vec2<i32>, base: i32, ${outputs.map((k) => `a${k}: vec4<i32>`).join(', ')}) {
  ${writes(kind, shape)}
}
@compute @workgroup_size(${GROUP_SIDE}, ${GROUP_SIDE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = i32(gid.z);
  let index = p.firstTile + slot;
  if (index >= p.grid.x * p.grid.y) { return; }
  let first = vec2<i32>(i32(gid.x) * ${span}, i32(gid.y));
  let corner = vec2<i32>(index % p.grid.x, index / p.grid.x) * p.tile;
  ${pixels.map((n) => `let l${n} = first + vec2<i32>(${n}, 0); let c${n} = l${n} - p.halo; let g${n} = corner + c${n};
  let live${n} = all(c${n} >= vec2<i32>(-p.reach)) && all(c${n} < p.tile + p.reach) && all(g${n} >= vec2<i32>(0)) && all(g${n} < p.frame);`).join('\n  ')}
  if (!(${pixels.map((n) => `live${n}`).join(' || ')})) { return; }
  let base = slot * p.window.x * p.window.y;
  ${pixels.map((n) => outputs.map((k) => read.kinds.map((sum) => `var ${sum}${n}_${k} = ${products.sum}(0);`).join(' ')).join(' ')).join('\n  ')}
  ${floating ? rows : `for (var t = 0; t < ${TAPS}; t++) {
    let d = vec2<i32>(t % 3 - 1, t / 3 - 1);
    ${pixels.map((n) => `let gg${n} = g${n} + d; let ok${n} = ${read.valid(n)}; let b${n} = select(0, ${read.index(n)}, ok${n});`).join('\n    ')}
    ${words.join('\n    ')}
  }`}
  ${pixels.map((n) => `if (live${n}) { finish(l${n}, g${n}, base, ${outputs.map((k) => read.accumulator(products.whole, n, k)).join(', ')}); }`).join('\n  ')}
}`;
}
