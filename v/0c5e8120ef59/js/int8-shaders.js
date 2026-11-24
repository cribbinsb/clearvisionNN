// The integer network in WGSL (docs/INTEGER.md is the arithmetic; the Rust
// reference in cvnn_runtime::integer is what these shaders are checked against).
//
// Every plane is four channels to a u32, a signed byte each; unsigned values
// are stored 128 below their value. One shader per kind of layer serves every
// tiling and every grouping of tiles: an invocation finds its tile from its z
// coordinate, and a tile's activations sit together in the scratch buffers, in
// a window of the tile, its halo and a border one pixel wide.
//
// One structure serves every form of the products (int8-forms.js). What a
// layer costs is reading: of samples and, above all, of weights. So:
//   - an invocation computes `span` pixels of a row, and goes row by row of
//     the 3x3: a row's samples (the span's and one either side) are read and
//     converted once, then each weight is read once and used for every pixel
//     and every tap of the row it serves;
//   - the rows are a loop, not written out three times: written out, a shader
//     takes many times longer to compile and (measured on an Apple GPU) often
//     runs slower. The words of a sample can be a loop too (`wordLoop`), which
//     shortens the shader by the number of words: a large model's shaders
//     otherwise take minutes to compile on some drivers;
//   - a trunk layer's output words can be shared out between `split`
//     invocations, each then holding that much less at once;
//   - no sample of a tile's activations is tested for lying outside what the
//     layer below computed: every layer writes stored zeros one pixel round
//     what it computes (hence the border), so what is read there is the zero
//     the arithmetic asks for. Only the stem, which reads the frame's own
//     input, tests for the frame's edge.
// The weights are bound as a uniform array where they fit: reading one is far
// cheaper than from a storage array on an NVIDIA GPU (measured).

import { FORMS, packingHelpers } from './int8-forms.js';

export const GROUP_SIDE = 8; // workgroup side, in invocations
export const BORDER = 1; // pixels of stored zeros round a window: as far as a 3x3 layer reads beyond what it writes
const ROWS = 3;
const ACROSS = 3; // taps of a row
const PAD_UNSIGNED = '0x80808080u'; // four stored zeros of an unsigned plane
const PAD_SIGNED = '0u';

const COMMON = `
struct Params { frame: vec2<i32>, grid: vec2<i32>, tile: vec2<i32>, window: vec2<i32>, inset: i32, reach: i32, reachBelow: i32, firstTile: i32 }
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

/** The weights of a layer with `count` elements of `type`: a uniform array, or a storage one. */
const weightsBinding = (type, count, uniform) => (uniform
  ? `@group(0) @binding(2) var<uniform> wts: array<${type}, ${count}>;`
  : `@group(0) @binding(2) var<storage, read> wts: array<${type}>;`);

/**
 * What a kind of layer reads. A sample is named by where it lies: `l` in the
 * tile's window, `c` in the tile, `g` in the frame. `index` is its first word
 * in `src`; `valid`, where there is one, is the test it must pass to be read
 * rather than taken as zero. `runs` are its words in runs of one signedness:
 * the stem reads the frame's input words, unsigned planes first and then
 * signed, summed apart; the others a tile's activations, all unsigned.
 */
function reading(kind, shape, edgeTests) {
  const inFrame = 'all(g >= vec2<i32>(0)) && all(g < p.frame)';
  if (kind === 'stem') {
    return {
      words: shape.inputWords,
      runs: [{ first: 0, end: shape.unsignedWords, signed: false }, { first: shape.unsignedWords, end: shape.inputWords, signed: true }],
      sums: ['u', 's'],
      valid: inFrame,
      index: `(g.y * p.frame.x + g.x) * ${shape.inputWords}`,
      accumulator: (whole, bias, n, k) => `${bias} + ${shape.stemGains[0]} * ${whole(`u${n}_${k}`, false)} + ${shape.stemGains[1]} * ${whole(`s${n}_${k}`, true)}`,
    };
  }
  return {
    words: shape.trunkWords,
    runs: [{ first: 0, end: shape.trunkWords, signed: false }],
    sums: ['u'],
    // The test the zeros written round each layer's output make unnecessary; kept as an option
    // so that what it costs can be measured on a GPU.
    valid: edgeTests ? `${inFrame} && all(c >= vec2<i32>(-p.reachBelow)) && all(c < p.tile + p.reachBelow)` : null,
    index: `(base + l.y * p.window.x + l.x) * ${shape.trunkWords}`,
    accumulator: (whole, bias, n, k) => `${bias} + ${whole(`u${n}_${k}`, false)}`,
  };
}

/**
 * One row of the 3x3 (`dy` from the pixel's own) for an invocation's `span`
 * pixels: the span's samples and one either side, word by word, each weight
 * read once for the pixels it serves; the words are looped over (`wordLoop`)
 * or written out one after another. `perPart` of the layer's `outputWords`
 * are this invocation's, from word `k0`.
 */
function rowOfTaps(read, form, { span, wordLoop }, outputWords, perPart) {
  const columns = range(span + 2);
  const placed = (j) => `let l = first + vec2<i32>(${j - 1}, dy); let c = l - p.inset; let g = corner + c;`;
  const place = read.valid
    ? columns.map((j) => `var ok${j}: bool; var b${j}: i32; { ${placed(j)} ok${j} = ${read.valid}; b${j} = select(0, ${read.index}, ok${j}); }`).join('\n    ')
    : `var row: i32; { ${placed(0)} row = ${read.index}; }`;
  // `word` is a number where the words are written out one by one, the loop's variable where they are looped over.
  const fetch = (j, word, signed) => (read.valid
    ? `select(${signed ? PAD_SIGNED : PAD_UNSIGNED}, src[b${j} + ${word}], ok${j})`
    : `src[row + ${j * read.words} + ${word}]`);
  const tap = (word, sum, dx) => `{ let w = ((dy * ${ACROSS} + ${ACROSS + dx}) * ${read.words} + ${word}) * ${outputWords} + k0;
        ${range(perPart).map((k) => `{ let m = wts[w + ${k}]; ${range(span).map((n) => `${sum}${n}_${k} += ${form.product('m', `v${n + dx}`)};`).join(' ')} }`).join('\n        ')} }`;
  const ofWord = (word, signed) => `{ ${columns.map((j) => `let v${j} = ${form.sample(fetch(j, word, signed), signed)};`).join(' ')}
      ${range(ACROSS).map((dx) => tap(word, signed ? 's' : 'u', dx)).join('\n      ')} }`;
  const ofRun = ({ first, end, signed }) => (wordLoop
    ? `for (var word = ${first}; word < ${end}; word++) ${ofWord('word', signed)}`
    : range(end - first).map((index) => ofWord(first + index, signed)).join('\n    '));
  return `${place}
    ${read.runs.filter(({ first, end }) => end > first).map(ofRun).join('\n    ')}`;
}

/**
 * What each kind of layer writes for the pixel at `l` in its tile's window
 * (`g` in the frame), given the accumulators a0, a1, ... of its output words
 * from word `k0`.
 */
function writes(kind, shape) {
  const words = shape.trunkWords;
  const at = `let o = (base + l.y * p.window.x + l.x) * ${words} + k0;`;
  const requantised = (k, extra = '') => `rounded(scaled(a${k}, par[${words} + k0 + ${k}])${extra}, par[${2 * words} + k0 + ${k}])`;
  const each = (perPart, write) => range(perPart).map(write).join('\n  ');
  if (kind === 'stem') return (perPart) => `${at}\n  ${each(perPart, (k) => `dst[o + ${k}] = pack8(${requantised(k)} - 128);`)}`;
  const own = (k) => ` + par[${3 * words} + k0 + ${k}] * (unpack8(src[o + ${k}]) + 128)`;
  if (kind === 'body') return (perPart) => `${at}\n  ${each(perPart, (k) => `dst[o + ${k}] = pack8(${requantised(k, own(k))} - 128);`)}`;
  if (kind === 'last') {
    // The last block's output and the stem's, brought onto the head's input scale.
    return (perPart) => `${at} let skipBy = par[${4 * words}];\n  ${each(perPart, (k) => `{ let g = clamp(${requantised(k, own(k))}, vec4<i32>(0), vec4<i32>(255));
    let f = unpack8(stem[o + ${k}]) + 128;
    dst[o + ${k}] = pack8(rounded(g * skipBy.y + f * skipBy.x, vec4<i32>(skipBy.z)) - 128); }`)}`;
  }
  const headWords = 1 + shape.stateWords;
  const head = (k) => `rounded(scaled(a${k}, par[${headWords + k}]), par[${2 * headWords + k}])`;
  const state = range(shape.stateWords).map((k) => `{ let v = ${head(1 + k)}; let m = min(abs(v), vec4<i32>(par[${3 * headWords}].x));
    dst[o + ${2 + k}] = pack8(sign(v) * vec4<i32>(bounded(m.x), bounded(m.y), bounded(m.z), bounded(m.w))); }`);
  return () => `let o = (g.y * p.frame.x + g.x) * ${shape.outputWords};
  let r = vec4<u32>(clamp(${head(0)}, vec4<i32>(-32768), vec4<i32>(32767)) & vec4<i32>(65535));
  dst[o] = r.x | (r.y << 16u); dst[o + 1] = r.z | (r.w << 16u);
  ${state.join('\n  ')}`;
}

/** How many invocations share out the output words of a `kind` of layer: the head's are too few to share. */
export const partsOf = (kind, split) => (kind === 'head' ? 1 : split);

/**
 * The compute shader of one kind of layer: 'stem', 'body', 'last' (the final
 * block, which also adds the long skip) or 'head'. `shape` is the model's
 * word counts (int8-model.js). `form` names how the products are formed
 * (int8-forms.js), `packedDot` says whether the packed built-ins exist,
 * `uniformWeights` whether the weights are bound as a uniform array, `span`
 * is the pixels of a row an invocation computes, `split` the invocations a
 * trunk layer's output words are shared between, `wordLoop` whether a
 * sample's words are looped over or written out, and `edgeTests` adds the
 * per-sample tests the zeros round a window stand in for. Throws if the form
 * cannot hold the layer's sums or `split` does not divide its words.
 */
export function layerShader(kind, shape, { form: formName, packedDot, uniformWeights, span, split, edgeTests, wordLoop }) {
  const form = FORMS[formName];
  const trunk = kind !== 'head';
  const outputWords = trunk ? shape.trunkWords : 1 + shape.stateWords;
  const parts = partsOf(kind, split);
  if (outputWords % parts) throw new Error(`A layer of ${outputWords} output words cannot be shared between ${parts} invocations.`);
  const perPart = outputWords / parts;
  const read = reading(kind, shape, edgeTests);
  if (!form.holds(ROWS * ACROSS * 4 * read.words)) throw new Error(`A layer this wide is not exact in ${form.label}.`);
  const pixels = range(span);
  const outputs = range(perPart);
  const bounded = kind === 'head' && shape.stateWords
    ? `@group(0) @binding(5) var<storage, read> tanhTable: array<u32>;
fn bounded(magnitude: i32) -> i32 { return i32((tanhTable[magnitude >> 2u] >> (u32(magnitude & 3) * 8u)) & 255u); }` : '';
  // A trunk layer leaves stored zeros one pixel round what it computes, the frame's edge included:
  // the layer above, which computes no more than this one, reads no further.
  const blank = trunk ? `
fn blank(l: vec2<i32>, c: vec2<i32>, base: i32, k0: i32) {
  if (any(c < vec2<i32>(-p.reach - ${BORDER})) || any(c >= p.tile + p.reach + ${BORDER})) { return; }
  let o = (base + l.y * p.window.x + l.x) * ${shape.trunkWords} + k0;
  ${outputs.map((k) => `dst[o + ${k}] = ${PAD_UNSIGNED};`).join(' ')}
}` : '';
  const blanked = (n) => (trunk ? `blank(l${n}, c${n}, base, k0);` : '');
  return `${form.enable}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
${weightsBinding(form.weights, ROWS * ACROSS * read.words * outputWords, uniformWeights)}
@group(0) @binding(3) var<storage, read> par: array<vec4<i32>>;
${kind === 'last' ? '@group(0) @binding(4) var<storage, read> stem: array<u32>;' : ''}
${bounded}
${packingHelpers(packedDot)}
${COMMON}
fn finish(l: vec2<i32>, g: vec2<i32>, base: i32, k0: i32, ${outputs.map((k) => `a${k}: vec4<i32>`).join(', ')}) {
  ${writes(kind, shape)(perPart)}
}${blank}
@compute @workgroup_size(${GROUP_SIDE}, ${GROUP_SIDE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = i32(gid.z) / ${parts};
  let k0 = (i32(gid.z) % ${parts}) * ${perPart};
  let index = p.firstTile + slot;
  let first = vec2<i32>(i32(gid.x) * ${span}, i32(gid.y));
  if (index >= p.grid.x * p.grid.y || any(first >= p.window)) { return; }
  let corner = vec2<i32>(index % p.grid.x, index / p.grid.x) * p.tile;
  let base = slot * p.window.x * p.window.y;
  ${pixels.map((n) => `let l${n} = first + vec2<i32>(${n}, 0); let c${n} = l${n} - p.inset; let g${n} = corner + c${n};
  let live${n} = all(c${n} >= vec2<i32>(-p.reach)) && all(c${n} < p.tile + p.reach) && all(g${n} >= vec2<i32>(0)) && all(g${n} < p.frame);`).join('\n  ')}
  if (!(${pixels.map((n) => `live${n}`).join(' || ')})) { ${pixels.map(blanked).join(' ')} return; }
  ${pixels.map((n) => outputs.map((k) => read.sums.map((sum) => `var ${sum}${n}_${k} = ${form.sum}(0);`).join(' ')).join(' ')).join('\n  ')}
  for (var dy = -1; dy <= 1; dy++) {
    ${rowOfTaps(read, form, { span, wordLoop }, outputWords, perPart)}
  }
  ${pixels.map((n) => `if (live${n}) { finish(l${n}, g${n}, base, k0, ${outputs.map((k) => read.accumulator(form.whole, `par[k0 + ${k}]`, n, k)).join(', ')}); } else { ${blanked(n)} }`).join('\n  ')}
}`;
}
