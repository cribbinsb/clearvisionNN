// The forms of the integer network's shaders: the ways a layer's sum of
// products (docs/INTEGER.md, section 2) can be computed on a GPU. Which is
// quick differs between GPUs by more than ten times, in opposite directions,
// so the page picks one from the GPU's maker (int8-plan.js). Everything after
// the sum (requantisation, the skips, the head) is integer arithmetic in every
// form.
//
//   integer  the packed 8-bit dot product. Exact.
//   float    32-bit float products of the bytes as stored. Exact: a sum of
//            products of two signed bytes cannot pass 2^24 (checked per layer).
//   half     16-bit float. Approximate: a 16-bit float keeps eleven binary
//            digits, so the sum is rounded as it grows (docs/INTEGER.md,
//            section 8, has the measured difference and its check).
//
// A form says how the weights are bound, what a sample and a sum are in WGSL,
// and how a sum becomes the whole number the integer arithmetic goes on with.

const CHANNELS = 4; // channels to a packed word
const BYTE = 8;

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

/**
 * A layer's packed weights as the float forms read them: for each vec4 of
 * words (four outputs' weights of one input word), a 4x4 matrix whose column
 * i holds the four outputs' weights of input channel i, so that the matrix
 * times a sample's four channels is their contribution to the four outputs.
 */
function matrixWeights(packed) {
  const floats = new Float32Array(packed.length * CHANNELS);
  packed.forEach((word, index) => {
    const output = index % CHANNELS;
    for (let channel = 0; channel < CHANNELS; channel += 1) {
      floats[(index - output) * CHANNELS + channel * CHANNELS + output] = (word << (24 - BYTE * channel)) >> 24;
    }
  });
  return floats;
}

// In the half form every factor is scaled to within one, so that no sum nears a 16-bit float's
// largest value: a weight by this, a signed sample by the same, an unsigned one by twice it.
const HALF_SCALE = 128;
const HALF_BIAS = 15; // of a 16-bit float's exponent
const HALF_FRACTION_BITS = 10;

/** `matrixWeights`, each weight over 128 and written as the bits of a 16-bit float (exact: a byte has seven digits). */
function halfMatrixWeights(packed) {
  const floats = matrixWeights(packed);
  const halves = new Uint16Array(floats.length);
  floats.forEach((weight, index) => {
    const magnitude = Math.abs(weight);
    if (magnitude === 0) return;
    // weight / 128 = 1.m * 2^(top - 7), `top` being the magnitude's highest set bit.
    const top = 31 - Math.clz32(magnitude);
    const exponent = top - Math.log2(HALF_SCALE) + HALF_BIAS;
    halves[index] = (weight < 0 ? 0x8000 : 0) | (exponent << HALF_FRACTION_BITS) | ((magnitude << (HALF_FRACTION_BITS - top)) & 0x3ff);
  });
  return halves;
}

// A sum of products of two signed bytes is exact in 32-bit float, in whatever
// order it is formed, while it cannot pass 2^24.
const FLOAT_EXACT = 2 ** 24;
const LARGEST_PRODUCT = 128 * 128;

/**
 * The forms by name. Of each:
 *   label      what the page's diagnostics call it
 *   exact      whether its sums are the integer ones, bit for bit
 *   inputs     how it reads an unsigned sample, which decides the bias it needs (int8-model.js):
 *              'stored' (the signed byte in the buffer) or 'plain' (its value)
 *   feature    the device feature it needs, if any; `enable` the WGSL directive for it
 *   weights    the WGSL type of four outputs' weights of one input word; `bind` makes them
 *   sum        the WGSL type of four outputs' sums
 *   sample     WGSL for a packed word of four channels as the products take it
 *   product    WGSL for weights `m` times sample `x`, as a `sum`
 *   whole      WGSL for a sum as a vec4<i32> of accumulator steps
 *   holds      whether a sum of `terms` products stays within what the form can hold
 * `signed` says whether a word is of signed planes (the state) rather than unsigned ones.
 */
export const FORMS = {
  integer: {
    label: 'integer', exact: true, inputs: 'stored', feature: null, enable: '',
    weights: 'vec4<u32>', bind: (packed) => packed, sum: 'vec4<i32>',
    sample: (word) => word,
    product: (m, x) => `dots(${x}, ${m})`,
    whole: (sum) => sum,
    holds: () => true, // the exporter refuses a model whose accumulator could leave 32 bits
  },
  float: {
    label: '32-bit float', exact: true, inputs: 'stored', feature: null, enable: '',
    weights: 'mat4x4<f32>', bind: matrixWeights, sum: 'vec4<f32>',
    sample: (word) => `vec4<f32>(unpack8(${word}))`,
    product: (m, x) => `${m} * ${x}`,
    whole: (sum) => `vec4<i32>(${sum})`,
    holds: (terms) => terms * LARGEST_PRODUCT <= FLOAT_EXACT,
  },
  half: {
    label: '16-bit float', exact: false, inputs: 'plain', feature: 'shader-f16', enable: 'enable f16;',
    weights: 'mat4x4<f16>', bind: halfMatrixWeights, sum: 'vec4<f16>',
    // An unsigned sample is read as its value, not as stored: most activations are zero, and a
    // zero then adds nothing at all to the sum, where the stored -128 would add a large term
    // for the bias to cancel and the rounding would be of that. Both conversions are exact.
    sample: (word, signed) => (signed
      ? `(vec4<f16>(unpack8(${word})) * ${1 / HALF_SCALE}h)`
      : `(vec4<f16>(unpack8(${word})) * ${1 / (2 * HALF_SCALE)}h + 0.5h)`),
    product: (m, x) => `${m} * ${x}`,
    whole: (sum, signed) => `vec4<i32>(round(vec4<f32>(${sum}) * ${HALF_SCALE * HALF_SCALE * (signed ? 1 : 2)}.0))`,
    holds: () => true, // every term is under one, and a layer has far fewer terms than the largest half
  },
};
