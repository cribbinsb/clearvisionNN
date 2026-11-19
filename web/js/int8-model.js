// An integer model file (docs/INTEGER.md) as the shaders need it: the layers'
// packed weights as they are, and their constants gathered per layer into the
// vec4<i32> arrays the shaders read. `cvnn_runtime::integer` has already
// validated the file by the time this runs.

const HEADER_WORDS = 16;
const MAGIC = 0x38494346; // "FCI8"
const SCHEMA_VERSION = 1;
const TAPS = 9;
const WORD = 4; // channels per packed word
const STORED_OFFSET = 128; // an unsigned value is stored this much below its value
const UNSIGNED_LEVELS = 255;
const SIGNED_LEVELS = 127;
const PICTURE_OUTPUTS = 4;
const PICTURE_WORDS = 2;

/** Sum of the four signed bytes packed in `word`. */
const byteSum = (word) => (word << 24 >> 24) + (word << 16 >> 24) + (word << 8 >> 24) + (word >> 24);

/**
 * A layer's bias for the stored form of its inputs: unsigned inputs are 128
 * below their value in the buffers, so each weight's share of that is added
 * here once. `gain(inputWord)` is what the layer multiplies that word's dot
 * products by (0 for a word of signed planes, which have no offset).
 */
function storedBias(conv, inputWords, outputs, gain) {
  const bias = Int32Array.from(conv.bias);
  for (let index = 0; index < TAPS * inputWords; index += 1) {
    const share = STORED_OFFSET * gain(index % inputWords);
    for (let output = 0; output < outputs; output += 1) {
      bias[output] += share * byteSum(conv.weights[index * outputs + output]);
    }
  }
  return bias;
}

/** `parts` (per-channel arrays, or short lists of constants) end to end, each padded to whole vec4s. */
function vectors(parts) {
  const padded = parts.map((part) => [...part, ...new Array((WORD - (part.length % WORD)) % WORD).fill(0)]);
  return Int32Array.from(padded.flat());
}

/** The model in `words` (an Int32Array of the whole file). */
export function parseIntegerModel(words) {
  if (words[0] !== MAGIC || words[1] !== SCHEMA_VERSION) throw new Error('Not an integer model this page can run.');
  const [, , , unsignedWords, signedWords, channels, blocks, , stateOutputs, halo, tile, tanhLength] = words;
  let at = HEADER_WORDS;
  const take = (count) => words.subarray(at, at += count);
  const slots = take((unsignedWords + signedWords) * WORD);
  const conv = (inputWords, outputs, identity) => ({
    weights: take(TAPS * inputWords * outputs),
    bias: take(outputs),
    multiplier: take(outputs),
    shift: take(outputs),
    identity: identity ? take(outputs) : [],
  });
  const inputWords = unsignedWords + signedWords;
  const trunkWords = channels / WORD;
  const stem = conv(inputWords, channels, false);
  const body = Array.from({ length: blocks }, () => conv(trunkWords, channels, true));
  const skip = take(3);
  const outputs = PICTURE_OUTPUTS + stateOutputs;
  const head = conv(trunkWords, outputs, false);
  const tanh = take(tanhLength / WORD);
  // The stem brings unsigned planes (1/255) and signed ones (1/127) onto one scale.
  const stemGain = (word) => (word < unsignedWords ? SIGNED_LEVELS : 0);
  const constants = (layer, bias, extra) => vectors([bias, layer.multiplier, layer.shift, layer.identity, extra]);
  const unsigned = () => 1;
  return {
    inputWords, unsignedWords, trunkWords, halo, tile,
    slots: [...slots], // per input byte, the input channel it holds; -1 for padding
    outputWords: PICTURE_WORDS + stateOutputs / WORD,
    stateWords: stateOutputs / WORD,
    stemGains: [SIGNED_LEVELS, UNSIGNED_LEVELS], // for words of unsigned planes, of signed planes
    // One entry per layer, in order: the stem, the body, the head.
    layers: [
      { weights: stem.weights, constants: constants(stem, storedBias(stem, inputWords, channels, stemGain), []) },
      ...body.map((block, index) => ({
        weights: block.weights,
        // The last block also adds the long skip.
        constants: constants(block, storedBias(block, trunkWords, channels, unsigned), index === blocks - 1 ? skip : []),
      })),
      { weights: head.weights, constants: constants(head, storedBias(head, trunkWords, outputs, unsigned), [tanhLength - 1]) },
    ],
    tanh,
  };
}
