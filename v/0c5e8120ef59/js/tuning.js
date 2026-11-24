// Diagnostic switches from the page's address, for finding out on someone's
// device why it is slow and what would be quicker: each overrides one choice
// the page otherwise makes for the GPU (int8-plan.js). They change speed only,
// except `form`, whose 16-bit float is not exact (docs/INTEGER.md, section 8).
//
//   ?form=integer|float|half   how the 8-bit network's products are formed
//   ?span=<n>                  pixels of a row one invocation computes
//   ?split=<n>                 invocations a layer's output words are shared between
//   ?words=looped|written      loop over a sample's words, or write them out one by one
//   ?edges=tested|bordered     test every sample for the tile's edge, or rely on the zero border instead
//   ?weights=storage           bind the weights as storage arrays, not uniform ones
//   ?packed=0                  spell the packed dot-product built-ins out
//   ?tile=<n>                  the tile side (0: the whole frame at once)
//   ?steps=1                   time each step of a frame on the GPU apart, and show them in the diagnostics

const NUMBERS = ['span', 'split', 'tile'];

/** The options (gpu-upscaler.js's) that `parameters`, a URLSearchParams, asks for. */
export function tuningFrom(parameters) {
  return {
    ...(parameters.has('form') && { form: parameters.get('form') }),
    ...Object.fromEntries(NUMBERS.filter((name) => parameters.has(name)).map((name) => [name, Number(parameters.get(name))])),
    ...(parameters.has('words') && { wordLoop: parameters.get('words') === 'looped' }),
    ...(parameters.has('edges') && { edgeTests: parameters.get('edges') === 'tested' }),
    ...(parameters.get('weights') === 'storage' && { uniformWeights: false }),
    ...(parameters.get('packed') === '0' && { packedDot: false }),
    ...(parameters.get('steps') === '1' && { stepTimes: true }),
  };
}
