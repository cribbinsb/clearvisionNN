// Diagnostic switches from the page's address, for finding out on someone's
// device why it is slow and what would be quicker: each overrides one choice
// the page otherwise makes for the GPU (int8-plan.js). They change speed only,
// except `form`, whose 16-bit float is not exact (docs/INTEGER.md, section 8).
//
//   ?form=integer|float|half   how the 8-bit network's products are formed
//   ?span=<n>                  pixels of a row one invocation computes
//   ?split=<n>                 invocations a layer's output words are shared between
//   ?edges=tested              test every sample for the tile's edge, as the zero border makes unnecessary
//   ?weights=storage           bind the weights as storage arrays, not uniform ones
//   ?packed=0                  spell the packed dot-product built-ins out
//   ?tile=<n>                  the tile side (0: the whole frame at once)

const NUMBERS = ['span', 'split', 'tile'];

/** The options (int8-network.js's) that `parameters`, a URLSearchParams, asks for. */
export function tuningFrom(parameters) {
  return {
    ...(parameters.has('form') && { form: parameters.get('form') }),
    ...Object.fromEntries(NUMBERS.filter((name) => parameters.has(name)).map((name) => [name, Number(parameters.get(name))])),
    ...(parameters.get('edges') === 'tested' && { edgeTests: true }),
    ...(parameters.get('weights') === 'storage' && { uniformWeights: false }),
    ...(parameters.get('packed') === '0' && { packedDot: false }),
  };
}
