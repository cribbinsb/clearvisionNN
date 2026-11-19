// How the integer network's shaders are written for a GPU: the form of the
// products (int8-forms.js) and the shape of the work (int8-shaders.js). The
// forms differ between GPUs by more than ten times, in opposite directions,
// and one frame of the wrong form takes seconds on some GPUs, so the choice
// is made from what the browser says of the GPU, never by trying forms out.
// docs/INTEGER.md, section 8, has the measurements behind each line here and
// says which lines are guesses.

import { FORMS } from './int8-forms.js';

// Makers whose GPUs run the packed integer dot product fastest. Measured on NVIDIA only.
const INTEGER_MAKERS = ['nvidia'];
// Makers whose GPUs run 16-bit float fastest: measured on Apple (an M5, sixteen times the integer
// form); the others are a guess, being the phone GPUs of the same tile-based kind, built round
// 16-bit float and without a packed dot product in hardware. The names are as Chrome gives them.
const HALF_MAKERS = ['apple', 'arm', 'qualcomm', 'img-tec', 'imgtec', 'samsung'];

// The shape of the work that each form runs fastest in, where it was measured.
const SHAPES = {
  integer: { span: 1, split: 1 },
  float: { span: 4, split: 1 },
  half: { span: 4, split: 2 },
};

/**
 * The form for a GPU of `maker` (the adapter's `vendor`) whose device has
 * `features`. An unknown maker gets 32-bit float: exact, needing nothing
 * optional, and the form whose worst case is mildest.
 */
function formFor(maker, features) {
  if (INTEGER_MAKERS.includes(maker)) return 'integer';
  if (HALF_MAKERS.includes(maker) && features.has(FORMS.half.feature)) return 'half';
  return 'float';
}

/**
 * The options for int8-network.js on `device`. `asked` are options that
 * override the choice (the page's diagnostic switches, a check's variants).
 */
export function planFor(device, asked = {}) {
  const form = asked.form ?? formFor(device.adapterInfo?.vendor ?? '', device.features);
  return { form, ...SHAPES[form], edgeTests: false, ...asked };
}

/** A plan in words, for the page's diagnostics. */
export function describePlan({ form, span, split, edgeTests }) {
  const shared = split > 1 ? `, outputs shared between ${split}` : '';
  return `${FORMS[form].label} products, ${span} to an invocation${shared}${edgeTests ? ', edges tested' : ''}`;
}
