// How the integer network's shaders are written for a GPU: the form of the
// products (int8-forms.js) and the shape of the work (int8-shaders.js). The
// forms differ between GPUs by more than ten times, in opposite directions,
// and one frame of the wrong form takes seconds on some GPUs, so the choice
// is made from what the browser says of the GPU, never by trying forms out.
// docs/INTEGER.md, section 8, has the measurements behind each line here and
// says which lines are guesses.

import { FORMS } from './int8-forms.js';

// Makers whose GPUs run the packed integer dot product fastest: measured on NVIDIA, about four
// times the float forms. AMD and Intel are a guess: desktop GPUs like NVIDIA's, and the form the
// page has always given them, but nobody has measured one.
const INTEGER_MAKERS = ['nvidia', 'amd', 'intel'];
// Makers whose GPUs run 16-bit float fastest: measured on Apple (an M5, sixteen times the integer
// form); the others are a guess, being phone GPUs of the same tile-based kind, built round
// 16-bit float and without a packed dot product in hardware. The names are as Chrome gives
// them (Imagination's has been spelt both ways).
const HALF_MAKERS = ['apple', 'arm', 'qualcomm', 'img-tec', 'imgtec', 'samsung'];

// The shape of the work for each form (int8-shaders.js says what each part is). Integer: measured
// on NVIDIA, where more pixels or parts to an invocation only slow it. Half: exactly the shape
// measured at 50 ms on the Apple M5, edge tests included; sharing the outputs and dropping the
// tests were quicker in a test of one layer chain there but are not yet measured in the page,
// so they stay switches (tuning.js). Float goes to GPUs nothing is known of: the pixels of the half form, since where
// float is the right form reads are likely the cost, and looped words, quicker on NVIDIA and
// much the quicker to compile.
const SHAPES = {
  integer: { span: 1, split: 1, wordLoop: false, edgeTests: false },
  float: { span: 4, split: 1, wordLoop: true, edgeTests: false },
  half: { span: 4, split: 1, wordLoop: false, edgeTests: true },
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
  if (!(form in FORMS)) throw new Error(`There is no "${form}" form of the shaders: ${Object.keys(FORMS).join(', ')}.`);
  return { form, ...SHAPES[form], ...asked };
}

/** A plan in words, for the page's diagnostics. */
export function describePlan({ form, span, split, wordLoop, edgeTests }) {
  const shared = split > 1 ? `, outputs shared between ${split}` : '';
  return `${FORMS[form].label} products, ${span} to an invocation${shared}${wordLoop ? ', words looped' : ''}${edgeTests ? ', edges tested' : ''}`;
}
