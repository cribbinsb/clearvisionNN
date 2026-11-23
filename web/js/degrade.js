// The "degrade the input" controls: what the simulated sender and channel do
// to the picture before the upscaler sees it, as plain settings. A list of
// ready-made inputs (degrade-presets.js) sets everything at once; the
// advanced controls set each step by itself. The grain profile of the output
// is read with them, because which one suits depends on the input: the
// ready-made input decides it, and there is no control of its own.

import { bitrateFor, supportedCodecs } from './codec.js';
import { CUSTOM, PRESET_CODECS, PRESETS } from './degrade-presets.js';
import { nanocodecAvailable } from './nanocodec.js';

// Bitrates span what the models were trained on, in bits per pixel at 30 frames a second.
const BITS_PER_PIXEL = [0.04, 0.08, 0.15, 0.3];
const QUANTISERS = [24, 28, 32, 36, 40, 44];
// The reductions the training data was made with; the first is the plain camera chain's.
const KERNELS = [['bicubic', 'Bicubic'], ['bilinear', 'Bilinear'], ['area', 'Area (box)'], ['lanczos', 'Lanczos']];
// Codec support is asked about at a typical input size; an unusual size can still fail at start.
const PROBE_SIZE = [640, 360];
// Each level slider with the element that shows its reading, and the decimals of the reading.
const SLIDERS = [['blur', 'blurValue', 1], ['noise', 'noiseValue', 1], ['denoise', 'denoiseValue', 1]];
// The controls a preset sets and whose change makes the input a custom one.
const ADVANCED = ['blur', 'noise', 'denoise', 'kernel', 'codec', 'rate', 'keyint'];
const FIRST_KEY_ONLY = '0'; // the key-frame choice of a real-time call

/** A codec with how it runs here and whether its quantiser can be fixed. */
function describeCodec(codec) {
  const kind = codec.acceleration.replace('prefer-', '');
  return `${codec.label} (${kind}${codec.quantiser ? ', fixed quantiser available' : ', bitrate only'})`;
}

// nanocodec's encoders, which report each frame's quantiser on the H.26x scale.
const NANOCODECS = [
  { id: 'nanocodec-h264', label: 'nanocodec H.264', name: 'h264' },
  { id: 'nanocodec-hevc', label: 'nanocodec HEVC', name: 'hevc' },
].map((codec) => ({ ...codec, engine: 'nanocodec', h26x: true, quantiser: true }));

const fill = (select, options) => select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));

/**
 * Fill the controls from what this browser can do. `controls` holds the
 * preset, the blur, noise and denoise sliders (each with its `...Value` reading)
 * and the kernel, codec, rate and keyint elements; `note` gets a sentence about
 * what is available. Returns `{ read, refresh }`.
 */
export async function setUpDegrade(controls, note) {
  const browser = await supportedCodecs(...PROBE_SIZE);
  const own = await nanocodecAvailable() ? NANOCODECS : [];
  const codecs = [...own, ...(browser ?? [])];
  fill(controls.kernel, KERNELS);
  fill(controls.codec, [['none', 'None'], ...codecs.map((codec) => [codec.id, codec.label])]);
  fill(controls.rate, [
    ...BITS_PER_PIXEL.map((value) => [`bpp:${value}`, '']),
    ...QUANTISERS.map((value) => [`qp:${value}`, `Fixed quantiser ${value}`]),
  ]);
  const chosen = () => codecs.find((codec) => codec.id === controls.codec.value) ?? null;
  const found = browser?.length ? `The browser's encoders: ${browser.map(describeCodec).join(', ')}.` : 'This browser offers no video encoder of its own.';
  const reported = own.length ? 'nanocodec reports the quantiser and type of every frame; the browser\'s encoders do not.' : 'nanocodec is not built here (tools/web/build-nanocodec.sh), so no encoder reports its quantiser.';
  note.textContent = `${found} ${reported}`;
  controls.codec.disabled = !codecs.length;

  // A preset with a codec needs an H.264 encoder; without one it is listed but cannot be chosen.
  const presetCodec = PRESET_CODECS.find((id) => codecs.some((codec) => codec.id === id)) ?? null;
  fill(controls.preset, [...PRESETS, CUSTOM].map((preset) => [preset.id, preset.label]));
  for (const [index, preset] of PRESETS.entries()) {
    controls.preset.options[index].disabled = preset.bitsPerPixel !== null && !presetCodec;
  }
  const customOption = controls.preset.options[PRESETS.length];
  customOption.hidden = true;

  const showReadings = () => {
    for (const [slider, reading, decimals] of SLIDERS) controls[reading].textContent = Number(controls[slider].value).toFixed(decimals);
  };

  /** Set every advanced control to what the chosen preset stands for. */
  function applyPreset() {
    const preset = PRESETS.find((candidate) => candidate.id === controls.preset.value);
    if (!preset) return;
    customOption.hidden = true;
    for (const [slider] of SLIDERS) controls[slider].value = preset[slider];
    controls.kernel.value = preset.kernel;
    controls.codec.value = preset.bitsPerPixel === null ? 'none' : presetCodec;
    if (preset.bitsPerPixel !== null) controls.rate.value = `bpp:${preset.bitsPerPixel}`;
    controls.keyint.value = FIRST_KEY_ONLY;
    showReadings();
  }
  controls.preset.addEventListener('change', applyPreset);
  for (const name of ADVANCED) {
    controls[name].addEventListener('change', () => {
      customOption.hidden = false;
      controls.preset.value = CUSTOM.id;
    });
  }
  // A slider restarts the run when released; until then only its reading follows it.
  for (const [slider] of SLIDERS) controls[slider].addEventListener('input', showReadings);
  applyPreset();

  /** Bring labels and availability in line with the codec and the `width` x `height` input. */
  function refresh(width, height) {
    const codec = chosen();
    showReadings();
    controls.rate.disabled = !codec;
    controls.keyint.disabled = !codec;
    for (const option of controls.rate.options) {
      const [kind, value] = option.value.split(':');
      if (kind === 'bpp') option.text = `${value} bits per pixel · ${Math.round(bitrateFor(Number(value), width, height) / 1000)} kbit/s`;
      else option.disabled = Boolean(codec) && !codec.quantiser;
    }
    if (controls.rate.selectedOptions[0].disabled) controls.rate.selectedIndex = 0;
  }

  /**
   * The settings for `width` x `height` input: the camera's blur, noise and
   * denoise levels and its reduction kernel, then the codec (or null), its
   * rate and key-frame interval; and `grain`, the profile of the grain the
   * output of such an input is shown with (docs/INTEGER.md, section 9).
   */
  function read(width, height) {
    const [kind, value] = controls.rate.value.split(':');
    const rate = { kind, value: Number(value) };
    const codec = chosen();
    return {
      blur: Number(controls.blur.value),
      sigma: Number(controls.noise.value),
      denoise: Number(controls.denoise.value),
      kernel: controls.kernel.value,
      codec,
      rate,
      keyInterval: Number(controls.keyint.value),
      grain: (PRESETS.find((preset) => preset.id === controls.preset.value) ?? CUSTOM).grain,
      askedKbps: codec && kind === 'bpp' ? bitrateFor(rate.value, width, height) / 1000 : null,
    };
  }
  return { read, refresh };
}
