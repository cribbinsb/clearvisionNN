// The ready-made inputs of the simple view: each a sensible combination of
// everything the advanced controls set one by one. The levels sit inside what
// the models were trained on (blur to 1.2 samples, noise 1 to 5 grey levels,
// denoise strength 2 to 12, 0.04 to 0.3 bits per pixel).

const WEBCAM = { blur: 0, noise: 2, denoise: 6, kernel: 'bicubic' };
// Ahead of an encoder the camera's denoise is kept light: at full strength it removed about as
// much still texture as the model restores (7 to 14 % of the still-texture score), and the
// encoder smooths noise itself.
const CALL = { ...WEBCAM, denoise: 2 };

// `bitsPerPixel` null means no codec; otherwise H.264 at that rate.
export const PRESETS = [
  { id: 'clean', label: 'Clean camera · only the 2× reduction', blur: 0, noise: 0, denoise: 0, kernel: 'bicubic', bitsPerPixel: null },
  { id: 'webcam', label: 'Typical webcam · some noise, the camera\'s own denoise', ...WEBCAM, bitsPerPixel: null },
  { id: 'soft', label: 'Soft, noisy camera · blur, more noise, a bilinear reduction', blur: 0.8, noise: 4, denoise: 8, kernel: 'bilinear', bitsPerPixel: null },
  { id: 'call-good', label: 'Video call, good link · webcam, then H.264 at 0.15 bits per pixel', ...CALL, bitsPerPixel: 0.15 },
  { id: 'call-poor', label: 'Video call, poor link · webcam, then H.264 at 0.04 bits per pixel', ...CALL, bitsPerPixel: 0.04 },
];

// What the list shows once an advanced control has been moved off a preset.
export const CUSTOM = { id: 'custom', label: 'As set under Advanced' };

// H.264 encoders a preset may use, the one that reports its quantiser first.
export const PRESET_CODECS = ['nanocodec-h264', 'avc'];
