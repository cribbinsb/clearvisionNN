// A real encoder and decoder between the sender and the upscaler, through
// WebCodecs, in a low-delay configuration: every frame goes in, and its
// decoded picture comes out, possibly a little later.

import { toI420 } from './planar.js';

// Frame rate the encoder is told, and bitrates are stated for.
export const FRAME_RATE = 30;
const MICROSECONDS = 1e6;
// 8-bit BT.709, limited range: what the runtime's colour conversion produces.
const COLOUR = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false };

// Codec strings chosen to cover the page's largest input (960 x 540) at 8 bits.
// `options` names the per-frame options dictionary of each codec; `h26x`
// marks those whose quantiser is on the 0-51 scale the page reports.
const CODECS = [
  { id: 'avc', label: 'H.264', string: 'avc1.42001f', options: 'avc', h26x: true },
  { id: 'hevc', label: 'HEVC', string: 'hev1.1.6.L93.B0', options: 'hevc', h26x: true },
  { id: 'vp9', label: 'VP9', string: 'vp09.00.31.08', options: 'vp9', h26x: false },
  { id: 'av1', label: 'AV1', string: 'av01.0.04M.08', options: 'av1', h26x: false },
];
// Software first: it behaves the same on every machine and exists without a GPU.
const ACCELERATIONS = ['prefer-software', 'prefer-hardware'];

/** Bits per second for `bitsPerPixel` at the nominal frame rate. */
export function bitrateFor(bitsPerPixel, width, height) {
  return Math.round(bitsPerPixel * width * height * FRAME_RATE);
}

function encoderConfig(codec, width, height, rate, acceleration) {
  const config = {
    codec: codec.string, width, height, framerate: FRAME_RATE,
    latencyMode: 'realtime', hardwareAcceleration: acceleration,
  };
  if (rate.kind === 'qp') return { ...config, bitrateMode: 'quantizer' };
  return { ...config, bitrateMode: 'constant', bitrate: bitrateFor(rate.value, width, height) };
}

async function supports(config) {
  try {
    return (await VideoEncoder.isConfigSupported(config)).supported;
  } catch {
    return false;
  }
}

/**
 * The codecs this browser can encode at `width` x `height`, each with the
 * acceleration that works and whether it takes a fixed quantiser; null where
 * the browser has no WebCodecs at all.
 */
export async function supportedCodecs(width, height) {
  if (!('VideoEncoder' in globalThis)) return null;
  const found = [];
  for (const codec of CODECS) {
    for (const acceleration of ACCELERATIONS) {
      if (await supports(encoderConfig(codec, width, height, { kind: 'bpp', value: 0.1 }, acceleration))) {
        const quantiser = await supports(encoderConfig(codec, width, height, { kind: 'qp' }, acceleration));
        found.push({ ...codec, acceleration, quantiser });
        break;
      }
    }
  }
  return found;
}

/** The timestamp (µs) the codec gives frame `index`; whole, as WebCodecs keeps it. */
export function timestampOf(index) {
  return Math.round((index * MICROSECONDS) / FRAME_RATE);
}

/**
 * An encoder feeding a decoder. `rate` is `{ kind: 'bpp' | 'qp', value }`;
 * `keyInterval` is frames between key frames (0: only the first).
 */
export function createCodec(codec, width, height, rate, keyInterval) {
  // The quantiser is known only when it was fixed, and is reported only for
  // the H.26x family, whose 0-51 scale a reader knows.
  const quantiser = codec.h26x && rate.kind === 'qp' ? rate.value : null;
  const decoded = []; // finished frames, oldest first
  const chunks = new Map(); // what the encoder said about each frame, by timestamp
  let failure = null;
  let wake = null;
  let copying = Promise.resolve(); // keeps decoded frames in order through the asynchronous copy
  const notify = () => { wake?.(); wake = null; };
  const fail = (error) => { failure = error; notify(); };
  const decoder = new VideoDecoder({
    output(frame) {
      const { timestamp } = frame;
      copying = copying.then(() => toI420(frame, width, height)).then((picture) => {
        decoded.push({ timestamp, frame: picture, quantiser, ...chunks.get(timestamp) });
        chunks.delete(timestamp);
        notify();
      }).catch(fail).finally(() => frame.close());
    },
    error: fail,
  });
  const encoder = new VideoEncoder({
    output(chunk, metadata) {
      // Without the hint a decoder may hold frames back in case later ones precede them.
      if (metadata?.decoderConfig) decoder.configure({ ...metadata.decoderConfig, optimizeForLatency: true });
      chunks.set(chunk.timestamp, { key: chunk.type === 'key', bytes: chunk.byteLength });
      decoder.decode(chunk);
    },
    error: fail,
  });
  encoder.configure(encoderConfig(codec, width, height, rate, codec.acceleration));
  return {
    /** Encode frame `index`, given as packed 4:2:0. */
    push(picture, index) {
      const frame = new VideoFrame(picture, {
        format: 'I420', codedWidth: width, codedHeight: height, timestamp: timestampOf(index), colorSpace: COLOUR,
      });
      const options = { keyFrame: keyInterval ? index % keyInterval === 0 : index === 0 };
      if (rate.kind === 'qp') options[codec.options] = { quantizer: rate.value };
      encoder.encode(frame, options);
      frame.close();
    },
    /** The oldest decoded frame not yet taken, waiting up to `timeout` ms for one; null if none came. */
    async next(timeout) {
      if (!decoded.length && !failure) {
        await new Promise((resolve) => { wake = resolve; setTimeout(resolve, timeout); });
      }
      if (failure) throw failure;
      return decoded.shift() ?? null;
    },
    close() {
      for (const coder of [encoder, decoder]) if (coder.state !== 'closed') coder.close();
    },
  };
}
