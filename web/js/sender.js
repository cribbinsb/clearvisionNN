// The simulated sender and channel: what happens to the original before the
// upscaler sees it. A camera's blur, sensor noise and denoise, then the 2x
// reduction (all in WebAssembly, in the order and with the filters the
// models were trained against), then optionally a real encode and decode.

import { Sender } from '../pkg/cvnn_web.js';
import { FRAME_RATE, createCodec, timestampOf } from './codec.js';
import { createNanocodec } from './nanocodec.js';

const NOISE_SEED = 1; // fixed, so a run can be repeated exactly
// How long to wait for a frame's decoded picture before moving on; an encoder
// that holds a frame back delivers it while the next one is being sent.
const DECODE_WAIT_MS = 100;
const RATE_WINDOW = FRAME_RATE; // frames the measured bitrate is averaged over

/**
 * Resolves to a sender for `width` x `height` originals under `degrade` (see
 * degrade.js). `send(original)` resolves to the input frame for the upscaler together with
 * the original it was made from and whether a frame was lost before it (`gap`), or to null
 * when the codec has nothing yet.
 */
export async function createSender(width, height, degrade) {
  const [inputWidth, inputHeight] = [width / 2, height / 2];
  const sender = new Sender(width, height, degrade.kernel, NOISE_SEED);
  const frame = new Uint8Array((inputWidth * inputHeight * 3) / 2);
  const make = degrade.codec?.engine === 'nanocodec' ? createNanocodec : createCodec;
  let codec = null;
  try {
    codec = degrade.codec
      && await make(degrade.codec, inputWidth, inputHeight, degrade.rate, degrade.keyInterval);
  } catch (error) {
    sender.free();
    throw error;
  }
  const waiting = []; // originals whose decoded frame has not arrived, oldest first
  const sizes = [];
  let index = 0;
  return {
    async send(original) {
      const start = performance.now();
      sender.send(original, degrade.blur, degrade.sigma, degrade.denoise, frame);
      const reduced = performance.now();
      if (!codec) return { original, frame, gap: false, coding: null, timings: { degrade: reduced - start, codec: 0 }, info: {} };
      waiting.push({ original, timestamp: timestampOf(index), degrade: reduced - start, sent: reduced });
      codec.push(frame, index);
      index += 1;
      const decoded = await codec.next(DECODE_WAIT_MS);
      if (!decoded) return null;
      // Frames come back in order; one the codec dropped is skipped with its original.
      let dropped = 0;
      while (waiting[0].timestamp !== decoded.timestamp) {
        waiting.shift();
        dropped += 1;
      }
      const entry = waiting.shift();
      sizes.push(decoded.bytes);
      if (sizes.length > RATE_WINDOW) sizes.shift();
      const bits = sizes.reduce((sum, bytes) => sum + 8 * bytes, 0);
      const { quantiser } = decoded;
      const told = degrade.tellModel && quantiser !== null;
      return {
        original: entry.original,
        frame: decoded.frame,
        // The frame before this one never arrived: what the upscaler remembers is not its predecessor.
        gap: dropped > 0,
        // A decoder knows the quantiser only where the codec reports it; the page may withhold it.
        coding: told ? { quantiser, key: decoded.key } : null,
        timings: { degrade: entry.degrade, codec: performance.now() - entry.sent },
        info: { key: decoded.key, quantiser, told, kbps: (bits * FRAME_RATE) / sizes.length / 1000 },
      };
    },
    release() {
      codec?.close();
      sender.free();
    },
  };
}
