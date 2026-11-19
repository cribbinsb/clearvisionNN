// The simulated channel: what happens to the camera's frame (camera-pool.js)
// before the upscaler sees it. Optionally a real encode and decode; this
// pairs each decoded frame with the original it came from.

import { FRAME_RATE, createCodec, timestampOf } from './codec.js';
import { createNanocodec } from './nanocodec.js';

// How long to wait for a frame's decoded picture before moving on; an encoder
// that holds a frame back delivers it while the next one is being sent.
const DECODE_WAIT_MS = 100;
const RATE_WINDOW = FRAME_RATE; // coded frames the measured bitrate is taken over

/**
 * Resolves to a sender for `width` x `height` originals under `degrade` (see
 * degrade.js). `send(shot, time)` takes what the camera made of an original
 * captured at `time` (ms on the source's clock): `{ picture, frame, timings }`.
 * It resolves to the input frame for the upscaler together with the original
 * it was made from (an ImageBitmap the caller owns) and whether a frame was
 * lost before it (`gap`), or to null when the codec has nothing yet.
 */
export async function createSender(width, height, degrade) {
  const [inputWidth, inputHeight] = [width / 2, height / 2];
  const make = degrade.codec?.engine === 'nanocodec' ? createNanocodec : createCodec;
  const codec = degrade.codec
    && await make(degrade.codec, inputWidth, inputHeight, degrade.rate, degrade.keyInterval);
  const waiting = []; // originals whose decoded frame has not arrived, oldest first
  const packets = []; // { bytes, time } of the newest coded frames, oldest first
  let index = 0;
  return {
    async send({ picture, frame, timings }, time) {
      if (!codec) return { original: picture, frame, gap: false, coding: null, timings: { ...timings, codec: 0 }, info: {} };
      waiting.push({ original: picture, timestamp: timestampOf(index), timings, sent: performance.now(), time });
      codec.push(frame, index, time);
      index += 1;
      const decoded = await codec.next(DECODE_WAIT_MS);
      if (!decoded) return null;
      // Frames come back in order; one the codec dropped is skipped with its original.
      let dropped = 0;
      while (waiting[0].timestamp !== decoded.timestamp) {
        waiting.shift().original.close();
        dropped += 1;
      }
      const entry = waiting.shift();
      packets.push({ bytes: decoded.bytes, time: entry.time });
      if (packets.length > RATE_WINDOW) packets.shift();
      // The rate on the source's clock: the bits sent after the window's first frame, over the time
      // between their captures.
      const bits = packets.slice(1).reduce((sum, packet) => sum + 8 * packet.bytes, 0);
      const seconds = (entry.time - packets[0].time) / 1000;
      const { quantiser } = decoded;
      const told = degrade.tellModel && quantiser !== null;
      return {
        original: entry.original,
        // The codec may reuse its picture for the next frame, which can arrive before this one is upscaled.
        frame: decoded.frame.slice(),
        // The frame before this one never arrived: what the upscaler remembers is not its predecessor.
        gap: dropped > 0,
        // A decoder knows the quantiser only where the codec reports it; the page may withhold it.
        coding: told ? { quantiser, key: decoded.key } : null,
        timings: { ...entry.timings, codec: performance.now() - entry.sent },
        info: { key: decoded.key, quantiser, told, kbps: seconds > 0 ? bits / seconds / 1000 : undefined },
      };
    },
    release() {
      codec?.close();
      waiting.forEach((entry) => entry.original.close());
    },
  };
}
