// nanocodec as the codec between the sender and the upscaler: its encoder
// reports the quantiser and type of every frame, which the browser's own
// encoders do not, so the model can be told how each frame was coded. The
// module is large and is fetched only when one of its codecs is chosen.

import { FRAME_RATE, bitrateFor, timestampOf } from './codec.js';

const MODULE = new URL('../nanocodec/nc_web.js', import.meta.url).href;
// The fastest rung of the speed ladder the H.26x encoders share: the sender is real-time.
const PRESET = 'faster';
// 8-bit BT.709, limited range: what the runtime's colour conversion produces.
const COLOUR = 'bt709';
// The encoder's longest key-frame interval, used for "the first only".
const LONGEST_KEY_INTERVAL = 30 * FRAME_RATE;
// Quantiser the encoder starts from before rate control has measured anything.
const STARTING_QUANTISER = 32;
// The encoder's rate buffer, as the training encodes had: one second at the target rate.
const RATE_BUFFER_MS = 1000;
const MICROSECONDS_PER_MS = 1000;

let loading = null;

/** Whether the module has been built and published beside the page. */
export async function nanocodecAvailable() {
  try {
    return (await fetch(MODULE, { method: 'HEAD' })).ok;
  } catch {
    return false;
  }
}

async function load() {
  loading ??= import(MODULE).then(async (module) => {
    await module.default();
    return module;
  });
  return loading;
}

/**
 * An encoder feeding a decoder, with the interface of `createCodec`
 * (codec.js). `codec.name` is nanocodec's name for the codec.
 */
export async function createNanocodec(codec, width, height, rate, keyInterval) {
  const { BrowserEncoder, BrowserDecoder, SourceFrame } = await load();
  const fixed = rate.kind === 'qp';
  const encoder = new BrowserEncoder(codec.name, width, height, fixed ? rate.value : STARTING_QUANTISER,
    FRAME_RATE, keyInterval || LONGEST_KEY_INTERVAL, PRESET, undefined, false, COLOUR);
  if (!fixed) {
    const kbps = Math.round(bitrateFor(rate.value, width, height) / 1000);
    encoder.set_rate_control(kbps, kbps, RATE_BUFFER_MS);
  }
  const decoder = new BrowserDecoder(codec.name);
  const luma = width * height;
  const layout = new Uint32Array([0, width, luma, width / 2, luma + luma / 4, width / 2]);
  const source = new SourceFrame(width, height);
  const decoded = []; // finished frames, oldest first
  let last = null; // the newest decoded picture, shown again when the encoder repeats a frame

  return {
    /**
     * Encode and decode frame `index`, given as packed 4:2:0. The encoder's
     * offer to see the second frame before coding the first is not taken up:
     * it would put every picture one frame behind its original.
     */
    push(picture, index) {
      source.from_planes(picture, 'I420', COLOUR, width, height, layout);
      // The frame is coded as captured now: when the page cannot keep up with the source and frames
      // are missed, rate control gives this one the bits of the time since the last, as a live
      // sender's does. Coded as one of an unbroken 30 a second it would be starved and go blocky.
      const packet = encoder.encode_at(source, Math.round(performance.now() * MICROSECONDS_PER_MS));
      // An empty packet repeats the previous picture: that is what a receiver shows.
      if (!encoder.repeat) {
        decoder.decode(packet);
        last = decoder.decoded_i420();
      }
      decoded.push({
        timestamp: timestampOf(index), frame: last, bytes: packet.length,
        key: encoder.key, quantiser: codec.h26x ? encoder.quantizer : null,
      });
    },
    /** The oldest decoded frame not yet taken, or null; nothing here is asynchronous. */
    async next() {
      return decoded.shift() ?? null;
    },
    close() {
      for (const part of [encoder, decoder, source]) part.free();
    },
  };
}
