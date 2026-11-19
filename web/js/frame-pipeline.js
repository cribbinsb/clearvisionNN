// One frame through the whole pipeline: the simulated sender makes the input
// from the original, and the upscaler makes the output from the input. It
// runs in a worker (pipeline-worker.js). Two frames may be inside at once:
// the sender works on one while the GPU finishes the one before.

import { createSender } from './sender.js';
import { createUpscaler } from './upscaler.js';
import { runtimeReady } from './wasm.js';

// Consecutive frames the codec may return nothing for before it is taken to have stopped.
const MAX_MISSED = 30;

/** `picture` as a bitmap the caller owns: a copy of pixels the processor holds, or the bitmap itself. */
const asBitmap = (picture) => (picture instanceof ImageData ? createImageBitmap(picture) : picture);

/**
 * A pipeline for `width` x `height` originals: `model` on `backendId`, the
 * input degraded as `degrade` says.
 */
export async function createPipeline(model, backendId, width, height, degrade) {
  await runtimeReady();
  const upscaler = await createUpscaler(model, backendId, width / 2, height / 2);
  let sender;
  try {
    sender = await createSender(width, height, degrade);
  } catch (error) {
    await upscaler.release();
    throw error;
  }
  // The original's pixels are read back through a canvas: the sender works on bytes.
  const reader = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
  let missed = 0;
  // Each half handles frames strictly in order; the two halves of different frames overlap.
  let sending = Promise.resolve();
  let upscaling = Promise.resolve();

  /** The sender's half: read `picture`, which is consumed, and make the input frame from it. */
  async function send(picture) {
    const start = performance.now();
    reader.drawImage(picture, 0, 0);
    picture.close();
    const pixels = reader.getImageData(0, 0, width, height).data;
    const read = performance.now() - start;
    const sent = await sender.send(pixels);
    // The sender reuses its frame for the next original, which may arrive before this one is upscaled.
    return sent && { ...sent, frame: sent.frame.slice(), timings: { read, ...sent.timings } };
  }

  /** The receiver's half: upscale what the sender produced, if it produced anything. */
  async function upscale(sent) {
    if (!sent) {
      missed += 1;
      if (missed > MAX_MISSED) throw new Error('The codec stopped producing frames.');
      return null;
    }
    missed = 0;
    // After a lost frame the motion and the history would describe different moments.
    if (sent.gap) upscaler.restart();
    const result = await upscaler.process(sent.frame, sent.coding);
    const [original, input, output] = await Promise.all(
      [new ImageData(sent.original, width, height), result.conventional, result.upscaled].map(asBitmap),
    );
    return { original, input, output, timings: { ...sent.timings, ...result.timings }, info: sent.info };
  }

  return {
    /**
     * Process one original, an ImageBitmap this call consumes. Resolves to
     * the original, the bicubic upscale of the input and the network's
     * output, all of one moment and each an ImageBitmap the caller owns,
     * with the time each step took (ms); or to null when the codec is still
     * holding the frame. Calls may overlap; results come in the order asked.
     */
    process(picture) {
      const sent = sending.then(() => send(picture));
      // A failure is reported through the result; the chains only keep the order.
      sending = sent.catch(() => {});
      const result = upscaling.then(async () => upscale(await sent));
      upscaling = result.catch(() => {});
      return result;
    },
    async release() {
      try {
        sender.release();
      } finally {
        await upscaler.release();
      }
    },
  };
}
