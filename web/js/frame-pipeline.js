// One frame through the whole pipeline: the simulated camera and channel make
// the input from the original, and the upscaler makes the output from the
// input. It runs in a worker (pipeline-worker.js), with the camera in workers
// of its own (camera-pool.js). Several frames are inside at once: the cameras
// work on the newest while the channel and the GPU finish the ones before.

import { createCameraPool } from './camera-pool.js';
import { createSender } from './sender.js';
import { createUpscaler } from './upscaler.js';
import { runtimeReady } from './wasm.js';

// Consecutive frames the codec may return nothing for before it is taken to have stopped.
const MAX_MISSED = 30;

/** `picture` as a bitmap the caller owns: a copy of pixels the processor holds, or the bitmap itself. */
const asBitmap = (picture) => (picture instanceof ImageData ? createImageBitmap(picture) : picture);

/**
 * A pipeline for `width` x `height` originals: `model` on `backendId`, the
 * input degraded as `degrade` says; `tuning` goes to the upscaler (upscaler.js).
 */
export async function createPipeline(model, backendId, width, height, degrade, tuning = {}) {
  await runtimeReady();
  const upscaler = await createUpscaler(model, backendId, width / 2, height / 2, tuning);
  let sender;
  try {
    sender = await createSender(width, height, degrade);
  } catch (error) {
    await upscaler.release();
    throw error;
  }
  const camera = createCameraPool(width, height, degrade);
  let missed = 0;
  // The channel and the upscaler each handle frames strictly in order; the camera's work on
  // different frames, and the two halves of different frames, overlap.
  let sending = Promise.resolve();
  let upscaling = Promise.resolve();

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
    const [input, output] = await Promise.all([result.conventional, result.upscaled].map(asBitmap));
    // `variant`: how a GPU upscaler's shaders came to be written, for the page's diagnostics.
    const info = { ...sent.info, ...(result.variant && { variant: result.variant }) };
    return { original: sent.original, input, output, timings: { ...sent.timings, ...result.timings }, info };
  }

  return {
    /**
     * Process one original, an ImageBitmap this call consumes, captured at
     * `time` (ms on the source's clock). Resolves to
     * the original, the bicubic upscale of the input and the network's
     * output, all of one moment and each an ImageBitmap the caller owns,
     * with the time each step took (ms); or to null when the codec is still
     * holding the frame. Calls may overlap; results come in the order asked.
     */
    process(picture, time) {
      const shot = camera.capture(picture);
      const sent = sending.then(async () => sender.send(await shot, time));
      // A failure is reported through the result; the chains only keep the order.
      sending = sent.catch(() => {});
      const result = upscaling.then(async () => upscale(await sent));
      upscaling = result.catch(() => {});
      return result;
    },
    async release() {
      try {
        camera.release();
        sender.release();
      } finally {
        await upscaler.release();
      }
    },
  };
}
