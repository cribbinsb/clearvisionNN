// The upscaler entirely on the GPU, for a model in integer form: the motion
// search runs in WebAssembly on the small luma plane, and everything else
// (the conventional upscale, warping history, the network, colour) is the
// page's own shaders. The pictures come back as bitmaps the GPU still holds;
// no pixel is read back to the processor.

import { IntegerNetwork, MotionSearch, coding_plane_levels as codingLevels } from '../pkg/cvnn_web.js';
import { requestDevice } from './gpu-device.js';
import { createGpuRuntime } from './gpu-runtime.js';
import { parseIntegerModel } from './int8-model.js';
import { createSchedule, groupCandidates } from './int8-schedule.js';

const BLOCK = 8; // side of a motion block, in input samples
const UNKNOWN_QUANTISER = -1; // what the WebAssembly side takes for "coding not known"

/** The model file's words, refused unless the reference implementation accepts the file. */
async function loadWords(model) {
  const words = new Int32Array(await (await fetch(model.url)).arrayBuffer());
  new IntegerNetwork(words).free();
  return words;
}

/**
 * An upscaler (see upscaler.js) for `model`, an integer export, and `width` x
 * `height` input frames. `options` are int8-network.js's, for checks.
 */
export async function createGpuUpscaler(model, width, height, options = {}) {
  const parsed = parseIntegerModel(await loadWords(model));
  const search = new MotionSearch(width, height);
  const pixels = width * height;
  const vectors = new Int32Array(2 * Math.ceil(width / BLOCK) * Math.ceil(height / BLOCK));
  const canvases = [0, 1].map(() => new OffscreenCanvas(2 * width, 2 * height));
  let gpu = null; // { device, runtime, schedule, lost }
  let stored = false; // whether the GPU holds the previous frame's output and state
  let shown = []; // the last frame's pictures, closed when the next replaces them
  let settling = Promise.resolve(); // resolves when the GPU has finished the newest frame submitted
  let network = 0; // ms from submitting a frame to the GPU finishing it, for the newest finished

  async function start() {
    const device = await requestDevice();
    const runtime = createGpuRuntime(device, parsed, model, width, height, canvases, options);
    const started = { device, runtime, schedule: createSchedule(groupCandidates(runtime.tiles)), lost: false };
    device.lost.then(() => { started.lost = true; });
    return started;
  }

  function stop() {
    gpu.runtime.destroy();
    gpu.device.destroy();
  }

  try {
    gpu = await start();
  } catch (error) {
    search.free();
    throw error;
  }
  return {
    /** See upscaler.js. The pictures are ImageBitmaps, closed by the next call. */
    async process(frame, coding) {
      const start0 = performance.now();
      const reset = search.push(frame.subarray(0, pixels), vectors);
      if (!model.align) vectors.fill(0); // history is used where it lies
      const levels = codingLevels(coding ? coding.quantiser : UNKNOWN_QUANTISER, Boolean(coding?.key));
      if (gpu.lost) {
        // A lost device took the history with it: start again on a new one, as after a cut.
        stop();
        gpu = await start();
        stored = false;
      }
      const prepared = performance.now();
      const { schedule } = gpu;
      const done = gpu.runtime.submit(frame, vectors, stored && !reset, levels, schedule.current());
      shown.forEach((picture) => picture.close());
      // Taken at once: the canvases hold this frame only until they are next drawn to.
      shown = canvases.map((canvas) => canvas.transferToImageBitmap());
      stored = true;
      // The pictures are handed back without waiting for the GPU to finish them: whoever draws
      // them waits exactly as long as needed. Only the frame before is waited for, so that work
      // never queues up behind a GPU that cannot keep pace.
      const earlier = settling;
      settling = done.then((computing) => {
        network = performance.now() - prepared;
        // The tuner goes by the GPU's own time where that is known: the wall
        // clock also counts waiting behind other work, which no grouping changes.
        schedule.record(computing ?? network);
      });
      await earlier;
      const [conventional, upscaled] = shown;
      // `network` is the newest frame the GPU has finished: this one's is not known yet.
      return { conventional, upscaled, timings: { prepare: prepared - start0, network, finish: 0 } };
    },
    /** Forget earlier frames: the next one starts a new stream. */
    restart() {
      search.restart();
      stored = false;
    },
    /** Read back what the last frame produced (gpu-runtime.js): for checks only. */
    read: (names) => gpu.runtime.read(names),
    async release() {
      await settling.catch(() => {});
      shown.forEach((picture) => picture.close());
      stop();
      search.free();
    },
  };
}
