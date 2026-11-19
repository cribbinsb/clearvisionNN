// The upscaler entirely on the GPU, for a model in integer form: the motion
// search runs in WebAssembly on the small luma plane, and everything else
// (the conventional upscale, warping history, the network, colour) is the
// page's own shaders. The pictures come back as bitmaps the GPU still holds;
// no pixel is read back to the processor.

import { IntegerNetwork, MotionSearch, coding_plane_levels as codingLevels } from '../pkg/cvnn_web.js';
import { requestDevice } from './gpu-device.js';
import { canvasArrives } from './gpu-handover.js';
import { createGpuRuntime } from './gpu-runtime.js';
import { parseIntegerModel } from './int8-model.js';
import { createSchedule, groupCandidates } from './int8-schedule.js';

const BLOCK = 8; // side of a motion block, in input samples
const PICTURE_BUFFERS = ['conventional', 'output']; // the GPU's buffers of the two pictures (gpu-runtime.js)
const UNKNOWN_QUANTISER = -1; // what the WebAssembly side takes for "coding not known"
// GPU makers whose hardware has no packed 8-bit dot product and runs integer arithmetic slowly:
// there the network's products are formed in float (int8-shaders.js), about ten times quicker on
// an Apple M5 (measured there). The choice is by maker, not by timing: one frame of the wrong
// form takes seconds on such a GPU.
const FLOAT_PRODUCT_VENDORS = ['apple'];
const FLOAT_SPAN = 4; // pixels of a row an invocation computes there: twice as quick again on an M5

/** The model file's words, refused unless the reference implementation accepts the file. */
async function loadWords(model) {
  const words = new Int32Array(await (await fetch(model.url)).arrayBuffer());
  new IntegerNetwork(words).free();
  return words;
}

/**
 * An upscaler (see upscaler.js) for `model`, an integer export, and `width` x
 * `height` input frames. `options` are int8-network.js's, for checks; how the
 * products are formed is otherwise chosen from the GPU's maker. `fastGpu` asks
 * for the fast GPU of two; `present` fixes how the pictures leave the GPU
 * ('canvas' or 'copy') where that is not to be found out.
 */
export async function createGpuUpscaler(model, width, height, { fastGpu = false, present, ...options } = {}) {
  const parsed = parseIntegerModel(await loadWords(model));
  const search = new MotionSearch(width, height);
  const pixels = width * height;
  const vectors = new Int32Array(2 * Math.ceil(width / BLOCK) * Math.ceil(height / BLOCK));
  const canvases = [0, 1].map(() => new OffscreenCanvas(2 * width, 2 * height));
  let gpu = null; // { device, runtime, schedule, lost }
  let stored = false; // whether the GPU holds the previous frame's output and state
  let shown = []; // the last frame's pictures, closed when the next replaces them

  async function start() {
    const device = await requestDevice({ fast: fastGpu });
    // On the fast GPU of two the pictures may not survive being handed to the browser as
    // canvases; where they do not (or as asked), they are copied out through the processor.
    const copied = present === 'copy' || (present !== 'canvas' && fastGpu && !(await canvasArrives(device, 2 * width, 2 * height)));
    const slowIntegers = FLOAT_PRODUCT_VENDORS.includes(device.adapterInfo?.vendor);
    // 16-bit float where the GPU has it, 32-bit float otherwise.
    const halfProducts = options.halfProducts ?? (slowIntegers && options.floatProducts === undefined && device.features.has('shader-f16'));
    const floatProducts = options.floatProducts ?? (slowIntegers && !halfProducts);
    // With float products, several pixels to an invocation: each weight read then serves them all.
    const span = options.span ?? (floatProducts || halfProducts ? FLOAT_SPAN : 1);
    try {
      const runtime = createGpuRuntime(device, parsed, model, width, height, canvases, { ...options, floatProducts, halfProducts, span });
      const variant = `${halfProducts ? '16-bit float' : floatProducts ? 'float' : 'integer'} products, ${span} to an invocation${copied ? ', pictures copied out' : ''}`;
      const started = { device, runtime, variant, copied, schedule: createSchedule(groupCandidates(runtime.tiles)), lost: false };
      device.lost.then(() => { started.lost = true; });
      return started;
    } catch (error) {
      device.destroy();
      throw error;
    }
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
      const done = gpu.runtime.submit(frame, vectors, stored && !reset, levels, gpu.schedule.current());
      shown.forEach((picture) => picture.close());
      // Taken before the wait: the canvases hold this frame only until they are next drawn to.
      const drawn = canvases.map((canvas) => canvas.transferToImageBitmap());
      // The pictures are handed back only once the GPU has finished them: a browser shows a
      // picture drawn before then as black.
      const computing = await done;
      stored = !gpu.lost;
      if (gpu.copied) {
        // The canvases' pictures would arrive black: the same pixels are read back from the GPU's
        // buffers (packed RGBA words, which are an image's bytes as they stand) and made pictures here.
        drawn.forEach((picture) => picture.close());
        const words = await gpu.runtime.read(PICTURE_BUFFERS);
        shown = await Promise.all(words.map((picture) => createImageBitmap(
          new ImageData(new Uint8ClampedArray(picture.buffer, picture.byteOffset, picture.byteLength), 2 * width, 2 * height),
        )));
      } else {
        shown = drawn;
      }
      const finished = performance.now();
      // The tuner goes by the GPU's own time where that is known: the wall
      // clock also counts waiting behind other work, which no grouping changes.
      gpu.schedule.record(computing ?? finished - prepared);
      const [conventional, upscaled] = shown;
      // `compute` is the GPU's own time for the network, where the device reports it; `network`
      // also counts waiting for the GPU, which other programs using it lengthen.
      const timings = { prepare: prepared - start0, network: finished - prepared, finish: 0 };
      if (computing !== null) timings.compute = computing;
      return { conventional, upscaled, timings, variant: gpu.variant };
    },
    /** Forget earlier frames: the next one starts a new stream. */
    restart() {
      search.restart();
      stored = false;
    },
    /** Read back what the last frame produced (gpu-runtime.js): for checks only. */
    read: (names) => gpu.runtime.read(names),
    async release() {
      shown.forEach((picture) => picture.close());
      stop();
      search.free();
    },
  };
}
