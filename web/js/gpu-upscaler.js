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
const TRIAL_FRAMES = 3; // frames each way of writing the shaders is timed on, the first not counted
const TRIAL_GIVE_UP = 3; // a way this many times slower than the best so far is dropped at once
const FIRST_FRAME_GIVE_UP = 5; // and one whose very first frame is this many times too slow, after that frame
// The network's GPU time above which the page's own shaders are not worth running: a float model
// on the browser's WebGPU would be quicker. Some GPUs take seconds where others take a millisecond.
const TOO_SLOW_MS = 100;
/** What `error.code` is when the shaders work but are too slow to use (app.js then tries a float model). */
export const SLOW_GPU = 'slow-gpu';

/**
 * The ways the network's products can be formed (int8-shaders.js), each with a name for the
 * diagnostics: in packed integers, quickest where the GPU has hardware for them, and in exact
 * float, far quicker where it has not (an Apple GPU).
 */
const VARIANTS = [
  { name: 'integer products', options: {} },
  { name: 'float products', options: { floatProducts: true } },
];

/** The model file's words, refused unless the reference implementation accepts the file. */
async function loadWords(model) {
  const words = new Int32Array(await (await fetch(model.url)).arrayBuffer());
  new IntegerNetwork(words).free();
  return words;
}

/**
 * An upscaler (see upscaler.js) for `model`, an integer export, and `width` x
 * `height` input frames. `options` are int8-network.js's, for checks, and
 * `tooSlowMs`, the GPU time a frame above which the upscaler refuses to start
 * (its error has the code `SLOW_GPU`), and `remembered`, a Map kept by the
 * caller in which the quickest way found for a model and size is kept, so
 * that only the first start on a device pays for the trial.
 */
export async function createGpuUpscaler(model, width, height, { tooSlowMs = TOO_SLOW_MS, remembered = new Map(), ...options } = {}) {
  const parsed = parseIntegerModel(await loadWords(model));
  const search = new MotionSearch(width, height);
  const pixels = width * height;
  const vectors = new Int32Array(2 * Math.ceil(width / BLOCK) * Math.ceil(height / BLOCK));
  const blank = new Uint8Array((pixels * 3) / 2); // a frame for the trials
  const memory = `${model.directory} ${width}x${height}`; // what a trial's outcome is remembered under
  const canvases = [0, 1].map(() => new OffscreenCanvas(2 * width, 2 * height));
  let gpu = null; // { device, runtime, schedule, lost }
  let stored = false; // whether the GPU holds the previous frame's output and state
  let shown = []; // the last frame's pictures, closed when the next replaces them

  /** Time `runtime` on blank frames: the GPU's own milliseconds for the quickest after the first. */
  async function trial(runtime, limit) {
    const group = createSchedule(groupCandidates(runtime.tiles)).current();
    const levels = codingLevels(UNKNOWN_QUANTISER, false);
    let quickest = Infinity;
    for (let frame = 0; frame < TRIAL_FRAMES; frame += 1) {
      const submitted = performance.now();
      const computing = await runtime.submit(blank, vectors, false, levels, group);
      const took = computing ?? performance.now() - submitted;
      // The first frame may also finish compiling the shaders, so it is not counted, unless it is
      // so slow that the way is already out of the running: on some GPUs one takes seconds.
      if (frame === 0 && took > FIRST_FRAME_GIVE_UP * Math.min(limit, tooSlowMs)) return took;
      if (frame > 0) quickest = Math.min(quickest, took);
      if (frame > 0 && quickest > TRIAL_GIVE_UP * limit) break;
    }
    return quickest;
  }

  /**
   * The runtime written the way this GPU runs fastest. The network's shaders can bind their
   * weights and form their dot products in more than one way; which is quickest differs between
   * GPU makers by orders of magnitude, so each way is timed here and the best kept. Options given
   * by the caller (a check) are used as they are.
   */
  async function quickestRuntime(device) {
    if (Object.keys(options).length) return { runtime: createGpuRuntime(device, parsed, model, width, height, canvases, options), variant: 'as asked' };
    let best = null; // { runtime, variant, time }
    // What was found quickest for this model and size before is used again without a trial.
    const known = remembered.get(memory);
    if (known) return { runtime: createGpuRuntime(device, parsed, model, width, height, canvases, known.options), variant: known.variant };
    for (const variant of VARIANTS) {
      let runtime;
      try {
        runtime = createGpuRuntime(device, parsed, model, width, height, canvases, variant.options);
      } catch {
        continue; // a way this device cannot build is simply not a candidate
      }
      const time = await trial(runtime, best?.time ?? Infinity);
      if (best && time >= best.time) {
        runtime.destroy();
      } else {
        best?.runtime.destroy();
        best = { runtime, variant: `${variant.name}, ${time.toFixed(1)} ms`, time, options: variant.options };
      }
    }
    if (!best) throw new Error('This GPU could not build the network\'s shaders.');
    if (best.time > tooSlowMs) {
      best.runtime.destroy();
      const slow = new Error(`The 8-bit shaders take ${best.time.toFixed(0)} ms a frame on this GPU, which is too slow to use.`);
      slow.code = SLOW_GPU;
      throw slow;
    }
    remembered.set(memory, { options: best.options, variant: best.variant });
    return best;
  }

  async function start() {
    const device = await requestDevice();
    try {
      const { runtime, variant } = await quickestRuntime(device);
      const started = { device, runtime, variant, schedule: createSchedule(groupCandidates(runtime.tiles)), lost: false };
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
      shown = canvases.map((canvas) => canvas.transferToImageBitmap());
      // The pictures are handed back only once the GPU has finished them: a browser shows a
      // picture drawn before then as black.
      const computing = await done;
      stored = !gpu.lost;
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
