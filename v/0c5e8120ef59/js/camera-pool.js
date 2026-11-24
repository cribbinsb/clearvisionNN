// The simulated camera, as a few workers taking frames in turn
// (camera-worker.js): a frame's blur, noise, denoise and reduction take longer
// than everything else done to it on the processor, and frames do not depend
// on each other, so they are made side by side and handed on in order.

const NOISE_SEED = 1; // fixed, so a run can be repeated exactly; each worker adds its number
// Two keep a 30 frames a second clip at speed with every step switched on; each
// more adds a frame to what is inside the pipeline at once (controller.js).
export const CAMERAS = 2;

/**
 * A camera for `width` x `height` originals set as `degrade` says (degrade.js).
 * `capture(picture)` takes an ImageBitmap and resolves to `{ picture, frame,
 * timings }`: the same picture, the half-size input made from it as 8-bit
 * 4:2:0, and the time reading and making took (ms). Frames go to the workers
 * in turn, so a run's noise repeats exactly.
 */
export function createCameraPool(width, height, degrade) {
  const waiting = new Map(); // calls not yet answered, by id
  let nextId = 0;
  let turn = 0;

  function call(worker, type, payload, moved = []) {
    return new Promise((resolve, reject) => {
      waiting.set(nextId, { resolve, reject });
      worker.postMessage({ id: nextId, type, ...payload }, moved);
      nextId += 1;
    });
  }

  const workers = Array.from({ length: CAMERAS }, () => {
    const worker = new Worker(new URL('./camera-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      const { resolve, reject } = waiting.get(data.id);
      waiting.delete(data.id);
      if (data.error === undefined) resolve(data.value);
      else reject(new Error(data.error));
    };
    return worker;
  });
  const settings = { width, height, kernel: degrade.kernel, blur: degrade.blur, noise: degrade.sigma, denoise: degrade.denoise };
  const opened = Promise.all(workers.map((worker, number) => call(worker, 'open', { ...settings, seed: NOISE_SEED + number })));

  return {
    async capture(picture) {
      const worker = workers[turn];
      turn = (turn + 1) % workers.length;
      await opened;
      return call(worker, 'capture', { picture }, [picture]);
    },
    release() {
      workers.forEach((worker) => worker.terminate());
      for (const { reject } of waiting.values()) reject(new Error('The camera was released.'));
      waiting.clear();
    },
  };
}
