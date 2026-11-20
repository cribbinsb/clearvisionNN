// One worker of the simulated camera (camera-pool.js): it reads an original's
// pixels and makes the half-size input frame from them, with the camera's
// blur, noise and denoise. This is the heaviest work on the processor, and
// frames do not depend on each other, so several of these run side by side.

import { Sender } from '../pkg/cvnn_web.js';
import { runtimeReady } from './wasm.js';

/** Answer `scope`'s messages: `open` a camera of one size, then `capture` each picture sent. */
function serve(scope) {
  let camera = null; // { sender, reader, width, height, settings }
  const handlers = {
    async open({ width, height, kernel, seed, blur, noise, denoise }) {
      await runtimeReady();
      // The original's pixels are read back through a canvas: the camera works on bytes.
      const reader = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
      camera = { sender: new Sender(width, height, kernel, seed), reader, width, height, levels: [blur, noise, denoise] };
      return [null, []];
    },
    capture({ picture }) {
      const { sender, reader, width, height, levels } = camera;
      const start = performance.now();
      reader.drawImage(picture, 0, 0);
      const pixels = reader.getImageData(0, 0, width, height).data;
      const read = performance.now();
      // A new array each time: it leaves with the answer.
      const frame = new Uint8Array((width * height * 3) / 8);
      sender.send(pixels, ...levels, frame);
      const timings = { read: read - start, degrade: performance.now() - read };
      // The picture goes back untouched, to be shown as the original.
      return [{ picture, frame, timings }, [picture, frame.buffer]];
    },
  };
  scope.onmessage = async ({ data }) => {
    try {
      const [value, moved] = await handlers[data.type](data);
      scope.postMessage({ id: data.id, value }, moved);
    } catch (error) {
      scope.postMessage({ id: data.id, error: error?.message ?? String(error) });
    }
  };
}

serve(self);
