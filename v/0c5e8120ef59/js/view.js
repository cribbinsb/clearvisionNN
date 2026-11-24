// What the canvas shows: one of the three pictures of a frame (the original,
// the input as its bicubic upscale, the network's output), the input and
// output either side of a divider, or the magnifier. All but the magnifier
// can be zoomed and panned (zoom.js); the divider stays where it is on screen.

import { createMagnifier } from './magnify.js';
import { createZoom } from './zoom.js';

const LAYERS = ['original', 'input', 'output'];

/**
 * A view drawing into `canvas`; the divider position is read from `slider`
 * (0-1000). `show(frame)` takes over a frame's pictures, each an ImageBitmap,
 * and frees them when the next frame replaces them; `set` changes
 * what is shown without waiting for the next frame.
 */
export function createView(canvas, slider) {
  const context = canvas.getContext('2d');
  // The newest frame's pictures are kept as they came, so that any part of them
  // can be drawn at any scale and the view can change between frames; only
  // what the chosen view shows is ever drawn.
  let pictures = null;
  const state = { mode: 'split', zoom: 4 };
  const magnifier = createMagnifier(canvas, () => state.mode === 'magnify', draw);

  const zoom = createZoom(canvas, () => state.mode !== 'magnify', draw);

  /** Draw the canvas's columns `from` to `to` from `picture`, through the zoom. */
  function drawColumns(picture, from, to) {
    if (to <= from) return;
    const [x, y, width, height] = zoom.region(canvas.width, canvas.height);
    const each = width / canvas.width; // picture pixels per canvas pixel
    context.drawImage(picture, x + from * each, y, (to - from) * each, height, from, 0, to - from, canvas.height);
  }

  function drawSplit() {
    const divider = Math.round((slider.value / slider.max) * canvas.width);
    drawColumns(pictures.input, 0, divider);
    drawColumns(pictures.output, divider, canvas.width);
    context.fillStyle = '#ffffff';
    context.fillRect(divider, 0, 1, canvas.height);
  }

  function draw() {
    if (!pictures) return;
    if (state.mode === 'magnify') {
      magnifier.draw(context, pictures, state.zoom);
      return;
    }
    // Enlarged, each picture pixel is a square: nothing is smoothed over.
    context.imageSmoothingEnabled = !zoom.zoomed();
    if (state.mode === 'split') drawSplit();
    else drawColumns(pictures[state.mode], 0, canvas.width);
  }

  slider.addEventListener('input', draw);
  return {
    resize(width, height) {
      canvas.width = width;
      canvas.height = height;
    },
    show(frame) {
      if (pictures) LAYERS.forEach((name) => pictures[name].close());
      pictures = Object.fromEntries(LAYERS.map((name) => [name, frame[name]]));
      draw();
    },
    set(changes) {
      Object.assign(state, changes);
      draw();
    },
  };
}
