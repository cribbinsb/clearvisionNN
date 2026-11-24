// The magnifier: four quadrants. Top left is the whole original with the
// chosen region marked; the other three show that region of the original,
// the input and the output, each output pixel enlarged to a square so that
// nothing is smoothed over.

const LABEL_SHARE = 1 / 28; // label height as a share of a quadrant's height
const MARK = '#ffd54a';

function label(context, text, x, y, size) {
  context.font = `${size}px system-ui, sans-serif`;
  context.fillStyle = 'rgba(0, 0, 0, 0.65)';
  context.fillRect(x, y, context.measureText(text).width + size, size * 1.5);
  context.fillStyle = '#ffffff';
  context.fillText(text, x + size / 2, y + size * 1.1);
}

/**
 * A magnifier on `canvas`. The region follows the pointer when it is pressed
 * or dragged in the top-left quadrant while `isActive()`; `redraw` is called
 * after each move.
 */
export function createMagnifier(canvas, isActive, redraw) {
  const centre = { x: 0.5, y: 0.5 }; // of the region, as a share of the picture

  function follow(event) {
    if (!isActive() || !(event.buttons & 1)) return;
    const box = canvas.getBoundingClientRect();
    const [x, y] = [(event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height];
    if (x > 0.5 || y > 0.5) return; // only the overview chooses the region
    centre.x = 2 * x;
    centre.y = 2 * y;
    redraw();
  }
  canvas.addEventListener('pointerdown', follow);
  canvas.addEventListener('pointermove', follow);

  return {
    /** Draw the four quadrants from `layers` (pictures named original, input, output) at `zoom`. */
    draw(context, layers, zoom) {
      const [width, height] = [canvas.width / 2, canvas.height / 2];
      const [w, h] = [Math.round(width / zoom), Math.round(height / zoom)];
      const corner = (share, size, full) => Math.min(Math.max(Math.round(share * full - size / 2), 0), full - size);
      const [x, y] = [corner(centre.x, w, canvas.width), corner(centre.y, h, canvas.height)];
      context.imageSmoothingEnabled = true;
      context.drawImage(layers.original, 0, 0, width, height);
      context.strokeStyle = MARK;
      context.lineWidth = 2;
      context.strokeRect(x / 2, y / 2, w / 2, h / 2);
      context.imageSmoothingEnabled = false;
      const quadrants = [['original', width, 0], ['input', 0, height], ['output', width, height]];
      for (const [name, left, top] of quadrants) {
        context.drawImage(layers[name], x, y, w, h, left, top, width, height);
      }
      context.imageSmoothingEnabled = true;
      context.fillStyle = '#ffffff';
      context.fillRect(width - 1, 0, 2, canvas.height);
      context.fillRect(0, height - 1, canvas.width, 2);
      const size = Math.max(12, Math.round(height * LABEL_SHARE));
      label(context, 'Original · whole picture', 0, 0, size);
      label(context, `Original ×${zoom}`, width + 1, 0, size);
      label(context, `Input (bicubic) ×${zoom}`, 0, height + 1, size);
      label(context, `Output ×${zoom}`, width + 1, height + 1, size);
    },
  };
}
