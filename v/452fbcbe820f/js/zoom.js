// Zoom and pan of the whole-picture views (original, input, output, and the
// two either side of the divider): the wheel or a pinch zooms about the
// pointer, a drag pans, a double click shows the whole picture again. The
// state is the scale and the picture point at the view's centre, in shares of
// the picture, so it survives a change of size. Adapted from nanocodec's
// view-zoom.js (BSD-3-Clause, same owner).

const MAX_ZOOM = 32;
// One wheel notch (deltaY 100) zooms by this factor.
const WHEEL_ZOOM_PER_100 = 1.25;
const FIT = Object.freeze({ scale: 1, u: 0.5, v: 0.5 });

/**
 * Zoom and pan on `canvas`, followed while `isActive()`; `redraw` is called
 * after each change. `region(width, height)` gives the part of a picture of
 * that size to show, as [x, y, width, height]; `zoomed()` whether that is
 * less than all of it.
 */
export function createZoom(canvas, isActive, redraw) {
  let state = { ...FIT };
  const pointers = new Map(); // pointers that are down, by id, with their last position
  let pinch = null; // distance between two pointers at their last move

  function set({ scale, u, v }) {
    // The shown region stays inside the picture.
    const half = 0.5 / scale;
    const clamp = (value) => Math.max(half, Math.min(1 - half, value));
    state = { scale, u: clamp(u), v: clamp(v) };
    canvas.style.cursor = scale > 1 && isActive() ? 'grab' : '';
    redraw();
  }

  /** The picture point under a pointer event, in shares of the picture. */
  function pointAt(event) {
    const box = canvas.getBoundingClientRect();
    const half = 0.5 / state.scale;
    return {
      u: state.u - half + (event.clientX - box.left) / box.width / state.scale,
      v: state.v - half + (event.clientY - box.top) / box.height / state.scale,
    };
  }

  /** Zoom by `factor`, the picture point `at` staying where it is on screen. */
  function zoomBy(factor, at) {
    const scale = Math.max(FIT.scale, Math.min(MAX_ZOOM, state.scale * factor));
    const ratio = state.scale / scale;
    set({ scale, u: at.u + (state.u - at.u) * ratio, v: at.v + (state.v - at.v) * ratio });
  }

  canvas.addEventListener('wheel', (event) => {
    // At the whole picture, scrolling down scrolls the page instead.
    if (!isActive() || (state.scale === FIT.scale && event.deltaY >= 0)) return;
    event.preventDefault();
    zoomBy(WHEEL_ZOOM_PER_100 ** (-event.deltaY / 100), pointAt(event));
  }, { passive: false });
  canvas.addEventListener('dblclick', () => { if (isActive()) set(FIT); });
  canvas.addEventListener('pointerdown', (event) => {
    if (!isActive() || event.button > 0) return;
    pointers.set(event.pointerId, event);
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    const last = pointers.get(event.pointerId);
    if (!last) return;
    pointers.set(event.pointerId, event);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const distance = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (pinch) zoomBy(distance / pinch, state);
      pinch = distance;
    } else {
      const box = canvas.getBoundingClientRect();
      const [du, dv] = [(event.clientX - last.clientX) / box.width, (event.clientY - last.clientY) / box.height];
      set({ ...state, u: state.u - du / state.scale, v: state.v - dv / state.scale });
    }
  });
  const end = (event) => {
    pointers.delete(event.pointerId);
    pinch = null;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  return {
    region(width, height) {
      const half = 0.5 / state.scale;
      return [(state.u - half) * width, (state.v - half) * height, width / state.scale, height / state.scale];
    },
    zoomed: () => state.scale > FIT.scale,
  };
}
