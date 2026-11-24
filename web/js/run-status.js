// What the page is doing, where it cannot be missed: a coloured badge with the
// run's state beside the status line, and a watchdog that says so when frames
// stop arriving. The pipeline runs in a worker, so the page's own thread stays
// free to notice a GPU that has stalled or is far too slow.

// Without a frame for this long the run is reported as stalled. Longer than the
// largest model's shaders take to compile on a slow GPU.
const STALL_MS = 8000;
const CHECK_MS = 1000;
const LABELS = {
  loading: 'Loading', starting: 'Starting', running: 'Running', paused: 'Paused',
  idle: 'Finished', failed: 'Failed', stalled: 'Not responding',
};
const STALLED = 'No frame has arrived for several seconds. The GPU may be too slow for this model, or its driver may have stalled: choose a smaller model, or reload the page.';

/** A status display on the `badge` and `line` elements; `onStall(stalled)` hears the watchdog change its mind. */
export function createRunStatus(badge, line, onStall) {
  let state = 'loading';
  let message = '';
  let note = '';
  let lastBeat = performance.now();
  let stalled = false;

  function draw() {
    const shown = stalled ? 'stalled' : state;
    badge.dataset.state = shown;
    badge.textContent = LABELS[shown];
    line.textContent = stalled ? STALLED : [message, note].filter(Boolean).join(' \u00b7 ');
    line.classList.toggle('error', state === 'failed' || stalled);
  }

  setInterval(() => {
    const expecting = state === 'starting' || state === 'running';
    const late = expecting && performance.now() - lastBeat > STALL_MS;
    if (late !== stalled) { stalled = late; draw(); onStall?.(late); }
  }, CHECK_MS);
  draw();

  return {
    /** Enter `next` (a key of LABELS) with `text` on the status line. */
    set(next, text) {
      state = next; message = text; lastBeat = performance.now(); stalled = false;
      draw();
    },
    /** A frame arrived; `remark` is anything unusual about the run ('' for nothing). */
    beat(remark = '') {
      lastBeat = performance.now();
      if (!stalled && remark === note) return;
      stalled = false; note = remark;
      draw();
    },
    failed: () => state === 'failed',
    text: () => line.textContent,
  };
}
