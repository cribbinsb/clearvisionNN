// The run lifecycle: at most one source and one pipeline exist at a time,
// starts never overlap, and whatever was acquired is released on
// every path, including failures part-way through starting. A run can be
// paused, stepped a frame at a time and resumed.

import { CAMERAS } from './camera-pool.js';

const WARM_UP_FRAMES = 5; // the first frames compile shaders and fill caches; they are not timed
// Frames inside the pipeline at once: one with each of the camera's workers while the channel and
// the GPU finish the one before.
const IN_FLIGHT = CAMERAS + 1;

/**
 * A controller that runs sessions built by `open(settings)`, which resolves to
 * `{ source, pipeline }`. `onFrame(result, timed)` receives every processed
 * frame and takes over its pictures; `onState(state, detail)` every change of
 * 'idle' | 'starting' | 'running' | 'paused' | 'failed'.
 */
export function createController({ open, onFrame, onState }) {
  let current = null; // { source, pipeline, stopped, paused, steps, wake, finished }
  let queue = Promise.resolve(); // serialises start requests

  async function release(session) {
    session.source?.stop();
    await session.pipeline?.release();
  }

  /** Let a paused session's loop look at its flags again. */
  function wake(session) {
    session.wake?.();
    session.wake = null;
  }

  async function loop(session, frameLimit) {
    const { source, pipeline } = session;
    const pending = []; // frames inside the pipeline, oldest first
    let shown = 0;
    const feed = async () => {
      const asked = performance.now();
      const { picture, time } = await source.grab();
      const grabbed = performance.now();
      pending.push(pipeline.process(picture, time).then((result) => result && {
        ...result,
        // What the page's own thread spent fetching the frame, and how long the pipeline held it.
        timings: { ...result.timings, grab: grabbed - asked, turnaround: performance.now() - grabbed },
      }));
    };
    // A pipeline may hold a frame back (a codec does) and return nothing for it.
    const showOldest = async () => {
      const result = await pending.shift();
      if (!result) return;
      onFrame(result, shown >= WARM_UP_FRAMES);
      shown += 1;
    };
    /** Paused: show what is inside, then wait; a step lets one more frame through. */
    const hold = async () => {
      source.pause();
      while (pending.length) await showOldest();
      if (session.steps === 0) await new Promise((resolve) => { session.wake = resolve; });
      if (session.stopped) return;
      await source.resume();
      if (!session.paused) return;
      session.steps -= 1;
      await source.nextFrame();
      await feed();
    };
    try {
      while (!session.stopped && shown < frameLimit) {
        if (session.paused) {
          await hold();
        } else {
          await feed();
          if (pending.length >= IN_FLIGHT) await showOldest();
          await source.nextFrame();
        }
      }
      onState('idle');
    } catch (error) {
      onState('failed', error);
    } finally {
      // Frames still inside are let through and their pictures freed, so the pipeline is idle when released.
      for (const outcome of await Promise.allSettled(pending)) outcome.value?.close();
    }
  }

  async function stopCurrent() {
    const session = current;
    if (!session) return;
    current = null;
    session.stopped = true;
    wake(session);
    try {
      await session.finished;
    } finally {
      await release(session);
    }
  }

  async function startNew(settings, frameLimit) {
    await stopCurrent();
    onState('starting');
    const session = { source: null, pipeline: null, stopped: false, paused: false, steps: 0, wake: null, finished: null };
    try {
      Object.assign(session, await open(settings, session));
    } catch (error) {
      await release(session);
      onState('failed', error);
      throw error;
    }
    current = session;
    onState('running', session);
    session.finished = loop(session, frameLimit);
    return session;
  }

  const enqueue = (task) => {
    const result = queue.then(task);
    queue = result.catch(() => {});
    return result;
  };

  /** Pause or resume the session that is running, if any. */
  function setPaused(paused) {
    if (!current || current.paused === paused) return;
    current.paused = paused;
    current.steps = 0;
    wake(current);
    onState(paused ? 'paused' : 'running', current);
  }

  return {
    /** Stop whatever runs, then run `settings` for up to `frameLimit` frames; resolves when the run ends. */
    // Only the start-up is queued: queueing the run itself would hold every later start behind it.
    start: (settings, frameLimit = Infinity) => enqueue(() => startNew(settings, frameLimit))
      .then((session) => session.finished),
    /** Hold the run at the frame it is on: what is inside the pipeline is shown, then nothing moves. */
    pause: () => setPaused(true),
    resume: () => setPaused(false),
    /** While paused, let one more frame through. */
    step() {
      if (!current?.paused) return;
      current.steps += 1;
      wake(current);
    },
  };
}
