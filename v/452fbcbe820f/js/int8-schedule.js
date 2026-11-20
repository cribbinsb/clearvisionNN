// Which grouping of tiles to run with, chosen by timing on this device: each
// candidate is tried for a few frames, the one with the quickest frame is
// kept, and the choice is tried again now and then or when frames slow down.
// Candidates are compared by their quickest frames because whatever else the
// machine is doing only ever adds time: on a shared GPU the medians of a few
// frames differ by more than the groupings do, the quickest frames do not.

const PROBE_FRAMES = 6; // frames timed per candidate; the first is discarded as warm-up
const REPROBE_AFTER = 600; // frames a choice stands before it is checked again
const DRIFT = 1.3; // a smoothed frame time this many times the choice's median when probed starts a new probe
const SMOOTHING = 0.1; // weight of the newest frame in the smoothed frame time
const HYSTERESIS = 0.95; // a rival must take under this share of the incumbent's time to replace it
const FEWEST_TILES = 2; // the smallest group worth trying: below it dispatch overhead dominates

const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];

/** Group sizes to try for `tiles` tiles: all of them at once, then halves of that. */
export function groupCandidates(tiles) {
  const candidates = [];
  for (let size = tiles; size >= Math.min(tiles, FEWEST_TILES); size = Math.ceil(size / 2)) {
    candidates.push(size);
    if (size === 1) break;
  }
  return candidates;
}

/**
 * A chooser among `candidates` (group sizes). Call `current()` for the group
 * size of the next frame and `record(ms)` with the time that frame took.
 */
export function createSchedule(candidates) {
  const newProbe = () => ({ index: 0, times: [], quickest: [], medians: [] });
  let chosen = candidates[0]; // all the tiles at once until a rival has shown itself faster
  let probing = candidates.length > 1 ? newProbe() : null;
  let settledFor = 0;
  let usual = Infinity; // the choice's median when it was probed
  let smoothed = Infinity;

  function finishProbe() {
    const best = probing.quickest.indexOf(Math.min(...probing.quickest));
    if (probing.quickest[best] < HYSTERESIS * probing.quickest[candidates.indexOf(chosen)]) chosen = candidates[best];
    usual = probing.medians[candidates.indexOf(chosen)];
    // Starting from the usual time, it takes several slow frames to count as drift, not one.
    smoothed = usual;
    probing = null;
    settledFor = 0;
  }

  return {
    current: () => (probing ? candidates[probing.index] : chosen),
    record(ms) {
      if (probing) {
        probing.times.push(ms);
        if (probing.times.length < PROBE_FRAMES) return;
        const timed = probing.times.slice(1);
        probing.quickest.push(Math.min(...timed));
        probing.medians.push(median(timed));
        probing.times = [];
        probing.index += 1;
        if (probing.index === candidates.length) finishProbe();
        return;
      }
      if (candidates.length === 1) return;
      settledFor += 1;
      smoothed += SMOOTHING * (ms - smoothed);
      if (settledFor >= REPROBE_AFTER || smoothed > DRIFT * usual) probing = newProbe();
    },
  };
}
