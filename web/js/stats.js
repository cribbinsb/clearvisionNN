// Smoothed timings and facts about the stream, shown under the view.

const SMOOTHING = 0.1; // weight of the newest frame in each running average
const UPSCALER_STEPS = ['prepare', 'network', 'finish'];
const SENDER_STEPS = ['degrade', 'codec'];

/** What the codec did to the newest frame, in words. */
function describeCoding(info) {
  if (info.key === undefined) return 'no codec';
  const told = info.told ? 'the model is told' : 'the model is not told';
  const quantiser = info.quantiser === null ? 'quantiser not reported by this encoder' : `QP ${info.quantiser}, ${told}`;
  return `${info.key ? 'key' : 'delta'} frame, ${quantiser}`;
}

/** A recorder that averages per-frame timings and writes them into the page's `cells`. */
export function createStats(cells) {
  const averages = {};
  let lastFrame = null;
  let lastInfo = {};
  const update = (key, value) => {
    averages[key] = key in averages ? averages[key] + SMOOTHING * (value - averages[key]) : value;
  };
  return {
    /** `timings` in ms by step; `info` is what the sender reported for the frame; `asked` the bitrate requested (kbit/s) or null. */
    record(timings, info, asked) {
      Object.entries(timings).forEach(([key, value]) => update(key, value));
      const now = performance.now();
      if (lastFrame !== null) update('interval', now - lastFrame);
      lastFrame = now;
      lastInfo = info;
      for (const key of [...UPSCALER_STEPS, ...SENDER_STEPS]) cells[key].textContent = `${averages[key].toFixed(1)} ms`;
      const upscaler = UPSCALER_STEPS.reduce((sum, key) => sum + averages[key], 0);
      cells.upscalerFps.textContent = (1000 / upscaler).toFixed(1);
      if ('interval' in averages) cells.fps.textContent = (1000 / averages.interval).toFixed(1);
      const target = asked === null ? '' : ` (asked for ${asked.toFixed(0)})`;
      cells.bitrate.textContent = info.kbps === undefined ? '–' : `${info.kbps.toFixed(0)} kbit/s${target}`;
      cells.coding.textContent = describeCoding(info);
    },
    snapshot: () => ({ ...averages, ...lastInfo }),
    reset() {
      Object.keys(averages).forEach((key) => delete averages[key]);
      lastFrame = null;
      lastInfo = {};
    },
  };
}
