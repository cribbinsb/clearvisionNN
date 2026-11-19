// A WebGPU device for the page's own shaders.

// Chrome can answer "no adapter" while its GPU process is still starting, so
// the request is repeated a few times before the page gives up.
const ADAPTER_ATTEMPTS = 8;
const ATTEMPT_INTERVAL_MS = 250;

// Asked for where the adapter offers them: timestamp queries time the GPU itself (gpu-timer.js).
const OPTIONAL_FEATURES = ['timestamp-query'];

const pause = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

async function requestAdapter() {
  if (!('gpu' in navigator)) return null;
  for (let attempt = 0; attempt < ADAPTER_ATTEMPTS; attempt += 1) {
    // On a machine with two GPUs the default may be the slower, integrated one.
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) return adapter;
    await pause(ATTEMPT_INTERVAL_MS);
  }
  return null;
}

/** Whether this browser can give the page a WebGPU device at all. */
export async function hasWebGpu() {
  return Boolean(await requestAdapter());
}

/** Resolves to the GPU the browser offers, in words, or null if it offers none: for the page's diagnostics. */
export async function describeGpu() {
  const adapter = await requestAdapter();
  if (!adapter) return null;
  const { vendor, architecture, description } = adapter.info ?? {};
  return [vendor, architecture, description].filter(Boolean).join(' ') || 'unnamed adapter';
}

/** Resolves to a new device; rejects if the browser offers no adapter. */
export async function requestDevice() {
  const adapter = await requestAdapter();
  if (!adapter) throw new Error('This browser has no usable WebGPU adapter.');
  return adapter.requestDevice({ requiredFeatures: OPTIONAL_FEATURES.filter((feature) => adapter.features.has(feature)) });
}
