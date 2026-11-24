// The "Debug info" button: everything needed to work out, from somewhere else,
// why the page fails or is slow on a machine, as text its user can copy and
// send. It says what the browser and its GPU are, what the video decoders
// accept, what the page was running and how that went, and it keeps a short
// log of the run in the browser's storage, so that after a page that locked
// up is reloaded the report still shows what the session before it was doing
// when it stopped. It holds no picture and nothing that identifies a person.

const STORE = 'upscaler-debug-log';
// Events kept per session: a run's start-up and several changes of settings.
const MAX_EVENTS = 80;
// A probe that has not answered in this long is reported as such, not waited
// for: a stalled GPU process is exactly what the report must be able to describe.
const PROBE_MS = 3000;
// The bundled clips (H.264 High, level 5.0) and the simulated sender's stream, at their sizes.
const DECODES = [
  { codec: 'avc1.640032', codedWidth: 1280, codedHeight: 720 },
  { codec: 'avc1.42001f', codedWidth: 640, codedHeight: 360 },
];
const ACCELERATIONS = ['no-preference', 'prefer-hardware', 'prefer-software'];
const CLIENT_HINTS = ['platform', 'platformVersion', 'architecture', 'bitness', 'model', 'fullVersionList'];

const text = (error) => `failed: ${error?.message ?? error}`;

/** `probe()`'s result, or words saying that it threw or did not answer. */
async function attempt(probe) {
  let timer;
  const late = new Promise((resolve) => { timer = setTimeout(() => resolve('no answer after 3 s'), PROBE_MS); });
  try {
    return await Promise.race([Promise.resolve().then(probe), late]);
  } catch (error) {
    return text(error);
  } finally {
    clearTimeout(timer);
  }
}

function read() {
  try {
    return JSON.parse(localStorage.getItem(STORE)) ?? [];
  } catch {
    return [];
  }
}

/**
 * A log of this session's events, each with the milliseconds since the page
 * loaded, written through to storage as it grows; `previous` is the log of
 * the session before, which ends in 'page closed' if that one ended normally.
 */
export function createDebugLog() {
  const previous = read();
  const events = [];
  const add = (what) => {
    events.push(`${Math.round(performance.now())} ms: ${what}`);
    if (events.length > MAX_EVENTS) events.shift();
    try {
      localStorage.setItem(STORE, JSON.stringify(events));
    } catch {
      // Storage refused (private browsing, say): this session's log is still reported.
    }
  };
  add(`page loaded ${new Date().toISOString()}`);
  addEventListener('error', (event) => add(`error: ${event.message} (${event.filename}:${event.lineno})`));
  addEventListener('unhandledrejection', (event) => add(`unhandled rejection: ${event.reason?.message ?? event.reason}`));
  addEventListener('pagehide', () => add('page closed'));
  return { add, events, previous };
}

/** The adapter `options` asks for; its long lists only when `full`. */
async function describeAdapter(options, full) {
  const adapter = await navigator.gpu.requestAdapter(options);
  if (!adapter) return 'none offered';
  const info = adapter.info ?? {};
  const named = {
    vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
    software: info.isFallbackAdapter ?? adapter.isFallbackAdapter,
  };
  if (!full) return named;
  const limits = {};
  for (const name in adapter.limits) limits[name] = adapter.limits[name];
  return { ...named, features: [...adapter.features].sort().join(' '), limits };
}

function webgl() {
  const context = document.createElement('canvas').getContext('webgl2');
  if (!context) return 'none';
  const names = context.getExtension('WEBGL_debug_renderer_info');
  // The renderer string names the driver and the graphics API the browser draws through.
  return {
    vendor: context.getParameter(names ? names.UNMASKED_VENDOR_WEBGL : context.VENDOR),
    renderer: context.getParameter(names ? names.UNMASKED_RENDERER_WEBGL : context.RENDERER),
  };
}

async function decoders() {
  if (typeof VideoDecoder !== 'function') return 'no VideoDecoder';
  const found = {};
  for (const config of DECODES) {
    for (const hardwareAcceleration of ACCELERATIONS) {
      const name = `${config.codec} ${config.codedWidth}x${config.codedHeight} ${hardwareAcceleration}`;
      found[name] = await attempt(async () => (await VideoDecoder.isConfigSupported({ ...config, hardwareAcceleration })).supported);
    }
  }
  return found;
}

/** The report, as text. `run()` gives what the page is doing now; `log` is this page's debug log. */
export async function debugReport(run, log) {
  const { userAgentData } = navigator;
  const report = {
    page: { url: location.href, build: import.meta.url, time: new Date().toISOString() },
    browser: {
      userAgent: navigator.userAgent,
      hints: userAgentData ? await attempt(() => userAgentData.getHighEntropyValues(CLIENT_HINTS)) : 'none',
      processors: navigator.hardwareConcurrency, memoryGb: navigator.deviceMemory,
      screen: `${screen.width}x${screen.height} at ${devicePixelRatio}`, isolated: crossOriginIsolated,
    },
    webgpu: 'gpu' in navigator ? {
      canvasFormat: navigator.gpu.getPreferredCanvasFormat(),
      adapter: await attempt(() => describeAdapter(undefined, true)),
      highPerformanceAdapter: await attempt(() => describeAdapter({ powerPreference: 'high-performance' }, false)),
    } : 'not in this browser',
    webgl: await attempt(webgl),
    videoDecoders: await decoders(),
    videoEncoder: typeof VideoEncoder === 'function',
    run: await attempt(run),
    log: log.events,
    previousSession: log.previous,
  };
  return JSON.stringify(report, null, 1);
}

/**
 * Wire `button` to write the report into `output` (a textarea, shown then) and
 * to the clipboard, saying which in `note`.
 */
export function setUpDebugButton(button, output, note, run, log) {
  button.addEventListener('click', async () => {
    button.disabled = true;
    note.textContent = 'Collecting…';
    output.value = await debugReport(run, log);
    output.hidden = false;
    try {
      await navigator.clipboard.writeText(output.value);
      note.textContent = 'Copied to the clipboard: paste it into a message. It holds no picture and no personal data.';
    } catch {
      output.select();
      note.textContent = 'Copy the text below and send it. It holds no picture and no personal data.';
    }
    button.disabled = false;
  });
}
