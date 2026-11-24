// The page: reads the controls, opens sessions through the controller and
// shows frames and timings. It runs from the moment it loads. Changing the
// source, size, model, backend or anything under "Degrade the input" starts
// again with the new settings; changing the view never does, so a recurrent
// model keeps its history while the pictures are examined.

import { backendsFor, usableBackends } from './backends.js';
import { COMPARISONS } from './comparison-upscale.js';
import { SOFTWARE_ADAPTER, describeGpu } from './gpu-device.js';
import { CLIP_HEIGHT, CLIP_WIDTH, CLIPS, FIRST_CLIP, clipUrl } from './clips.js';
import { createController } from './controller.js';
import { setUpDegrade } from './degrade.js';
import { describe, loadModels } from './models.js';
import { createPipelineHost } from './remote-pipeline.js';
import { createRunStatus } from './run-status.js';
import { openSource } from './source.js';
import { createStats } from './stats.js';
import { tuningFrom } from './tuning.js';
import { runUnattended } from './unattended.js';
import { createView } from './view.js';

const element = (id) => document.getElementById(id);
const byId = (ids) => Object.fromEntries(ids.map((id) => [id, element(id)]));
const controls = {
  ...byId(['source', 'file', 'size', 'model', 'backend', 'split', 'zoom', 'preset', 'blur', 'blurValue', 'noise', 'noiseValue', 'denoise', 'denoiseValue', 'kernel', 'codec', 'rate', 'keyint', 'comparison', 'pause', 'step']),
  view: [...document.querySelectorAll('input[name="view"]')],
};
// The codec's time has a cell id of its own: `codec` is the codec control.
const statCells = () => ({ ...byId(['fps', 'upscalerFps', 'prepare', 'network', 'compute', 'wait', 'finish', 'degrade', 'bitrate', 'coding']), codec: element('codec-time') });
const diagnostics = element('diagnostics');
const status = createRunStatus(element('state'), element('status'));
const credit = element('credit');
const view = createView(element('view'), controls.split);
const stats = createStats(statCells());
const host = createPipelineHost();

let gpu = null; // the GPU the browser offers, in words
const tuning = tuningFrom(new URLSearchParams(location.search));
let models = [];
let backends = []; // those this browser can use
let degrade = null;
let asked = null; // the bitrate the current run asked the encoder for, kbit/s
let framesShown = 0;
let source = null; // the running session's source, asked each frame for anything unusual about it

function fill(select, options) {
  select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
}

/** What the controls ask for, as plain data. Sizes are the original's; the input is half. */
function readSettings() {
  const [kind, clipId] = controls.source.value.split(':');
  const clip = CLIPS.find((candidate) => candidate.id === clipId);
  if (clip) controls.size.value = `${CLIP_WIDTH}x${CLIP_HEIGHT}`; // clips play at their own size
  controls.size.disabled = Boolean(clip);
  const [width, height] = controls.size.value.split('x').map(Number);
  degrade.refresh(width / 2, height / 2);
  return {
    kind, clip, width, height,
    file: controls.file.files[0],
    model: models[controls.model.selectedIndex],
    backend: controls.backend.value,
    degrade: degrade.read(width / 2, height / 2),
  };
}

/** Acquire a session's resources into `session`, so the controller can release a partial one. */
async function open(settings, session) {
  const { kind, clip, width, height } = settings;
  source = null;
  session.source = await openSource(kind, width, height, { file: settings.file, url: clip && clipUrl(clip.id) });
  const { kernel } = COMPARISONS.find((comparison) => comparison.id === controls.comparison.value);
  session.pipeline = await host.createPipeline(settings.model, settings.backend, width, height, settings.degrade, tuning, kernel);
  session.size = [width, height];
  source = session.source;
  view.resize(width, height);
  return session;
}

const controller = createController({
  open,
  onFrame(result, timed) {
    const start = performance.now();
    view.show(result);
    framesShown += 1;
    status.beat(source?.note?.() ?? '');
    if (timed) stats.record({ ...result.timings, show: performance.now() - start }, result.info, asked);
    // Everything needed to say why a slow page is slow, in one line that can be read out or copied.
    if (timed) diagnostics.textContent = `${controls.backend.selectedOptions[0]?.text} · GPU: ${gpu ?? 'none offered'} · shaders: ${result.info.variant ?? 'not the page\'s'} · ms per frame: ${stats.summary()}`;
  },
  onState(state, detail) {
    // The transport follows the run: Pause while it moves, Resume and Step while it is held.
    const live = state === 'running' || state === 'paused';
    controls.pause.disabled = !live;
    controls.pause.textContent = state === 'paused' ? 'Resume' : 'Pause';
    controls.step.disabled = state !== 'paused';
    if (state === 'starting') status.set('starting', 'Loading the model and compiling its shaders…');
    if (live) {
      const [width, height] = detail.size;
      const held = state === 'paused' ? ' · paused' : '';
      const backend = controls.backend.selectedOptions[0]?.text ?? '';
      status.set(state, `${width} × ${height} original → ${width / 2} × ${height / 2} input → ${width} × ${height} output · ${backend}${held}`);
    }
    if (state === 'idle') status.set('idle', '');
    if (state === 'failed') status.set('failed', String(detail));
  },
});

function start(frameLimit) {
  const settings = readSettings();
  credit.textContent = settings.clip ? settings.clip.credit : '';
  asked = settings.degrade.askedKbps;
  stats.reset();
  framesShown = 0;
  return controller.start(settings, frameLimit);
}

/** Show the controls that belong to the chosen view, and tell the view. */
function applyView() {
  const mode = controls.view.find((radio) => radio.checked).value;
  controls.split.hidden = mode !== 'split';
  controls.zoom.closest('label').hidden = mode !== 'magnify';
  view.set({ mode, zoom: Number(controls.zoom.value) });
}

function outcome() {
  const failed = status.failed();
  return {
    ok: !failed, error: failed ? status.text() : undefined,
    backend: controls.backend.value, frames: framesShown, asked, ...stats.snapshot(),
  };
}

/** Offer the backends that can run the chosen model, keeping the current one if it is among them. */
function offerBackends() {
  const offered = backendsFor(models[controls.model.selectedIndex], backends);
  const current = controls.backend.value;
  fill(controls.backend, offered.map((backend) => [backend.id, backend.label]));
  if (offered.some((backend) => backend.id === current)) controls.backend.value = current;
}

/**
 * List the models and start on the first one this browser can run. The list
 * leads with the 8-bit models, which need WebGPU; without it the page says so
 * and starts on a float model instead.
 */
function offerModels() {
  fill(controls.model, models.map((model, index) => [index, describe(model)]));
  const runs = (model) => backendsFor(model, backends).length > 0;
  const runnable = models.findIndex(runs);
  const preferred = models.findIndex((model) => model.opensFirst && runs(model));
  controls.model.selectedIndex = Math.max(preferred >= 0 ? preferred : runnable, 0);
  const fallback = preferred < 0 && !runs(models.find((model) => model.opensFirst) ?? models[0]);
  element('modelNote').textContent = fallback
    ? 'This browser has no usable WebGPU, which the 8-bit models need, so the page is running a float model on a much slower backend. Use a current Chrome, Edge or Safari with WebGPU switched on.'
    : gpu?.endsWith(SOFTWARE_ADAPTER) ? 'This browser is running WebGPU in software, not on a GPU, so every model will be slow. Check that hardware acceleration is switched on in the browser\'s settings.' : '';
  offerBackends();
  // Registered before the listeners that restart a run, so a run never starts on a stale backend.
  controls.model.addEventListener('change', offerBackends);
}

async function main() {
  [models, backends, gpu] = await Promise.all([loadModels(), usableBackends(), describeGpu()]);
  offerModels();
  fill(controls.source, [
    ...CLIPS.map((clip) => [`clip:${clip.id}`, clip.title]),
    ['pattern', 'Test pattern'], ['camera', 'Camera'], ['file', 'Video file…'],
  ]);
  controls.source.value = `clip:${FIRST_CLIP}`;
  degrade = await setUpDegrade(controls, element('codecNote'));
  fill(controls.comparison, COMPARISONS.map((comparison) => [comparison.id, comparison.label]));
  // The comparison upscale is the page's own shader: without WebGPU there is only the upscaler's.
  controls.comparison.disabled = !backends.some((backend) => backend.needs === 'webgpu');
  for (const control of [...controls.view, controls.zoom]) control.addEventListener('change', applyView);
  applyView();
  const parameters = new URLSearchParams(location.search);
  if (parameters.has('auto')) {
    await runUnattended(parameters, controls, start, outcome, element('report'));
    return;
  }
  // A failed start already shows its error through onState.
  const restart = () => start().catch(() => {});
  const restarting = ['size', 'model', 'backend', 'file', 'preset', 'blur', 'noise', 'denoise', 'kernel', 'codec', 'rate', 'keyint', 'comparison'];
  for (const name of restarting) controls[name].addEventListener('change', restart);
  controls.pause.addEventListener('click', () => {
    if (controls.step.disabled) controller.pause();
    else controller.resume();
  });
  controls.step.addEventListener('click', () => controller.step());
  controls.source.addEventListener('change', () => {
    // A file has to be chosen first; choosing it then triggers the restart.
    if (controls.source.value === 'file') controls.file.click();
    else restart();
  });
  restart();
}

main().catch((error) => status.set('failed', String(error)));
