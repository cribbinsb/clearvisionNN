// Unattended runs for tools/web/smoke.py: `?auto=<backend or default>&frames=<n>` plus
// any of preset, source, size, model, view, zoom, blur, noise, denoise, kernel, codec, rate and keyint sets
// the controls, runs that many frames and writes a JSON report into #report.

const DEFAULT_BACKEND = 'default'; // `auto=default` keeps the backend the page chose for the model
// The ready-made input first: choosing it sets the controls after it, which may then be set one by one.
const SELECTS = ['preset', 'source', 'size', 'codec', 'rate', 'keyint', 'zoom', 'blur', 'noise', 'denoise', 'kernel'];

/**
 * Apply `parameters` to `controls`, run `start(frames)` and report through
 * `report`. `outcome()` gives the facts to report once the run has ended.
 */
export async function runUnattended(parameters, controls, start, outcome, report) {
  try {
    for (const name of SELECTS) {
      if (!parameters.has(name)) continue;
      controls[name].value = parameters.get(name);
      // A select keeps its old value when given one it does not offer.
      const refused = controls[name] instanceof HTMLSelectElement && controls[name].value !== parameters.get(name);
      if (refused) throw new Error(`${name}=${parameters.get(name)} is not on offer here.`);
      if (name === 'preset') controls.preset.dispatchEvent(new Event('change'));
    }
    // The model first: choosing it decides which backends are on offer.
    if (parameters.has('model')) {
      controls.model.selectedIndex = Number(parameters.get('model'));
      controls.model.dispatchEvent(new Event('change'));
    }
    const backend = parameters.get('auto');
    if (backend !== DEFAULT_BACKEND) {
      controls.backend.value = backend;
      if (controls.backend.value !== backend) throw new Error(`The ${backend} backend cannot run this model here.`);
    }
    if (parameters.has('view')) controls.view.find((radio) => radio.value === parameters.get('view')).click();
    controls.zoom.dispatchEvent(new Event('change'));
    await start(Number(parameters.get('frames') ?? 20));
    report.textContent = JSON.stringify(outcome());
  } catch (error) {
    report.textContent = JSON.stringify({ ok: false, error: String(error) });
  }
  report.dataset.done = 'true';
}
