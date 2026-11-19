// The models the page can run: `models/manifest.json` lists directories, each
// holding a contract `model.json` and the network file it names: `model.onnx`
// (float arithmetic) or `model.i8` (8-bit integer), both written by the exporters.

const ROOT = new URL('../models/', import.meta.url);
const ONNX_FILE = 'model.onnx';

/**
 * Every listed model's contract, with `network` ('onnx' or 'integer': the
 * kind of its network file, which decides what can run it) and that file's URL.
 */
export async function loadModels() {
  const manifest = await (await fetch(new URL('manifest.json', ROOT))).json();
  return Promise.all(manifest.models.map(async (directory) => {
    const base = new URL(`${directory}/`, ROOT);
    const contract = await (await fetch(new URL('model.json', base))).json();
    // Contracts written before the integer form existed describe a graph without saying so.
    const network = contract.network ?? 'onnx';
    const file = network === 'integer' ? contract.integer.file : ONNX_FILE;
    return { ...contract, network, directory, url: new URL(file, base).href };
  }));
}

/** A label for the model list: arithmetic, size and design first, then which export it is. */
export function describe(model) {
  const arithmetic = model.network === 'integer' ? '8-bit' : model.precision === 'fp32' ? 'float' : `float (${model.precision})`;
  const kind = model.temporal === 'none' ? 'per-frame' : model.temporal;
  const thousands = Math.round(model.parameters / 1e3);
  return `${arithmetic} · ${model.size.toUpperCase()} · ${kind} — ${model.directory}, ${thousands}k`;
}
