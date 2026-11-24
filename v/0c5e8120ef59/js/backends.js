// What can upscale: for a model in 8-bit integer form, the page's own WebGPU
// shaders, either for the whole frame or for the network alone; for a float
// graph, ONNX Runtime Web's execution providers. The page offers those this
// browser can use.

import { hasWebGpu } from './gpu-device.js';
import { createGpuUpscaler } from './gpu-upscaler.js';
import { createIntegerNetwork } from './int8-backend.js';
import { createOnnxNetwork } from './onnx-network.js';

const webnn = (deviceType) => ({ name: 'webnn', deviceType, powerPreference: 'high-performance' });
const onnx = (provider) => (model, width, height, unused) => createOnnxNetwork(provider, model, width, height, unused);

// `runs` is the kind of network file a backend takes (a model's `network`);
// `needs` is the browser interface it cannot do without, if any.
// A backend is either a whole upscaler, `upscaler(model, width, height)`, or a
// network for the WebAssembly session to work around,
// `network(model, width, height, unusedChannels)` resolving to `run(input)`
// and `release()` (tensors.js gives the forms). Best first within each kind.
export const BACKENDS = [
  { id: 'webgpu-int8', label: 'WebGPU 8-bit', runs: 'integer', needs: 'webgpu', upscaler: createGpuUpscaler },
  { id: 'webgpu-int8-network', label: 'WebGPU 8-bit network, the rest in WebAssembly', runs: 'integer', needs: 'webgpu', network: createIntegerNetwork },
  { id: 'webgpu', label: 'WebGPU', runs: 'onnx', needs: 'webgpu', network: onnx('webgpu') },
  { id: 'webnn-gpu', label: 'WebNN (GPU)', runs: 'onnx', needs: 'webnn', network: onnx(webnn('gpu')) },
  { id: 'webnn-npu', label: 'WebNN (NPU)', runs: 'onnx', needs: 'webnn', network: onnx(webnn('npu')) },
  { id: 'webnn-cpu', label: 'WebNN (CPU)', runs: 'onnx', needs: 'webnn', network: onnx(webnn('cpu')) },
  { id: 'wasm', label: 'WebAssembly (CPU, slow)', runs: 'onnx', needs: null, network: onnx('wasm') },
];

/**
 * Resolves to the backends this browser can really use, best first: asked
 * once, when the page loads, because finding out about WebGPU takes a moment.
 */
export async function usableBackends() {
  const has = { webgpu: await hasWebGpu(), webnn: 'ml' in navigator };
  return BACKENDS.filter((backend) => backend.needs === null || has[backend.needs]);
}

/** Those of `backends` that can run `model`. */
export function backendsFor(model, backends) {
  return backends.filter((backend) => backend.runs === model.network);
}

/** The backend called `id`. */
export function backendOf(id) {
  return BACKENDS.find((backend) => backend.id === id);
}
