// The ONNX Runtime Web build every backend shares: one import, so its version
// and the location of its WebAssembly files are chosen in one place.

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

export { ort };
