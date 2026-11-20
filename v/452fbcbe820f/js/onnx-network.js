// The network as an ONNX graph run by ONNX Runtime Web on one of its
// execution providers: the float path every exported model has.

import { ort } from './ort.js';
import { createGpuRunner } from './webgpu-io.js';

/** `run(features)` with the tensors in the page's memory, for providers without a cheaper route. */
function createCpuRunner(session, shape) {
  return {
    async run(features) {
      const outputs = await session.run({ features: new ort.Tensor('float32', features, shape) });
      const entries = Object.entries(outputs).map(async ([name, tensor]) => [name, await tensor.getData()]);
      return Object.fromEntries(await Promise.all(entries));
    },
    destroy() {},
  };
}

/**
 * A network session on `provider` for `width` x `height` input. The size is
 * fixed at creation: WebNN needs static shapes, and the others plan better
 * with them. `unusedChannels` is the `[first, one past the last]` input
 * channels that are always zero.
 */
export async function createOnnxNetwork(provider, model, width, height, unusedChannels) {
  const onGpu = provider === 'webgpu';
  const session = await ort.InferenceSession.create(model.url, {
    executionProviders: [provider],
    freeDimensionOverrides: { height, width },
    graphOptimizationLevel: 'all',
    ...(onGpu ? { preferredOutputLocation: 'gpu-buffer' } : {}),
  });
  const shape = [1, model.input_channels, height, width];
  let runner;
  try {
    runner = onGpu ? await createGpuRunner(session, model, shape, unusedChannels) : createCpuRunner(session, shape);
  } catch (error) {
    await session.release();
    throw error;
  }
  return {
    async run(features) {
      const outputs = await runner.run(features);
      return { fine: outputs.fine, coarse: outputs.coarse ?? new Float32Array(0) };
    },
    async release() {
      runner.destroy();
      await session.release();
    },
  };
}
