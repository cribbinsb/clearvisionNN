// Tensor traffic for the WebGPU backend. Moving bytes between the page and the
// GPU costs several times the network's arithmetic, so the input lives in one
// GPU buffer that receives only the channels that change, and the outputs are
// read back through staging buffers kept between frames.

import { ort } from './ort.js';

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;

/** Whether `model` can run on `device`: half-precision graphs need 16-bit floats in shaders. */
function checkPrecision(model, device) {
  if (model.precision === 'fp16' && !device.features.has('shader-f16')) {
    // Without the feature every kernel fails validation and the outputs are silently zero.
    throw new Error(`${model.directory}: this GPU or its driver has no 16-bit float shaders `
      + '(WebGPU "shader-f16"); choose the full-precision model.');
  }
}

/** A staging buffer and the array its contents are copied to, both reused every frame. */
function createReadback(device, values) {
  const size = values * FLOAT_BYTES;
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const data = new Float32Array(values);
  return {
    copyFrom: (encoder, source) => encoder.copyBufferToBuffer(source, 0, staging, 0, size),
    async read() {
      await staging.mapAsync(GPUMapMode.READ);
      data.set(new Float32Array(staging.getMappedRange()));
      staging.unmap();
      return data;
    },
    destroy: () => staging.destroy(),
  };
}

/**
 * `run(features)` for a WebGPU `session` created with its outputs left on the
 * GPU. `shape` is the input's [1, channels, height, width]; channels
 * `unused[0]` up to `unused[1]` are never sent: the buffer starts zeroed, as they are.
 */
export async function createGpuRunner(session, model, shape, unused) {
  const device = await ort.env.webgpu.device;
  checkPrecision(model, device);
  const plane = shape[2] * shape[3];
  const values = shape[1] * plane;
  const buffer = device.createBuffer({
    size: values * FLOAT_BYTES,
    // COPY_SRC because the runtime may copy from a buffer it is given.
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const input = ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: shape });
  const sent = [[0, unused[0] * plane], [unused[1] * plane, values]].filter(([from, to]) => to > from);
  const readbacks = new Map(); // output name -> readback, made when its size is first seen

  function readbackFor(name, tensor) {
    if (!readbacks.has(name)) {
      readbacks.set(name, createReadback(device, tensor.dims.reduce((a, b) => a * b, 1)));
    }
    return readbacks.get(name);
  }

  return {
    /** Run one frame; the returned arrays are overwritten by the next call. */
    async run(features) {
      for (const [from, to] of sent) {
        device.queue.writeBuffer(buffer, from * FLOAT_BYTES, features, from, to - from);
      }
      const outputs = await session.run({ features: input });
      try {
        const encoder = device.createCommandEncoder();
        const pending = Object.entries(outputs).map(([name, tensor]) => {
          const readback = readbackFor(name, tensor);
          readback.copyFrom(encoder, tensor.gpuBuffer);
          return [name, readback];
        });
        device.queue.submit([encoder.finish()]);
        const arrays = await Promise.all(pending.map(([, readback]) => readback.read()));
        return Object.fromEntries(pending.map(([name], index) => [name, arrays[index]]));
      } finally {
        Object.values(outputs).forEach((tensor) => tensor.dispose());
      }
    },
    destroy() {
      buffer.destroy();
      readbacks.forEach((readback) => readback.destroy());
    },
  };
}
