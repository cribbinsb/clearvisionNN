// The 8-bit backend: the network in the page's own WebGPU shaders, for models
// exported in integer form. It takes and returns packed 8-bit planes, which
// the session packs and reads itself (tensors.js).

import { requestDevice } from './gpu-device.js';
import { parseIntegerModel } from './int8-model.js';
import { createInt8Network } from './int8-network.js';
import { createSchedule, groupCandidates } from './int8-schedule.js';

/**
 * A network (see backends.js) running `model`, an integer export, for `width`
 * x `height` input: `run(packed)` resolves to the packed outputs.
 */
export async function createIntegerNetwork(model, width, height) {
  const words = new Int32Array(await (await fetch(model.url)).arrayBuffer());
  const device = await requestDevice();
  let network;
  try {
    network = createInt8Network(device, parseIntegerModel(words), width, height);
  } catch (error) {
    device.destroy();
    throw error;
  }
  const schedule = createSchedule(groupCandidates(network.tiles));
  return {
    async run(packed) {
      const start = performance.now();
      const outputs = await network.run(packed, schedule.current());
      schedule.record(performance.now() - start);
      return outputs;
    },
    async release() {
      network.destroy();
      device.destroy();
    },
  };
}
