// Run in the demo's site by tools/web/shader_bench.py: the GPU's own time for
// one frame of the 8-bit network, for each of several ways of writing its
// shaders, the ways taking turns so that whatever else the GPU is doing falls
// on all alike.
async ({ directory, width, height, variants, burst, rounds }) => {
  const { requestDevice } = await import('/js/gpu-device.js');
  const { parseIntegerModel } = await import('/js/int8-model.js');
  const { createInt8Network } = await import('/js/int8-network.js');
  const { createPassTimer } = await import('/js/gpu-timer.js');
  const base = new URL(`/models/${directory}/`, location.origin);
  const contract = await (await fetch(new URL('model.json', base))).json();
  const model = parseIntegerModel(new Int32Array(await (await fetch(new URL(contract.integer.file, base))).arrayBuffer()));
  const device = await requestDevice();
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message.slice(0, 600)));
  const timer = createPassTimer(device);
  if (!timer) return { errors: ['This GPU device has no timestamp queries.'], results: [] };
  // What the input holds does not change the work: no shader branches on a sample.
  const input = new Uint32Array(width * height * model.inputWords).map(() => Math.random() * 2 ** 32);
  const made = variants.map((options) => {
    const network = createInt8Network(device, model, width, height, options);
    device.queue.writeBuffer(network.input, 0, input);
    return { options, network, times: [] };
  });
  /** Milliseconds a frame of `network`: `burst` frames in one pass, so that coarsened timestamps do not decide. */
  async function time(network) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: timer.writes() });
    for (let frame = 0; frame < burst; frame += 1) network.encode(pass);
    pass.end();
    timer.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    return (await timer.read()) / burst;
  }
  // One round more than is kept: the first compiles the shaders and warms the GPU up.
  for (let round = 0; round <= rounds; round += 1) {
    // Forwards and backwards by turns, so that no way always follows the same neighbour.
    for (const entry of round % 2 ? [...made].reverse() : made) {
      const ms = await time(entry.network);
      if (round > 0) entry.times.push(ms);
    }
  }
  const sorted = (values) => [...values].sort((a, b) => a - b);
  const at = (values, share) => Number(sorted(values)[Math.floor(share * (values.length - 1))].toFixed(3));
  return {
    errors,
    gpu: device.adapterInfo?.description ?? '',
    results: made.map(({ options, times }) => ({ options, min: at(times, 0), median: at(times, 0.5), p90: at(times, 0.9) })),
  };
}
