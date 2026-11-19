// Run in the demo page by tools/web/int8_check.py: the 8-bit WebGPU network
// against the WebAssembly reference on real frames, with the recurrent state
// fed back through the session frame after frame, and timings of each variant.
async ({ directory, clip, frames, width, height, tiles, timedFrames }) => {
  const { runtimeReady } = await import('/js/wasm.js');
  await runtimeReady();
  const { Upscaler, IntegerNetwork } = await import('/pkg/cvnn_web.js');
  const { requestDevice } = await import('/js/gpu-device.js');
  const { parseIntegerModel } = await import('/js/int8-model.js');
  const { createInt8Network } = await import('/js/int8-network.js');
  const { groupCandidates } = await import('/js/int8-schedule.js');
  const { createCameraPool } = await import('/js/camera-pool.js');
  const { openSource } = await import('/js/source.js');
  const { clipUrl } = await import('/js/clips.js');

  const base = new URL(`/models/${directory}/`, location.origin);
  const contract = await (await fetch(new URL('model.json', base))).json();
  const words = new Int32Array(await (await fetch(new URL(contract.integer.file, base))).arrayBuffer());
  const codec = new IntegerNetwork(words);
  const model = parseIntegerModel(words);
  const device = await requestDevice();
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message.slice(0, 300)));

  const upscaler = new Upscaler(width, height, contract.planes, contract.temporal, contract.align,
    contract.state_channels, contract.side_planes, contract.luma_only, contract.coding_planes,
    contract.confidence_plane);
  const source = await openSource('clip', 2 * width, 2 * height, { url: clipUrl(clip) });
  const camera = createCameraPool(2 * width, 2 * height, { sigma: 0, blur: 0, denoise: 0, kernel: 'bicubic' });
  const pixels = width * height;
  const packed = new Uint32Array(pixels * codec.input_words());
  const reference = new Uint32Array(pixels * codec.output_words());
  const picture = new Uint8ClampedArray(16 * pixels);

  // Every tiling asked for, each with every grouping the page would try, plus the spelt-out arithmetic.
  const variants = [];
  for (const tile of tiles) {
    const network = createInt8Network(device, model, width, height, { tile });
    for (const group of groupCandidates(network.tiles)) variants.push({ tile, group, network, packedDot: network.packedDot });
    const speltOut = createInt8Network(device, model, width, height, { tile, packedDot: false });
    variants.push({ tile, group: speltOut.tiles, network: speltOut, packedDot: false });
    // The float form of the products (what an Apple GPU runs) must give the same words, with the
    // weights bound either way.
    for (const uniformWeights of [true, false]) {
      const float = createInt8Network(device, model, width, height, { tile, floatProducts: true, uniformWeights });
      variants.push({ tile, group: float.tiles, network: float, packedDot: float.packedDot, float: uniformWeights ? 'uniform' : 'storage' });
    }
  }
  const results = variants.map(({ tile, group, packedDot, float }) => ({ tile, group, packedDot, float, differingWords: 0, times: [] }));
  let distinct = 0;
  for (let frame = 0; frame < frames; frame += 1) {
    const sent = await camera.capture((await source.grab()).picture);
    sent.picture.close();
    upscaler.prepare_packed(sent.frame, codec, packed);
    for (const tile of tiles) {
      codec.infer(packed, width, height, tile, reference);
      for (const [index, variant] of variants.entries()) {
        if (variant.tile !== tile) continue;
        const output = await variant.network.run(packed, variant.group);
        for (let word = 0; word < output.length; word += 1) results[index].differingWords += output[word] !== reference[word];
      }
    }
    distinct = Math.max(distinct, new Set(reference.subarray(0, 4096)).size);
    // The deployed tiling's reference output carries the stream on.
    codec.infer(packed, width, height, model.tile, reference);
    upscaler.finish_packed(codec, reference, picture);
    for (let wait = 0; wait < 3; wait += 1) await source.nextFrame();
  }
  // Timings on the last frame's input, the variants taking turns so that load drifts over all alike.
  for (let round = 0; round < timedFrames; round += 1) {
    for (const [index, variant] of variants.entries()) {
      const start = performance.now();
      await variant.network.run(packed, variant.group);
      results[index].times.push(performance.now() - start);
    }
  }
  source.stop();
  camera.release();
  const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.floor(q * (values.length - 1))];
  return {
    directory, width, height, frames, errors,
    packedDotAvailable: navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product') ?? false,
    distinctOutputWords: distinct,
    variants: results.map(({ times, ...rest }) => ({
      ...rest, ms: [0.1, 0.5, 0.9].map((q) => Number(quantile(times, q).toFixed(2))),
    })),
  };
}
