// Run in the demo's site by tools/web/int8_check.py: the 8-bit WebGPU network
// against the WebAssembly reference on real frames, with the recurrent state
// fed back through the session frame after frame. The exact forms of the
// shaders are compared word for word; the approximate one is measured, at the
// output and layer by layer.
async ({ directory, clip, frames, width, height, tiles, differences }) => {
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
  const { createDifference, readWords } = await import(URL.createObjectURL(new Blob([differences], { type: 'text/javascript' })));

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

  // The ways the shaders can be written, beyond the tiling: each form, and between them every
  // option in each of its settings (a large model's shaders take long to compile, so not every
  // combination). The approximate form is checked as the page runs it, and one other way.
  const HALF = { form: 'half', span: 4, split: 1, edgeTests: true };
  const ways = [
    {}, { packedDot: false, edgeTests: true }, { span: 4, split: 2, uniformWeights: false, wordLoop: true },
    { form: 'float', span: 4, wordLoop: true }, { form: 'float', span: 2, split: 2, edgeTests: true, uniformWeights: false },
    ...(device.features.has('shader-f16') ? [HALF, { ...HALF, split: 2, edgeTests: false, uniformWeights: false, wordLoop: true }] : []),
  ];
  const variants = [];
  for (const tile of tiles) {
    for (const [index, way] of ways.entries()) {
      const network = createInt8Network(device, model, width, height, { tile, ...way });
      // The default way with every grouping of tiles the page would try; the others with one.
      const groups = index === 0 ? groupCandidates(network.tiles) : [network.tiles];
      for (const group of groups) variants.push({ tile, group, way, network, difference: createDifference('output', model.outputWords) });
    }
  }
  // Layer by layer on the whole frame: the approximate form's layers each given the exact
  // form's activations (`alone`), and run on from its own (`through`).
  const layered = device.features.has('shader-f16') ? {
    exact: createInt8Network(device, model, width, height, { tile: 0 }),
    alone: createInt8Network(device, model, width, height, { tile: 0, ...HALF }),
    through: createInt8Network(device, model, width, height, { tile: 0, ...HALF }),
    layers: model.layers.map((_, layer) => {
      const layout = layer === model.layers.length - 1 ? 'output' : 'signed';
      return { alone: createDifference(layout, model.outputWords), through: createDifference(layout, model.outputWords) };
    }),
  } : null;

  async function compareLayers() {
    const { exact, alone, through, layers } = layered;
    for (const network of [exact, alone, through]) device.queue.writeBuffer(network.input, 0, packed);
    for (const [layer, differences] of layers.entries()) {
      const encoder = device.createCommandEncoder();
      exact.scratch.forEach((buffer, index) => encoder.copyBufferToBuffer(buffer, 0, alone.scratch[index], 0, buffer.size));
      const pass = encoder.beginComputePass();
      for (const network of [exact, alone, through]) network.encode(pass, network.tiles, [layer, layer + 1]);
      pass.end();
      device.queue.submit([encoder.finish()]);
      const [expected, gotAlone, gotThrough] = await Promise.all([exact, alone, through].map((network) => readWords(device, network.target(layer))));
      differences.alone.add(gotAlone, expected);
      differences.through.add(gotThrough, expected);
    }
  }

  let distinct = 0;
  for (let frame = 0; frame < frames; frame += 1) {
    const sent = await camera.capture((await source.grab()).picture);
    sent.picture.close();
    upscaler.prepare_packed(sent.frame, codec, packed);
    for (const tile of tiles) {
      codec.infer(packed, width, height, tile, reference);
      for (const variant of variants) {
        if (variant.tile === tile) variant.difference.add(await variant.network.run(packed, variant.group), reference);
      }
    }
    if (layered) await compareLayers();
    distinct = Math.max(distinct, new Set(reference.subarray(0, 4096)).size);
    // The deployed tiling's reference output carries the stream on.
    codec.infer(packed, width, height, model.tile, reference);
    upscaler.finish_packed(codec, reference, picture);
    for (let wait = 0; wait < 3; wait += 1) await source.nextFrame();
  }
  source.stop();
  camera.release();
  return {
    directory, width, height, frames, errors,
    packedDotAvailable: navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product') ?? false,
    halfAvailable: device.features.has('shader-f16'),
    distinctOutputWords: distinct,
    variants: variants.map(({ tile, group, way, difference }) => ({ tile, group, form: 'integer', ...way, ...difference.report() })),
    layers: layered ? layered.layers.map(({ alone, through }, layer) => ({ layer, alone: alone.report(), through: through.report() })) : [],
  };
}
