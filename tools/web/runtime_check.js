// Run in the demo's site by tools/web/runtime_check.py: the upscaler that
// runs entirely on the GPU against the integer reference compiled to
// WebAssembly, frame after frame of a real clip, comparing the network's
// input, its outputs (the state among them), the conventional upscale and the
// output picture word for word. Part-way through, the stream is restarted and
// the GPU device destroyed, and both must be recovered from exactly.
async ({ directory, clip, frames, width, height, tiles, restartAt, loseDeviceAt, quantiser }) => {
  const { runtimeReady } = await import('/js/wasm.js');
  await runtimeReady();
  const { IntegerReference, MotionSearch } = await import('/pkg/cvnn_web.js');
  const { loadModels } = await import('/js/models.js');
  const { createGpuUpscaler } = await import('/js/gpu-upscaler.js');
  const { createCameraPool } = await import('/js/camera-pool.js');
  const { openSource } = await import('/js/source.js');
  const { clipUrl } = await import('/js/clips.js');

  // Every device the page is given goes on the list of whoever is asking, so that one can be taken away.
  let devices = [];
  const request = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function requestDevice(...args) {
    const device = await request.apply(this, args);
    devices.push(device);
    return device;
  };

  const model = (await loadModels()).find((candidate) => candidate.directory === directory);
  const words = new Int32Array(await (await fetch(model.url)).arrayBuffer());
  const source = await openSource('clip', 2 * width, 2 * height, { url: clipUrl(clip) });
  const camera = createCameraPool(2 * width, 2 * height, { sigma: 0, blur: 0, denoise: 0, kernel: 'bicubic' });
  const pixels = width * height;
  const names = ['input', 'state', 'conventional', 'output'];
  const variants = [];
  for (const tile of tiles) {
    const reference = new IntegerReference(words, width, height, model.temporal, model.align,
      model.state_channels, model.coding_planes, model.confidence_plane);
    devices = [];
    const upscaler = await createGpuUpscaler(model, width, height, { tile });
    variants.push({ tile, reference, upscaler, search: new MotionSearch(width, height), differing: Object.fromEntries(names.map((name) => [name, 0])), devices });
  }
  const vectors = new Int32Array(2 * Math.ceil(width / 8) * Math.ceil(height / 8));
  const expected = [pixels * (words[3] + words[4]), pixels * (2 + words[8] / 4), 4 * pixels, 4 * pixels].map((length) => new Uint32Array(length));
  let moved = 0;
  for (let frame = 0; frame < frames; frame += 1) {
    const sent = await camera.capture((await source.grab()).picture);
    sent.picture.close();
    const planes = new Uint8Array(sent.frame);
    const coding = frame % 2 ? { quantiser, key: frame % 8 === 1 } : null; // told on alternate frames
    for (const variant of variants) {
      devices = variant.devices;
      if (frame === restartAt) {
        variant.upscaler.restart();
        variant.search.restart();
        variant.reference.restart();
      }
      if (frame === loseDeviceAt) {
        variant.devices.at(-1).destroy();
        await new Promise((resolve) => { setTimeout(resolve, 50); });
        variant.reference.restart(); // the history on the GPU went with the device
        variant.search.restart();
        variant.upscaler.restart();
      }
      const reset = variant.search.push(planes.subarray(0, pixels), vectors);
      if (!model.align) vectors.fill(0);
      moved = Math.max(moved, ...vectors.map(Math.abs));
      await variant.upscaler.process(planes, coding);
      const got = await variant.upscaler.read(names);
      variant.reference.step(planes, vectors, reset, coding ? coding.quantiser : -1, Boolean(coding?.key), variant.tile);
      variant.reference.read(...expected);
      names.forEach((name, index) => {
        for (let word = 0; word < expected[index].length; word += 1) variant.differing[name] += got[index][word] !== expected[index][word];
      });
    }
    for (let wait = 0; wait < 2; wait += 1) await source.nextFrame();
  }
  const report = {
    directory, width, height, frames, restartAt, loseDeviceAt,
    devicesUsed: variants.map((variant) => variant.devices.length),
    largestVectorInSteps: moved,
    variants: variants.map(({ tile, differing }) => ({ tile, differingWords: differing })),
  };
  for (const variant of variants) await variant.upscaler.release();
  source.stop();
  camera.release();
  GPUAdapter.prototype.requestDevice = request;
  return report;
}
