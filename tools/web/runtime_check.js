// Run in the demo's site by tools/web/runtime_check.py: the upscaler that
// runs entirely on the GPU against the integer reference compiled to
// WebAssembly, frame after frame of a real clip, comparing the network's
// input, its outputs (the state among them), the conventional upscale and the
// output picture: word for word for an exact form of the shaders, and by how
// much for the approximate one, whose differences the history carries from
// frame to frame. Part-way through, the stream is restarted and the GPU
// device destroyed, and both must be recovered from. The output picture is
// shown with grain of a profile that changes from frame to frame (`grains`,
// taken in turn): the pictures must agree with it on, and the network's
// input on the frames after shows that none of it reached the history.
async ({ directory, clip, frames, width, height, tiles, restartAt, loseDeviceAt, grains, differences }) => {
  const { runtimeReady } = await import('/js/wasm.js');
  await runtimeReady();
  const { IntegerReference, MotionSearch } = await import('/pkg/cvnn_web.js');
  const { loadModels } = await import('/js/models.js');
  const { createGpuUpscaler } = await import('/js/gpu-upscaler.js');
  const { requestDevice } = await import('/js/gpu-device.js');
  const { createCameraPool } = await import('/js/camera-pool.js');
  const { openSource } = await import('/js/source.js');
  const { clipUrl } = await import('/js/clips.js');
  const { createDifference } = await import(URL.createObjectURL(new Blob([differences], { type: 'text/javascript' })));

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
  const outputWords = 2 + words[8] / 4;
  // What is read back after each frame, and the layout of each (differences.js).
  const layouts = { input: 'signed', state: 'output', conventional: 'picture', output: 'picture' };
  const names = Object.keys(layouts);
  // Each tiling in the exact form and, where the GPU has 16-bit float, in the approximate one.
  const probe = await requestDevice();
  const forms = ['integer', ...(probe.features.has('shader-f16') ? ['half'] : [])];
  probe.destroy();
  // One reference stream for each tiling, which every form of that tiling is compared with.
  const streams = [];
  for (const tile of tiles) {
    const reference = new IntegerReference(words, width, height, model.temporal, model.align,
      model.state_channels, model.confidence_plane);
    const variants = [];
    for (const form of forms) {
      devices = [];
      const upscaler = await createGpuUpscaler(model, width, height, { tile, form });
      const differing = Object.fromEntries(names.map((name) => [name, createDifference(layouts[name], outputWords)]));
      variants.push({ tile, form, upscaler, differing, devices });
    }
    streams.push({ tile, reference, search: new MotionSearch(width, height), variants });
  }
  const variants = streams.flatMap((stream) => stream.variants);
  const vectors = new Int32Array(2 * Math.ceil(width / 8) * Math.ceil(height / 8));
  const expected = [pixels * (words[3] + words[4]), pixels * outputWords, 4 * pixels, 4 * pixels].map((length) => new Uint32Array(length));
  let moved = 0;
  for (let frame = 0; frame < frames; frame += 1) {
    const sent = await camera.capture((await source.grab()).picture);
    sent.picture.close();
    const planes = new Uint8Array(sent.frame);
    const grain = grains[frame % grains.length];
    for (const stream of streams) {
      // A lost device takes the history on the GPU with it, so the stream starts again as after a restart.
      if (frame === restartAt || frame === loseDeviceAt) {
        stream.reference.restart();
        stream.search.restart();
      }
      const reset = stream.search.push(planes.subarray(0, pixels), vectors);
      if (!model.align) vectors.fill(0);
      moved = Math.max(moved, ...vectors.map(Math.abs));
      stream.reference.set_grain(grain);
      stream.reference.step(planes, vectors, reset, stream.tile);
      stream.reference.read(...expected);
      for (const variant of stream.variants) {
        devices = variant.devices;
        if (frame === loseDeviceAt) {
          variant.devices.at(-1).destroy();
          await new Promise((resolve) => { setTimeout(resolve, 50); });
        }
        if (frame === restartAt || frame === loseDeviceAt) variant.upscaler.restart();
        variant.upscaler.setGrain(grain);
        await variant.upscaler.process(planes);
        const got = await variant.upscaler.read(names);
        names.forEach((name, index) => variant.differing[name].add(got[index], expected[index]));
      }
    }
    for (let wait = 0; wait < 2; wait += 1) await source.nextFrame();
  }
  const report = {
    directory, width, height, frames, restartAt, loseDeviceAt,
    devicesUsed: variants.map((variant) => variant.devices.length),
    largestVectorInSteps: moved,
    variants: variants.map(({ tile, form, differing }) => (
      { tile, form, differences: Object.fromEntries(names.map((name) => [name, differing[name].report()])) })),
  };
  for (const variant of variants) await variant.upscaler.release();
  source.stop();
  camera.release();
  GPUAdapter.prototype.requestDevice = request;
  return report;
}
