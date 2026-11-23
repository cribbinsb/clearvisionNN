// A whole frame on the GPU: the 4:2:0 planes and the block motion field go
// in, the conventional upscale and the output come out as pictures, and the
// recurrent state and the previous output never leave. Nothing is read back
// unless a caller asks (`read`, for checks).

import { createPassTimer } from './gpu-timer.js';
import { createInt8Network } from './int8-network.js';
import {
  BLOCK_ERRORS, CONVENTIONAL, FLOW, GROUP_SIDE, PRESENT, WIDE_CHROMA, assembleShader, finishShader,
} from './runtime-shaders.js';

const BLOCK = 8; // side of a motion block, in input samples
const WORD = 4; // bytes of a word
const FRAME_BYTES = 24; // the Frame uniform: size, valid, grain amplitude, frame index; padded to its alignment
const PICTURES = ['conventional', 'output'];
// The steps of a frame in the order they run; `flow` and `errors` only where there is history to warp.
const ORDER = ['wideChroma', 'conventional', 'flow', 'errors', 'assemble', 'network', 'finish'];

/**
 * `model` (int8-model.js), whose contract is `contract`, on `device` for
 * `width` x `height` input. `options` are int8-network.js's, and `stepTimes`:
 * each step is then a pass of its own, timed apart, to find which one a GPU
 * is slow at. `canvases` are two canvases of twice that size, drawn to by
 * `present`.
 */
export function createGpuRuntime(device, model, contract, width, height, canvases, { stepTimes = false, ...options } = {}) {
  const network = createInt8Network(device, model, width, height, options);
  const recurrent = contract.temporal === 'recurrent';
  const confident = recurrent && contract.confidence_plane;
  const pixels = width * height;
  const grid = [Math.ceil(width / BLOCK), Math.ceil(height / BLOCK)];
  const owned = [];
  const buffer = (bytes, usage) => {
    const made = device.createBuffer({ size: Math.ceil(bytes / WORD) * WORD, usage });
    owned.push(made);
    return made;
  };
  const storage = (words) => buffer(words * WORD, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const frame = buffer(FRAME_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  // Luma alternates between two buffers, so the previous frame's stays for the confidence plane.
  const luma = [storage(pixels / WORD), storage(pixels / WORD)];
  const buffers = {
    chroma: storage(pixels / 2 / WORD),
    wide: storage(pixels),
    base: storage(4 * pixels),
    colour: storage(4 * pixels),
    vectors: storage(2 * grid[0] * grid[1]),
    flow: storage(2 * pixels),
    errors: storage(grid[0] * grid[1]),
    previous: storage(4 * pixels),
    conventional: storage(4 * pixels),
    output: storage(4 * pixels),
    input: network.input,
    state: network.output,
  };
  const compute = (code) => device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
  });
  /** A pass over `size` samples: its pipeline bound to `resources` (binding 0 is always the frame). */
  const step = (code, size, resources) => {
    const pipeline = compute(code);
    const bind = (parity) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [frame, ...resources].map((resource, binding) => ({ binding, resource }))
        .filter(({ resource }) => resource !== null)
        .map(({ binding, resource }) => ({ binding, resource: { buffer: Array.isArray(resource) ? resource[parity] : resource } })),
    });
    const groups = [bind(0), bind(1)];
    return (pass, parity) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, groups[parity]);
      pass.dispatchWorkgroups(Math.ceil(size[0] / GROUP_SIDE), Math.ceil(size[1] / GROUP_SIDE));
    };
  };
  const input = [width, height];
  const fine = [2 * width, 2 * height];
  const now = luma; // [parity] is this frame's luma
  const before = [luma[1], luma[0]];
  const b = buffers;
  // A binding the model's shader does not use is left out (null): its layout has no place for it.
  const history = (resource) => (recurrent ? resource : null);
  const steps = {
    wideChroma: step(WIDE_CHROMA, input, [b.chroma, b.wide]),
    conventional: step(CONVENTIONAL, fine, [now, b.wide, b.base, b.colour, b.conventional]),
    flow: step(FLOW, input, [b.vectors, b.flow]),
    errors: step(BLOCK_ERRORS, grid, [now, before, b.flow, b.errors]),
    assemble: step(assembleShader(model, contract, network.packedDot), input, [
      now, history(b.base), history(b.flow), confident ? b.errors : null, history(b.previous),
      recurrent && model.stateWords ? b.state : null, b.input,
    ]),
    finish: step(finishShader(model.outputWords), input, [b.base, b.colour, b.state, b.previous, b.output]),
  };
  let tilesPerGroup = network.tiles;
  steps.network = (pass) => network.encode(pass, tilesPerGroup);
  const presenters = createPresenters(device, canvases, PICTURES.map((name) => buffers[name]), fine[0]);
  const settings = new Int32Array(FRAME_BYTES / WORD);
  const timer = createPassTimer(device, stepTimes ? ORDER.length : 1);
  let parity = 0;

  return {
    tiles: network.tiles,
    /**
     * Submit one frame: `planes` is the 4:2:0 frame (Y, U, V bytes), `vectors`
     * its block motion field (Int32Array, dx and dy per block), `valid`
     * whether the history on the GPU may be used, `group` the tiles per
     * dispatch, `grain` the amplitude of the
     * grain on the output picture (0: none) and `index` the frame's place in
     * the stream, which the grain is drawn from. Resolves when the GPU has
     * finished it, to `{ compute, steps }`: the milliseconds the GPU spent
     * computing it (null where that cannot be known) and, where steps are
     * timed apart, those of each by name (else null).
     */
    submit(planes, vectors, valid, group, grain, index) {
      tilesPerGroup = group;
      parity = 1 - parity;
      settings.set([width, height, recurrent && valid ? 1 : 0, grain, index | 0]);
      device.queue.writeBuffer(frame, 0, settings);
      device.queue.writeBuffer(luma[parity], 0, planes, 0, pixels);
      device.queue.writeBuffer(buffers.chroma, 0, planes, pixels, pixels / 2);
      device.queue.writeBuffer(buffers.vectors, 0, vectors);
      const encoder = device.createCommandEncoder();
      const warping = recurrent && valid;
      const due = ORDER.filter((name) => (name === 'flow' ? warping : name === 'errors' ? warping && confident : true));
      // One pass for the frame, or one for each step where they are timed apart.
      const passes = stepTimes && timer ? due.map((name) => [name]) : [due];
      passes.forEach((names) => {
        const pass = encoder.beginComputePass(timer ? { timestampWrites: timer.writes(stepTimes ? ORDER.indexOf(names[0]) : 0) } : {});
        names.forEach((name) => steps[name](pass, parity));
        pass.end();
      });
      timer?.resolve(encoder);
      presenters.forEach((present) => present(encoder));
      device.queue.submit([encoder.finish()]);
      if (!timer) return device.queue.onSubmittedWorkDone().then(() => ({ compute: null, steps: null }));
      return timer.read().then((times) => {
        if (!times) return { compute: null, steps: null };
        if (!stepTimes) return { compute: times[0], steps: null };
        const known = due.map((name) => [name, times[ORDER.indexOf(name)]]).filter(([, ms]) => ms !== null);
        // A step too short for the timestamps to tell apart counts as nothing.
        return { compute: known.reduce((sum, [, ms]) => sum + ms, 0), steps: Object.fromEntries(known) };
      });
    },
    /** Read back what the last frame produced, as Uint32Arrays: for checks only. */
    async read(names) {
      const staged = names.map((name) => {
        const source = buffers[name];
        const staging = device.createBuffer({ size: source.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
        device.queue.submit([encoder.finish()]);
        return staging;
      });
      return Promise.all(staged.map(async (staging) => {
        await staging.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(staging.getMappedRange().slice(0));
        staging.destroy();
        return words;
      }));
    },
    destroy() {
      network.destroy();
      timer?.destroy();
      owned.forEach((made) => made.destroy());
    },
  };
}

/** For each canvas, a function adding to an encoder the pass that draws its picture's RGBA words. */
function createPresenters(device, canvases, pictures, width) {
  const format = navigator.gpu.getPreferredCanvasFormat();
  const module = device.createShaderModule({ code: PRESENT });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vertex' },
    fragment: { module, entryPoint: 'fragment', targets: [{ format }] },
    primitive: { topology: 'triangle-strip' },
  });
  const size = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(size, 0, new Uint32Array([width, 0, 0, 0]));
  return canvases.map((canvas, index) => {
    const context = canvas.getContext('webgpu');
    context.configure({ device, format, alphaMode: 'opaque' });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: pictures[index] } }, { binding: 1, resource: { buffer: size } }],
    });
    return (encoder) => {
      const view = context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(4);
      pass.end();
    };
  });
}
