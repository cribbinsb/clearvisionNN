// The integer network on a GPU device: buffers, pipelines and the dispatches
// of one frame. Tiles are run in groups, each group through every layer
// before the next, so a group's activations are all that is live at once;
// the group size changes only the order of the work, never its result.

import { GROUP_SIDE, layerShader } from './int8-shaders.js';

const UNIFORM_STEP = 256; // bytes between parameter blocks: the alignment uniform offsets need
const PARAMS_BYTES = 48;
const WORD_BYTES = 4;

const hasPackedDot = () => navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product') ?? false;
const kindOf = (layer, layers) => (layer === 0 ? 'stem' : layer === layers - 1 ? 'head' : layer === layers - 2 ? 'last' : 'body');

/**
 * `model` (int8-model.js) on `device` for `width` x `height` input.
 * `run(input, tilesPerGroup)` resolves to the packed outputs; `encode` adds
 * the network to a pass of the caller's, for one that keeps the planes on the
 * GPU. `options` are
 * for checking and timing: `tile` overrides the model's tile side (0: the
 * whole frame as one tile, with no halo), `haloOverride` its halo,
 * `packedDot: false` uses the spelt-out arithmetic even where the packed
 * built-ins exist, and `uniformWeights: false` binds the weights as storage
 * even where they fit a uniform binding.
 */
export function createInt8Network(device, model, width, height, options = {}) {
  const { tile = model.tile, haloOverride = model.halo, packedDot = hasPackedDot() } = options;
  const layers = model.layers.length;
  // One choice for the whole model, so that a kind of layer has one shader.
  const largestWeights = Math.max(...model.layers.map((layer) => layer.weights.byteLength));
  const uniformWeights = options.uniformWeights ?? largestWeights <= device.limits.maxUniformBufferBindingSize;
  const halo = tile ? haloOverride : 0;
  const size = tile ? [tile, tile] : [width, height];
  const grid = [Math.ceil(width / size[0]), Math.ceil(height / size[1])];
  const window = [size[0] + 2 * halo, size[1] + 2 * halo];
  const tiles = grid[0] * grid[1];
  const owned = [];
  const buffer = (bytes, usage) => {
    const made = device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage });
    owned.push(made);
    return made;
  };
  const storage = (bytes) => buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const filled = (data, usage = GPUBufferUsage.STORAGE) => {
    const made = buffer(data.byteLength, usage | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(made, 0, data);
    return made;
  };
  const weights = (data) => filled(data, uniformWeights ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE);
  const pixels = width * height;
  const input = storage(pixels * model.inputWords * WORD_BYTES);
  const output = storage(pixels * model.outputWords * WORD_BYTES);
  const staging = buffer(output.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  // Sized for every tile at once, the largest group; smaller groups use the start of each.
  const scratch = [0, 1, 2].map(() => storage(tiles * window[0] * window[1] * model.trunkWords * WORD_BYTES));
  const tanh = model.stateWords ? filled(model.tanh) : null;
  const pipelines = {};
  for (const kind of ['stem', 'body', 'last', 'head']) {
    const module = device.createShaderModule({ code: layerShader(kind, model, packedDot, uniformWeights) });
    pipelines[kind] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  // scratch[0] keeps the stem's output for the long skip; the body alternates between the other two.
  const source = (layer) => (layer === 0 ? input : layer === 1 ? scratch[0] : scratch[1 + (layer % 2)]);
  const target = (layer) => (layer === layers - 1 ? output : layer === 0 ? scratch[0] : scratch[1 + ((layer + 1) % 2)]);
  const layerGroups = model.layers.map((data, layer) => {
    const kind = kindOf(layer, layers);
    const entries = [source(layer), target(layer), weights(data.weights), filled(data.constants)]
      .map((resource, binding) => ({ binding, resource: { buffer: resource } }));
    if (kind === 'last') entries.push({ binding: 4, resource: { buffer: scratch[0] } });
    if (kind === 'head' && tanh) entries.push({ binding: 5, resource: { buffer: tanh } });
    return device.createBindGroup({ layout: pipelines[kind].getBindGroupLayout(0), entries });
  });
  const reach = (layer) => Math.min(halo, layers - 1 - layer);
  const schedules = new Map(); // tiles per group -> the parameter bind groups of each group and layer

  function scheduleFor(perGroup) {
    if (!schedules.has(perGroup)) {
      const groups = Math.ceil(tiles / perGroup);
      const table = new Int32Array((groups * layers * UNIFORM_STEP) / 4);
      for (let group = 0; group < groups; group += 1) {
        for (let layer = 0; layer < layers; layer += 1) {
          const block = [width, height, ...grid, ...size, ...window, halo, reach(layer), layer ? reach(layer - 1) : 0, group * perGroup];
          table.set(block, ((group * layers + layer) * UNIFORM_STEP) / 4);
        }
      }
      const uniforms = buffer(table.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(uniforms, 0, table);
      const bind = (group, layer) => device.createBindGroup({
        layout: pipelines[kindOf(layer, layers)].getBindGroupLayout(1),
        entries: [{ binding: 0, resource: { buffer: uniforms, offset: (group * layers + layer) * UNIFORM_STEP, size: PARAMS_BYTES } }],
      });
      schedules.set(perGroup, Array.from({ length: groups }, (_, group) => model.layers.map((_data, layer) => bind(group, layer))));
    }
    return schedules.get(perGroup);
  }

  /** Append one frame of the network to `pass`, a compute pass: every group of tiles through every layer. */
  function encode(pass, tilesPerGroup = tiles) {
    const perGroup = Math.min(tiles, Math.max(1, tilesPerGroup));
    scheduleFor(perGroup).forEach((groupLayers, group) => {
      const depth = Math.min(perGroup, tiles - group * perGroup);
      groupLayers.forEach((parameters, layer) => {
        pass.setPipeline(pipelines[kindOf(layer, layers)]);
        pass.setBindGroup(0, layerGroups[layer]);
        pass.setBindGroup(1, parameters);
        pass.dispatchWorkgroups(Math.ceil(window[0] / GROUP_SIDE), Math.ceil(window[1] / GROUP_SIDE), depth);
      });
    });
  }

  const result = new Uint32Array(pixels * model.outputWords);
  return {
    tiles,
    packedDot,
    uniformWeights,
    // The packed planes the network reads and writes, for a caller that fills and uses them on the GPU.
    input,
    output,
    encode,
    /** Run one frame of packed input planes; the returned array is overwritten by the next call. */
    async run(packed, tilesPerGroup = tiles) {
      device.queue.writeBuffer(input, 0, packed);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      encode(pass, tilesPerGroup);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, staging, 0, output.size);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      result.set(new Uint32Array(staging.getMappedRange()));
      staging.unmap();
      return result;
    },
    destroy: () => owned.forEach((made) => made.destroy()),
  };
}
