// The upscaler proper: the WebAssembly session prepares the network's input
// from a decoded low-resolution frame, the chosen backend runs the network,
// and the session turns its outputs into the picture. The tensors pass in the
// form the model's network file calls for (tensors.js).

import { Upscaler } from '../pkg/cvnn_web.js';
import { backendOf } from './backends.js';
import { createExchange } from './tensors.js';

// The contract version this page and its WebAssembly module implement.
const SCHEMA_VERSION = 2;

function checkContract(model, upscaler, width, height) {
  if (model.schema_version !== SCHEMA_VERSION) {
    throw new Error(`${model.directory}: contract version ${model.schema_version}, page expects ${SCHEMA_VERSION}`);
  }
  if (width % model.size_multiple || height % model.size_multiple) {
    throw new Error(`Input size must be a multiple of ${model.size_multiple}.`);
  }
  if (upscaler.input_channels() !== model.input_channels) {
    throw new Error(`${model.directory}: contract and runtime disagree on the network's input`);
  }
}

/**
 * An upscaler for `model` on `backendId`, for `width` x `height` input
 * frames: `process(frame, coding)`, `restart()` and `release()`. A backend
 * that is a whole upscaler provides it; around one that is only a network,
 * the WebAssembly session does the rest.
 */
export function createUpscaler(model, backendId, width, height) {
  const backend = backendOf(backendId);
  if (backend.upscaler) return backend.upscaler(model, width, height);
  return createSessionUpscaler(model, backend, width, height);
}

/** The WebAssembly session around `backend`'s network. */
async function createSessionUpscaler(model, backend, width, height) {
  const upscaler = new Upscaler(width, height, model.planes, model.temporal, model.align,
    model.state_channels, model.side_planes, model.luma_only, model.coding_planes,
    model.confidence_plane);
  let exchange = null;
  let network;
  try {
    checkContract(model, upscaler, width, height);
    exchange = await createExchange(model, upscaler, width, height);
    network = await backend.network(model, width, height, upscaler.unused_input_channels());
  } catch (error) {
    exchange?.release();
    upscaler.free();
    throw error;
  }
  const outputWidth = upscaler.output_width();
  const outputHeight = upscaler.output_height();
  // Kept between frames: allocating arrays this large per frame costs more than filling them.
  const conventional = new ImageData(outputWidth, outputHeight);
  const upscaled = new ImageData(outputWidth, outputHeight);
  return {
    /**
     * Upscale one 4:2:0 frame. `coding` is `{ quantiser, key }` when the
     * decoder knows how the frame was coded, else null. Returns the bicubic
     * and the network's pictures and the time each step took (ms); the
     * pictures are overwritten by the next call.
     */
    async process(frame, coding) {
      const start = performance.now();
      if (coding) upscaler.set_coding(coding.quantiser, coding.key);
      else upscaler.clear_coding();
      const input = exchange.prepare(frame);
      const prepared = performance.now();
      let outputs;
      try {
        outputs = await network.run(input);
      } catch (error) {
        upscaler.restart(); // the prepared frame is abandoned; history no longer matches
        throw error;
      }
      const inferred = performance.now();
      upscaler.conventional(conventional.data);
      exchange.finish(outputs, upscaled.data);
      const finished = performance.now();
      return {
        conventional,
        upscaled,
        timings: { prepare: prepared - start, network: inferred - prepared, finish: finished - inferred },
      };
    },
    /** Forget earlier frames: the next one starts a new stream. */
    restart: () => upscaler.restart(),
    async release() {
      try {
        await network.release();
      } finally {
        exchange.release();
        upscaler.free();
      }
    },
  };
}
