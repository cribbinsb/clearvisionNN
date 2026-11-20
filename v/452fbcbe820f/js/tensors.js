// How a session and a network exchange a frame's tensors. The form follows
// the kind of the model's network file: float arrays for an ONNX graph,
// packed 8-bit planes for the integer form, which the WebAssembly session
// quantises and reads back itself so that no float tensor crosses into the page.

import { IntegerNetwork } from '../pkg/cvnn_web.js';

/** Float tensors: the input as channels x height x width, the outputs as `{ fine, coarse }`. */
function floatExchange(model, upscaler, width, height) {
  // Kept between frames: allocating arrays this large per frame costs more than filling them.
  const features = new Float32Array(model.input_channels * width * height);
  return {
    prepare(frame) {
      upscaler.prepare(frame, features);
      return features;
    },
    finish: (outputs, picture) => upscaler.finish(outputs.fine, outputs.coarse, picture),
    release() {},
  };
}

/** Packed planes both ways; the model file tells the session how to pack them. */
async function packedExchange(model, upscaler, width, height) {
  const words = new Int32Array(await (await fetch(model.url)).arrayBuffer());
  const codec = new IntegerNetwork(words); // validates the file
  const packed = new Uint32Array(codec.input_words() * width * height);
  return {
    prepare(frame) {
      upscaler.prepare_packed(frame, codec, packed);
      return packed;
    },
    finish: (outputs, picture) => upscaler.finish_packed(codec, outputs, picture),
    release: () => codec.free(),
  };
}

const EXCHANGES = { onnx: floatExchange, integer: packedExchange };

/**
 * The exchange for `model` around `upscaler` (the WebAssembly session):
 * `prepare(frame)` returns the network's input for a 4:2:0 frame,
 * `finish(outputs, picture)` writes the picture from the network's outputs,
 * and `release()` frees what the exchange holds.
 */
export function createExchange(model, upscaler, width, height) {
  return EXCHANGES[model.network](model, upscaler, width, height);
}
