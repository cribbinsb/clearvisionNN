/* tslint:disable */
/* eslint-disable */
/**
 * The amplitude the page's shaders multiply the grain's noise by under the
 * profile named `profile`; zero is no grain.
 *
 * # Errors
 *
 * If there is no profile of that name.
 */
export function grain_amplitude(profile: string): number;
/**
 * The amplitude of the profile a stream starts on, before any is named.
 */
export function default_grain_amplitude(): number;
/**
 * An exported integer model.
 */
export class IntegerNetwork {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Words per pixel of the packed input.
   */
  input_words(): number;
  /**
   * Words per pixel of the packed output.
   */
  output_words(): number;
  /**
   * The model in the words of a model file.
   *
   * # Errors
   *
   * If the file is not a valid integer model of the schema this module reads.
   */
  constructor(words: Int32Array);
  /**
   * The reference network on packed input planes, written to `output`;
   * `tile` 0 runs the whole frame as one tile. Slow: for checking the GPU.
   *
   * # Errors
   *
   * If an array is not the size the frame implies.
   */
  infer(packed: Uint32Array, width: number, height: number, tile: number, output: Uint32Array): void;
}
/**
 * The integer runtime the GPU's whole frame is checked against.
 */
export class IntegerReference {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * A reference for `width` x `height` frames of the luma-only model in
   * `words` (a model file), with the fields of its contract.
   *
   * # Errors
   *
   * If the file is not a valid integer model or does not match the contract.
   */
  constructor(words: Int32Array, width: number, height: number, temporal: string, align: boolean, state_channels: number, confidence_plane: boolean);
  /**
   * Copy what the last frame produced into the arrays given: the
   * network's packed input and outputs, and the conventional upscale and
   * the output as RGBA words.
   *
   * # Errors
   *
   * If an array is not the size of what it is to hold.
   */
  read(input: Uint32Array, outputs: Uint32Array, conventional: Uint32Array, picture: Uint32Array): void;
  /**
   * Upscale one 4:2:0 frame (Y, then U, then V) with the block motion
   * field `vectors` ([`MotionSearch::push`]); `tile` 0 runs the network on
   * the whole frame.
   *
   * # Errors
   *
   * If `frame` or `vectors` is not the configured size.
   */
  step(frame: Uint8Array, vectors: Int32Array, reset: boolean, tile: number): void;
  /**
   * Forget earlier frames.
   */
  restart(): void;
  /**
   * Show the frames from now on with the grain of the profile named
   * `profile` (a reference starts on `webcam`).
   *
   * # Errors
   *
   * If there is no profile of that name.
   */
  set_grain(profile: string): void;
}
/**
 * The motion search of one stream.
 */
export class MotionSearch {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * A search over `width` x `height` luma planes.
   *
   * # Errors
   *
   * If the analysis cannot work on that size.
   */
  constructor(width: number, height: number);
  /**
   * Analyse the next luma plane: write its block motion field to
   * `vectors` (dx then dy per block, row by row) and return whether
   * history from before this frame is to be discarded.
   *
   * # Errors
   *
   * If `luma` or `vectors` is not the configured size.
   */
  push(luma: Uint8Array, vectors: Int32Array): boolean;
  /**
   * Forget earlier frames.
   *
   * # Errors
   *
   * If the analysis cannot be restarted.
   */
  restart(): void;
}
/**
 * Makes the network's input picture from originals of one size.
 */
export class Sender {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * A sender for `width` x `height` originals (both multiples of four)
   * that reduces with the kernel called `kernel` ("area", "bilinear",
   * "bicubic" or "lanczos"); `seed` fixes the noise it will add.
   *
   * # Errors
   *
   * If there is no kernel of that name.
   */
  constructor(width: number, height: number, kernel: string, seed: number);
  /**
   * Reduce the RGBA `original` to half size as a camera would deliver it:
   * blurred by `blur` samples, with sensor noise of `noise` luma levels,
   * denoised at strength `denoise` (each skipped at zero). The result is
   * written to `frame` as 8-bit 4:2:0 (Y, then U, then V).
   *
   * # Errors
   *
   * If `original` or `frame` is not the configured size.
   */
  send(original: Uint8Array, blur: number, noise: number, denoise: number, frame: Uint8Array): void;
}
/**
 * One stream being upscaled in the page.
 */
export class Upscaler {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Write the conventional (bicubic) 2x upscale of the prepared frame to
   * `picture`, as RGBA.
   *
   * # Errors
   *
   * If `picture` is not the output size.
   */
  conventional(picture: Uint8ClampedArray): void;
  /**
   * Width of the output picture.
   */
  output_width(): number;
  /**
   * Height of the output picture.
   */
  output_height(): number;
  /**
   * Channels of the network's input tensor.
   */
  input_channels(): number;
  /**
   * Channels of the hidden-state output (0 for a per-frame model).
   */
  hidden_channels(): number;
  /**
   * The input channels that stay zero for the whole stream, as
   * `[first, one past the last]`: the page uploads only the others.
   */
  unused_input_channels(): Uint32Array;
  /**
   * A session for `width` x `height` input frames and a model with the
   * given contract (the fields of its `model.json`).
   *
   * # Errors
   *
   * If the size is not one the network and analysis can take.
   */
  constructor(width: number, height: number, planes: number, temporal: string, align: boolean, state_channels: number, side_planes: boolean, luma_only: boolean, confidence_plane: boolean);
  /**
   * Turn the network's outputs for the prepared frame into the 2x picture
   * and write it to `picture`, as RGBA. `coarse` is empty for a per-frame
   * model.
   *
   * # Errors
   *
   * If no frame is prepared, or a tensor or `picture` has the wrong size.
   */
  finish(fine: Float32Array, coarse: Float32Array, picture: Uint8ClampedArray): void;
  /**
   * Analyse the next frame, given as 8-bit 4:2:0 (Y, then U, then V) of
   * `width` x `height`, and write the network's input tensor (channels x
   * height x width) to `input`.
   *
   * # Errors
   *
   * If `frame` or `input` is not the configured size.
   */
  prepare(frame: Uint8Array, input: Float32Array): void;
  /**
   * Forget earlier frames: call after a seek, a source change or a failed
   * network call.
   *
   * # Errors
   *
   * If the analysis cannot be restarted.
   */
  restart(): void;
  /**
   * Show the pictures finished from now on with the grain of the profile
   * named `profile` on their luma (a session starts on `webcam`).
   *
   * # Errors
   *
   * If there is no profile of that name.
   */
  set_grain(profile: string): void;
  /**
   * As [`Upscaler::finish`], from the integer network's packed outputs.
   *
   * # Errors
   *
   * If no frame is prepared, or `packed` or `picture` has the wrong size.
   */
  finish_packed(codec: IntegerNetwork, packed: Uint32Array, picture: Uint8ClampedArray): void;
  /**
   * As [`Upscaler::prepare`], for a model in integer form: the network's
   * input as packed 8-bit planes, `codec.input_words()` words per pixel.
   *
   * # Errors
   *
   * If `frame` or `packed` is not the configured size.
   */
  prepare_packed(frame: Uint8Array, codec: IntegerNetwork, packed: Uint32Array): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_integernetwork_free: (a: number, b: number) => void;
  readonly __wbg_integerreference_free: (a: number, b: number) => void;
  readonly __wbg_motionsearch_free: (a: number, b: number) => void;
  readonly __wbg_sender_free: (a: number, b: number) => void;
  readonly __wbg_upscaler_free: (a: number, b: number) => void;
  readonly default_grain_amplitude: () => number;
  readonly grain_amplitude: (a: number, b: number) => [number, number, number];
  readonly integernetwork_infer: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => [number, number];
  readonly integernetwork_input_words: (a: number) => number;
  readonly integernetwork_new: (a: number, b: number) => [number, number, number];
  readonly integernetwork_output_words: (a: number) => number;
  readonly integerreference_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
  readonly integerreference_read: (a: number, b: any, c: any, d: any, e: any) => [number, number];
  readonly integerreference_restart: (a: number) => void;
  readonly integerreference_set_grain: (a: number, b: number, c: number) => [number, number];
  readonly integerreference_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly motionsearch_new: (a: number, b: number) => [number, number, number];
  readonly motionsearch_push: (a: number, b: number, c: number, d: any) => [number, number, number];
  readonly motionsearch_restart: (a: number) => [number, number];
  readonly sender_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
  readonly sender_send: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => [number, number];
  readonly upscaler_conventional: (a: number, b: any) => [number, number];
  readonly upscaler_finish: (a: number, b: number, c: number, d: number, e: number, f: any) => [number, number];
  readonly upscaler_finish_packed: (a: number, b: number, c: number, d: number, e: any) => [number, number];
  readonly upscaler_hidden_channels: (a: number) => number;
  readonly upscaler_input_channels: (a: number) => number;
  readonly upscaler_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
  readonly upscaler_output_height: (a: number) => number;
  readonly upscaler_output_width: (a: number) => number;
  readonly upscaler_prepare: (a: number, b: number, c: number, d: any) => [number, number];
  readonly upscaler_prepare_packed: (a: number, b: number, c: number, d: number, e: any) => [number, number];
  readonly upscaler_restart: (a: number) => [number, number];
  readonly upscaler_set_grain: (a: number, b: number, c: number) => [number, number];
  readonly upscaler_unused_input_channels: (a: number) => [number, number];
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
