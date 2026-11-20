let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedInt32ArrayMemory0 = null;

function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;

function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
/**
 * The values of the coding planes on the network's 8-bit grid: known,
 * quantiser, key frame. A negative `quantiser` means nothing is known.
 * @param {number} quantiser
 * @param {boolean} key_frame
 * @returns {Int32Array}
 */
export function coding_plane_levels(quantiser, key_frame) {
    const ret = wasm.coding_plane_levels(quantiser, key_frame);
    var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let WASM_VECTOR_LEN = 0;

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

const IntegerNetworkFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_integernetwork_free(ptr >>> 0, 1));
/**
 * An exported integer model.
 */
export class IntegerNetwork {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntegerNetworkFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_integernetwork_free(ptr, 0);
    }
    /**
     * Words per pixel of the packed input.
     * @returns {number}
     */
    input_words() {
        const ret = wasm.integernetwork_input_words(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Words per pixel of the packed output.
     * @returns {number}
     */
    output_words() {
        const ret = wasm.integernetwork_output_words(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The model in the words of a model file.
     *
     * # Errors
     *
     * If the file is not a valid integer model of the schema this module reads.
     * @param {Int32Array} words
     */
    constructor(words) {
        const ptr0 = passArray32ToWasm0(words, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.integernetwork_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        IntegerNetworkFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The reference network on packed input planes, written to `output`;
     * `tile` 0 runs the whole frame as one tile. Slow: for checking the GPU.
     *
     * # Errors
     *
     * If an array is not the size the frame implies.
     * @param {Uint32Array} packed
     * @param {number} width
     * @param {number} height
     * @param {number} tile
     * @param {Uint32Array} output
     */
    infer(packed, width, height, tile, output) {
        const ptr0 = passArray32ToWasm0(packed, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.integernetwork_infer(this.__wbg_ptr, ptr0, len0, width, height, tile, output);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) IntegerNetwork.prototype[Symbol.dispose] = IntegerNetwork.prototype.free;

const IntegerReferenceFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_integerreference_free(ptr >>> 0, 1));
/**
 * The integer runtime the GPU's whole frame is checked against.
 */
export class IntegerReference {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IntegerReferenceFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_integerreference_free(ptr, 0);
    }
    /**
     * A reference for `width` x `height` frames of the luma-only model in
     * `words` (a model file), with the fields of its contract.
     *
     * # Errors
     *
     * If the file is not a valid integer model or does not match the contract.
     * @param {Int32Array} words
     * @param {number} width
     * @param {number} height
     * @param {string} temporal
     * @param {boolean} align
     * @param {number} state_channels
     * @param {boolean} coding_planes
     * @param {boolean} confidence_plane
     */
    constructor(words, width, height, temporal, align, state_channels, coding_planes, confidence_plane) {
        const ptr0 = passArray32ToWasm0(words, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(temporal, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.integerreference_new(ptr0, len0, width, height, ptr1, len1, align, state_channels, coding_planes, confidence_plane);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        IntegerReferenceFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Copy what the last frame produced into the arrays given: the
     * network's packed input and outputs, and the conventional upscale and
     * the output as RGBA words.
     *
     * # Errors
     *
     * If an array is not the size of what it is to hold.
     * @param {Uint32Array} input
     * @param {Uint32Array} outputs
     * @param {Uint32Array} conventional
     * @param {Uint32Array} picture
     */
    read(input, outputs, conventional, picture) {
        const ret = wasm.integerreference_read(this.__wbg_ptr, input, outputs, conventional, picture);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Upscale one 4:2:0 frame (Y, then U, then V) with the block motion
     * field `vectors` ([`MotionSearch::push`]); `tile` 0 runs the network on
     * the whole frame. A negative `quantiser` means the coding is unknown.
     *
     * # Errors
     *
     * If `frame` or `vectors` is not the configured size.
     * @param {Uint8Array} frame
     * @param {Int32Array} vectors
     * @param {boolean} reset
     * @param {number} quantiser
     * @param {boolean} key_frame
     * @param {number} tile
     */
    step(frame, vectors, reset, quantiser, key_frame, tile) {
        const ptr0 = passArray8ToWasm0(frame, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(vectors, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.integerreference_step(this.__wbg_ptr, ptr0, len0, ptr1, len1, reset, quantiser, key_frame, tile);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Forget earlier frames.
     */
    restart() {
        wasm.integerreference_restart(this.__wbg_ptr);
    }
}
if (Symbol.dispose) IntegerReference.prototype[Symbol.dispose] = IntegerReference.prototype.free;

const MotionSearchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_motionsearch_free(ptr >>> 0, 1));
/**
 * The motion search of one stream.
 */
export class MotionSearch {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MotionSearchFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_motionsearch_free(ptr, 0);
    }
    /**
     * A search over `width` x `height` luma planes.
     *
     * # Errors
     *
     * If the analysis cannot work on that size.
     * @param {number} width
     * @param {number} height
     */
    constructor(width, height) {
        const ret = wasm.motionsearch_new(width, height);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        MotionSearchFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Analyse the next luma plane: write its block motion field to
     * `vectors` (dx then dy per block, row by row) and return whether
     * history from before this frame is to be discarded.
     *
     * # Errors
     *
     * If `luma` or `vectors` is not the configured size.
     * @param {Uint8Array} luma
     * @param {Int32Array} vectors
     * @returns {boolean}
     */
    push(luma, vectors) {
        const ptr0 = passArray8ToWasm0(luma, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.motionsearch_push(this.__wbg_ptr, ptr0, len0, vectors);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Forget earlier frames.
     *
     * # Errors
     *
     * If the analysis cannot be restarted.
     */
    restart() {
        const ret = wasm.motionsearch_restart(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) MotionSearch.prototype[Symbol.dispose] = MotionSearch.prototype.free;

const SenderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sender_free(ptr >>> 0, 1));
/**
 * Makes the network's input picture from originals of one size.
 */
export class Sender {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SenderFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sender_free(ptr, 0);
    }
    /**
     * A sender for `width` x `height` originals (both multiples of four)
     * that reduces with the kernel called `kernel` ("area", "bilinear",
     * "bicubic" or "lanczos"); `seed` fixes the noise it will add.
     *
     * # Errors
     *
     * If there is no kernel of that name.
     * @param {number} width
     * @param {number} height
     * @param {string} kernel
     * @param {number} seed
     */
    constructor(width, height, kernel, seed) {
        const ptr0 = passStringToWasm0(kernel, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sender_new(width, height, ptr0, len0, seed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        SenderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Reduce the RGBA `original` to half size as a camera would deliver it:
     * blurred by `blur` samples, with sensor noise of `noise` luma levels,
     * denoised at strength `denoise` (each skipped at zero). The result is
     * written to `frame` as 8-bit 4:2:0 (Y, then U, then V).
     *
     * # Errors
     *
     * If `original` or `frame` is not the configured size.
     * @param {Uint8Array} original
     * @param {number} blur
     * @param {number} noise
     * @param {number} denoise
     * @param {Uint8Array} frame
     */
    send(original, blur, noise, denoise, frame) {
        const ptr0 = passArray8ToWasm0(original, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.sender_send(this.__wbg_ptr, ptr0, len0, blur, noise, denoise, frame);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) Sender.prototype[Symbol.dispose] = Sender.prototype.free;

const UpscalerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_upscaler_free(ptr >>> 0, 1));
/**
 * One stream being upscaled in the page.
 */
export class Upscaler {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UpscalerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_upscaler_free(ptr, 0);
    }
    /**
     * Say how the frames prepared from now on were coded: the average
     * quantiser (0 to 51) and whether each is a key frame. Only a model with
     * coding planes uses it.
     * @param {number} quantiser
     * @param {boolean} key_frame
     */
    set_coding(quantiser, key_frame) {
        wasm.upscaler_set_coding(this.__wbg_ptr, quantiser, key_frame);
    }
    /**
     * Say that nothing is known about how the frames were coded (the state
     * a session starts in), or that they never met a codec.
     */
    clear_coding() {
        wasm.upscaler_clear_coding(this.__wbg_ptr);
    }
    /**
     * Write the conventional (bicubic) 2x upscale of the prepared frame to
     * `picture`, as RGBA.
     *
     * # Errors
     *
     * If `picture` is not the output size.
     * @param {Uint8ClampedArray} picture
     */
    conventional(picture) {
        const ret = wasm.upscaler_conventional(this.__wbg_ptr, picture);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Width of the output picture.
     * @returns {number}
     */
    output_width() {
        const ret = wasm.upscaler_output_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Height of the output picture.
     * @returns {number}
     */
    output_height() {
        const ret = wasm.upscaler_output_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Channels of the network's input tensor.
     * @returns {number}
     */
    input_channels() {
        const ret = wasm.upscaler_input_channels(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Channels of the hidden-state output (0 for a per-frame model).
     * @returns {number}
     */
    hidden_channels() {
        const ret = wasm.upscaler_hidden_channels(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The input channels that stay zero for the whole stream, as
     * `[first, one past the last]`: the page uploads only the others.
     * @returns {Uint32Array}
     */
    unused_input_channels() {
        const ret = wasm.upscaler_unused_input_channels(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * A session for `width` x `height` input frames and a model with the
     * given contract (the fields of its `model.json`).
     *
     * # Errors
     *
     * If the size is not one the network and analysis can take.
     * @param {number} width
     * @param {number} height
     * @param {number} planes
     * @param {string} temporal
     * @param {boolean} align
     * @param {number} state_channels
     * @param {boolean} side_planes
     * @param {boolean} luma_only
     * @param {boolean} coding_planes
     * @param {boolean} confidence_plane
     */
    constructor(width, height, planes, temporal, align, state_channels, side_planes, luma_only, coding_planes, confidence_plane) {
        const ptr0 = passStringToWasm0(temporal, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.upscaler_new(width, height, planes, ptr0, len0, align, state_channels, side_planes, luma_only, coding_planes, confidence_plane);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        UpscalerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Turn the network's outputs for the prepared frame into the 2x picture
     * and write it to `picture`, as RGBA. `coarse` is empty for a per-frame
     * model.
     *
     * # Errors
     *
     * If no frame is prepared, or a tensor or `picture` has the wrong size.
     * @param {Float32Array} fine
     * @param {Float32Array} coarse
     * @param {Uint8ClampedArray} picture
     */
    finish(fine, coarse, picture) {
        const ptr0 = passArrayF32ToWasm0(fine, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(coarse, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.upscaler_finish(this.__wbg_ptr, ptr0, len0, ptr1, len1, picture);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Analyse the next frame, given as 8-bit 4:2:0 (Y, then U, then V) of
     * `width` x `height`, and write the network's input tensor (channels x
     * height x width) to `input`.
     *
     * # Errors
     *
     * If `frame` or `input` is not the configured size.
     * @param {Uint8Array} frame
     * @param {Float32Array} input
     */
    prepare(frame, input) {
        const ptr0 = passArray8ToWasm0(frame, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.upscaler_prepare(this.__wbg_ptr, ptr0, len0, input);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Forget earlier frames: call after a seek, a source change or a failed
     * network call.
     *
     * # Errors
     *
     * If the analysis cannot be restarted.
     */
    restart() {
        const ret = wasm.upscaler_restart(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * As [`Upscaler::finish`], from the integer network's packed outputs.
     *
     * # Errors
     *
     * If no frame is prepared, or `packed` or `picture` has the wrong size.
     * @param {IntegerNetwork} codec
     * @param {Uint32Array} packed
     * @param {Uint8ClampedArray} picture
     */
    finish_packed(codec, packed, picture) {
        _assertClass(codec, IntegerNetwork);
        const ptr0 = passArray32ToWasm0(packed, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.upscaler_finish_packed(this.__wbg_ptr, codec.__wbg_ptr, ptr0, len0, picture);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * As [`Upscaler::prepare`], for a model in integer form: the network's
     * input as packed 8-bit planes, `codec.input_words()` words per pixel.
     *
     * # Errors
     *
     * If `frame` or `packed` is not the configured size.
     * @param {Uint8Array} frame
     * @param {IntegerNetwork} codec
     * @param {Uint32Array} packed
     */
    prepare_packed(frame, codec, packed) {
        const ptr0 = passArray8ToWasm0(frame, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        _assertClass(codec, IntegerNetwork);
        const ret = wasm.upscaler_prepare_packed(this.__wbg_ptr, ptr0, len0, codec.__wbg_ptr, packed);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) Upscaler.prototype[Symbol.dispose] = Upscaler.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_Error_e83987f665cf5504 = function(arg0, arg1) {
        const ret = Error(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg___wbindgen_throw_b855445ff6a94295 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_length_40c759d395b54595 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_58bec3c3f0487eb5 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_69bca3cb64fc8748 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_98176481f29eb789 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_length_d5532ee3f1122b8a = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_set_8fa23c1f8b7e534d = function(arg0, arg1, arg2) {
        arg0.set(getArrayI32FromWasm0(arg1, arg2));
    };
    imports.wbg.__wbg_set_9e6516df7b7d0f19 = function(arg0, arg1, arg2) {
        arg0.set(getArrayU8FromWasm0(arg1, arg2));
    };
    imports.wbg.__wbg_set_e07f5bde8e6f3f0d = function(arg0, arg1, arg2) {
        arg0.set(getArrayU8FromWasm0(arg1, arg2));
    };
    imports.wbg.__wbg_set_eaa55bcb7597ecca = function(arg0, arg1, arg2) {
        arg0.set(getArrayF32FromWasm0(arg1, arg2));
    };
    imports.wbg.__wbg_set_fed9f2d08c1c01ae = function(arg0, arg1, arg2) {
        arg0.set(getArrayU32FromWasm0(arg1, arg2));
    };
    imports.wbg.__wbg_subarray_480600f3d6a9f26c = function(arg0, arg1, arg2) {
        const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
        ;
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedFloat32ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('cvnn_web_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
