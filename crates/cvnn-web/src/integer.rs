//! The integer network for the page: the session's two halves for a model in
//! integer form, exchanging packed 8-bit planes with the page's 8-bit backend
//! instead of float tensors, and the reference network that backend is
//! checked against.
//!
//! Bindings only: the logic is `cvnn_runtime::integer`'s.

use cvnn_runtime::integer::{IntegerModel, PICTURE_OUTPUTS};
use js_sys::{Uint8ClampedArray, Uint32Array};
use wasm_bindgen::prelude::*;

use crate::{Upscaler, deliver, to_js_error};

/// An exported integer model.
#[wasm_bindgen]
#[derive(Debug)]
pub struct IntegerNetwork {
    model: IntegerModel,
    /// Packed planes on their way to the page.
    packed: Vec<u32>,
    /// The network's outputs made float again, on their way into the session.
    fine: Vec<f32>,
    coarse: Vec<f32>,
}

#[wasm_bindgen]
impl IntegerNetwork {
    /// The model in the words of a model file.
    ///
    /// # Errors
    ///
    /// If the file is not a valid integer model of the schema this module reads.
    #[wasm_bindgen(constructor)]
    pub fn new(words: &[i32]) -> Result<IntegerNetwork, JsError> {
        let model = IntegerModel::from_words(words).map_err(|e| to_js_error(&e))?;
        Ok(Self {
            model,
            packed: Vec::new(),
            fine: Vec::new(),
            coarse: Vec::new(),
        })
    }

    /// Words per pixel of the packed input.
    #[must_use]
    pub fn input_words(&self) -> usize {
        self.model.input_words()
    }

    /// Words per pixel of the packed output.
    #[must_use]
    pub fn output_words(&self) -> usize {
        self.model.output_words()
    }

    /// The reference network on packed input planes, written to `output`;
    /// `tile` 0 runs the whole frame as one tile. Slow: for checking the GPU.
    ///
    /// # Errors
    ///
    /// If an array is not the size the frame implies.
    pub fn infer(
        &self,
        packed: &[u32],
        width: usize,
        height: usize,
        tile: usize,
        output: &Uint32Array,
    ) -> Result<(), JsError> {
        let result = self.model.infer(packed, width, height, tile);
        let words = result.map_err(|e| to_js_error(&e))?;
        deliver(&words, output.length(), |words| output.copy_from(words))
    }
}

#[wasm_bindgen]
impl Upscaler {
    /// As [`Upscaler::prepare`], for a model in integer form: the network's
    /// input as packed 8-bit planes, `codec.input_words()` words per pixel.
    ///
    /// # Errors
    ///
    /// If `frame` or `packed` is not the configured size.
    pub fn prepare_packed(
        &mut self,
        frame: &[u8],
        codec: &mut IntegerNetwork,
        packed: &Uint32Array,
    ) -> Result<(), JsError> {
        let (y, u, v) = self.planes(frame)?;
        let tensor = self.session.prepare(y, u, v).map_err(|e| to_js_error(&e))?;
        codec.packed.resize(packed.length() as usize, 0);
        codec
            .model
            .pack_input(tensor, &mut codec.packed)
            .map_err(|e| to_js_error(&e))?;
        deliver(&codec.packed, packed.length(), |words| {
            packed.copy_from(words);
        })
    }

    /// As [`Upscaler::finish`], from the integer network's packed outputs.
    ///
    /// # Errors
    ///
    /// If no frame is prepared, or `packed` or `picture` has the wrong size.
    pub fn finish_packed(
        &mut self,
        codec: &mut IntegerNetwork,
        packed: &[u32],
        picture: &Uint8ClampedArray,
    ) -> Result<(), JsError> {
        let pixels = self.width * self.height;
        codec.fine.resize(PICTURE_OUTPUTS * pixels, 0.0);
        codec
            .coarse
            .resize(codec.model.state_outputs() * pixels, 0.0);
        codec
            .model
            .unpack_output(
                packed,
                self.width,
                self.height,
                &mut codec.fine,
                &mut codec.coarse,
            )
            .map_err(|e| to_js_error(&e))?;
        self.finish_picture(&codec.fine, &codec.coarse, picture)
    }
}
