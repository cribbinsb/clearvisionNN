//! The fully integer path for the page. The GPU does everything around and
//! in the network; what stays here is the motion search, which needs only
//! the small luma plane ([`MotionSearch`]), and the reference the GPU is
//! checked against ([`IntegerReference`]).
//!
//! Bindings only: the logic is `cvnn_preproc`'s and `cvnn_runtime::fixed`'s.

use cvnn_preproc::{PreprocConfig, Preprocessor};
use cvnn_runtime::fixed::{BlockVector, FrameInput, IntegerRuntime, block_vectors};
use cvnn_runtime::grain::GrainProfile;
use cvnn_runtime::integer::IntegerModel;
use cvnn_runtime::{Input, ModelSpec};
use js_sys::{Int32Array, Uint32Array};
use wasm_bindgen::prelude::*;

use crate::{deliver, grain, to_js_error};

/// The amplitude the page's shaders multiply the grain's noise by under the
/// profile named `profile`; zero is no grain.
///
/// # Errors
///
/// If there is no profile of that name.
#[wasm_bindgen]
pub fn grain_amplitude(profile: &str) -> Result<i32, JsError> {
    Ok(grain(profile)?.amplitude())
}

/// The amplitude of the profile a stream starts on, before any is named.
#[wasm_bindgen]
#[must_use]
pub fn default_grain_amplitude() -> i32 {
    GrainProfile::default().amplitude()
}

/// The motion search of one stream.
#[wasm_bindgen]
#[derive(Debug)]
pub struct MotionSearch {
    preprocessor: Preprocessor,
    /// The newest field, two words per block: dx then dy, in fixed-point steps.
    vectors: Vec<i32>,
}

#[wasm_bindgen]
impl MotionSearch {
    /// A search over `width` x `height` luma planes.
    ///
    /// # Errors
    ///
    /// If the analysis cannot work on that size.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> Result<MotionSearch, JsError> {
        let preprocessor = Preprocessor::new(width, height, PreprocConfig::default());
        Ok(Self {
            preprocessor: preprocessor.map_err(|e| to_js_error(&e))?,
            vectors: Vec::new(),
        })
    }

    /// Analyse the next luma plane: write its block motion field to
    /// `vectors` (dx then dy per block, row by row) and return whether
    /// history from before this frame is to be discarded.
    ///
    /// # Errors
    ///
    /// If `luma` or `vectors` is not the configured size.
    pub fn push(&mut self, luma: &[u8], vectors: &Int32Array) -> Result<bool, JsError> {
        let features = self.preprocessor.push(luma).map_err(|e| to_js_error(&e))?;
        self.vectors.clear();
        self.vectors
            .extend(block_vectors(&features.motion).into_iter().flatten());
        deliver(&self.vectors, vectors.length(), |words| {
            vectors.copy_from(words);
        })?;
        Ok(features.reset)
    }

    /// Forget earlier frames.
    ///
    /// # Errors
    ///
    /// If the analysis cannot be restarted.
    pub fn restart(&mut self) -> Result<(), JsError> {
        self.preprocessor.restart().map_err(|e| to_js_error(&e))
    }
}

/// The integer runtime the GPU's whole frame is checked against.
#[wasm_bindgen]
#[derive(Debug)]
pub struct IntegerReference {
    runtime: IntegerRuntime,
}

#[wasm_bindgen]
impl IntegerReference {
    /// A reference for `width` x `height` frames of the luma-only model in
    /// `words` (a model file), with the fields of its contract.
    ///
    /// # Errors
    ///
    /// If the file is not a valid integer model or does not match the contract.
    #[wasm_bindgen(constructor)]
    pub fn new(
        words: &[i32],
        width: usize,
        height: usize,
        temporal: &str,
        align: bool,
        state_channels: usize,
        confidence_plane: bool,
    ) -> Result<IntegerReference, JsError> {
        let model = IntegerModel::from_words(words).map_err(|e| to_js_error(&e))?;
        let spec = ModelSpec {
            planes: 1,
            temporal: temporal.parse().map_err(|e| to_js_error(&e))?,
            align,
            state_channels,
            input: Input::Luma {
                confidence: confidence_plane,
            },
        };
        let runtime = IntegerRuntime::new(model, spec, width, height);
        Ok(Self {
            runtime: runtime.map_err(|e| to_js_error(&e))?,
        })
    }

    /// Forget earlier frames.
    pub fn restart(&mut self) {
        self.runtime.restart();
    }

    /// Show the frames from now on with the grain of the profile named
    /// `profile` (a reference starts on `webcam`).
    ///
    /// # Errors
    ///
    /// If there is no profile of that name.
    pub fn set_grain(&mut self, profile: &str) -> Result<(), JsError> {
        self.runtime.set_grain(grain(profile)?);
        Ok(())
    }

    /// Upscale one 4:2:0 frame (Y, then U, then V) with the block motion
    /// field `vectors` ([`MotionSearch::push`]); `tile` 0 runs the network on
    /// the whole frame.
    ///
    /// # Errors
    ///
    /// If `frame` or `vectors` is not the configured size.
    pub fn step(
        &mut self,
        frame: &[u8],
        vectors: &[i32],
        reset: bool,
        tile: usize,
    ) -> Result<(), JsError> {
        let luma = frame.len() * 2 / 3;
        let (y, colour) = frame.split_at(luma);
        let (u, v) = colour.split_at(luma / 4);
        let motion: Vec<BlockVector> = vectors.as_chunks::<2>().0.to_vec();
        let input = FrameInput {
            y,
            u,
            v,
            motion: &motion,
            reset,
        };
        self.runtime.step(&input, tile).map_err(|e| to_js_error(&e))
    }

    /// Copy what the last frame produced into the arrays given: the
    /// network's packed input and outputs, and the conventional upscale and
    /// the output as RGBA words.
    ///
    /// # Errors
    ///
    /// If an array is not the size of what it is to hold.
    pub fn read(
        &self,
        input: &Uint32Array,
        outputs: &Uint32Array,
        conventional: &Uint32Array,
        picture: &Uint32Array,
    ) -> Result<(), JsError> {
        let runtime = &self.runtime;
        let pairs = [
            (runtime.packed_input(), input),
            (runtime.outputs(), outputs),
            (runtime.conventional(), conventional),
            (runtime.picture(), picture),
        ];
        for (words, target) in pairs {
            deliver(words, target.length(), |words| {
                target.copy_from(words);
            })?;
        }
        Ok(())
    }
}
