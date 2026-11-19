//! The runtime session for the browser. The page hands a decoded 4:2:0 frame
//! to [`Upscaler::prepare`], runs the network on the returned tensor with
//! `WebGPU`, `WebNN` or WebAssembly, and hands the outputs to
//! [`Upscaler::finish`] to get the picture back as RGBA.
//!
//! Bindings only: the logic is `cvnn-runtime`'s. Results are written into typed
//! arrays the page keeps between frames: allocating megabytes of fresh array
//! per frame costs the page more than the arithmetic does.

use cvnn_runtime::coding::FrameCoding;
use cvnn_runtime::colour::yuv444_to_rgba;
use cvnn_runtime::{Input, ModelSpec, SR_SCALE, Session};
use js_sys::{Float32Array, Uint8ClampedArray};
use wasm_bindgen::prelude::*;

mod fixed;
mod integer;
mod sender;

const RGBA_BYTES: usize = 4;
/// Luma samples per chroma sample along each axis of a 4:2:0 frame.
const CHROMA_STEP: usize = 2;

/// The Y, U and V planes of a 4:2:0 frame.
type Planes420<'a> = (&'a [u8], &'a [u8], &'a [u8]);

/// One stream being upscaled in the page.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Upscaler {
    session: Session,
    width: usize,
    height: usize,
    /// A picture on its way to the page.
    rgba: Vec<u8>,
}

/// Copy `samples` into the page's `target`, which must be the same length.
fn deliver<T>(samples: &[T], length: u32, copy: impl FnOnce(&[T])) -> Result<(), JsError> {
    if length as usize != samples.len() {
        let message = format!("array holds {length} values, expected {}", samples.len());
        return Err(JsError::new(&message));
    }
    copy(samples);
    Ok(())
}

fn to_js_error(error: &impl core::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

#[wasm_bindgen]
impl Upscaler {
    /// A session for `width` x `height` input frames and a model with the
    /// given contract (the fields of its `model.json`).
    ///
    /// # Errors
    ///
    /// If the size is not one the network and analysis can take.
    #[wasm_bindgen(constructor)]
    #[allow(
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools,
        reason = "one argument per field of the contract"
    )]
    pub fn new(
        width: usize,
        height: usize,
        planes: usize,
        temporal: &str,
        align: bool,
        state_channels: usize,
        side_planes: bool,
        luma_only: bool,
        coding_planes: bool,
        confidence_plane: bool,
    ) -> Result<Upscaler, JsError> {
        let spec = ModelSpec {
            planes,
            temporal: temporal.parse().map_err(|e| to_js_error(&e))?,
            align,
            state_channels,
            input: Input::from_contract(luma_only, side_planes, coding_planes, confidence_plane)
                .map_err(|e| to_js_error(&e))?,
        };
        let session = Session::new(spec, width, height).map_err(|e| to_js_error(&e))?;
        let output_samples = SR_SCALE * width * SR_SCALE * height;
        Ok(Self {
            session,
            width,
            height,
            rgba: vec![0; output_samples * RGBA_BYTES],
        })
    }

    /// Channels of the network's input tensor.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.session.spec().input_channels()
    }

    /// The input channels that stay zero for the whole stream, as
    /// `[first, one past the last]`: the page uploads only the others.
    #[must_use]
    pub fn unused_input_channels(&self) -> Vec<usize> {
        let unused = self.session.spec().unused_input_channels();
        vec![unused.start, unused.end]
    }

    /// Channels of the hidden-state output (0 for a per-frame model).
    #[must_use]
    pub fn hidden_channels(&self) -> usize {
        self.session.spec().hidden_channels()
    }

    /// Analyse the next frame, given as 8-bit 4:2:0 (Y, then U, then V) of
    /// `width` x `height`, and write the network's input tensor (channels x
    /// height x width) to `input`.
    ///
    /// # Errors
    ///
    /// If `frame` or `input` is not the configured size.
    pub fn prepare(&mut self, frame: &[u8], input: &Float32Array) -> Result<(), JsError> {
        let (y, u, v) = self.planes(frame)?;
        let tensor = self.session.prepare(y, u, v).map_err(|e| to_js_error(&e))?;
        deliver(tensor, input.length(), |tensor| input.copy_from(tensor))
    }

    /// Say how the frames prepared from now on were coded: the average
    /// quantiser (0 to 51) and whether each is a key frame. Only a model with
    /// coding planes uses it.
    pub fn set_coding(&mut self, quantiser: f32, key_frame: bool) {
        self.session.set_coding(Some(FrameCoding {
            quantiser,
            key_frame,
        }));
    }

    /// Say that nothing is known about how the frames were coded (the state
    /// a session starts in), or that they never met a codec.
    pub fn clear_coding(&mut self) {
        self.session.set_coding(None);
    }

    /// Forget earlier frames: call after a seek, a source change or a failed
    /// network call.
    ///
    /// # Errors
    ///
    /// If the analysis cannot be restarted.
    pub fn restart(&mut self) -> Result<(), JsError> {
        self.session.restart().map_err(|e| to_js_error(&e))
    }

    /// Write the conventional (bicubic) 2x upscale of the prepared frame to
    /// `picture`, as RGBA.
    ///
    /// # Errors
    ///
    /// If `picture` is not the output size.
    pub fn conventional(&mut self, picture: &Uint8ClampedArray) -> Result<(), JsError> {
        yuv444_to_rgba(self.session.conventional(), &mut self.rgba);
        deliver(&self.rgba, picture.length(), |rgba| picture.copy_from(rgba))
    }

    /// Turn the network's outputs for the prepared frame into the 2x picture
    /// and write it to `picture`, as RGBA. `coarse` is empty for a per-frame
    /// model.
    ///
    /// # Errors
    ///
    /// If no frame is prepared, or a tensor or `picture` has the wrong size.
    pub fn finish(
        &mut self,
        fine: &[f32],
        coarse: &[f32],
        picture: &Uint8ClampedArray,
    ) -> Result<(), JsError> {
        self.finish_picture(fine, coarse, picture)
    }

    /// Width of the output picture.
    #[must_use]
    pub fn output_width(&self) -> usize {
        SR_SCALE * self.width
    }

    /// Height of the output picture.
    #[must_use]
    pub fn output_height(&self) -> usize {
        SR_SCALE * self.height
    }
}

impl Upscaler {
    /// The Y, U and V planes of a 4:2:0 `frame` of the configured size.
    fn planes<'a>(&self, frame: &'a [u8]) -> Result<Planes420<'a>, JsError> {
        let luma = self.width * self.height;
        let chroma = luma / (CHROMA_STEP * CHROMA_STEP);
        if frame.len() != luma + 2 * chroma {
            return Err(JsError::new("frame is not the configured size"));
        }
        let (y, colour) = frame.split_at(luma);
        let (u, v) = colour.split_at(chroma);
        Ok((y, u, v))
    }

    /// Finish the prepared frame from the network's float outputs, into `picture`.
    fn finish_picture(
        &mut self,
        fine: &[f32],
        coarse: &[f32],
        picture: &Uint8ClampedArray,
    ) -> Result<(), JsError> {
        let output = self.session.finish(fine, coarse);
        yuv444_to_rgba(output.map_err(|e| to_js_error(&e))?, &mut self.rgba);
        deliver(&self.rgba, picture.length(), |rgba| picture.copy_from(rgba))
    }
}
