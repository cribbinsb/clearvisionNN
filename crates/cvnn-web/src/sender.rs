//! The simulated sender for the page: it turns the full-size original into
//! the low-resolution 4:2:0 frame a real sender would encode, with optional
//! blur, sensor noise and the camera's denoise. Bindings only: the logic is
//! `cvnn-runtime`'s.

use cvnn_runtime::SR_SCALE;
use cvnn_runtime::camera::{Camera, Exposure};
use cvnn_runtime::colour::{Yuv420, rgba_to_yuv420};
use cvnn_runtime::kernel::Kernel;
use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;

const RGBA_BYTES: usize = 4;

/// Makes the network's input picture from originals of one size.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Sender {
    width: usize,
    original_len: usize,
    camera: Camera,
    /// The reduced picture, RGBA.
    reduced: Vec<u8>,
    frame: Yuv420,
}

#[wasm_bindgen]
impl Sender {
    /// A sender for `width` x `height` originals (both multiples of four)
    /// that reduces with the kernel called `kernel` ("area", "bilinear",
    /// "bicubic" or "lanczos"); `seed` fixes the noise it will add.
    ///
    /// # Errors
    ///
    /// If there is no kernel of that name.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, kernel: &str, seed: u32) -> Result<Sender, JsError> {
        let kernel =
            Kernel::from_name(kernel).ok_or_else(|| JsError::new("unknown reduction kernel"))?;
        let camera = Camera::new(width, height, kernel, seed);
        Ok(Self {
            width,
            original_len: width * height * RGBA_BYTES,
            reduced: vec![0; camera.reduced_len()],
            camera,
            frame: Yuv420::new(width / SR_SCALE, height / SR_SCALE),
        })
    }

    /// Reduce the RGBA `original` to half size as a camera would deliver it:
    /// blurred by `blur` samples, with sensor noise of `noise` luma levels,
    /// denoised at strength `denoise` (each skipped at zero). The result is
    /// written to `frame` as 8-bit 4:2:0 (Y, then U, then V).
    ///
    /// # Errors
    ///
    /// If `original` or `frame` is not the configured size.
    pub fn send(
        &mut self,
        original: &[u8],
        blur: f32,
        noise: f32,
        denoise: f32,
        frame: &Uint8Array,
    ) -> Result<(), JsError> {
        if original.len() != self.original_len {
            return Err(JsError::new("original is not the configured size"));
        }
        let Yuv420 { y, u, v } = &self.frame;
        if frame.length() as usize != y.len() + u.len() + v.len() {
            return Err(JsError::new("frame array is not the 4:2:0 size"));
        }
        self.camera.capture(
            original,
            Exposure {
                blur,
                noise,
                denoise,
            },
            &mut self.reduced,
        );
        rgba_to_yuv420(&self.reduced, self.width / SR_SCALE, &mut self.frame);
        let Yuv420 { y, u, v } = &self.frame;
        frame.subarray(0, y.len() as u32).copy_from(y);
        let chroma_end = (y.len() + u.len()) as u32;
        frame.subarray(y.len() as u32, chroma_end).copy_from(u);
        frame.subarray(chroma_end, frame.length()).copy_from(v);
        Ok(())
    }
}
