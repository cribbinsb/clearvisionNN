//! Everything around the network at run time (DESIGN.md section 8.1): the
//! conventional upscale, the pre-process, warping history onto the current
//! frame, assembling the network's input, and blending its output.
//!
//! The network itself runs elsewhere (ONNX Runtime, a browser's `WebGPU` or
//! `WebNN`). A [`Session`] therefore works in two halves around that call:
//! [`Session::prepare`] returns the input tensor, and [`Session::finish`]
//! takes the network's outputs and returns the picture. It is the only
//! implementation of this logic outside training, and `cvnn-py` checks it
//! against the training model.

#![forbid(unsafe_code)]

pub mod blur;
pub mod camera;
pub mod coding;
pub mod colour;
mod compose;
pub mod confidence;
pub mod denoise;
mod fastmath;
pub mod fixed;
mod flow;
pub mod grid;
mod history;
pub mod integer;
pub mod kernel;
pub mod noise;
mod planes;
pub mod reduce;
pub mod resample;
mod session;
mod spec;
pub mod warp;

pub use planes::Planes;
pub use session::Session;
pub use spec::{Input, ModelSpec, Temporal};

/// Output size over input size.
pub const SR_SCALE: usize = 2;
