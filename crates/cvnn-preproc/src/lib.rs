//! The pre-process that runs in front of the network (DESIGN.md section
//! 8.1): for each incoming frame, a motion field towards the previous frame
//! and the side planes of section 8.3.
//!
//! Motion search, stillness and the noise estimate are nanocodec's source
//! analysis (`nc-analysis`). This crate adds what super-resolution needs on
//! top: [`subpel`] refinement of the whole-sample vectors, and the [`reset`]
//! signal. It is the only implementation: native code, the browser
//! (WebAssembly) and training (Python bindings) all call it.

#![forbid(unsafe_code)]

mod config;
mod luma;
mod preprocessor;
pub mod reset;
pub mod side_planes;
pub mod subpel;

pub use config::PreprocConfig;
pub use preprocessor::Preprocessor;
pub use reset::ResetConfig;
pub use side_planes::{PLANES, SIDE_CHANNELS, write_side_planes};
