//! clearvisionNN: fast, causal 2x video super-resolution.
//!
//! This crate gathers the others under one name: [`common`] (shared types),
//! [`preproc`] (the motion search and side planes that run in front of the
//! network) and [`runtime`] (everything around the network per stream, and the
//! reference implementation of the 8-bit integer network). With the `models`
//! feature it also carries the released models, as [`models`]; those are
//! licensed for non-commercial use, unlike the code.

#![forbid(unsafe_code)]

pub use cvnn_common as common;
/// The released 8-bit models (the `models` feature; non-commercial licence).
#[cfg(feature = "models")]
pub use cvnn_models as models;
pub use cvnn_preproc as preproc;
pub use cvnn_runtime as runtime;
