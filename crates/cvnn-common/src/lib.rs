//! Types shared by every clearvisionNN crate: the block grid the pre-process
//! works on, the per-frame features it hands to the network, and errors.
//!
//! Lowest layer of the workspace: depends on nothing else in it.

#![forbid(unsafe_code)]

mod block_grid;
mod error;
mod features;

pub use block_grid::{BLOCK_SIZE, BlockGrid};
pub use error::{Error, Result};
pub use features::{FrameFeatures, MotionVector};
