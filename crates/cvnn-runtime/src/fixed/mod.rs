//! Everything around the integer network in fixed point (docs/INTEGER.md,
//! "The runtime"): the conventional upscale, the motion field, warping
//! history onto the current frame, the confidence plane, adding the
//! network's detail, and colour conversion. With `integer`, this is the whole
//! of a frame in integers, so that a GPU can compute it without anything
//! crossing back to the processor, and be checked against it bit for bit.
//!
//! [`IntegerRuntime`] is the reference: one frame in, one picture out, the
//! recurrent state and the previous output kept inside.

mod colour;
mod confidence;
mod conventional;
mod motion;
mod runtime;
mod upscale;
mod warp;

pub use motion::{BlockVector, block_vectors, vector_steps};
pub use runtime::{FrameInput, IntegerRuntime, coding_levels};

/// Steps of a picture sample per 8-bit level, wherever more than 8 bits are
/// kept: the conventional upscale, the previous output, the output itself.
/// It is the integer network's residual step, so the two add without rounding.
pub const PICTURE_STEPS: i32 = crate::integer::RESIDUAL_STEPS;
/// The largest picture sample: level 255.
pub const PICTURE_MAX: i32 = 255 * PICTURE_STEPS;
/// A block's motion vector counts in this many steps per input sample.
pub const VECTOR_STEPS: i32 = 32;
