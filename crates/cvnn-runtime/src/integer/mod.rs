//! The network in 8-bit integer arithmetic (docs/INTEGER.md is the
//! specification): the reference every other implementation of it, the page's
//! WGSL shaders first of all, is tested against bit for bit.
//!
//! [`IntegerModel`] is an exported model file, validated on load.
//! [`IntegerModel::pack_input`] turns the float network input the session
//! prepares into packed 8-bit planes, [`IntegerModel::infer`] runs the
//! network on them, and [`IntegerModel::unpack_output`] turns its packed
//! outputs back into the float outputs the session finishes a frame from.
//! `infer` is written to be read and checked, not to be fast.

mod arith;
mod codec;
mod format;
mod infer;
mod layers;
mod tiles;

pub use format::IntegerModel;

/// First word of a model file: "FCI8", little-endian.
pub const MAGIC: i32 = 0x3849_4346;
/// Version of the file layout and of the arithmetic it implies; raise on any change to either.
pub const SCHEMA_VERSION: i32 = 1;
/// Words of the fixed header at the start of a model file.
pub const HEADER_WORDS: usize = 16;
/// Channels packed into one 32-bit word, a byte each.
pub const WORD_CHANNELS: usize = 4;
/// Steps of an unsigned plane (pictures, flags, activations) from 0 to 1 or to the layer's top.
pub const UNSIGNED_LEVELS: i32 = 255;
/// Steps of a signed plane (the recurrent state) from 0 to 1.
pub const SIGNED_LEVELS: i32 = 127;
/// An unsigned value is stored as a signed byte, this much below its value,
/// so that one signed packed dot product serves every plane.
pub const STORED_OFFSET: i32 = 128;
/// The multiplier of a requantisation is below 2 to this power, which keeps
/// its product with the accumulator's upper half inside 32 bits.
pub const MULTIPLIER_BITS: u32 = 14;
/// The picture outputs count in this many steps per 8-bit level: fine enough
/// that their rounding (1/64 of a level) cannot be seen, and within 16 bits.
pub const RESIDUAL_STEPS: i32 = 32;
/// The state outputs enter the tanh table in this many steps per unit.
pub const TANH_STEPS: i32 = 1024;
/// Output words per pixel holding the picture outputs: four 16-bit values.
pub const PICTURE_WORDS: usize = 2;
/// Picture outputs per input pixel: the 2x2 output pixels it covers.
pub const PICTURE_OUTPUTS: usize = 4;
/// Bits of each state value, a field of the file so that a later model can
/// carry a finer state; 8 is the only width this implementation runs.
pub const STATE_BITS: i32 = 8;
