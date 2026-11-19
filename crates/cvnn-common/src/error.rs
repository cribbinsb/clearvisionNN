//! The error type of the clearvisionNN crates.

use core::fmt;

/// What can go wrong in the pre-process and runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Frame dimensions the analysis cannot work on.
    InvalidDimensions {
        /// Requested width in luma samples.
        width: usize,
        /// Requested height in luma samples.
        height: usize,
        /// Why they were rejected.
        reason: String,
    },
    /// A buffer whose length does not match the configured size.
    FrameSizeMismatch {
        /// Samples a frame of the configured size holds.
        expected: usize,
        /// Samples the caller supplied.
        actual: usize,
    },
    /// A model contract the runtime cannot honour.
    InvalidSpec(String),
    /// A call made at the wrong point of a two-step sequence.
    OutOfOrder(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions {
                width,
                height,
                reason,
            } => {
                write!(f, "invalid frame size {width}x{height}: {reason}")
            }
            Error::FrameSizeMismatch { expected, actual } => {
                write!(f, "buffer has {actual} samples, expected {expected}")
            }
            Error::InvalidSpec(why) => write!(f, "invalid model contract: {why}"),
            Error::OutOfOrder(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result with the clearvisionNN [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
