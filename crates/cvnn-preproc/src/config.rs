//! Settings of the pre-process.

use crate::reset::ResetConfig;

/// What the [`crate::Preprocessor`] computes beyond nanocodec's analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocConfig {
    /// Refine whole-sample vectors to sub-sample precision
    /// ([`crate::subpel`]). The warp of 2x history needs it: a whole-sample
    /// error at input resolution is two samples at output resolution.
    pub subpel: bool,
    /// When history is declared unusable.
    pub reset: ResetConfig,
}

impl Default for PreprocConfig {
    fn default() -> Self {
        Self {
            subpel: true,
            reset: ResetConfig::default(),
        }
    }
}
