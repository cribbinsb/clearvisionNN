//! What the pre-process reports for one frame: the motion field and the
//! side planes the network is given (DESIGN.md section 8.3).

use crate::block_grid::BlockGrid;

/// Displacement of a block, in luma samples of the analysed frame, with
/// sub-sample precision. The matching block of the previous frame is at
/// `position + vector`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MotionVector {
    /// Horizontal displacement (positive: the match is to the right).
    pub dx: f32,
    /// Vertical displacement (positive: the match is below).
    pub dy: f32,
}

/// The pre-process output for one frame, on the 8x8 block grid.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameFeatures {
    /// Motion towards the previous frame. Zero when there is no reference.
    pub motion: BlockGrid<MotionVector>,
    /// Mean absolute luma difference per sample at the chosen vector.
    pub match_error: BlockGrid<f32>,
    /// Consecutive frames each block has been reliably still (0: not still).
    pub still_run: BlockGrid<u8>,
    /// Running sensor-noise estimate, in luma sample units.
    pub noise_sigma: f32,
    /// Mean luma of the frame.
    pub mean_luma: f32,
    /// History from before this frame should be discarded: the first frame,
    /// a scene cut or an abrupt lighting change.
    pub reset: bool,
}
