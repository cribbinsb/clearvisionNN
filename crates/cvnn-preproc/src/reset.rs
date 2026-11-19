//! When history must be discarded: a scene cut, a stream restart or an
//! abrupt lighting change. The network is trained to be safe after a reset
//! (DESIGN.md section 8.5), so false positives cost a little quality and
//! misses cost ghosting; the limits lean towards resetting.

/// Limits of the reset decision. Provisional: to be tuned on the flicker
/// suite's scene-cut and lighting-change clips.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResetConfig {
    /// A change of frame mean luma beyond this (in sample units) resets.
    pub mean_luma_jump: f32,
    /// A block whose match error exceeds this (mean absolute difference per
    /// sample) counts as failed.
    pub failed_block_error: f32,
    /// A reset is declared when more than this share of blocks failed.
    pub failed_block_share: f32,
}

impl Default for ResetConfig {
    fn default() -> Self {
        Self {
            mean_luma_jump: 24.0,
            failed_block_error: 12.0,
            failed_block_share: 0.6,
        }
    }
}

impl ResetConfig {
    /// Whether a frame with these measurements starts new history.
    #[must_use]
    pub fn is_reset(&self, previous_mean_luma: f32, mean_luma: f32, match_errors: &[f32]) -> bool {
        if (mean_luma - previous_mean_luma).abs() > self.mean_luma_jump {
            return true;
        }
        let failed = match_errors
            .iter()
            .filter(|&&e| e > self.failed_block_error)
            .count();
        failed as f32 > self.failed_block_share * match_errors.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_frames_keep_history() {
        assert!(!ResetConfig::default().is_reset(100.0, 101.0, &[0.5, 1.0, 2.0]));
    }

    #[test]
    fn a_brightness_jump_resets() {
        assert!(ResetConfig::default().is_reset(60.0, 120.0, &[0.5; 8]));
    }

    #[test]
    fn widespread_match_failure_resets() {
        assert!(ResetConfig::default().is_reset(100.0, 100.0, &[30.0, 30.0, 30.0, 1.0]));
    }
}
