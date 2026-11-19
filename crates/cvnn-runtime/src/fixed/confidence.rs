//! The confidence plane in integers: how well the motion field predicts the
//! current luma from the previous one, per block, spread over the frame.

use cvnn_common::BLOCK_SIZE;

use super::motion::spread;
use super::warp::{Tap, WEIGHT_BITS};

/// Mean block errors, in 8-bit levels, at which confidence is full and at
/// which it is gone (`crate::confidence`'s thresholds).
const TRUSTED_LEVELS: i32 = 3;
const UNTRUSTED_LEVELS: i32 = 10;
/// Errors are kept in 2^-12 of a level: the scale of an unrounded warped sample.
const ERROR_BITS: u32 = 2 * WEIGHT_BITS;
/// The spread between block centres multiplies by 256.
const SPREAD_BITS: u32 = 8;
const FULL: i32 = 255;

/// Each block's mean absolute difference between `current` and the previous
/// luma warped onto it, over the samples the block really has.
pub(super) fn block_errors(
    current: &[u8],
    previous: &[u8],
    taps: &[Tap],
    (height, width): (usize, usize),
) -> Vec<i32> {
    let (rows, cols) = (height.div_ceil(BLOCK_SIZE), width.div_ceil(BLOCK_SIZE));
    let mut sums = vec![0; rows * cols];
    for (sample, tap) in taps.iter().enumerate() {
        let carried = tap.sum(|at| i32::from(previous[at]), width);
        let block = (sample / width / BLOCK_SIZE) * cols + (sample % width) / BLOCK_SIZE;
        sums[block] += ((i32::from(current[sample]) << ERROR_BITS) - carried).abs();
    }
    for (block, sum) in sums.iter_mut().enumerate() {
        let tall = BLOCK_SIZE.min(height - (block / cols) * BLOCK_SIZE);
        let wide = BLOCK_SIZE.min(width - (block % cols) * BLOCK_SIZE);
        *sum /= (tall * wide) as i32;
    }
    sums
}

/// The confidence of one sample from its spread block error: 255 at or below
/// the trusted error, falling linearly to 0 at the untrusted one.
pub(super) fn from_error(spread_error: i32) -> i32 {
    let error = (spread_error + (1 << (SPREAD_BITS - 1))) >> SPREAD_BITS;
    let span = (UNTRUSTED_LEVELS - TRUSTED_LEVELS) << ERROR_BITS;
    let above = ((UNTRUSTED_LEVELS << ERROR_BITS) - error).max(0);
    ((FULL * above + span / 2) / span).min(FULL)
}

/// The confidence plane (0 to 255) of a frame from its block errors.
pub(super) fn plane(errors: &[i32], (height, width): (usize, usize), out: &mut [i32]) {
    let grid = (height.div_ceil(BLOCK_SIZE), width.div_ceil(BLOCK_SIZE));
    spread(errors, grid, (height, width), out);
    for value in out.iter_mut() {
        *value = from_error(*value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_falls_from_full_to_none_between_the_thresholds() {
        let levels = |error: i32| from_error(error << (ERROR_BITS + SPREAD_BITS));
        assert_eq!(
            (levels(0), levels(3), levels(10), levels(40)),
            (255, 255, 0, 0)
        );
        assert!((120..=135).contains(&from_error(13 << (ERROR_BITS + SPREAD_BITS - 1))));
    }
}
