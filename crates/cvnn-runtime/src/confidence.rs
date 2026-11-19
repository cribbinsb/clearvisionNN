//! How well the motion field predicts the current frame from the previous
//! one: the plane that lets a recurrent network distrust its history where
//! motion estimation fails (water, occlusions, an undetected cut).

use cvnn_common::BLOCK_SIZE;

use crate::grid::{GridFilter, GridSpreader};

/// Mean prediction error of a block (samples in [0, 1]) at or below which the
/// motion is fully trusted: about the residual sensor noise leaves.
pub const TRUSTED_ERROR: f32 = 3.0 / 255.0;
/// Mean prediction error at or above which the motion is not trusted at all.
pub const UNTRUSTED_ERROR: f32 = 10.0 / 255.0;

/// Computes confidence planes, with working buffers kept between frames.
#[derive(Debug, Default)]
pub(crate) struct Confidence {
    /// Mean absolute prediction error of each block.
    blocks: Vec<f32>,
    spreader: GridSpreader,
}

impl Confidence {
    /// Write to `out` the confidence of each sample of a `height` x `width`
    /// frame: 1 where `carried` (the previous frame brought onto this one)
    /// matches `current`, falling to 0 as the block's mean error grows.
    pub(crate) fn write(
        &mut self,
        current: &[f32],
        carried: &[f32],
        (height, width): (usize, usize),
        out: &mut [f32],
    ) {
        let (rows, cols) = (height.div_ceil(BLOCK_SIZE), width.div_ceil(BLOCK_SIZE));
        self.blocks.clear();
        self.blocks.resize(rows * cols, 0.0);
        let lines = current.chunks_exact(width).zip(carried.chunks_exact(width));
        for (y, (now, before)) in lines.enumerate() {
            let sums = &mut self.blocks[(y / BLOCK_SIZE) * cols..][..cols];
            let pairs = now.chunks(BLOCK_SIZE).zip(before.chunks(BLOCK_SIZE));
            for (sum, (now, before)) in sums.iter_mut().zip(pairs) {
                *sum += now
                    .iter()
                    .zip(before)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>();
            }
        }
        // Blocks on the right and bottom edges may overhang the frame: each
        // mean is over the samples the block really has.
        for (row, sums) in self.blocks.chunks_exact_mut(cols).enumerate() {
            let tall = BLOCK_SIZE.min(height - row * BLOCK_SIZE);
            for (col, sum) in sums.iter_mut().enumerate() {
                let wide = BLOCK_SIZE.min(width - col * BLOCK_SIZE);
                *sum /= (tall * wide) as f32;
            }
        }
        let size = (height, width);
        self.spreader
            .spread(&self.blocks, (rows, cols), size, GridFilter::Bilinear, out);
        let span = UNTRUSTED_ERROR - TRUSTED_ERROR;
        for value in out.iter_mut() {
            *value = (1.0 - (*value - TRUSTED_ERROR) / span).clamp(0.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: usize = 2 * BLOCK_SIZE;

    #[test]
    fn a_perfect_prediction_is_fully_trusted() {
        let frame = vec![0.5; SIDE * SIDE];
        let mut out = vec![0.0; SIDE * SIDE];
        Confidence::default().write(&frame, &frame, (SIDE, SIDE), &mut out);
        assert!(out.iter().all(|&value| (value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn a_failed_prediction_is_not_trusted() {
        let (now, before) = (vec![0.5; SIDE * SIDE], vec![0.0; SIDE * SIDE]);
        let mut out = vec![1.0; SIDE * SIDE];
        Confidence::default().write(&now, &before, (SIDE, SIDE), &mut out);
        assert!(out.iter().all(|&value| value == 0.0));
    }
}
