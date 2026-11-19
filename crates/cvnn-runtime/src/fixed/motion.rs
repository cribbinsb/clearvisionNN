//! The block motion field in fixed point, and the per-sample flows made of it.

use cvnn_common::{BLOCK_SIZE, BlockGrid, MotionVector};

use super::VECTOR_STEPS;
use super::upscale::bilinear2x;

/// A block's displacement, each component in 1 / `VECTOR_STEPS` of an input sample.
pub type BlockVector = [i32; 2];

/// Steps of a per-sample flow per sample: a block vector's steps times the
/// 256ths of the bilinear spread between block centres.
pub(super) const FLOW_BITS: u32 = 13;
/// Weights between two block centres count in 16ths: twice the block size.
const SPREAD: i32 = 2 * BLOCK_SIZE as i32;

/// A motion vector on the fixed-point grid, each component to the nearest step.
#[must_use]
pub fn vector_steps(vector: MotionVector) -> BlockVector {
    let steps = |value: f32| (value * VECTOR_STEPS as f32).round() as i32;
    [steps(vector.dx), steps(vector.dy)]
}

/// The pre-process's motion field on the fixed-point grid, row by row.
#[must_use]
pub fn block_vectors(motion: &BlockGrid<MotionVector>) -> Vec<BlockVector> {
    motion.cells().iter().copied().map(vector_steps).collect()
}

/// The two cells a sample lies between along an axis of `cells` blocks, and
/// the weight of the second, in 16ths.
fn spread_tap(index: usize, cells: usize) -> (usize, usize, i32) {
    // The sample's position between block centres, in 16ths of a block.
    let position = 2 * index.cast_signed() + 1 - BLOCK_SIZE.cast_signed();
    let base = position >> 4;
    let cell = |at: isize| at.clamp(0, cells.cast_signed() - 1).cast_unsigned();
    (cell(base), cell(base + 1), (position - (base << 4)) as i32)
}

/// A block grid (`rows` x `cols`) spread bilinearly over a `height` x `width`
/// frame: each sample is the exact weighted sum, scaled by 256.
pub(super) fn spread(
    grid: &[i32],
    (rows, cols): (usize, usize),
    (height, width): (usize, usize),
    out: &mut [i32],
) {
    for (y, row) in out.chunks_exact_mut(width).enumerate().take(height) {
        let (top, bottom, down) = spread_tap(y, rows);
        for (x, value) in row.iter_mut().enumerate() {
            let (left, right, across) = spread_tap(x, cols);
            let along = |line: usize| {
                grid[line * cols + left] * (SPREAD - across) + grid[line * cols + right] * across
            };
            *value = along(top) * (SPREAD - down) + along(bottom) * down;
        }
    }
}

/// The motion of one frame as per-sample flows.
#[derive(Debug, Default)]
pub(super) struct Flow {
    /// One plane per component (dx, dy), in 2^-FLOW_BITS of a sample of the
    /// plane's own resolution: input, and output.
    pub(super) coarse: [Vec<i32>; 2],
    pub(super) fine: [Vec<i32>; 2],
}

impl Flow {
    /// Take the motion of `vectors`, the block field of `size` input frames.
    pub(super) fn set(&mut self, vectors: &[BlockVector], size: (usize, usize)) {
        let grid = (size.0.div_ceil(BLOCK_SIZE), size.1.div_ceil(BLOCK_SIZE));
        let samples = size.0 * size.1;
        for component in 0..2 {
            let blocks: Vec<i32> = vectors.iter().map(|vector| vector[component]).collect();
            let coarse = &mut self.coarse[component];
            coarse.resize(samples, 0);
            spread(&blocks, grid, size, coarse);
            let fine = &mut self.fine[component];
            fine.resize(4 * samples, 0);
            bilinear2x(coarse, size, fine);
            // Bilinear sums carry 16; an output sample is half an input one:
            // together a division by 8, to the nearest.
            for value in fine.iter_mut() {
                *value = (*value + 4) >> 3;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_field_gives_a_uniform_flow_at_both_resolutions() {
        let mut flow = Flow::default();
        flow.set(&[[VECTOR_STEPS, -VECTOR_STEPS / 2]; 4], (16, 16));
        assert!(flow.coarse[0].iter().all(|&dx| dx == 1 << FLOW_BITS));
        assert!(flow.fine[0].iter().all(|&dx| dx == 2 << FLOW_BITS));
        assert!(flow.fine[1].iter().all(|&dy| dy == -(1 << FLOW_BITS)));
    }

    #[test]
    fn spreading_meets_block_values_at_block_centres() {
        let mut out = [0; 16];
        spread(&[0, 8], (1, 2), (1, 16), &mut out);
        // Block centres are between samples 3 and 4, and 11 and 12.
        assert_eq!((out[3], out[12], out[7] + out[8]), (0, 8 * 256, 8 * 256));
    }
}
