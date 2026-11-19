//! An owned luma frame with edge-replicating access, for the block
//! differences the refinement measures outside nanocodec's search.

use cvnn_common::BLOCK_SIZE;

/// One frame's luma samples, row-major.
#[derive(Debug, Clone)]
pub(crate) struct LumaFrame {
    width: usize,
    height: usize,
    samples: Vec<u8>,
}

impl LumaFrame {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            samples: vec![0; width * height],
        }
    }

    pub(crate) fn copy_from(&mut self, samples: &[u8]) {
        self.samples.copy_from_slice(samples);
    }

    pub(crate) fn samples(&self) -> &[u8] {
        &self.samples
    }

    /// The sample at (`x`, `y`), coordinates clamped to the frame.
    fn clamped(&self, x: isize, y: isize) -> u8 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.samples[y * self.width + x]
    }

    /// Mean absolute difference between block (`col`, `row`) of `self` and
    /// the block of `reference` displaced by (`dx`, `dy`) whole samples.
    /// Only the part of the block inside the frame counts.
    pub(crate) fn block_mean_abs_diff(
        &self,
        reference: &LumaFrame,
        (col, row): (usize, usize),
        (dx, dy): (isize, isize),
    ) -> f32 {
        let (x0, y0) = (col * BLOCK_SIZE, row * BLOCK_SIZE);
        let (x1, y1) = (
            (x0 + BLOCK_SIZE).min(self.width),
            (y0 + BLOCK_SIZE).min(self.height),
        );
        let mut sum = 0u32;
        for y in y0..y1 {
            let current = &self.samples[y * self.width + x0..y * self.width + x1];
            for (x, &sample) in (x0..x1).zip(current) {
                let matched = reference.clamped(x as isize + dx, y as isize + dy);
                sum += u32::from(sample.abs_diff(matched));
            }
        }
        sum as f32 / ((x1 - x0) * (y1 - y0)) as f32
    }
}
