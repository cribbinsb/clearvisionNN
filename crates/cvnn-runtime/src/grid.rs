//! Spreading a block grid over a frame.

use cvnn_common::BLOCK_SIZE;

/// How a block grid is spread over the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridFilter {
    /// Every sample takes its block's value.
    Nearest,
    /// Values are interpolated between block centres.
    Bilinear,
}

impl GridFilter {
    /// The two grid cells sample `index` lies between along an axis of `len`
    /// cells, and the weight of the second.
    fn tap(self, index: usize, len: usize) -> (usize, usize, f32) {
        match self {
            Self::Nearest => (index / BLOCK_SIZE, index / BLOCK_SIZE, 0.0),
            Self::Bilinear => {
                let position = (index as f32 + 0.5) / BLOCK_SIZE as f32 - 0.5;
                let floor = position.floor();
                let clamped = |cell: isize| cell.clamp(0, len as isize - 1) as usize;
                let base = floor as isize;
                (clamped(base), clamped(base + 1), position - floor)
            }
        }
    }
}

/// Spreads block grids over frames, with a working buffer kept between calls.
#[derive(Debug, Default)]
pub struct GridSpreader {
    /// Each grid row spread along the frame's width.
    wide_rows: Vec<f32>,
}

impl GridSpreader {
    /// Spread a block grid (`rows` x `cols`) over a `height` x `width` frame.
    /// Blocks on the right and bottom edges may overhang the frame.
    pub fn spread(
        &mut self,
        grid: &[f32],
        (rows, cols): (usize, usize),
        (height, width): (usize, usize),
        filter: GridFilter,
        out: &mut [f32],
    ) {
        self.wide_rows.resize(rows * width, 0.0);
        for (cells, wide) in grid
            .chunks_exact(cols)
            .zip(self.wide_rows.chunks_exact_mut(width))
        {
            for (x, value) in wide.iter_mut().enumerate() {
                let (left, right, fraction) = filter.tap(x, cols);
                *value = cells[left] * (1.0 - fraction) + cells[right] * fraction;
            }
        }
        for (y, target) in out.chunks_exact_mut(width).enumerate().take(height) {
            let (top, bottom, fraction) = filter.tap(y, rows);
            let upper = &self.wide_rows[top * width..][..width];
            let lower = &self.wide_rows[bottom * width..][..width];
            for ((value, &above), &below) in target.iter_mut().zip(upper).zip(lower) {
                *value = above * (1.0 - fraction) + below * fraction;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_repeats_each_block() {
        let grid = [1.0, 2.0];
        let mut out = [0.0f32; 12 * 4];
        GridSpreader::default().spread(&grid, (1, 2), (4, 12), GridFilter::Nearest, &mut out);
        assert!((out[7] - 1.0).abs() < 1e-6 && (out[8] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn bilinear_meets_block_values_at_block_centres() {
        let grid = [0.0, 8.0];
        let mut out = [0.0f32; 16];
        GridSpreader::default().spread(&grid, (1, 2), (1, 16), GridFilter::Bilinear, &mut out);
        // Block centres are at 3.5 and 11.5; the midpoint between them is 7.5.
        assert!(out[3].abs() < 1e-6 && (out[12] - 8.0).abs() < 1e-6);
        assert!((out[7] + out[8] - 8.0).abs() < 1e-5);
    }
}
