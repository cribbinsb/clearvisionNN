//! Bilinear warping by a per-sample motion field.
//!
//! The work is split in two so that the first half becomes SIMD: turning
//! positions into sample indices and weights is plain arithmetic over a row;
//! only the lookups that follow are inherently one sample at a time.

use crate::fastmath::floor;

/// Where a run of output samples reads from, one entry per sample.
#[derive(Debug, Default)]
struct Taps {
    /// Index of the upper-left of the four source samples. Never on the last
    /// column or row, so the other three always exist.
    index: Vec<u32>,
    /// Weights of the right-hand and of the lower samples.
    fx: Vec<f32>,
    fy: Vec<f32>,
}

/// A plane's size and the flow's scale, as the tap arithmetic needs them.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    width: usize,
    /// The last column and row, as positions are clamped to them.
    max_x: f32,
    max_y: f32,
    /// What flow values are multiplied by to be in this plane's samples.
    flow_scale: f32,
}

impl Geometry {
    /// For planes of `height` x `width`, at least 2 x 2.
    fn new((height, width): (usize, usize), flow_scale: f32) -> Self {
        Self {
            width,
            max_x: (width - 1) as f32,
            max_y: (height - 1) as f32,
            flow_scale,
        }
    }
}

impl Taps {
    fn resize(&mut self, len: usize) {
        self.index.resize(len, 0);
        self.fx.resize(len, 0.0);
        self.fy.resize(len, 0.0);
    }

    /// Set the taps from `start` on to those of row `row`, displaced by the
    /// flow rows `dx` and `dy`.
    fn set_row(&mut self, start: usize, row: usize, dx: &[f32], dy: &[f32], geometry: Geometry) {
        let Geometry {
            width,
            max_x,
            max_y,
            flow_scale,
        } = geometry;
        let (last_x, last_y) = (max_x - 1.0, max_y - 1.0);
        let targets = self.index[start..]
            .iter_mut()
            .zip(&mut self.fx[start..])
            .zip(&mut self.fy[start..]);
        let flows = dx.iter().zip(dy).enumerate();
        for (((index, fx), fy), (column, (&dx, &dy))) in targets.zip(flows) {
            let x = (column as f32 + flow_scale * dx).clamp(0.0, max_x);
            let y = (row as f32 + flow_scale * dy).clamp(0.0, max_y);
            // A position on the last column or row is taken as a full step
            // from the one before it.
            let ((x0, whole_x), (y0, whole_y)) = (floor(x.min(last_x)), floor(y.min(last_y)));
            *index = y0 * width as u32 + x0;
            *fx = x - whole_x;
            *fy = y - whole_y;
        }
    }

    /// Sample `src`, a plane `width` wide, at every tap.
    fn sample(&self, src: &[f32], width: usize, out: &mut [f32]) {
        let taps = self.index.iter().zip(&self.fx).zip(&self.fy);
        for (value, ((&index, &fx), &fy)) in out.iter_mut().zip(taps) {
            // One slice spanning all four samples: a single bounds check.
            let window = &src[index as usize..][..width + 2];
            let above = window[0] * (1.0 - fx) + window[1] * fx;
            let below = window[width] * (1.0 - fx) + window[width + 1] * fx;
            *value = above * (1.0 - fy) + below * fy;
        }
    }
}

/// Warps single planes, with working buffers kept between calls.
#[derive(Debug, Default)]
pub struct Warper {
    row: Taps,
}

impl Warper {
    /// Sample `src` (`height` x `width`) at every position displaced by
    /// `flow_scale` times the flow (`dx`, `dy`), bilinearly, taking the
    /// nearest edge sample outside the plane. With the pre-process's
    /// convention this brings the previous frame onto the current one.
    pub fn warp(
        &mut self,
        src: &[f32],
        size: (usize, usize),
        (dx, dy): (&[f32], &[f32]),
        flow_scale: f32,
        out: &mut [f32],
    ) {
        let geometry = Geometry::new(size, flow_scale);
        let width = size.1;
        self.row.resize(width);
        let flows = dx.chunks_exact(width).zip(dy.chunks_exact(width));
        for (row, (target, (dx, dy))) in out.chunks_exact_mut(width).zip(flows).enumerate() {
            self.row.set_row(0, row, dx, dy, geometry);
            self.row.sample(src, width, target);
        }
    }
}

/// A flow prepared once for warping several planes of the same size.
#[derive(Debug, Default)]
pub struct WarpMap {
    width: usize,
    taps: Taps,
}

impl WarpMap {
    /// Prepare the warp [`Warper::warp`] would do with these arguments.
    pub fn set(&mut self, size: (usize, usize), (dx, dy): (&[f32], &[f32]), flow_scale: f32) {
        let geometry = Geometry::new(size, flow_scale);
        self.width = size.1;
        self.taps.resize(size.0 * size.1);
        let flows = dx.chunks_exact(self.width).zip(dy.chunks_exact(self.width));
        for (row, (dx, dy)) in flows.enumerate() {
            self.taps.set_row(row * self.width, row, dx, dy, geometry);
        }
    }

    /// Warp `src` into `out`, both of the size given to [`WarpMap::set`].
    pub fn apply(&self, src: &[f32], out: &mut [f32]) {
        self.taps.sample(src, self.width, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_moves_by_the_flow() {
        let mut src = [0.0f32; 64];
        src[5 * 8 + 6] = 1.0;
        let flow: (&[f32], &[f32]) = (&[1.0; 64], &[0.5; 64]);
        let mut out = [0.0f32; 64];
        Warper::default().warp(&src, (8, 8), flow, 2.0, &mut out);
        assert!((out[4 * 8 + 4] - 1.0).abs() < 1e-6);
        let mut map = WarpMap::default();
        map.set((8, 8), flow, 2.0);
        let mut mapped = [0.0f32; 64];
        map.apply(&src, &mut mapped);
        assert_eq!(out, mapped);
    }

    #[test]
    fn fractional_positions_are_interpolated() {
        let src = [0.0f32, 1.0, 2.0, 3.0];
        let mut out = [0.0f32; 4];
        Warper::default().warp(&src, (2, 2), (&[0.5; 4], &[0.0; 4]), 1.0, &mut out);
        assert_eq!(out, [0.5, 1.0, 2.5, 3.0]);
    }

    #[test]
    fn positions_outside_the_plane_take_the_edge_sample() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        Warper::default().warp(&src, (2, 2), (&[9.0; 4], &[9.0; 4]), 1.0, &mut out);
        assert_eq!(out, [4.0; 4]);
        Warper::default().warp(&src, (2, 2), (&[-9.0; 4], &[-9.0; 4]), 1.0, &mut out);
        assert_eq!(out, [1.0; 4]);
    }
}
