//! The block motion field as per-sample warps at input and at output
//! resolution.

use cvnn_common::{FrameFeatures, MotionVector};

use crate::SR_SCALE;
use crate::grid::{GridFilter, GridSpreader};
use crate::resample::Resampler;
use crate::warp::{WarpMap, Warper};

/// The motion field of one frame, ready to warp history onto it.
#[derive(Debug, Default)]
pub(crate) struct Flow {
    /// Input frame size (height, width).
    size: (usize, usize),
    /// One component (dx, dy) per sample, in input samples, at input and at
    /// output resolution.
    coarse_field: [Vec<f32>; 2],
    fine_field: [Vec<f32>; 2],
    /// The warp at input resolution, shared by every hidden-state channel.
    coarse: WarpMap,
    /// For planes at output resolution: there are few (the reconstructed
    /// planes), so their warp is not prepared ahead.
    fine: Warper,
    blocks: Vec<f32>,
    spreader: GridSpreader,
}

impl Flow {
    /// Take the motion of `features`, for input frames of `size`.
    pub(crate) fn set(
        &mut self,
        features: &FrameFeatures,
        size: (usize, usize),
        resampler: &mut Resampler,
    ) {
        let grid = (features.motion.rows(), features.motion.cols());
        let samples = size.0 * size.1;
        let components: [fn(&MotionVector) -> f32; 2] = [|v| v.dx, |v| v.dy];
        for (component, pick) in components.into_iter().enumerate() {
            self.blocks.clear();
            self.blocks.extend(features.motion.cells().iter().map(pick));
            let coarse = &mut self.coarse_field[component];
            coarse.resize(samples, 0.0);
            self.spreader
                .spread(&self.blocks, grid, size, GridFilter::Bilinear, coarse);
            let fine = &mut self.fine_field[component];
            fine.resize(samples * SR_SCALE * SR_SCALE, 0.0);
            resampler.bilinear2x(coarse, size, fine);
        }
        let [dx, dy] = &self.coarse_field;
        self.coarse.set(size, (dx, dy), 1.0);
        self.size = size;
    }

    /// Warp a plane at input resolution.
    pub(crate) fn warp_coarse(&self, src: &[f32], out: &mut [f32]) {
        self.coarse.apply(src, out);
    }

    /// Warp a plane at output resolution.
    pub(crate) fn warp_fine(&mut self, src: &[f32], out: &mut [f32]) {
        let size = (SR_SCALE * self.size.0, SR_SCALE * self.size.1);
        let [dx, dy] = &self.fine_field;
        // Vectors are in input samples; at output resolution they are longer.
        self.fine.warp(src, size, (dx, dy), SR_SCALE as f32, out);
    }
}
