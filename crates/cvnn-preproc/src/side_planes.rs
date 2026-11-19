//! The side planes the network is given (DESIGN.md section 8.3), scaled to
//! roughly [0, 1], each followed by a flag channel saying whether it was
//! supplied.
//!
//! The scales are part of the network's input contract: changing one
//! invalidates trained weights. This is the only place they are defined;
//! training and every runtime call it.

use cvnn_common::FrameFeatures;

/// Planes, in channel order: match error; match error in multiples of the
/// noise level; noise level; frames a block has been still.
pub const PLANES: usize = 4;
/// Channels written per block: each plane and its flag.
pub const SIDE_CHANNELS: usize = 2 * PLANES;

/// Divisor of the raw match error (luma units).
const RAW_ERROR_SCALE: f32 = 16.0;
/// Divisor of the match error once expressed in multiples of the noise level.
const NOISE_UNIT_ERROR_SCALE: f32 = 8.0;
/// Divisor of the noise sigma (luma units).
const NOISE_SCALE: f32 = 16.0;
/// Divisor of the still-run length (frames).
const STILL_RUN_SCALE: f32 = 32.0;
/// Added to the noise level before dividing by it: 8-bit quantisation alone
/// gives differences of this order.
const NOISE_FLOOR: f32 = 1.0;

/// Write the side planes of `features` into `out`, channel-major
/// (`SIDE_CHANNELS` x rows x cols). `withheld[i]` blanks plane `i` and clears
/// its flag, as when a deployment cannot supply it.
///
/// # Panics
///
/// If `out` is not `SIDE_CHANNELS * rows * cols` long.
pub fn write_side_planes(features: &FrameFeatures, withheld: [bool; PLANES], out: &mut [f32]) {
    let blocks = features.match_error.cells().len();
    assert_eq!(
        out.len(),
        SIDE_CHANNELS * blocks,
        "side-plane buffer has the wrong size"
    );
    let sigma = features.noise_sigma;
    for (index, (&error, &still)) in features
        .match_error
        .cells()
        .iter()
        .zip(features.still_run.cells())
        .enumerate()
    {
        let values = [
            error / RAW_ERROR_SCALE,
            error / (sigma + NOISE_FLOOR) / NOISE_UNIT_ERROR_SCALE,
            sigma / NOISE_SCALE,
            f32::from(still) / STILL_RUN_SCALE,
        ];
        for (plane, value) in values.into_iter().enumerate() {
            let supplied = !withheld[plane];
            out[(2 * plane) * blocks + index] = if supplied { value.clamp(0.0, 1.0) } else { 0.0 };
            out[(2 * plane + 1) * blocks + index] = f32::from(u8::from(supplied));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cvnn_common::{BlockGrid, MotionVector};

    fn features() -> FrameFeatures {
        FrameFeatures {
            motion: BlockGrid::for_frame(16, 8, MotionVector::default()),
            match_error: BlockGrid::for_frame(16, 8, 4.0),
            still_run: BlockGrid::for_frame(16, 8, 16),
            noise_sigma: 3.0,
            mean_luma: 100.0,
            reset: false,
        }
    }

    #[test]
    fn planes_are_scaled_and_flagged() {
        let mut out = [0.0; SIDE_CHANNELS * 2];
        write_side_planes(&features(), [false; PLANES], &mut out);
        assert_eq!(
            out,
            [
                0.25, 0.25, 1.0, 1.0, 0.125, 0.125, 1.0, 1.0, 0.1875, 0.1875, 1.0, 1.0, 0.5, 0.5,
                1.0, 1.0
            ]
        );
    }

    #[test]
    fn a_withheld_plane_is_zero_with_a_cleared_flag() {
        let mut out = [9.0; SIDE_CHANNELS * 2];
        write_side_planes(&features(), [true, false, false, false], &mut out);
        assert_eq!(out[..4], [0.0; 4]);
        assert_eq!(out[4], 0.125);
    }
}
