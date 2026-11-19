//! The arithmetic between the network's tensors and pictures: packing history
//! into channels, and blending a new estimate with history.

use crate::SR_SCALE;
use crate::fastmath::sigmoid;

/// Fold each `SR_SCALE` x `SR_SCALE` block of a `2 * height` x `2 * width`
/// plane into channels, in the order of `PyTorch`'s `pixel_unshuffle` (row
/// offset, then column offset). `out` holds `SR_SCALE * SR_SCALE` planes of
/// `height` x `width`.
pub fn pixel_unshuffle(src: &[f32], (height, width): (usize, usize), out: &mut [f32]) {
    let plane = height * width;
    for (offset, target) in out.chunks_exact_mut(plane).enumerate() {
        let (down, across) = (offset / SR_SCALE, offset % SR_SCALE);
        let rows = src
            .chunks_exact(SR_SCALE * width)
            .skip(down)
            .step_by(SR_SCALE);
        for (target, row) in target.chunks_exact_mut(width).zip(rows) {
            let samples = row[across..].iter().step_by(SR_SCALE);
            for (value, &sample) in target.iter_mut().zip(samples) {
                *value = sample;
            }
        }
    }
}

/// `out = weight * previous + (1 - weight) * (base + detail)`, the weight
/// being `sigmoid(logit)`. A convex blend: the result stays between history
/// and the new estimate.
pub fn blend(base: &[f32], detail: &[f32], previous: &[f32], logit: &[f32], out: &mut [f32]) {
    let estimates = base.iter().zip(detail);
    let histories = previous.iter().zip(logit);
    for ((value, (&base, &detail)), (&previous, &logit)) in
        out.iter_mut().zip(estimates).zip(histories)
    {
        let weight = sigmoid(logit);
        *value = weight * previous + (1.0 - weight) * (base + detail);
    }
}

/// `out = base + detail`: the blend when there is no usable history.
pub fn add_detail(base: &[f32], detail: &[f32], out: &mut [f32]) {
    for ((value, &base), &detail) in out.iter_mut().zip(base).zip(detail) {
        *value = base + detail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_orders_channels_by_row_then_column() {
        let src = [0.0, 1.0, 2.0, 3.0];
        let mut out = [9.0; 4];
        pixel_unshuffle(&src, (1, 1), &mut out);
        assert_eq!(out, src);
    }

    #[test]
    fn unshuffle_keeps_each_offset_in_its_own_plane() {
        // A 4 x 4 plane whose samples are their own index.
        let src: Vec<f32> = (0u8..16).map(f32::from).collect();
        let mut out = [0.0; 16];
        pixel_unshuffle(&src, (2, 2), &mut out);
        assert_eq!(out[..4], [0.0, 2.0, 8.0, 10.0]);
        assert_eq!(out[12..], [5.0, 7.0, 13.0, 15.0]);
    }

    #[test]
    fn blend_is_convex() {
        let mut out = [0.0];
        blend(&[0.2], &[0.1], &[0.9], &[0.0], &mut out);
        assert!((out[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn without_history_the_estimate_stands() {
        let mut out = [0.0];
        add_detail(&[0.2], &[0.1], &mut out);
        assert!((out[0] - 0.3).abs() < 1e-6);
    }
}
