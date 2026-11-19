//! The 2x reduction that makes a network input from a full-size picture: the
//! antialiased filters the training data was reduced with (Pillow's
//! `Image.resize`), reproduced in its 8-bit fixed-point arithmetic so that a
//! picture reduced here is what the network was trained on.

use crate::SR_SCALE;
use crate::kernel::Kernel;

const RGBA_BYTES: usize = 4;
const COLOUR_BYTES: usize = 3;
const OPAQUE: u8 = 255;
/// Fractional bits of the fixed-point weights: what is left of 32 bits after
/// an 8-bit sample and two bits of headroom for the kernel's negative lobes.
const WEIGHT_BITS: u32 = 22;
const HALF: i32 = 1 << (WEIGHT_BITS - 1);

/// The input samples one output sample is a weighted sum of.
#[derive(Debug, Clone)]
struct Taps {
    first: usize,
    weights: Vec<i32>,
}

/// Taps for every output sample of a line of `length` input samples. The
/// kernel is stretched by the reduction, which is what antialiases it; at the
/// ends it is cut at the picture's edge and its weights rescaled.
fn taps(length: usize, kernel: Kernel) -> Vec<Taps> {
    let scale = SR_SCALE as f64;
    let support = kernel.support() * scale;
    (0..length / SR_SCALE)
        .map(|index| {
            let centre = (index as f64 + 0.5) * scale;
            let first = (centre - support + 0.5).max(0.0) as usize;
            let end = ((centre + support + 0.5) as usize).min(length);
            let raw: Vec<f64> = (first..end)
                .map(|sample| kernel.weight((sample as f64 - centre + 0.5) / scale))
                .collect();
            let total: f64 = raw.iter().sum();
            let fixed = |weight: &f64| {
                let scaled = weight / total * f64::from(1_u32 << WEIGHT_BITS);
                (scaled + 0.5_f64.copysign(scaled)) as i32
            };
            Taps {
                first,
                weights: raw.iter().map(fixed).collect(),
            }
        })
        .collect()
}

fn to_byte(sum: i32) -> u8 {
    (sum >> WEIGHT_BITS).clamp(0, i32::from(OPAQUE)) as u8
}

/// Halves pictures of one size, keeping its tables and scratch between frames.
#[derive(Debug, Clone)]
pub struct Reducer {
    width: usize,
    height: usize,
    across: Vec<Taps>,
    down: Vec<Taps>,
    /// The picture after the horizontal pass: half width, full height. It is
    /// rounded to 8 bits between the passes, as Pillow rounds it.
    narrowed: Vec<u8>,
}

impl Reducer {
    /// A reducer for `width` x `height` pictures (both even), filtering with `kernel`.
    #[must_use]
    pub fn new(width: usize, height: usize, kernel: Kernel) -> Self {
        Self {
            width,
            height,
            across: taps(width, kernel),
            down: taps(height, kernel),
            narrowed: vec![0; width / SR_SCALE * height * RGBA_BYTES],
        }
    }

    /// Write the half-size picture of `rgba` to `reduced`, both RGBA; alpha
    /// is ignored and the result is opaque.
    ///
    /// # Panics
    ///
    /// If either slice is not the size the reducer was made for.
    pub fn reduce(&mut self, rgba: &[u8], reduced: &mut [u8]) {
        let (half_width, half_height) = (self.width / SR_SCALE, self.height / SR_SCALE);
        assert_eq!(rgba.len(), self.width * self.height * RGBA_BYTES);
        assert_eq!(reduced.len(), half_width * half_height * RGBA_BYTES);
        let rows = rgba.chunks_exact(self.width * RGBA_BYTES);
        let narrow_rows = self.narrowed.chunks_exact_mut(half_width * RGBA_BYTES);
        for (row, narrow) in rows.zip(narrow_rows) {
            let pixels = row.as_chunks::<RGBA_BYTES>().0;
            let narrow_pixels = narrow.as_chunks_mut::<RGBA_BYTES>().0;
            for (taps, out) in self.across.iter().zip(narrow_pixels) {
                let mut sums = [HALF; COLOUR_BYTES];
                for (weight, pixel) in taps.weights.iter().zip(&pixels[taps.first..]) {
                    for (sum, &sample) in sums.iter_mut().zip(pixel) {
                        *sum += weight * i32::from(sample);
                    }
                }
                for (byte, sum) in out.iter_mut().zip(sums) {
                    *byte = to_byte(sum);
                }
            }
        }
        let stride = half_width * RGBA_BYTES;
        let mut sums = vec![HALF; stride];
        for (taps, out) in self.down.iter().zip(reduced.chunks_exact_mut(stride)) {
            sums.fill(HALF);
            let lines = self.narrowed[taps.first * stride..].chunks_exact(stride);
            for (weight, line) in taps.weights.iter().zip(lines) {
                for (sum, &sample) in sums.iter_mut().zip(line) {
                    *sum += weight * i32::from(sample);
                }
            }
            let pixels = out.as_chunks_mut::<RGBA_BYTES>().0.iter_mut();
            for (pixel, sum) in pixels.zip(sums.as_chunks::<RGBA_BYTES>().0) {
                for channel in 0..COLOUR_BYTES {
                    pixel[channel] = to_byte(sum[channel]);
                }
                pixel[COLOUR_BYTES] = OPAQUE;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_weights_are_the_antialiased_cubic() {
        let weights = &taps(64, Kernel::Bicubic)[8].weights;
        let unit = 1 << (WEIGHT_BITS - 8);
        let expected: Vec<i32> = [-3, -9, 29, 111, 111, 29, -9, -3]
            .iter()
            .map(|w| w * unit)
            .collect();
        assert_eq!(weights, &expected);
        assert_eq!(taps(64, Kernel::Bicubic)[8].first, 13);
    }

    #[test]
    fn the_area_kernel_is_the_mean_of_two_samples() {
        let taps = taps(64, Kernel::Area);
        assert_eq!(taps[8].first, 16);
        assert_eq!(taps[8].weights, vec![1 << (WEIGHT_BITS - 1); 2]);
    }

    #[test]
    fn a_flat_picture_stays_flat_to_its_edges() {
        let (width, height) = (16, 12);
        let rgba: Vec<u8> = (0..width * height).flat_map(|_| [200, 37, 5, 0]).collect();
        let mut reduced = vec![0; width / 2 * (height / 2) * RGBA_BYTES];
        Reducer::new(width, height, Kernel::Bicubic).reduce(&rgba, &mut reduced);
        let pixels = reduced.as_chunks::<RGBA_BYTES>().0;
        assert!(pixels.iter().all(|pixel| *pixel == [200, 37, 5, 255]));
    }
}
