//! A camera's spatial noise reduction for a simulated sender: an
//! edge-preserving (bilateral) filter on brightness that flattens fine
//! texture and keeps edges, as the training degradation applies it after the
//! sensor noise (there with `OpenCV`'s `bilateralFilter`).

use crate::colour::{LUMA_RANGE, MAX_SAMPLE, luma_of};

const RGBA_BYTES: usize = 4;
const COLOUR_BYTES: usize = 3;
/// The filter's neighbourhood reaches this many samples: a diameter of five.
const RADIUS: isize = 2;
/// Deviation of the neighbourhood's spatial weighting, in samples.
const SPATIAL_SIGMA: f32 = 2.0;
/// The strength is given on the limited-range luma scale the training data is
/// stored in; RGB spans that range stretched to 0-255.
const LEVEL_TO_RGB: f32 = MAX_SAMPLE / LUMA_RANGE;

const LEVELS: usize = 256;
/// Fractional bits of the fixed-point weights: a weight times a sample, summed
/// over the neighbourhood, stays well inside 32 bits.
const WEIGHT_BITS: u32 = 12;

/// The neighbourhood: each sample within [`RADIUS`] of the centre, as its
/// column and row in a plane padded by [`RADIUS`], and its spatial weight.
fn neighbourhood() -> Vec<(usize, usize, f32)> {
    let mut taps = Vec::new();
    for dy in -RADIUS..=RADIUS {
        for dx in -RADIUS..=RADIUS {
            let squared = (dx * dx + dy * dy) as f32;
            if squared <= (RADIUS * RADIUS) as f32 {
                let spatial = (-squared / (2.0 * SPATIAL_SIGMA * SPATIAL_SIGMA)).exp();
                taps.push(((dx + RADIUS) as usize, (dy + RADIUS) as usize, spatial));
            }
        }
    }
    taps
}

/// Mirror `position` about the end samples of a line of `length` (`OpenCV`'s default border).
fn mirrored(position: isize, length: usize) -> usize {
    let last = length as isize - 1;
    let folded = position.abs();
    (if folded > last {
        2 * last - folded
    } else {
        folded
    })
    .clamp(0, last) as usize
}

/// Denoises pictures of one size, keeping its tables and scratch between frames.
#[derive(Debug, Clone)]
pub struct Denoise {
    width: usize,
    height: usize,
    taps: Vec<(usize, usize, f32)>,
    /// A neighbour's weight by how far its brightness is from the centre's: a
    /// table per neighbour, the spatial weight folded in, for one strength.
    weights: Vec<[u32; LEVELS]>,
    /// The strength `weights` was made for.
    strength: f32,
    /// The picture's brightness, 0-255, with [`RADIUS`] mirrored samples all
    /// round, so that no neighbour needs an edge test.
    luma: Vec<u8>,
    /// Per sample of one row: the weighted sum of its neighbours, and the weights' sum.
    sums: Vec<u32>,
    totals: Vec<u32>,
}

impl Denoise {
    /// A denoiser for `width` x `height` pictures.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let border = 2 * RADIUS as usize;
        Self {
            width,
            height,
            taps: neighbourhood(),
            weights: Vec::new(),
            strength: 0.0,
            luma: vec![0; (width + border) * (height + border)],
            sums: vec![0; width],
            totals: vec![0; width],
        }
    }

    /// Make the weight tables for `strength`, unless they are already for it.
    fn tabulate(&mut self, strength: f32) {
        if !self.weights.is_empty() && self.strength.to_bits() == strength.to_bits() {
            return;
        }
        let sigma = strength * LEVEL_TO_RGB;
        let one = (1_u32 << WEIGHT_BITS) as f32;
        let table = |spatial: f32| {
            std::array::from_fn(|difference| {
                let squared = (difference * difference) as f32;
                (spatial * (-squared / (2.0 * sigma * sigma)).exp() * one).round() as u32
            })
        };
        self.weights = self
            .taps
            .iter()
            .map(|&(_, _, spatial)| table(spatial))
            .collect();
        self.strength = strength;
    }

    /// Filter the brightness of `rgba` in place: differences below about
    /// `strength` luma levels are smoothed away, larger ones (edges) are
    /// kept. Colour is left alone.
    ///
    /// # Panics
    ///
    /// If `rgba` is not the size the denoiser was made for.
    pub fn apply(&mut self, rgba: &mut [u8], strength: f32) {
        let (width, height, radius) = (self.width, self.height, RADIUS as usize);
        assert_eq!(rgba.len(), width * height * RGBA_BYTES);
        self.tabulate(strength);
        let padded_width = width + 2 * radius;
        for (row, line) in self.luma.chunks_exact_mut(padded_width).enumerate() {
            let y = mirrored(row as isize - RADIUS, height);
            let pixels = rgba[y * width * RGBA_BYTES..][..width * RGBA_BYTES]
                .as_chunks::<RGBA_BYTES>()
                .0;
            for (luma, pixel) in line[radius..].iter_mut().zip(pixels) {
                *luma = luma_of(pixel[0], pixel[1], pixel[2]);
            }
            for offset in 1..=radius {
                line[radius - offset] = line[radius + offset.min(width - 1)];
                line[radius + width - 1 + offset] =
                    line[radius + (width - 1).saturating_sub(offset)];
            }
        }
        for (y, out) in rgba.chunks_exact_mut(width * RGBA_BYTES).enumerate() {
            let centres = &self.luma[(y + radius) * padded_width + radius..][..width];
            let (sums, totals) = (&mut self.sums[..width], &mut self.totals[..width]);
            sums.fill(0);
            totals.fill(0);
            for (&(column, row, _), weights) in self.taps.iter().zip(&self.weights) {
                let neighbours = &self.luma[(y + row) * padded_width + column..][..width];
                for x in 0..width {
                    let weight = weights[usize::from(neighbours[x].abs_diff(centres[x]))];
                    sums[x] += weight * u32::from(neighbours[x]);
                    totals[x] += weight;
                }
            }
            let pixels = out.as_chunks_mut::<RGBA_BYTES>().0.iter_mut();
            for ((pixel, &centre), (&sum, &total)) in
                pixels.zip(centres).zip(sums.iter().zip(totals.iter()))
            {
                // The centre's own weight is one, so the total is never zero.
                let shift = ((sum + total / 2) / total) as i32 - i32::from(centre);
                for sample in &mut pixel[..COLOUR_BYTES] {
                    *sample = (i32::from(*sample) + shift).clamp(0, 255) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::SensorNoise;

    fn deviation(rgba: &[u8], columns: std::ops::Range<usize>, width: usize) -> f32 {
        let values: Vec<f32> = rgba
            .chunks_exact(width * RGBA_BYTES)
            .flat_map(|row| columns.clone().map(|x| f32::from(row[x * RGBA_BYTES])))
            .collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
    }

    #[test]
    fn noise_on_a_flat_area_is_reduced_and_an_edge_is_kept() {
        let (width, height) = (64, 48);
        let clean: Vec<u8> = (0..width * height)
            .flat_map(|i| {
                if i % width < width / 2 {
                    [60, 60, 60, 255]
                } else {
                    [180, 180, 180, 255]
                }
            })
            .collect();
        let mut noisy = vec![0; clean.len()];
        SensorNoise::new(3).add(&clean, 3.0, &mut noisy);
        let before = deviation(&noisy, 4..28, width);
        Denoise::new(width, height).apply(&mut noisy, 10.0);
        let after = deviation(&noisy, 4..28, width);
        assert!(after < 0.5 * before, "deviation {before} -> {after}");
        let row = &noisy[20 * width * RGBA_BYTES..];
        let (left, right) = (
            row[(width / 2 - 1) * RGBA_BYTES],
            row[(width / 2) * RGBA_BYTES],
        );
        assert!(
            i32::from(right) - i32::from(left) > 100,
            "edge {left} | {right}"
        );
    }
}
