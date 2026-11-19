//! Sensor noise for a simulated camera: a fresh, repeatable realisation added
//! to the luma of every frame, as the training degradation adds it before the
//! picture is reduced and encoded.

use crate::colour::{LUMA_RANGE, MAX_SAMPLE};

const RGBA_BYTES: usize = 4;
const COLOUR_BYTES: usize = 3;
const MAX_LEVEL: i32 = 255;
/// Bytes of one random word; their sum approximates a normal draw.
const DRAWS: usize = 4;
/// Mean and standard deviation of the sum of `DRAWS` uniform bytes.
const SUM_MEAN: i32 = DRAWS as i32 * 255 / 2;
const SUM_DEVIATION: f32 = 147.800_54;
/// Fractional bits of the fixed-point scale from a draw to a shift in levels.
const SCALE_BITS: u32 = 16;
/// The level is given on the limited-range luma scale the training data is
/// stored in; RGB spans that range stretched to 0-255.
const LEVEL_TO_RGB: f32 = MAX_SAMPLE / LUMA_RANGE;

/// A noise source. The same seed gives the same sequence of frames.
#[derive(Debug, Clone)]
pub struct SensorNoise {
    state: u32,
}

impl SensorNoise {
    /// A source that starts from `seed`.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        // Xorshift never leaves zero.
        Self { state: seed.max(1) }
    }

    /// An approximately normal draw with deviation [`SUM_DEVIATION`]: the
    /// sum of four uniform bytes about its mean, whose tails end at 3.5
    /// deviations.
    fn draw(&mut self) -> i32 {
        let mut word = self.state;
        word ^= word << 13;
        word ^= word >> 17;
        word ^= word << 5;
        self.state = word;
        let sum: u32 = word.to_le_bytes().iter().map(|&byte| u32::from(byte)).sum();
        sum as i32 - SUM_MEAN
    }

    /// Write `rgba` to `noisy` with noise of standard deviation `sigma` (luma
    /// levels, 0-255 scale) added to each pixel's brightness; colour is left
    /// alone and the result is clipped to 8 bits.
    ///
    /// # Panics
    ///
    /// If the slices differ in length.
    pub fn add(&mut self, rgba: &[u8], sigma: f32, noisy: &mut [u8]) {
        assert_eq!(rgba.len(), noisy.len());
        // Integer arithmetic per pixel: this runs on every sample of the original.
        let scale = (sigma * LEVEL_TO_RGB / SUM_DEVIATION * (1_u32 << SCALE_BITS) as f32) as i32;
        let half = 1 << (SCALE_BITS - 1);
        let pixels = rgba.as_chunks::<RGBA_BYTES>().0.iter();
        for (pixel, out) in pixels.zip(noisy.as_chunks_mut::<RGBA_BYTES>().0) {
            let shift = (self.draw() * scale + half) >> SCALE_BITS;
            for channel in 0..COLOUR_BYTES {
                out[channel] = (i32::from(pixel[channel]) + shift).clamp(0, MAX_LEVEL) as u8;
            }
            out[COLOUR_BYTES] = pixel[COLOUR_BYTES];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_has_the_asked_level_and_repeats_from_its_seed() {
        let grey = [128u8, 128, 128, 255].repeat(1 << 16);
        let (mut first, mut second) = (grey.clone(), grey.clone());
        let sigma = 4.0;
        SensorNoise::new(7).add(&grey, sigma, &mut first);
        SensorNoise::new(7).add(&grey, sigma, &mut second);
        assert_eq!(first, second);
        let shifts: Vec<f32> = first
            .iter()
            .step_by(RGBA_BYTES)
            .map(|&v| f32::from(v) - 128.0)
            .collect();
        let mean = shifts.iter().sum::<f32>() / shifts.len() as f32;
        let deviation =
            (shifts.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / shifts.len() as f32).sqrt();
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!(
            (deviation - sigma * LEVEL_TO_RGB).abs() < 0.15,
            "deviation {deviation}"
        );
    }
}
