//! Optical blur for a simulated camera: a Gaussian over the full-size
//! picture, as the training degradation blurs it before noise and reduction
//! (there with `OpenCV`'s `GaussianBlur`; here the same kernel in fixed point).

const RGBA_BYTES: usize = 4;
/// The kernel reaches this many deviations either side, as `OpenCV`'s does for 8-bit pictures.
const REACH: f32 = 3.0;
/// Fractional bits of the fixed-point weights: with an 8-bit sample a
/// weighted sum then fits 16 bits, which is what vector arithmetic is quickest at.
const WEIGHT_BITS: u32 = 8;
const HALF: u16 = 1 << (WEIGHT_BITS - 1);

/// The kernel's weights, centre first, in fixed point; they and their
/// mirror images sum to exactly one.
fn weights(sigma: f32) -> Vec<u16> {
    let radius = (REACH * sigma).round().max(1.0) as usize;
    let raw: Vec<f32> = (0..=radius)
        .map(|offset| (-((offset * offset) as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total = raw[0] + 2.0 * raw[1..].iter().sum::<f32>();
    let one = 1_u16 << WEIGHT_BITS;
    let mut fixed: Vec<u16> = raw
        .iter()
        .map(|weight| (weight / total * f32::from(one)).round() as u16)
        .collect();
    fixed[0] = one - 2 * fixed[1..].iter().sum::<u16>();
    fixed
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

/// Blurs pictures of one size, keeping its scratch between frames.
#[derive(Debug, Clone)]
pub struct Blur {
    width: usize,
    height: usize,
    /// One row with its ends mirrored outwards, so the horizontal pass never tests for an edge.
    padded: Vec<u8>,
    /// Weighted sums of one row or line, all four channels alike.
    sums: Vec<u16>,
    /// The picture after the horizontal pass, rounded to 8 bits.
    across: Vec<u8>,
}

impl Blur {
    /// A blur for `width` x `height` pictures.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            padded: Vec::new(),
            sums: vec![0; width * RGBA_BYTES],
            across: vec![0; width * height * RGBA_BYTES],
        }
    }

    /// Write `rgba` blurred by a Gaussian of deviation `sigma` (samples) to
    /// `blurred`. Alpha is blurred like the colours, which leaves an opaque
    /// picture opaque.
    ///
    /// # Panics
    ///
    /// If either slice is not the size the blur was made for.
    pub fn apply(&mut self, rgba: &[u8], sigma: f32, blurred: &mut [u8]) {
        assert_eq!(rgba.len(), self.width * self.height * RGBA_BYTES);
        assert_eq!(blurred.len(), rgba.len());
        let weights = weights(sigma);
        let radius = weights.len() - 1;
        let taps = 2 * radius + 1;
        let weight = |offset: usize| weights[offset.abs_diff(radius)];
        let stride = self.width * RGBA_BYTES;
        self.padded.resize(stride + 2 * radius * RGBA_BYTES, 0);
        // Each pass adds one tap at a time across a whole line: plain loops the compiler vectorises.
        for (row, carried) in rgba
            .chunks_exact(stride)
            .zip(self.across.chunks_exact_mut(stride))
        {
            for (index, pixel) in self
                .padded
                .as_chunks_mut::<RGBA_BYTES>()
                .0
                .iter_mut()
                .enumerate()
            {
                let from = mirrored(index as isize - radius as isize, self.width) * RGBA_BYTES;
                pixel.copy_from_slice(&row[from..from + RGBA_BYTES]);
            }
            self.sums.fill(HALF);
            for offset in 0..taps {
                let shifted = &self.padded[offset * RGBA_BYTES..][..stride];
                for (sum, &sample) in self.sums.iter_mut().zip(shifted) {
                    *sum += weight(offset) * u16::from(sample);
                }
            }
            for (carry, &sum) in carried.iter_mut().zip(&self.sums) {
                *carry = (sum >> WEIGHT_BITS) as u8;
            }
        }
        for (y, out) in blurred.chunks_exact_mut(stride).enumerate() {
            self.sums.fill(HALF);
            for offset in 0..taps {
                let line = mirrored(y as isize + offset as isize - radius as isize, self.height);
                let carried = &self.across[line * stride..][..stride];
                for (sum, &carry) in self.sums.iter_mut().zip(carried) {
                    *sum += weight(offset) * u16::from(carry);
                }
            }
            for (byte, &sum) in out.iter_mut().zip(&self.sums) {
                *byte = (sum >> WEIGHT_BITS) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_picture_stays_flat_to_its_edges() {
        let (width, height) = (12, 10);
        let rgba = [200u8, 37, 5, 255].repeat(width * height);
        let mut blurred = vec![0; rgba.len()];
        Blur::new(width, height).apply(&rgba, 1.2, &mut blurred);
        assert_eq!(blurred, rgba);
    }

    #[test]
    fn a_point_spreads_into_the_gaussian_and_keeps_its_light() {
        let (width, height) = (15, 15);
        let mut rgba = [0u8, 0, 0, 255].repeat(width * height);
        rgba[(7 * width + 7) * RGBA_BYTES] = 240;
        let mut blurred = vec![0; rgba.len()];
        Blur::new(width, height).apply(&rgba, 1.0, &mut blurred);
        let red = |x: usize, y: usize| i32::from(blurred[(y * width + x) * RGBA_BYTES]);
        // The centre of a unit-deviation Gaussian holds about 1/(2 pi) of the light.
        assert!((red(7, 7) - 38).abs() <= 1, "centre {}", red(7, 7));
        assert_eq!(red(6, 7), red(8, 7));
        assert_eq!(red(7, 6), red(8, 7));
        let total: i32 = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| red(x, y))
            .sum();
        assert!((total - 240).abs() <= 8, "total {total}");
    }
}
