//! The grain's noise: a field of zero-mean samples that depends only on the
//! frame index and the sample's position, so every implementation (this one,
//! the page's shaders, training) can produce the same grain with no state but
//! a frame counter. White noise from an integer hash, lightly blurred so that
//! it reads as grain rather than as salt and pepper (docs/INTEGER.md,
//! section 9).

/// Odd multipliers that spread a row or column number over all 32 bits
/// before it is mixed into a key by exclusive-or: the golden-ratio prime, and
/// a constant of `MurmurHash3`. Were the coordinates simply added to the key,
/// two rows whose keys happened to lie close would be shifted copies of each
/// other; this way two keys give the same sample only one sample at a time.
const ROW: u32 = 0x9E37_79B1;
const COLUMN: u32 = 0x85EB_CA6B;
/// Bytes of a hashed word; their sum approximates a normal draw.
const DRAWS: i32 = 4;
/// Twice the mean of a byte, so that the mean of the sum stays an integer.
const BYTE_RANGE: i32 = 255;
/// The blur along each axis, in sixteenths: about the light blur a camera's
/// own processing leaves on its noise, with weights a shift can divide by.
const KERNEL: [i32; 3] = [3, 10, 3];
/// The blurred sum keeps this many bits fewer than it has: enough is left
/// (a deviation of about 1090) and the product with an amplitude then fits 32 bits.
const FIELD_SHIFT: u32 = 4;

/// A bijective scramble of a 32-bit word in which every input bit changes
/// about half the output bits: the two-round multiply and xor-shift finaliser
/// "lowbias32" found by Chris Wellons' hash-prospector search (public domain).
const fn scramble(mut word: u32) -> u32 {
    word ^= word >> 16;
    word = word.wrapping_mul(0x7FEB_352D);
    word ^= word >> 15;
    word = word.wrapping_mul(0x846C_A68B);
    word ^ (word >> 16)
}

/// The key of a frame, from which each of its rows takes its own.
const fn frame_key(frame: u32) -> u32 {
    scramble(frame)
}

/// The key of row `y` (which may lie outside the picture) of a frame.
const fn row_key(frame_key: u32, y: i32) -> u32 {
    scramble(frame_key ^ y.cast_unsigned().wrapping_mul(ROW))
}

/// The white sample at column `x` of a row: the sum of the four bytes of its
/// hashed word about their mean. Zero mean, variance 21845, within 510.
fn white(row_key: u32, x: i32) -> i32 {
    let word = scramble(row_key ^ x.cast_unsigned().wrapping_mul(COLUMN));
    let sum: i32 = word.to_le_bytes().iter().map(|&byte| i32::from(byte)).sum();
    sum - DRAWS * BYTE_RANGE / 2
}

/// Row `y` blurred along itself, for columns `0..out.len()`; `samples` is
/// scratch for the white samples, which start one column before the picture.
fn blurred_row(frame_key: u32, y: i32, samples: &mut [i32], out: &mut [i32]) {
    let key = row_key(frame_key, y);
    for (sample, x) in samples.iter_mut().zip(-1..) {
        *sample = white(key, x);
    }
    for (value, taps) in out.iter_mut().zip(samples.windows(KERNEL.len())) {
        *value = KERNEL.iter().zip(taps).map(|(k, tap)| k * tap).sum();
    }
}

/// Write the noise field of frame `frame` for a `height` x `width` picture:
/// per sample, the white noise of its 3x3 neighbourhood weighted by the
/// kernel both ways and rounded to 1/16 of that sum. The white noise is
/// defined beyond the picture too, so its edges are no special case.
pub(super) fn field(frame: u32, (height, width): (usize, usize), out: &mut [i32]) {
    let key = frame_key(frame);
    let mut samples = vec![0; width + KERNEL.len() - 1];
    // The blurred rows above, at and below the one being written, in turn.
    let mut rows = [(); KERNEL.len()].map(|()| vec![0; width]);
    for (row, y) in rows.iter_mut().zip(-1..).take(KERNEL.len() - 1) {
        blurred_row(key, y, &mut samples, row);
    }
    for ((line, y), newest) in out.chunks_exact_mut(width).take(height).zip(0..).zip(2..) {
        blurred_row(key, y + 1, &mut samples, &mut rows[newest % KERNEL.len()]);
        let taps: [&[i32]; 3] =
            core::array::from_fn(|tap| &*rows[(newest + 1 + tap) % KERNEL.len()]);
        for (x, value) in line.iter_mut().enumerate() {
            let sum: i32 = KERNEL.iter().zip(taps).map(|(k, row)| k * row[x]).sum();
            *value = (sum + (1 << (FIELD_SHIFT - 1))) >> FIELD_SHIFT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (usize, usize) = (192, 256);
    /// The deviation the specification gives the field: sqrt(21845) x 118 / 16.
    const DEVIATION: f64 = 1090.03;

    fn of(frame: u32) -> Vec<i32> {
        let mut out = vec![0; SIZE.0 * SIZE.1];
        field(frame, SIZE, &mut out);
        out
    }

    fn mean(values: &[i32]) -> f64 {
        values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64
    }

    /// The correlation of two zero-mean fields.
    fn correlation(a: &[i32], b: &[i32]) -> f64 {
        let dot = |x: &[i32], y: &[i32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(&p, &q)| f64::from(p) * f64::from(q))
                .sum()
        };
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
    }

    #[test]
    fn the_hash_is_the_published_one() {
        // From a calculation of the definition in another language, not from this code.
        assert_eq!(scramble(1), 0x6889_90C0);
        assert_eq!(scramble(0xDEAD_BEEF), 0xE628_C683);
    }

    #[test]
    fn the_field_has_no_mean_and_the_specified_deviation() {
        let noise = of(7);
        let deviation =
            (noise.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / noise.len() as f64).sqrt();
        assert!(
            mean(&noise).abs() < 0.03 * DEVIATION,
            "mean {}",
            mean(&noise)
        );
        assert!(
            (deviation / DEVIATION - 1.0).abs() < 0.03,
            "deviation {deviation}"
        );
        assert!(noise.iter().all(|v| v.abs() <= 8160));
    }

    #[test]
    fn a_frame_index_always_gives_the_same_field_and_another_a_fresh_one() {
        assert_eq!(of(3), of(3));
        for (a, b) in [(3, 4), (0, 1), (u32::MAX, 0)] {
            let related = correlation(&of(a), &of(b));
            assert!(related.abs() < 0.03, "frames {a} and {b}: {related}");
        }
    }

    #[test]
    fn neighbours_are_related_only_as_the_blur_makes_them() {
        let noise = of(11);
        let (height, width) = SIZE;
        let shifted = |by: usize| correlation(&noise[..height * width - by], &noise[by..]);
        // Of the kernel with itself, one and two samples along: 60/118 and 9/118.
        assert!((shifted(1) - 60.0 / 118.0).abs() < 0.03, "{}", shifted(1));
        assert!(
            (shifted(width) - 60.0 / 118.0).abs() < 0.03,
            "{}",
            shifted(width)
        );
        assert!((shifted(2) - 9.0 / 118.0).abs() < 0.03, "{}", shifted(2));
        assert!(shifted(3).abs() < 0.03 && shifted(3 * width).abs() < 0.03);
    }

    #[test]
    fn a_picture_is_a_window_on_one_endless_field() {
        // A narrower, shorter picture sees the same samples: nothing depends on the size.
        let (height, width) = (SIZE.0 / 3, SIZE.1 / 5);
        let (wide, mut narrow) = (of(5), vec![0; height * width]);
        field(5, (height, width), &mut narrow);
        for (row, line) in narrow.chunks_exact(width).enumerate() {
            assert_eq!(line, &wide[row * SIZE.1..][..width]);
        }
    }
}
