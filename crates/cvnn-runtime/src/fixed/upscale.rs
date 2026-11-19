//! The 2x upscales in integers: the kernels of `crate::resample`, whose
//! weights are exact in 256ths (bicubic) and quarters (bilinear), so the sums
//! here are the float ones without their rounding.

/// Weights of the four bicubic taps for a sample three quarters and one
/// quarter of the way past the second tap, in 256ths (Keys' cubic, a = -0.75).
const BICUBIC: [[i32; 4]; 2] = [[-9, 67, 225, -27], [-27, 225, 67, -9]];
/// Weights of the two bilinear taps for the same two positions, in quarters.
const BILINEAR: [[i32; 2]; 2] = [[1, 3], [3, 1]];
/// Bits of a pass of each kernel: its weights sum to 2 to this power.
pub(super) const BICUBIC_BITS: u32 = 8;
pub(super) const BILINEAR_BITS: u32 = 2;

/// One axis of a 2x kernel: output sample `index` as taps into a row of `len` source samples.
fn taps<const N: usize>(phases: &[[i32; N]; 2], index: usize, len: usize) -> [(usize, i32); N] {
    // Even samples sit a quarter before source sample index / 2, odd ones a
    // quarter after it; the taps start one (bilinear) or two (bicubic) before.
    let first = (index / 2 + index % 2).cast_signed() - (N / 2).cast_signed();
    core::array::from_fn(|tap| {
        let source = (first + tap.cast_signed()).clamp(0, len.cast_signed() - 1);
        (source.cast_unsigned(), phases[index % 2][tap])
    })
}

/// A `height` x `width` plane upscaled 2x by a separable kernel of `N` taps:
/// each output is the exact weighted sum, scaled by the kernel's two passes.
fn upscale<const N: usize>(
    phases: &[[i32; N]; 2],
    source: &[i32],
    (height, width): (usize, usize),
    out: &mut [i32],
) {
    for (y, row) in out.chunks_exact_mut(2 * width).enumerate() {
        let rows = taps(phases, y, height);
        for (x, value) in row.iter_mut().enumerate() {
            let columns = taps(phases, x, width);
            *value = rows
                .iter()
                .map(|&(sy, wy)| {
                    let line = &source[sy * width..][..width];
                    wy * columns.iter().map(|&(sx, wx)| wx * line[sx]).sum::<i32>()
                })
                .sum();
        }
    }
}

/// Bicubic 2x: `out` is the sum scaled by 2^(2 * `BICUBIC_BITS`).
pub(super) fn bicubic2x(source: &[i32], size: (usize, usize), out: &mut [i32]) {
    upscale(&BICUBIC, source, size, out);
}

/// Bilinear 2x: `out` is the sum scaled by 2^(2 * `BILINEAR_BITS`).
pub(super) fn bilinear2x(source: &[i32], size: (usize, usize), out: &mut [i32]) {
    upscale(&BILINEAR, source, size, out);
}

/// `value / 2^shift` to the nearest, halves upwards.
pub(super) fn round_shift(value: i32, shift: u32) -> i32 {
    (value + (1 << (shift - 1))) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_plane_stays_flat_at_full_scale() {
        let mut out = [0; 48];
        bicubic2x(&[100; 12], (3, 4), &mut out);
        assert!(out.iter().all(|&value| value == 100 << (2 * BICUBIC_BITS)));
        bilinear2x(&[100; 12], (3, 4), &mut out);
        assert!(out.iter().all(|&value| value == 100 << (2 * BILINEAR_BITS)));
    }

    #[test]
    fn bilinear_samples_sit_a_quarter_of_the_way_between_neighbours() {
        let mut out = [0; 8];
        bilinear2x(&[0, 16], (1, 2), &mut out);
        assert_eq!(out[..4], [0, 64, 192, 256]);
    }
}
