//! The conventional 2x upscale of a 4:2:0 frame in picture steps: what a
//! player would show without the network, and what its detail is added to.

use super::upscale::{BICUBIC_BITS, BILINEAR_BITS, bicubic2x, bilinear2x, round_shift};
use super::{PICTURE_MAX, PICTURE_STEPS};

/// An 8-bit level is this many bits coarser than a picture step.
pub(super) const LEVEL_SHIFT: u32 = PICTURE_STEPS.trailing_zeros();

/// Sums scaled by 2^`bits` per level, as picture steps within the picture's range.
fn to_steps(sums: &mut [i32], bits: u32) {
    for value in sums.iter_mut() {
        *value = round_shift(*value, bits - LEVEL_SHIFT).clamp(0, PICTURE_MAX);
    }
}

/// Write to `out` (luma, then the two chroma planes, each at twice `size`)
/// the upscale of `luma` (`size`) and `chroma` (half of it each way): luma
/// bicubic; chroma to luma size bilinearly, as in training, then bicubic.
pub(super) fn conventional(
    luma: &[u8],
    chroma: [&[u8]; 2],
    size: (usize, usize),
    out: &mut [Vec<i32>; 3],
) {
    let widen = |samples: &[u8]| {
        samples
            .iter()
            .map(|&sample| i32::from(sample))
            .collect::<Vec<i32>>()
    };
    let [y, colours @ ..] = out;
    bicubic2x(&widen(luma), size, y);
    to_steps(y, 2 * BICUBIC_BITS);
    let mut at_luma_size = vec![0; size.0 * size.1];
    for (target, samples) in colours.iter_mut().zip(chroma) {
        bilinear2x(&widen(samples), (size.0 / 2, size.1 / 2), &mut at_luma_size);
        bicubic2x(&at_luma_size, size, target);
        to_steps(target, 2 * (BICUBIC_BITS + BILINEAR_BITS));
    }
}
