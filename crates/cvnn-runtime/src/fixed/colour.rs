//! 4:4:4 YUV in picture steps to RGBA, in integers: `crate::colour`'s BT.709
//! limited-range conversion with its coefficients in 2^-14.

use super::PICTURE_STEPS;

/// Bits of the coefficients.
const BITS: u32 = 14;
/// 255 / 219: limited-range luma to full range.
const LUMA: i32 = 19_077;
/// 2 (1 - Kr) 255 / 224 and 2 (1 - Kb) 255 / 224: chroma into red and blue.
const RED_FROM_CR: i32 = 29_372;
const BLUE_FROM_CB: i32 = 34_610;
/// Kr 2 (1 - Kr) / Kg and Kb 2 (1 - Kb) / Kg, times 255 / 224: chroma out of green.
const GREEN_FROM_CR: i32 = 8_731;
const GREEN_FROM_CB: i32 = 3_494;
const LUMA_OFFSET: i32 = 16 * PICTURE_STEPS;
const CHROMA_OFFSET: i32 = 128 * PICTURE_STEPS;
/// Picture steps and coefficient bits together: what a product is divided by to give a level.
const SHIFT: u32 = BITS + PICTURE_STEPS.trailing_zeros();
const OPAQUE: u32 = 0xff00_0000;

/// One pixel as a little-endian RGBA word, from samples in picture steps.
pub(super) fn rgba(y: i32, u: i32, v: i32) -> u32 {
    let luma = LUMA * (y - LUMA_OFFSET);
    let (cb, cr) = (u - CHROMA_OFFSET, v - CHROMA_OFFSET);
    let level = |value: i32| {
        ((value + (1 << (SHIFT - 1))) >> SHIFT)
            .clamp(0, 255)
            .cast_unsigned()
    };
    let red = level(luma + RED_FROM_CR * cr);
    let green = level(luma - GREEN_FROM_CR * cr - GREEN_FROM_CB * cb);
    let blue = level(luma + BLUE_FROM_CB * cb);
    red | green << 8 | blue << 16 | OPAQUE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_range_grey_becomes_full_range_grey() {
        let neutral = 128 * PICTURE_STEPS;
        assert_eq!(rgba(16 * PICTURE_STEPS, neutral, neutral), 0xff00_0000);
        assert_eq!(rgba(235 * PICTURE_STEPS, neutral, neutral), 0xffff_ffff);
        assert_eq!(rgba(126 * PICTURE_STEPS, neutral, neutral) & 0xff, 128);
    }
}
