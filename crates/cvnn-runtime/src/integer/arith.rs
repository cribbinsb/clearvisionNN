//! The integer primitives of the specification. Everything wraps at 32 bits,
//! as the shaders' arithmetic does; a model file that passed the exporter's
//! range checks never reaches the wrap.

use super::{STORED_OFFSET, UNSIGNED_LEVELS};

/// Bits of the accumulator multiplied separately, so no product leaves 32 bits.
const SPLIT_BITS: u32 = 16;
const SPLIT_MASK: i32 = (1 << SPLIT_BITS) - 1;

/// `accumulator * multiplier / 2^16`, rounded down.
pub(super) fn scale(accumulator: i32, multiplier: i32) -> i32 {
    let high = (accumulator >> SPLIT_BITS).wrapping_mul(multiplier);
    let low = (accumulator & SPLIT_MASK).wrapping_mul(multiplier) >> SPLIT_BITS;
    high.wrapping_add(low)
}

/// `value / 2^shift`, rounded to nearest with halves towards positive infinity.
pub(super) fn round_shift(value: i32, shift: u32) -> i32 {
    value.wrapping_add((1 << shift) >> 1) >> shift
}

/// An accumulator on its output scale; `extra` is added before the final
/// rounding (the identity branch, already multiplied).
pub(super) fn requantise(accumulator: i32, multiplier: i32, shift: u32, extra: i32) -> i32 {
    round_shift(scale(accumulator, multiplier).wrapping_add(extra), shift)
}

/// The rectifier and the 8-bit ceiling in one clamp.
pub(super) fn to_unsigned(value: i32) -> u8 {
    value.clamp(0, UNSIGNED_LEVELS) as u8
}

/// The value of an unsigned plane from its stored byte.
pub(super) fn unsigned_from_stored(stored: i8) -> i32 {
    i32::from(stored) + STORED_OFFSET
}

/// The stored byte of an unsigned value (0 to 255).
pub(super) fn stored_from_unsigned(value: i32) -> u8 {
    (value - STORED_OFFSET) as i8 as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_is_the_exact_floor_of_the_wide_product() {
        for &(accumulator, multiplier) in &[(123_456_789, 16_383), (-987_654_321, 9_001), (-1, 1)] {
            let wide = (i64::from(accumulator) * i64::from(multiplier)).div_euclid(1 << SPLIT_BITS);
            assert_eq!(i64::from(scale(accumulator, multiplier)), wide);
        }
    }

    #[test]
    fn halves_round_towards_positive_infinity() {
        assert_eq!(
            (round_shift(3, 1), round_shift(-3, 1), round_shift(5, 0)),
            (2, -1, 5)
        );
    }

    #[test]
    fn stored_bytes_round_trip() {
        for value in [0, 1, 127, 128, 255] {
            assert_eq!(
                unsigned_from_stored(stored_from_unsigned(value) as i8),
                value
            );
        }
    }
}
