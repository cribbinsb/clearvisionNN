//! Per-sample functions written so that loops over them become SIMD.
//!
//! The standard library's `exp` and `tanh` are calls the compiler cannot
//! vectorise, and a float-to-integer `as` cast saturates, which x86 has no
//! vector instruction for; the runtime evaluates millions of each per frame.
//! The `exp`-based functions here are accurate to about 1e-6, far inside the
//! tolerance the runtime is checked against the training model with.

use core::f32::consts::{LN_2, LOG2_E};

/// Arguments are clamped to this magnitude: `exp` stays finite, and the
/// functions built on it have long since saturated.
const MAX_ARGUMENT: f32 = 80.0;
/// 1.5 * 2^23. A float this large has no fraction bits, so adding it rounds
/// the other term to an integer (ties to even), and that integer is then the
/// difference of the low bits of the sum's representation: rounding and
/// conversion without a float-to-integer instruction.
const ROUNDER: f32 = 12_582_912.0;
/// Magnitudes below this (2^22) round correctly that way.
#[cfg(test)]
const ROUNDER_RANGE: f32 = 4_194_304.0;
/// IEEE 754 single precision: exponent bias and mantissa width.
const EXPONENT_BIAS: u32 = 127;
const MANTISSA_BITS: u32 = 23;
/// Taylor coefficients of `exp(r) - 1 - r` from `r^2` to `r^7`, highest
/// first. With `|r| <= ln(2) / 2` the first omitted term is below 6e-9.
const TAYLOR: [f32; 6] = [
    1.0 / 5040.0,
    1.0 / 720.0,
    1.0 / 120.0,
    1.0 / 24.0,
    1.0 / 6.0,
    1.0 / 2.0,
];

/// `e^x`, with `x` clamped to +/- [`MAX_ARGUMENT`].
#[inline]
fn exp(x: f32) -> f32 {
    let x = x.clamp(-MAX_ARGUMENT, MAX_ARGUMENT);
    // x = n ln 2 + r with |r| <= ln(2) / 2, so e^x = 2^n e^r.
    let shifted = x * LOG2_E + ROUNDER;
    let n = shifted - ROUNDER;
    let r = x - n * LN_2;
    let series = TAYLOR.iter().fold(0.0, |sum, &term| sum * r + term);
    let exponent = shifted
        .to_bits()
        .wrapping_sub(ROUNDER.to_bits())
        .wrapping_add(EXPONENT_BIAS);
    (series * r * r + r + 1.0) * f32::from_bits(exponent << MANTISSA_BITS)
}

/// The integer nearest to `value` (ties to even), which must be below 2^22
/// in magnitude and not negative.
#[inline]
pub(crate) fn round(value: f32) -> u32 {
    (value + ROUNDER).to_bits().wrapping_sub(ROUNDER.to_bits())
}

/// The whole part of `value`, as an integer and as a float, under the same
/// conditions as [`round`].
#[inline]
pub(crate) fn floor(value: f32) -> (u32, f32) {
    let shifted = value + ROUNDER;
    let rounded = shifted - ROUNDER;
    let nearest = shifted.to_bits().wrapping_sub(ROUNDER.to_bits());
    if rounded > value {
        (nearest.wrapping_sub(1), rounded - 1.0)
    } else {
        (nearest, rounded)
    }
}

/// The logistic function `1 / (1 + e^-x)`.
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// The hyperbolic tangent.
#[inline]
pub(crate) fn tanh(x: f32) -> f32 {
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 2e-6;

    #[test]
    fn functions_match_the_standard_library() {
        for step in -4000..=4000 {
            let x = step as f32 * 0.01;
            let logistic = 1.0 / (1.0 + (-x).exp());
            assert!((sigmoid(x) - logistic).abs() < TOLERANCE, "sigmoid({x})");
            assert!((tanh(x) - x.tanh()).abs() < TOLERANCE, "tanh({x})");
        }
    }

    #[test]
    fn rounding_and_floor_match_the_standard_library() {
        for step in 0..=4000 {
            let x = step as f32 * 0.37;
            assert_eq!(round(x), x.round_ties_even() as u32, "round({x})");
            assert_eq!(floor(x), (x.floor() as u32, x.floor()), "floor({x})");
        }
        let large = ROUNDER_RANGE - 1.0;
        assert_eq!(floor(large), (large as u32, large));
    }

    #[test]
    fn extreme_arguments_saturate() {
        assert_eq!(sigmoid(1e9), 1.0);
        assert!(sigmoid(-1e9) < 1e-30);
        assert_eq!(tanh(1e9), 1.0);
        assert_eq!(tanh(-1e9), -1.0);
    }
}
