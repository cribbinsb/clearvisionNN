//! The fixed-point amplitude a noise sample is multiplied by, from a
//! profile's level (docs/INTEGER.md, section 9).

use super::GrainProfile;
use crate::fixed::{PICTURE_MAX, round_shift};

/// The amplitude of one grey level. It carries the constant that makes the
/// noise field's deviation (1090.03) one grey level of 32 picture steps after
/// the final shift: 32 x 2^17 / 1090.03.
const AMPLITUDE_ONE: i32 = 3848;
/// A level counts 2^-8 of a grey level.
const LEVEL_BITS: u32 = 8;
const LEVEL_STEPS: f32 = (1 << LEVEL_BITS) as f32;
/// Bits an amplitude times a noise sample is rounded by, to picture steps.
const GRAIN_BITS: u32 = 17;

impl GrainProfile {
    /// The amplitude every sample's noise is multiplied by: what an
    /// implementation that cannot run this crate (a shader) is handed, so
    /// that the step from a level to fixed point exists once. Zero means no grain.
    #[must_use]
    pub fn amplitude(self) -> i32 {
        let level = (self.level() * LEVEL_STEPS).round_ties_even() as i32;
        round_shift(AMPLITUDE_ONE * level, LEVEL_BITS)
    }
}

/// A noise sample at an amplitude, in picture steps.
pub(super) fn scaled(amplitude: i32, noise: i32) -> i32 {
    round_shift(amplitude * noise, GRAIN_BITS)
}

/// A clean sample with its grain, kept inside the picture's range.
#[must_use]
pub fn shown(clean: i32, grain: i32) -> i32 {
    (clean + grain).clamp(0, PICTURE_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The largest noise sample: 510 x 256 / 16.
    const NOISE_MAX: i64 = 8160;

    #[test]
    fn the_amplitudes_are_those_of_the_specification() {
        assert_eq!(GrainProfile::None.amplitude(), 0);
        // 0.35 grey levels is 90 steps of 1/256 (0.3516); 0.75 is 192, exactly.
        assert_eq!(GrainProfile::Produced.amplitude(), 1353);
        assert_eq!(GrainProfile::Webcam.amplitude(), 2886);
    }

    #[test]
    fn no_profile_leaves_32_bits() {
        for (_, profile) in GrainProfile::NAMED {
            let product = i64::from(profile.amplitude()) * NOISE_MAX + (1 << GRAIN_BITS);
            assert!(product < i64::from(i32::MAX));
        }
    }

    #[test]
    fn grain_is_clipped_to_the_picture() {
        assert_eq!(shown(100, -300), 0);
        assert_eq!(shown(PICTURE_MAX - 5, 40), PICTURE_MAX);
        assert_eq!(shown(4000, -40), 3960);
        assert_eq!(scaled(AMPLITUDE_ONE, 1090), 32);
    }
}
