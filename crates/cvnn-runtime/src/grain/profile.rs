//! How much grain: one of a few fixed profiles, chosen by the kind of source.
//! There is no level to set: the amount that suits a picture cannot be read
//! from the input, and a number a viewer must tune is a feature that does not
//! work. The two levels below are the only place the amounts are written.

use core::str::FromStr;

use cvnn_common::{Error, Result};

/// Grey levels of grain (the deviation of the noise added to the displayed
/// 8-bit luma) for clean or produced content, and for a webcam or video
/// call. Both came out of a sweep on held-out clips of each kind: the level
/// at which the output stops looking waxy without the grain itself being
/// seen. A webcam takes about twice what produced content does, since its
/// sensor noise was the larger part of what the network removed.
const PRODUCED_LEVEL: f32 = 0.35;
/// See [`PRODUCED_LEVEL`].
const WEBCAM_LEVEL: f32 = 0.75;

/// The grain a stream's pictures are shown with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrainProfile {
    /// No grain: the picture is word for word the network's. For checks
    /// against the picture without grain, and for a caller that adds its own.
    None,
    /// Clean or produced content (film, television, screen capture).
    Produced,
    /// A webcam or video call: the project's main case, so the profile a
    /// stream has until it is given another.
    #[default]
    Webcam,
}

impl GrainProfile {
    /// Every profile, with the name the bindings and the page use for it.
    pub const NAMED: [(&'static str, Self); 3] = [
        ("none", Self::None),
        ("produced", Self::Produced),
        ("webcam", Self::Webcam),
    ];

    /// The profile's grain in grey levels of the displayed 8-bit luma.
    #[must_use]
    pub fn level(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Produced => PRODUCED_LEVEL,
            Self::Webcam => WEBCAM_LEVEL,
        }
    }
}

impl FromStr for GrainProfile {
    type Err = Error;

    /// A profile by its name in [`GrainProfile::NAMED`].
    fn from_str(name: &str) -> Result<Self> {
        let found = Self::NAMED.iter().find(|(known, _)| *known == name);
        found
            .map(|&(_, profile)| profile)
            .ok_or_else(|| Error::InvalidSpec(format!("unknown grain profile '{name}'")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_starts_on_the_webcam_profile() {
        assert_eq!(GrainProfile::default(), GrainProfile::Webcam);
    }

    #[test]
    fn every_profile_is_found_by_its_name_and_nothing_else_is() {
        for (name, profile) in GrainProfile::NAMED {
            assert_eq!(name.parse::<GrainProfile>().unwrap(), profile);
        }
        assert!("1.5".parse::<GrainProfile>().is_err());
    }

    #[test]
    fn the_levels_are_those_of_the_sweep() {
        let levels = GrainProfile::NAMED.map(|(_, profile)| profile.level());
        assert_eq!(levels, [0.0, 0.35, 0.75]);
    }
}
