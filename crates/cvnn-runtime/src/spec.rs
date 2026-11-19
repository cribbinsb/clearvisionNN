//! What a trained network expects and produces: the contract exported beside
//! its weights.

use core::ops::Range;
use core::str::FromStr;

use cvnn_common::{Error, Result};
use cvnn_preproc::SIDE_CHANNELS;

use crate::SR_SCALE;
use crate::coding::CODING_CHANNELS;

/// Planes of a picture (Y, U, V).
pub const PICTURE_CHANNELS: usize = 3;
/// The per-pixel blend weight the temporal design predicts.
const BLEND_CHANNELS: usize = 1;
/// The channel telling the network whether history is usable.
const VALID_CHANNELS: usize = 1;
/// The plane telling the network how well motion predicts the frame.
const CONFIDENCE_CHANNELS: usize = 1;

/// How a model uses earlier frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temporal {
    /// Each frame alone.
    None,
    /// The recursive blend: the network weighs its new estimate against the
    /// aligned previous output.
    Blend,
    /// A recurrent state and the previous output are inputs; the output is
    /// the conventional upscale plus the network's detail.
    Recurrent,
}

impl FromStr for Temporal {
    type Err = Error;

    /// The name a contract (`model.json`) uses.
    fn from_str(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "blend" => Ok(Self::Blend),
            "recurrent" => Ok(Self::Recurrent),
            other => Err(Error::InvalidSpec(format!(
                "unknown temporal design '{other}'"
            ))),
        }
    }
}

/// What the network is shown of the picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// Y, U and V at input resolution, then the side-plane channels, which
    /// are zero unless the model was trained with them.
    Picture {
        /// Whether the side planes are filled.
        side_planes: bool,
    },
    /// Luma alone: no chroma and no side-plane channels.
    Luma {
        /// The network is also given how each frame was coded.
        coding: bool,
        /// The network is also given how well motion predicts the frame
        /// (recurrent design only).
        confidence: bool,
    },
}

impl Input {
    /// The input a contract's fields describe.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSpec`] if they contradict each other.
    #[allow(
        clippy::fn_params_excessive_bools,
        reason = "one argument per field of the contract"
    )]
    pub fn from_contract(
        luma_only: bool,
        side_planes: bool,
        coding_planes: bool,
        confidence_plane: bool,
    ) -> Result<Self> {
        let reason = match (luma_only, side_planes, coding_planes || confidence_plane) {
            (true, false, _) => {
                return Ok(Self::Luma {
                    coding: coding_planes,
                    confidence: confidence_plane,
                });
            }
            (false, _, false) => return Ok(Self::Picture { side_planes }),
            (true, true, _) => "a luma-only model takes no side planes",
            (false, _, true) => "only a luma-only model takes coding or confidence planes",
        };
        Err(Error::InvalidSpec(reason.to_owned()))
    }
}

/// The shape of a model, mirroring the training configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// Leading picture planes the network reconstructs (1: luma; 3: all).
    pub planes: usize,
    /// How earlier frames are used.
    pub temporal: Temporal,
    /// Warp history by the motion field before use (temporal only).
    pub align: bool,
    /// Hidden-state channels carried between frames (temporal only).
    pub state_channels: usize,
    /// What the network is shown of the picture. The recurrent design is
    /// always luma-only; the blend never is.
    pub input: Input,
}

/// Most hidden-state channels a contract may ask for: far above any trained
/// model, low enough that a corrupt contract cannot exhaust memory.
const MAX_STATE_CHANNELS: usize = 256;

impl ModelSpec {
    /// Check that the contract describes a model this runtime implements.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSpec`] naming the field that is out of range.
    pub fn validate(&self) -> Result<()> {
        if self.planes != 1 && self.planes != PICTURE_CHANNELS {
            let why = format!(
                "planes must be 1 (luma) or {PICTURE_CHANNELS}, got {}",
                self.planes
            );
            return Err(Error::InvalidSpec(why));
        }
        let temporal = self.temporal != Temporal::None;
        if temporal && !(1..=MAX_STATE_CHANNELS).contains(&self.state_channels) {
            let why = format!("state_channels must be 1 to {MAX_STATE_CHANNELS}");
            return Err(Error::InvalidSpec(why));
        }
        let reason = match (self.input, self.temporal) {
            (Input::Luma { .. }, Temporal::Blend) => "the blend design is not luma-only",
            (Input::Picture { .. }, Temporal::Recurrent) => "the recurrent design is luma-only",
            (Input::Luma { .. }, _) if self.planes != 1 => "a luma-only model reconstructs luma",
            (Input::Luma { confidence, .. }, Temporal::None) if confidence => {
                "only the recurrent design takes a confidence plane"
            }
            _ => return Ok(()),
        };
        Err(Error::InvalidSpec(reason.to_owned()))
    }

    /// Hidden-state channels actually carried.
    #[must_use]
    pub fn hidden_channels(&self) -> usize {
        if self.temporal == Temporal::None {
            0
        } else {
            self.state_channels
        }
    }

    /// Channels of the network's input, at input resolution.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        let previous = self.planes * SR_SCALE * SR_SCALE;
        let history = match self.temporal {
            Temporal::None => 0,
            Temporal::Blend => previous + self.state_channels + VALID_CHANNELS,
            Temporal::Recurrent => self.state_channels + previous,
        };
        let picture = match self.input {
            Input::Picture { .. } => PICTURE_CHANNELS + SIDE_CHANNELS,
            Input::Luma { coding, confidence } => {
                1 + usize::from(confidence) * CONFIDENCE_CHANNELS
                    + usize::from(coding) * CODING_CHANNELS
            }
        };
        picture + history
    }

    /// Input channels this model never fills: they are zero for the whole
    /// stream, so a caller that keeps the tensor somewhere costly to reach (a
    /// GPU buffer) need not send them again each frame.
    #[must_use]
    pub fn unused_input_channels(&self) -> Range<usize> {
        match self.input {
            Input::Picture { side_planes: false } => {
                PICTURE_CHANNELS..PICTURE_CHANNELS + SIDE_CHANNELS
            }
            Input::Picture { side_planes: true } | Input::Luma { .. } => 0..0,
        }
    }

    /// Channels of the network's fine output, at output resolution.
    #[must_use]
    pub fn fine_channels(&self) -> usize {
        let blend = usize::from(self.temporal == Temporal::Blend) * BLEND_CHANNELS;
        self.planes + blend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(planes: usize, state_channels: usize) -> ModelSpec {
        ModelSpec {
            planes,
            temporal: Temporal::Blend,
            align: true,
            state_channels,
            input: Input::Picture { side_planes: true },
        }
    }

    fn recurrent() -> ModelSpec {
        ModelSpec {
            temporal: Temporal::Recurrent,
            input: Input::Luma {
                coding: true,
                confidence: true,
            },
            ..spec(1, 16)
        }
    }

    #[test]
    fn the_recurrent_input_is_luma_confidence_state_previous_output_and_coding() {
        let spec = recurrent();
        assert!(spec.validate().is_ok());
        assert_eq!(spec.input_channels(), 1 + 1 + 16 + 4 + CODING_CHANNELS);
        assert_eq!((spec.fine_channels(), spec.hidden_channels()), (1, 16));
        assert!(spec.unused_input_channels().is_empty());
    }

    #[test]
    fn the_recurrent_design_must_be_luma_only() {
        let chroma_too = ModelSpec {
            input: Input::Picture { side_planes: false },
            ..recurrent()
        };
        assert!(chroma_too.validate().is_err());
        assert!(Input::from_contract(true, true, false, false).is_err());
        assert!(Input::from_contract(false, false, true, false).is_err());
        assert!("recurrent".parse::<Temporal>().is_ok() && "other".parse::<Temporal>().is_err());
    }

    #[test]
    fn only_luma_or_all_planes_are_accepted() {
        assert!(spec(1, 8).validate().is_ok() && spec(3, 8).validate().is_ok());
        assert!(spec(4, 8).validate().is_err() && spec(0, 8).validate().is_err());
    }

    #[test]
    fn only_a_model_without_side_planes_leaves_channels_unused() {
        let with_side = spec(1, 8);
        let without_side = ModelSpec {
            input: Input::Picture { side_planes: false },
            ..with_side
        };
        assert!(with_side.unused_input_channels().is_empty());
        let unused = without_side.unused_input_channels();
        assert_eq!(unused, PICTURE_CHANNELS..PICTURE_CHANNELS + SIDE_CHANNELS);
        assert!(unused.end < without_side.input_channels());
    }

    #[test]
    fn a_temporal_model_needs_a_sane_state_size() {
        assert!(spec(1, 0).validate().is_err() && spec(1, 100_000).validate().is_err());
    }
}
