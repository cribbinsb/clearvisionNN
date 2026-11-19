//! How a frame was coded, as the network is told it: the planes a receiver
//! fills from the slice headers of the stream it decodes.

/// Planes per frame: whether the values are known, the quantiser on a 0-1
/// scale, and whether the frame is a key frame.
pub const CODING_CHANNELS: usize = 3;
/// The largest quantiser of H.264 and H.265 at 8 bits; it sets the 0-1 scale.
pub const MAX_QP: f32 = 51.0;

/// What a decoder knows about one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameCoding {
    /// The frame's average quantiser, 0 to [`MAX_QP`].
    pub quantiser: f32,
    /// Whether the frame is a key frame.
    pub key_frame: bool,
}

/// The value of each coding plane; all zero when nothing is known, as for
/// video that never met a codec.
#[must_use]
pub fn coding_planes(coding: Option<FrameCoding>) -> [f32; CODING_CHANNELS] {
    coding.map_or([0.0; CODING_CHANNELS], |known| {
        let key = f32::from(u8::from(known.key_frame));
        [1.0, known.quantiser / MAX_QP, key]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_coding_is_all_zero() {
        assert_eq!(coding_planes(None), [0.0; CODING_CHANNELS]);
    }

    #[test]
    fn known_coding_scales_the_quantiser() {
        let coding = FrameCoding {
            quantiser: MAX_QP,
            key_frame: true,
        };
        assert_eq!(coding_planes(Some(coding)), [1.0, 1.0, 1.0]);
    }
}
