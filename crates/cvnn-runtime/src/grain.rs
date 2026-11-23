//! Grain: fresh pseudo-random noise added to the luma of the picture that is
//! shown, so that camera material does not look waxy once the network has
//! taken its sensor noise away. It is never added to the picture kept as
//! history or fed back to the network, so a stream with grain and one without
//! differ in nothing but what is shown.
//!
//! `shown = picture + level x unit grain(frame, x, y)`, the level being that
//! of the stream's profile ([`GrainProfile`]): all of it integer arithmetic
//! that docs/INTEGER.md (section 9) defines, so that the float session, the
//! integer runtime and the page's shaders show the same grain. This module
//! is its reference.

mod amplitude;
mod noise;
mod profile;

pub use amplitude::shown;
pub use profile::GrainProfile;

use crate::SR_SCALE;

/// The grain of one stream: its profile and the index of the next frame.
#[derive(Debug, Clone)]
pub struct Grain {
    /// Height and width of the input; the grain is at twice that.
    size: (usize, usize),
    profile: GrainProfile,
    /// The index of the next frame: frames since the stream started. It
    /// wraps at 2^32, after which the grain repeats.
    frame: u32,
    /// The last frame's grain, in picture steps.
    steps: Vec<i32>,
}

impl Grain {
    /// The grain of the default profile, for `height` x `width` input frames.
    #[must_use]
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            size: (height, width),
            profile: GrainProfile::default(),
            frame: 0,
            steps: Vec::new(),
        }
    }

    /// Change the profile from the next frame on. The frame index runs on.
    pub fn set(&mut self, profile: GrainProfile) {
        self.profile = profile;
    }

    /// Start a new stream: the next frame is frame 0 again.
    pub fn restart(&mut self) {
        self.seek(0);
    }

    /// Make the next frame the one of index `frame`, to show a known frame's grain.
    pub fn seek(&mut self, frame: u32) {
        self.frame = frame;
    }

    /// The grain of the next frame, at twice the input size, in picture steps
    /// (1/32 of a grey level), to add to the luma shown ([`shown`]); `None`
    /// when there is none to add. Every call is a frame, with or without grain.
    pub fn next_frame(&mut self) -> Option<&[i32]> {
        let frame = self.frame;
        self.frame = frame.wrapping_add(1);
        let amplitude = self.profile.amplitude();
        if amplitude == 0 {
            return None;
        }
        let fine = (SR_SCALE * self.size.0, SR_SCALE * self.size.1);
        self.steps.resize(fine.0 * fine.1, 0);
        noise::field(frame, fine, &mut self.steps);
        for step in &mut self.steps {
            *step = amplitude::scaled(amplitude, *step);
        }
        Some(&self.steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEIGHT: usize = 64;
    const WIDTH: usize = 96;
    /// Picture steps per grey level.
    const STEPS: f64 = 32.0;

    fn grain(profile: GrainProfile) -> Grain {
        let mut grain = Grain::new(HEIGHT, WIDTH);
        grain.set(profile);
        grain
    }

    #[test]
    fn the_none_profile_adds_nothing() {
        assert!(grain(GrainProfile::None).next_frame().is_none());
    }

    #[test]
    fn the_grain_is_the_profile_s_grey_levels_about_zero() {
        for profile in [GrainProfile::Produced, GrainProfile::Webcam] {
            let (mut grain, expected) = (grain(profile), f64::from(profile.level()));
            let steps = grain.next_frame().unwrap();
            assert_eq!(steps.len(), 4 * HEIGHT * WIDTH);
            let mean = steps.iter().map(|&v| f64::from(v)).sum::<f64>() / steps.len() as f64;
            assert!(mean.abs() < 0.03 * STEPS * expected, "mean {mean}");
            let squares = steps.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            let levels = (squares / steps.len() as f64).sqrt() / STEPS;
            assert!(
                (levels / expected - 1.0).abs() < 0.03,
                "{levels} grey levels"
            );
        }
    }

    #[test]
    fn every_frame_has_its_own_grain_and_an_index_always_the_same() {
        let mut stream = Grain::new(HEIGHT, WIDTH);
        let first = stream.next_frame().unwrap().to_vec();
        let second = stream.next_frame().unwrap().to_vec();
        assert_ne!(first, second);
        stream.restart();
        assert_eq!(stream.next_frame().unwrap(), first);
        stream.seek(1);
        assert_eq!(stream.next_frame().unwrap(), second);
        // A frame without grain is a frame all the same: the index runs on.
        let mut late = grain(GrainProfile::None);
        assert!(late.next_frame().is_none());
        late.set(GrainProfile::Webcam);
        assert_eq!(late.next_frame().unwrap(), second);
    }

    #[test]
    fn the_index_wraps() {
        let mut stream = Grain::new(HEIGHT, WIDTH);
        let first = stream.next_frame().unwrap().to_vec();
        stream.seek(u32::MAX);
        assert_ne!(stream.next_frame().unwrap(), first);
        assert_eq!(stream.next_frame().unwrap(), first);
    }
}
