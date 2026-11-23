//! Grain on the picture a session hands back: the same grain, sample for
//! sample, that the integer runtime shows (`crate::grain`), added to the
//! float luma after history has taken its copy.

use super::Session;
use crate::fixed::PICTURE_MAX;
use crate::grain::GrainProfile;

/// A picture step (in which the grain is made) on the session's scale of 0 to 1.
const STEP: f32 = 1.0 / PICTURE_MAX as f32;

impl Session {
    /// The grain the luma of the pictures finished from now on carries (the
    /// default profile until this is called). The network's input and the
    /// history never see it.
    pub fn set_grain(&mut self, profile: GrainProfile) {
        self.grain.set(profile);
    }

    /// Add the frame's grain to the output luma. History has been stored by
    /// now, and the float picture is not clipped here: colour conversion does that.
    pub(super) fn add_grain(&mut self) {
        let Some(steps) = self.grain.next_frame() else {
            return;
        };
        for (sample, &step) in self.output.plane_mut(0).iter_mut().zip(steps) {
            *sample += step as f32 * STEP;
        }
    }
}
