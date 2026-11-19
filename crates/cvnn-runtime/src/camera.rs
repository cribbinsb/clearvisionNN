//! A simulated camera for the page's sender: what happens to a full-size
//! original before a real sender would encode it, in the order the training
//! degradation applies it: optical blur, sensor noise, the camera's own
//! denoise, then the 2x reduction.

use crate::SR_SCALE;
use crate::blur::Blur;
use crate::denoise::Denoise;
use crate::kernel::Kernel;
use crate::noise::SensorNoise;
use crate::reduce::Reducer;

const RGBA_BYTES: usize = 4;

/// What the camera does to a frame. A level of zero switches its step off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exposure {
    /// Deviation of the optical blur, in samples of the original.
    pub blur: f32,
    /// Deviation of the sensor noise, in luma levels.
    pub noise: f32,
    /// Strength of the camera's denoise, in luma levels.
    pub denoise: f32,
}

/// A camera for originals of one size; the same seed gives the same noise.
#[derive(Debug, Clone)]
pub struct Camera {
    blur: Blur,
    noise: SensorNoise,
    denoise: Denoise,
    reducer: Reducer,
    /// The picture on its way through the steps; each step that runs writes the other.
    stages: [Vec<u8>; 2],
}

impl Camera {
    /// A camera for `width` x `height` originals (both even) that reduces with `kernel`.
    #[must_use]
    pub fn new(width: usize, height: usize, kernel: Kernel, seed: u32) -> Self {
        let stage = vec![0; width * height * RGBA_BYTES];
        Self {
            blur: Blur::new(width, height),
            noise: SensorNoise::new(seed),
            denoise: Denoise::new(width, height),
            reducer: Reducer::new(width, height, kernel),
            stages: [stage.clone(), stage],
        }
    }

    /// Bytes of the reduced picture this camera writes.
    #[must_use]
    pub fn reduced_len(&self) -> usize {
        self.stages[0].len() / (SR_SCALE * SR_SCALE)
    }

    /// Write the half-size picture a camera set to `exposure` makes of the
    /// RGBA `original` to `reduced`, also RGBA.
    ///
    /// # Panics
    ///
    /// If either slice is not the size the camera was made for.
    pub fn capture(&mut self, original: &[u8], exposure: Exposure, reduced: &mut [u8]) {
        assert_eq!(original.len(), self.stages[0].len());
        let [first, second] = &mut self.stages;
        // Where the picture is: nowhere yet (the original), or one of the stages.
        let mut held: Option<&mut Vec<u8>> = None;
        let mut spare = Some(first);
        let mut other = Some(second);
        if exposure.blur > 0.0 {
            let into = spare.take().expect("a free stage");
            self.blur.apply(original, exposure.blur, into);
            held = Some(into);
        }
        if exposure.noise > 0.0 {
            let into = if held.is_some() {
                other.take()
            } else {
                spare.take()
            }
            .expect("a free stage");
            self.noise.add(
                held.as_deref().map_or(original, Vec::as_slice),
                exposure.noise,
                into,
            );
            held = Some(into);
        }
        if exposure.denoise > 0.0 {
            let picture = held.take().unwrap_or_else(|| {
                let into = spare.take().expect("a free stage");
                into.copy_from_slice(original);
                into
            });
            self.denoise.apply(picture, exposure.denoise);
            held = Some(picture);
        }
        self.reducer
            .reduce(held.as_deref().map_or(original, Vec::as_slice), reduced);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFF: Exposure = Exposure {
        blur: 0.0,
        noise: 0.0,
        denoise: 0.0,
    };

    fn textured(width: usize, height: usize) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let level = if (i % width / 2 + i / width / 2).is_multiple_of(2) {
                    70
                } else {
                    190
                };
                [level, level, level, 255]
            })
            .collect()
    }

    #[test]
    fn with_every_step_off_the_camera_only_reduces() {
        let (width, height) = (32, 24);
        let original = textured(width, height);
        let mut camera = Camera::new(width, height, Kernel::Bicubic, 1);
        let mut captured = vec![0; camera.reduced_len()];
        camera.capture(&original, OFF, &mut captured);
        let mut reduced = vec![0; captured.len()];
        Reducer::new(width, height, Kernel::Bicubic).reduce(&original, &mut reduced);
        assert_eq!(captured, reduced);
    }

    #[test]
    fn every_combination_of_steps_runs_and_changes_the_picture() {
        let (width, height) = (32, 24);
        let original = textured(width, height);
        let mut plain = vec![0; width * height];
        Camera::new(width, height, Kernel::Area, 1).capture(&original, OFF, &mut plain);
        for steps in 1..8_u8 {
            let level = |bit: u8, value: f32| if steps & bit == 0 { 0.0 } else { value };
            let exposure = Exposure {
                blur: level(1, 1.0),
                noise: level(2, 4.0),
                denoise: level(4, 60.0),
            };
            let mut captured = vec![0; plain.len()];
            Camera::new(width, height, Kernel::Area, 1).capture(&original, exposure, &mut captured);
            assert_ne!(captured, plain, "steps {steps:03b}");
        }
    }
}
