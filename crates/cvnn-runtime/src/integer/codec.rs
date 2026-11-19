//! Between the float tensors of the session and the packed 8-bit planes of
//! the integer network: the input quantised on its way in, the outputs made
//! float again on their way out.

use cvnn_common::{Error, Result};

use super::arith::stored_from_unsigned;
use super::format::IntegerModel;
use super::{
    PICTURE_OUTPUTS, PICTURE_WORDS, RESIDUAL_STEPS, SIGNED_LEVELS, UNSIGNED_LEVELS, WORD_CHANNELS,
};
use crate::SR_SCALE;

fn check_len(actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::FrameSizeMismatch { expected, actual })
    }
}

/// `value` on a grid of `levels` steps per unit, to the nearest step (halves
/// to even, as the training simulation rounds), within `low..=high`.
fn to_grid(value: f32, levels: i32, low: i32, high: i32) -> i32 {
    ((value * levels as f32).round_ties_even() as i32).clamp(low, high)
}

/// One float plane as stored bytes: unsigned planes hold 0 to 1, signed ones -1 to 1.
fn store_plane(plane: &[f32], unsigned: bool, stored: &mut [u8]) {
    for (byte, &value) in stored.iter_mut().zip(plane) {
        *byte = if unsigned {
            stored_from_unsigned(to_grid(value, UNSIGNED_LEVELS, 0, UNSIGNED_LEVELS))
        } else {
            to_grid(value, SIGNED_LEVELS, -SIGNED_LEVELS, SIGNED_LEVELS) as i8 as u8
        };
    }
}

impl IntegerModel {
    /// Quantise the float network input (`channels` x height x width,
    /// channel-major) into `packed`: `input_words()` words per pixel.
    ///
    /// # Errors
    ///
    /// [`Error::FrameSizeMismatch`] if the two do not describe one frame.
    pub fn pack_input(&self, features: &[f32], packed: &mut [u32]) -> Result<()> {
        let words = self.input_words();
        let plane = packed.len() / words;
        check_len(packed.len(), plane * words)?;
        check_len(features.len(), plane * self.input_channels)?;
        // A plane at a time, then four planes to a word: both loops run over
        // contiguous memory, which is what makes megabytes per frame affordable.
        let mut stored = vec![0; WORD_CHANNELS * plane];
        for (word, slots) in self.slots.as_chunks::<WORD_CHANNELS>().0.iter().enumerate() {
            let unsigned = word < self.unsigned_words;
            for (bytes, slot) in stored.chunks_exact_mut(plane).zip(slots) {
                match slot {
                    Some(channel) => {
                        let source = &features[channel * plane..(channel + 1) * plane];
                        store_plane(source, unsigned, bytes);
                    }
                    None => bytes.fill(if unsigned { stored_from_unsigned(0) } else { 0 }),
                }
            }
            let (first, rest) = stored.split_at(plane);
            let (second, rest) = rest.split_at(plane);
            let (third, fourth) = rest.split_at(plane);
            for (pixel, target) in packed.iter_mut().skip(word).step_by(words).enumerate() {
                *target =
                    u32::from_le_bytes([first[pixel], second[pixel], third[pixel], fourth[pixel]]);
            }
        }
        Ok(())
    }

    /// The float outputs the session finishes a frame from: `fine`, the
    /// residual at output resolution (2 height x 2 width), and `coarse`, the
    /// state (`state_outputs()` x height x width).
    ///
    /// # Errors
    ///
    /// [`Error::FrameSizeMismatch`] if a buffer is not the size `width` and `height` imply.
    pub fn unpack_output(
        &self,
        packed: &[u32],
        width: usize,
        height: usize,
        fine: &mut [f32],
        coarse: &mut [f32],
    ) -> Result<()> {
        let (plane, words) = (width * height, self.output_words());
        check_len(packed.len(), plane * words)?;
        check_len(fine.len(), plane * PICTURE_OUTPUTS)?;
        check_len(coarse.len(), plane * self.state_outputs())?;
        let residual_unit = 1.0 / (UNSIGNED_LEVELS * RESIDUAL_STEPS) as f32;
        let state_unit = 1.0 / SIGNED_LEVELS as f32;
        let fine_width = SR_SCALE * width;
        for (y, row) in packed.chunks_exact(width * words).enumerate() {
            for output in 0..PICTURE_OUTPUTS {
                let start = (SR_SCALE * y + output / SR_SCALE) * fine_width + output % SR_SCALE;
                let targets = fine[start..].iter_mut().step_by(SR_SCALE);
                let sources = row.iter().skip(output / 2).step_by(words);
                for (target, &word) in targets.zip(sources) {
                    let half = (word >> (16 * (output % 2))) as u16;
                    *target = f32::from(half as i16) * residual_unit;
                }
            }
        }
        for (channel, target) in coarse.chunks_exact_mut(plane.max(1)).enumerate() {
            let sources = packed.iter().skip(PICTURE_WORDS + channel / WORD_CHANNELS);
            for (value, &word) in target.iter_mut().zip(sources.step_by(words)) {
                let stored = word.to_le_bytes()[channel % WORD_CHANNELS];
                *value = f32::from(stored as i8) * state_unit;
            }
        }
        Ok(())
    }
}

impl IntegerModel {
    /// Pack input planes already on their 8-bit grids (`channels` x height x
    /// width: unsigned planes 0 to 255, signed ones -127 to 127) into `packed`.
    ///
    /// # Errors
    ///
    /// [`Error::FrameSizeMismatch`] if the two do not describe one frame.
    pub fn pack_levels(&self, levels: &[i32], packed: &mut [u32]) -> Result<()> {
        let words = self.input_words();
        let plane = packed.len() / words;
        check_len(packed.len(), plane * words)?;
        check_len(levels.len(), plane * self.input_channels)?;
        for (pixel, target) in packed.chunks_exact_mut(words).enumerate() {
            let slots = self.slots.as_chunks::<WORD_CHANNELS>().0;
            for (word, (target, slots)) in target.iter_mut().zip(slots).enumerate() {
                let unsigned = word < self.unsigned_words;
                *target = u32::from_le_bytes(slots.map(|slot| {
                    let level = slot.map_or(0, |channel| levels[channel * plane + pixel]);
                    if unsigned {
                        stored_from_unsigned(level)
                    } else {
                        level as i8 as u8
                    }
                }));
            }
        }
        Ok(())
    }

    /// One pixel's outputs from its packed words: the four picture outputs
    /// (residuals in 1/32 level, for the 2x2 output pixels row by row), and
    /// through `state` each state output with its channel, -127 to 127.
    pub fn unpack_pixel(
        &self,
        words: &[u32],
        mut state: impl FnMut(usize, i32),
    ) -> [i32; PICTURE_OUTPUTS] {
        for (word, &packed) in words[PICTURE_WORDS..].iter().enumerate() {
            for (byte, stored) in packed.to_le_bytes().into_iter().enumerate() {
                state(word * WORD_CHANNELS + byte, i32::from(stored as i8));
            }
        }
        core::array::from_fn(|output| {
            i32::from((words[output / 2] >> (16 * (output % 2))) as u16 as i16)
        })
    }
}
