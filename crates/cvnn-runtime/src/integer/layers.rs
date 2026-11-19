//! The layers of an integer model as read from its file.

use cvnn_common::Result;

use super::WORD_CHANNELS;
use super::format::{MULTIPLIER_RANGE, Reader, TAPS};

/// Largest shift a requantisation may use: beyond it a 32-bit value is all rounding.
const SHIFT_RANGE: core::ops::Range<i32> = 0..31;
/// Identity multipliers stay below this so their product with an 8-bit
/// activation, added to a scaled accumulator, stays inside 32 bits.
const IDENTITY_RANGE: core::ops::Range<i32> = 0..1 << 22;
/// Skip multipliers times an 8-bit activation, two of them summed, stay inside 32 bits.
const SKIP_RANGE: core::ops::Range<i32> = 0..1 << 22;

/// A 3x3 convolution with its per-output-channel requantisation.
#[derive(Debug, Clone)]
pub(super) struct Conv {
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    /// Signed 8-bit weights, `[tap][input][output]`.
    pub(super) weights: Vec<i32>,
    /// In accumulator units.
    pub(super) bias: Vec<i32>,
    pub(super) multiplier: Vec<i32>,
    pub(super) shift: Vec<u32>,
    /// The identity branch's multiplier per channel; empty without one.
    pub(super) identity: Vec<i32>,
}

impl Conv {
    /// The next layer in the file: packed weights `[tap][input word][output]`,
    /// then bias, multiplier and shift per output, then the identity
    /// multipliers if the layer has that branch.
    pub(super) fn read(
        reader: &mut Reader,
        inputs: usize,
        outputs: usize,
        identity: bool,
    ) -> Result<Self> {
        let words = inputs / WORD_CHANNELS;
        let packed = reader.take(TAPS * words * outputs)?;
        let mut weights = vec![0; TAPS * inputs * outputs];
        for (index, word) in packed.iter().enumerate() {
            let (tap_word, output) = (index / outputs, index % outputs);
            for (byte, weight) in word.to_le_bytes().into_iter().enumerate() {
                let input = tap_word * WORD_CHANNELS + byte;
                weights[input * outputs + output] = i32::from(weight as i8);
            }
        }
        let bias = reader.take(outputs)?.to_vec();
        let multiplier = reader.bounded(outputs, MULTIPLIER_RANGE)?;
        let shift = reader.bounded(outputs, SHIFT_RANGE)?;
        let identity = if identity {
            reader.bounded(outputs, IDENTITY_RANGE)?
        } else {
            Vec::new()
        };
        Ok(Self {
            inputs,
            outputs,
            weights,
            bias,
            multiplier,
            shift: shift.into_iter().map(i32::cast_unsigned).collect(),
            identity,
        })
    }

    /// The weights of one tap and input channel, per output.
    pub(super) fn tap(&self, tap: usize, input: usize) -> &[i32] {
        let start = (tap * self.inputs + input) * self.outputs;
        &self.weights[start..start + self.outputs]
    }
}

/// The long skip: stem output plus body output, brought onto one scale.
#[derive(Debug, Clone, Copy)]
pub(super) struct SkipAdd {
    pub(super) stem: i32,
    pub(super) body: i32,
    pub(super) shift: u32,
}

impl SkipAdd {
    pub(super) fn read(reader: &mut Reader) -> Result<Self> {
        let multipliers = reader.bounded(2, SKIP_RANGE)?;
        let shift = reader.bounded(1, SHIFT_RANGE)?[0];
        Ok(Self {
            stem: multipliers[0],
            body: multipliers[1],
            shift: shift.cast_unsigned(),
        })
    }
}
