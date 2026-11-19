//! The reference network: plain loops over one tile at a time, exactly the
//! arithmetic of docs/INTEGER.md.

use cvnn_common::{Error, Result};

use super::arith::{requantise, round_shift, to_unsigned, unsigned_from_stored};
use super::format::{IntegerModel, TAPS};
use super::layers::Conv;
use super::tiles::{Rect, Tiling};
use super::{PICTURE_OUTPUTS, PICTURE_WORDS, SIGNED_LEVELS, UNSIGNED_LEVELS, WORD_CHANNELS};

/// Side of the kernel.
const KERNEL: isize = 3;

/// The 8-bit activations of one tile's window, `channels` per pixel.
struct Activations {
    origin: (isize, isize),
    stride: usize,
    channels: usize,
    data: Vec<u8>,
}

impl Activations {
    fn at(&self, x: isize, y: isize) -> usize {
        let (column, row) = (
            (x - self.origin.0).cast_unsigned(),
            (y - self.origin.1).cast_unsigned(),
        );
        (row * self.stride + column) * self.channels
    }
}

/// `conv`'s accumulators at (x, y): the bias plus every tap inside `valid`
/// (taps outside read zeros), with `value(x, y, input)` supplying the inputs.
fn accumulate(
    conv: &Conv,
    accumulators: &mut [i32],
    (x, y): (isize, isize),
    valid: Rect,
    value: impl Fn(isize, isize, usize) -> i32,
) {
    accumulators.copy_from_slice(&conv.bias);
    for tap in 0..TAPS {
        let offset = tap.cast_signed();
        let (sx, sy) = (x + offset % KERNEL - 1, y + offset / KERNEL - 1);
        if !valid.contains(sx, sy) {
            continue;
        }
        for input in 0..conv.inputs {
            let sample = value(sx, sy, input);
            for (accumulator, weight) in accumulators.iter_mut().zip(conv.tap(tap, input)) {
                *accumulator = accumulator.wrapping_add(sample.wrapping_mul(*weight));
            }
        }
    }
}

impl IntegerModel {
    /// Run the network on packed input planes (`input_words()` per pixel, row
    /// by row) and return its packed outputs (`output_words()` per pixel).
    /// `tile` is the tile side, or 0 for the whole frame as one tile.
    ///
    /// # Errors
    ///
    /// [`Error::FrameSizeMismatch`] if `input` is not `width` x `height`.
    pub fn infer(
        &self,
        input: &[u32],
        width: usize,
        height: usize,
        tile: usize,
    ) -> Result<Vec<u32>> {
        let expected = width * height * self.input_words();
        if input.len() != expected {
            return Err(Error::FrameSizeMismatch {
                expected,
                actual: input.len(),
            });
        }
        let tiling = Tiling::new(width, height, tile, self.halo(), self.body.len() + 2);
        let mut output = vec![0; width * height * self.output_words()];
        let channels = self.stem.outputs;
        let mut buffers: [Vec<u8>; 3] =
            core::array::from_fn(|_| vec![0; tiling.window_pixels() * channels]);
        for core in tiling.cores() {
            self.infer_tile(&tiling, core, input, &mut buffers, &mut output);
        }
        Ok(output)
    }

    fn infer_tile(
        &self,
        tiling: &Tiling,
        core: Rect,
        input: &[u32],
        buffers: &mut [Vec<u8>; 3],
        output: &mut [u32],
    ) {
        let (origin, stride) = tiling.window(core);
        let channels = self.stem.outputs;
        let window = |data| Activations {
            origin,
            stride,
            channels,
            data,
        };
        let [stem_data, first, second] = buffers;
        let mut stem = window(core::mem::take(stem_data));
        self.stem_layer(tiling, tiling.region(core, 0), input, &mut stem);
        let (mut current, mut next) = (
            window(core::mem::take(first)),
            window(core::mem::take(second)),
        );
        for (index, conv) in self.body.iter().enumerate() {
            let layer = index + 1;
            let source = if index == 0 { &stem } else { &current };
            let region = tiling.region(core, layer);
            body_layer(conv, source, tiling.region(core, index), region, &mut next);
            core::mem::swap(&mut current, &mut next);
        }
        self.add_skip(&stem, tiling.region(core, self.body.len()), &mut current);
        self.head_layer(tiling, core, &current, output);
        (*stem_data, *first, *second) = (stem.data, current.data, next.data);
    }

    /// The stem: mixed unsigned and signed planes, read from the frame itself.
    fn stem_layer(&self, tiling: &Tiling, region: Rect, input: &[u32], out: &mut Activations) {
        let (frame, words) = (tiling.frame(), self.input_words());
        let width = frame.x1.cast_unsigned();
        let conv = &self.stem;
        let mut accumulators = vec![0; conv.outputs];
        // Both kinds of plane on the one scale of 1 / (255 * 127).
        let value = |x: isize, y: isize, slot: usize| {
            let word = input
                [(y.cast_unsigned() * width + x.cast_unsigned()) * words + slot / WORD_CHANNELS];
            let stored = word.to_le_bytes()[slot % WORD_CHANNELS] as i8;
            if slot / WORD_CHANNELS < self.unsigned_words {
                unsigned_from_stored(stored) * SIGNED_LEVELS
            } else {
                i32::from(stored) * UNSIGNED_LEVELS
            }
        };
        for (x, y) in region.pixels() {
            accumulate(conv, &mut accumulators, (x, y), frame, value);
            let at = out.at(x, y);
            for (channel, &accumulator) in accumulators.iter().enumerate() {
                let scaled = requantise(
                    accumulator,
                    conv.multiplier[channel],
                    conv.shift[channel],
                    0,
                );
                out.data[at + channel] = to_unsigned(scaled);
            }
        }
    }

    /// The stem's output added to the body's, on the head's input scale.
    fn add_skip(&self, stem: &Activations, region: Rect, body: &mut Activations) {
        let skip = self.skip;
        for (x, y) in region.pixels() {
            let at = body.at(x, y);
            for channel in 0..body.channels {
                let sum = i32::from(body.data[at + channel]) * skip.body
                    + i32::from(stem.data[at + channel]) * skip.stem;
                body.data[at + channel] = to_unsigned(round_shift(sum, skip.shift));
            }
        }
    }

    /// The head: picture outputs as 16-bit residuals, state outputs through the tanh table.
    fn head_layer(&self, tiling: &Tiling, core: Rect, source: &Activations, output: &mut [u32]) {
        let conv = &self.head;
        let valid = tiling.region(core, self.body.len());
        let (width, words) = (tiling.frame().x1.cast_unsigned(), self.output_words());
        let mut accumulators = vec![0; conv.outputs];
        let mut scaled = vec![0; conv.outputs];
        for (x, y) in core.pixels() {
            let value = |sx, sy, input| i32::from(source.data[source.at(sx, sy) + input]);
            accumulate(conv, &mut accumulators, (x, y), valid, value);
            for (channel, target) in scaled.iter_mut().enumerate() {
                *target = requantise(
                    accumulators[channel],
                    conv.multiplier[channel],
                    conv.shift[channel],
                    0,
                );
            }
            let pixel = (y.cast_unsigned() * width + x.cast_unsigned()) * words;
            let (pictures, state) = scaled.split_at(PICTURE_OUTPUTS);
            for (word, pair) in pictures.as_chunks::<2>().0.iter().enumerate() {
                output[pixel + word] = residual_half(pair[0]) | residual_half(pair[1]) << 16;
            }
            for (word, values) in state.as_chunks::<WORD_CHANNELS>().0.iter().enumerate() {
                let bytes = values.map(|value| self.bounded_state(value));
                output[pixel + PICTURE_WORDS + word] = u32::from_le_bytes(bytes);
            }
        }
    }

    /// tanh of a state output, as a stored signed byte.
    fn bounded_state(&self, value: i32) -> u8 {
        let last = self.tanh.len() - 1;
        let magnitude = i32::from(self.tanh[(value.unsigned_abs() as usize).min(last)]);
        (magnitude * value.signum()) as i8 as u8
    }
}

/// One body layer: convolution plus the identity branch, rectified to 8 bits.
fn body_layer(conv: &Conv, source: &Activations, valid: Rect, region: Rect, out: &mut Activations) {
    let mut accumulators = vec![0; conv.outputs];
    for (x, y) in region.pixels() {
        let value = |sx, sy, input| i32::from(source.data[source.at(sx, sy) + input]);
        accumulate(conv, &mut accumulators, (x, y), valid, value);
        let (own, at) = (source.at(x, y), out.at(x, y));
        for (channel, &accumulator) in accumulators.iter().enumerate() {
            let identity =
                conv.identity[channel].wrapping_mul(i32::from(source.data[own + channel]));
            let scaled = requantise(
                accumulator,
                conv.multiplier[channel],
                conv.shift[channel],
                identity,
            );
            out.data[at + channel] = to_unsigned(scaled);
        }
    }
}

/// A picture output saturated to 16 bits, as half of an output word.
fn residual_half(value: i32) -> u32 {
    u32::from(value.clamp(i16::MIN.into(), i16::MAX.into()) as i16 as u16)
}
