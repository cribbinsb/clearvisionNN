//! The reference for a whole frame in integers: planes in, picture out, with
//! the recurrent state and the previous output kept between frames.

use cvnn_common::{BLOCK_SIZE, Error, Result};

use super::PICTURE_MAX;
use super::colour::rgba;
use super::confidence::{block_errors, plane as confidence_plane};
use super::conventional::{LEVEL_SHIFT, conventional};
use super::motion::{BlockVector, Flow};
use super::upscale::round_shift;
use super::warp::taps;
use crate::SR_SCALE;
use crate::grain::{Grain, GrainProfile, shown};
use crate::integer::{IntegerModel, UNSIGNED_LEVELS};
use crate::spec::{Input, ModelSpec, Temporal};

/// Outputs per input pixel: the 2x2 block it becomes.
const FOLD: usize = SR_SCALE * SR_SCALE;

/// One decoded frame and what is known about it.
#[derive(Debug, Clone, Copy)]
pub struct FrameInput<'a> {
    /// Luma, `height` x `width`.
    pub y: &'a [u8],
    /// Chroma, each `height / 2` x `width / 2`.
    pub u: &'a [u8],
    /// See `u`.
    pub v: &'a [u8],
    /// The block motion field towards the previous frame ([`super::block_vectors`]).
    pub motion: &'a [BlockVector],
    /// History from before this frame is to be discarded.
    pub reset: bool,
}

/// A stream being upscaled entirely in integers around an integer model.
#[derive(Debug)]
pub struct IntegerRuntime {
    model: IntegerModel,
    spec: ModelSpec,
    size: (usize, usize),
    flow: Flow,
    /// The conventional upscale: luma and the two chroma planes at output
    /// size, in picture steps.
    base: [Vec<i32>; 3],
    /// The previous output luma (picture steps), state (channel-major, signed
    /// levels) and input luma; meaningful once `stored`.
    previous: Vec<i32>,
    state: Vec<i32>,
    luma_before: Vec<u8>,
    stored: bool,
    /// The network's input planes on their 8-bit grids, then packed, and its packed outputs.
    levels: Vec<i32>,
    packed: Vec<u32>,
    outputs: Vec<u32>,
    /// The conventional upscale and the output as RGBA words.
    conventional: Vec<u32>,
    picture: Vec<u32>,
    /// What is added to the output shown, and never to `previous`.
    grain: Grain,
}

impl IntegerRuntime {
    /// A runtime for `width` x `height` input frames of the luma-only model
    /// `model`, whose contract is `spec`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSpec`] if contract and model disagree or the design is
    /// not luma-only; [`Error::InvalidDimensions`] for sizes that cannot be warped or halved.
    pub fn new(model: IntegerModel, spec: ModelSpec, width: usize, height: usize) -> Result<Self> {
        spec.validate()?;
        let luma_only =
            matches!(spec.input, Input::Luma { .. }) && spec.temporal != Temporal::Blend;
        let agree = model.input_channels() == spec.input_channels()
            && model.state_outputs() == spec.hidden_channels();
        if !(luma_only && agree) {
            let why = "the integer runtime needs a luma-only model whose file matches its contract";
            return Err(Error::InvalidSpec(why.to_owned()));
        }
        if width < 2 || height < 2 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            let reason = "width and height must be even and at least 2".to_owned();
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason,
            });
        }
        let (plane, fine) = (width * height, FOLD * width * height);
        Ok(Self {
            spec,
            size: (height, width),
            flow: Flow::default(),
            base: core::array::from_fn(|_| vec![0; fine]),
            previous: vec![0; fine],
            state: vec![0; spec.hidden_channels() * plane],
            luma_before: vec![0; plane],
            stored: false,
            levels: vec![0; spec.input_channels() * plane],
            packed: vec![0; model.input_words() * plane],
            outputs: Vec::new(),
            conventional: vec![0; fine],
            picture: vec![0; fine],
            grain: Grain::new(height, width),
            model,
        })
    }

    /// Forget earlier frames: the next one starts a new stream.
    pub fn restart(&mut self) {
        self.stored = false;
        self.grain.restart();
    }

    /// The grain the pictures of the frames from now on are shown with (the
    /// default profile until this is called). Nothing but
    /// [`IntegerRuntime::picture`] changes.
    pub fn set_grain(&mut self, profile: GrainProfile) {
        self.grain.set(profile);
    }

    /// Upscale one frame; `tile` is the network's tile side (0: whole frame).
    ///
    /// # Errors
    ///
    /// [`Error::FrameSizeMismatch`] if a plane or the motion field is not the configured size.
    pub fn step(&mut self, frame: &FrameInput, tile: usize) -> Result<()> {
        let (height, width) = self.size;
        let plane = height * width;
        let blocks = height.div_ceil(BLOCK_SIZE) * width.div_ceil(BLOCK_SIZE);
        for (actual, expected) in [
            (frame.y.len(), plane),
            (frame.u.len(), plane / 4),
            (frame.v.len(), plane / 4),
            (frame.motion.len(), blocks),
        ] {
            if actual != expected {
                return Err(Error::FrameSizeMismatch { expected, actual });
            }
        }
        conventional(frame.y, [frame.u, frame.v], self.size, &mut self.base);
        let [y, u, v] = &self.base;
        for (pixel, word) in self.conventional.iter_mut().enumerate() {
            *word = rgba(y[pixel], u[pixel], v[pixel]);
        }
        self.write_input(frame);
        self.model.pack_levels(&self.levels, &mut self.packed)?;
        self.outputs = self.model.infer(&self.packed, width, height, tile)?;
        self.finish(frame.y);
        Ok(())
    }

    /// The network's input planes, in the contract's channel order: luma;
    /// confidence; state; the previous output folded to input size.
    fn write_input(&mut self, frame: &FrameInput) {
        let (height, width) = self.size;
        let plane = height * width;
        let (luma, rest) = self.levels.split_at_mut(plane);
        for (level, &sample) in luma.iter_mut().zip(frame.y) {
            *level = i32::from(sample);
        }
        let Input::Luma { confidence } = self.spec.input else {
            return; // refused at construction
        };
        if self.spec.temporal == Temporal::Recurrent {
            let (confident, history) = rest.split_at_mut(usize::from(confidence) * plane);
            let (state, folded) = history.split_at_mut(self.state.len());
            // The previous output brought onto this frame, at output size, in picture steps.
            let mut carried = self.base[0].clone();
            if self.stored && !frame.reset {
                // A model that does not align its history takes it where it lies.
                let still = vec![[0; 2]; frame.motion.len()];
                let motion = if self.spec.align {
                    frame.motion
                } else {
                    &still
                };
                self.flow.set(motion, self.size);
                let [dx, dy] = &self.flow.coarse;
                let coarse = taps(self.size, [dx, dy]);
                if confidence {
                    let errors = block_errors(frame.y, &self.luma_before, &coarse, self.size);
                    confidence_plane(&errors, self.size, confident);
                }
                for (target, source) in state
                    .chunks_exact_mut(plane)
                    .zip(self.state.chunks_exact(plane))
                {
                    for (level, tap) in target.iter_mut().zip(&coarse) {
                        *level = tap.sample(|at| source[at], width);
                    }
                }
                let [dx, dy] = &self.flow.fine;
                let fine = taps((SR_SCALE * height, SR_SCALE * width), [dx, dy]);
                for (sample, tap) in carried.iter_mut().zip(&fine) {
                    *sample = tap.sample(|at| self.previous[at], SR_SCALE * width);
                }
            } else {
                // No usable history: the conventional upscale stands in, the
                // state is empty, and there is no prediction to distrust.
                confident.fill(UNSIGNED_LEVELS);
                state.fill(0);
            }
            for (offset, target) in folded.chunks_exact_mut(plane).enumerate() {
                for (pixel, level) in target.iter_mut().enumerate() {
                    let row = SR_SCALE * (pixel / width) + offset / SR_SCALE;
                    let column = SR_SCALE * (pixel % width) + offset % SR_SCALE;
                    let steps = carried[row * SR_SCALE * width + column];
                    *level = round_shift(steps, LEVEL_SHIFT).clamp(0, UNSIGNED_LEVELS);
                }
            }
        }
    }

    /// Add the network's detail to the conventional upscale, convert to
    /// RGBA with the frame's grain, and keep what the next frame needs.
    /// The grain is only in the picture: what is kept is clean.
    fn finish(&mut self, luma: &[u8]) {
        let (height, width) = self.size;
        let plane = height * width;
        let words = self.model.output_words();
        let [base, u, v] = &self.base;
        let grain = self.grain.next_frame();
        for (pixel, outputs) in self.outputs.chunks_exact(words).enumerate() {
            let state = &mut self.state;
            let keep = |channel: usize, level: i32| state[channel * plane + pixel] = level;
            let residuals = self.model.unpack_pixel(outputs, keep);
            for (offset, residual) in residuals.into_iter().enumerate() {
                let row = SR_SCALE * (pixel / width) + offset / SR_SCALE;
                let at = row * SR_SCALE * width + SR_SCALE * (pixel % width) + offset % SR_SCALE;
                let sample = (base[at] + residual).clamp(0, PICTURE_MAX);
                self.previous[at] = sample;
                let lit = grain.map_or(sample, |steps| shown(sample, steps[at]));
                self.picture[at] = rgba(lit, u[at], v[at]);
            }
        }
        self.luma_before.copy_from_slice(luma);
        self.stored = true;
    }

    /// The network's packed input for the last frame.
    #[must_use]
    pub fn packed_input(&self) -> &[u32] {
        &self.packed
    }

    /// The network's packed outputs for the last frame.
    #[must_use]
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }

    /// The last frame's output as RGBA words (red in the low byte), at twice the input size.
    #[must_use]
    pub fn picture(&self) -> &[u32] {
        &self.picture
    }

    /// The last frame's conventional upscale, as [`IntegerRuntime::picture`].
    #[must_use]
    pub fn conventional(&self) -> &[u32] {
        &self.conventional
    }

    /// The last frame's output luma in picture steps (1/32 of an 8-bit
    /// level), as it is kept for the next frame: without grain.
    #[must_use]
    pub fn luma(&self) -> &[i32] {
        &self.previous
    }

    /// The model this runtime runs.
    #[must_use]
    pub fn model(&self) -> &IntegerModel {
        &self.model
    }
}
