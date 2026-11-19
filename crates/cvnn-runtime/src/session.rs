//! One video stream being upscaled: the two halves around the network call.

use cvnn_common::{Error, FrameFeatures, Result};
use cvnn_preproc::{PLANES, PreprocConfig, Preprocessor, SIDE_CHANNELS, write_side_planes};

mod luma;

use crate::SR_SCALE;
use crate::compose::{add_detail, blend, pixel_unshuffle};
use crate::grid::{GridFilter, GridSpreader};
use crate::history::{HiddenForm, History};
use crate::planes::Planes;
use crate::resample::Resampler;
use crate::spec::{Input, ModelSpec, PICTURE_CHANNELS, Temporal};

const MAX_SAMPLE: f32 = 255.0;
/// The network halves its input twice, so both dimensions must divide by this.
const NETWORK_ALIGN: usize = 4;

/// Upscales one stream frame by frame around an external network.
#[derive(Debug)]
pub struct Session {
    spec: ModelSpec,
    width: usize,
    height: usize,
    preprocessor: Preprocessor,
    /// The network's input for the frame being processed.
    input: Planes,
    /// The conventional upscale of the frame being processed.
    base: Planes,
    /// History on the current frame's grid (temporal models).
    previous: Planes,
    /// Whether that history is usable for the frame being processed.
    valid: bool,
    history: History,
    output: Planes,
    resampler: Resampler,
    spreader: GridSpreader,
    /// One chroma plane as floats, before it is brought to luma size.
    chroma: Vec<f32>,
    /// The side planes on the block grid.
    side: Vec<f32>,
    /// What only a luma-only model needs.
    luma: luma::LumaState,
    /// `prepare` has run and `finish` has not.
    awaiting_network: bool,
}

fn check_len(actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::FrameSizeMismatch { expected, actual })
    }
}

impl Session {
    /// A session for `width` x `height` input frames (both multiples of
    /// [`NETWORK_ALIGN`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSpec`] for a contract this runtime does not implement;
    /// [`Error::InvalidDimensions`] for sizes the network or analysis cannot take.
    pub fn new(spec: ModelSpec, width: usize, height: usize) -> Result<Self> {
        spec.validate()?;
        if !width.is_multiple_of(NETWORK_ALIGN) || !height.is_multiple_of(NETWORK_ALIGN) {
            let reason = format!("width and height must be multiples of {NETWORK_ALIGN}");
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason,
            });
        }
        let (fine_height, fine_width) = (SR_SCALE * height, SR_SCALE * width);
        Ok(Self {
            spec,
            width,
            height,
            preprocessor: Preprocessor::new(width, height, PreprocConfig::default())?,
            input: Planes::zeros(spec.input_channels(), height, width),
            base: Planes::zeros(PICTURE_CHANNELS, fine_height, fine_width),
            previous: Planes::zeros(spec.planes, fine_height, fine_width),
            valid: false,
            history: History::new(spec.planes, spec.hidden_channels(), height, width),
            output: Planes::zeros(PICTURE_CHANNELS, fine_height, fine_width),
            resampler: Resampler::default(),
            spreader: GridSpreader::default(),
            chroma: Vec::new(),
            side: Vec::new(),
            luma: luma::LumaState::new(spec, height, width),
            awaiting_network: false,
        })
    }

    /// Forget everything about earlier frames: the next frame starts a new
    /// stream. For seeks, source changes, and recovery after a failed
    /// network call.
    ///
    /// # Errors
    ///
    /// As [`Session::new`].
    pub fn restart(&mut self) -> Result<()> {
        self.preprocessor.restart()?;
        self.history.clear();
        self.awaiting_network = false;
        Ok(())
    }

    /// The model contract this session was built for.
    #[must_use]
    pub fn spec(&self) -> ModelSpec {
        self.spec
    }

    /// The conventional 2x upscale of the most recently prepared frame, for
    /// comparison with the network's result: 4:4:4 YUV in [0, 1].
    #[must_use]
    pub fn conventional(&self) -> &Planes {
        &self.base
    }

    /// Analyse the next frame (8-bit 4:2:0 planes) and return the network's
    /// input: `spec.input_channels()` x height x width, channel-major.
    ///
    /// Each prepared frame must be finished before the next is prepared: the
    /// analysis has already moved on to it, so skipping its output would leave
    /// motion and history describing different frames. If the network call
    /// fails, call [`Session::restart`].
    ///
    /// # Errors
    ///
    /// [`Error::OutOfOrder`] if the previous frame was not finished;
    /// [`Error::FrameSizeMismatch`] if a plane has the wrong length.
    pub fn prepare(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<&[f32]> {
        if self.awaiting_network {
            return Err(Error::OutOfOrder(
                "prepare called before the previous frame was finished",
            ));
        }
        let size = (self.height, self.width);
        check_len(u.len(), y.len() / 4)?;
        check_len(v.len(), y.len() / 4)?;
        let features = self.preprocessor.push(y)?;
        if let Input::Luma { coding, confidence } = self.spec.input {
            self.prepare_luma(y, u, v, &features, coding, confidence);
        } else {
            self.write_picture(y, u, v);
            self.write_side_planes(&features);
            for channel in 0..PICTURE_CHANNELS {
                let (source, target) = (self.input.plane(channel), self.base.plane_mut(channel));
                self.resampler.bicubic2x(source, size, target);
            }
            if self.spec.temporal == Temporal::Blend {
                self.write_history(&features);
            }
        }
        self.awaiting_network = true;
        Ok(self.input.data())
    }

    /// Luma as given; chroma brought to luma size bilinearly, as in training.
    fn write_picture(&mut self, y: &[u8], u: &[u8], v: &[u8]) {
        let to_unit = |sample: &u8| f32::from(*sample) / MAX_SAMPLE;
        for (target, sample) in self.input.plane_mut(0).iter_mut().zip(y) {
            *target = to_unit(sample);
        }
        let half = (self.height / 2, self.width / 2);
        for (channel, plane) in [(1, u), (2, v)] {
            self.chroma.clear();
            self.chroma.extend(plane.iter().map(to_unit));
            let target = self.input.plane_mut(channel);
            self.resampler.bilinear2x(&self.chroma, half, target);
        }
    }

    fn write_side_planes(&mut self, features: &FrameFeatures) {
        if self.spec.input != (Input::Picture { side_planes: true }) {
            return; // the channels stay zero, as in training
        }
        let grid = (features.motion.rows(), features.motion.cols());
        let blocks = grid.0 * grid.1;
        self.side.resize(SIDE_CHANNELS * blocks, 0.0);
        write_side_planes(features, [false; PLANES], &mut self.side);
        let frame = (self.height, self.width);
        for (channel, plane) in self.side.chunks_exact(blocks).enumerate() {
            let target = self.input.plane_mut(PICTURE_CHANNELS + channel);
            self.spreader
                .spread(plane, grid, frame, GridFilter::Nearest, target);
        }
    }

    /// History channels: the aligned previous output folded to input
    /// resolution, the aligned hidden state, and the validity plane.
    fn write_history(&mut self, features: &FrameFeatures) {
        let (spec, size) = (self.spec, (self.height, self.width));
        let plane = self.input.plane_len();
        let folded = SR_SCALE * SR_SCALE * plane;
        let start = (PICTURE_CHANNELS + SIDE_CHANNELS) * plane;
        let (previous, rest) = self.input.data_mut()[start..].split_at_mut(spec.planes * folded);
        let (hidden, valid) = rest.split_at_mut(spec.state_channels * plane);
        self.valid = !features.reset && self.history.is_stored();
        if self.valid {
            let resampler = &mut self.resampler;
            self.history
                .aligned(features, spec.align, resampler, &mut self.previous, hidden);
        } else {
            // No usable history: the conventional upscale stands in.
            let len = self.previous.data().len();
            self.previous
                .data_mut()
                .copy_from_slice(&self.base.data()[..len]);
            hidden.fill(0.0);
        }
        for (channel, target) in previous.chunks_exact_mut(folded).enumerate() {
            pixel_unshuffle(self.previous.plane(channel), size, target);
        }
        valid.fill(f32::from(u8::from(self.valid)));
    }

    /// Turn the network's outputs for the prepared frame into the picture:
    /// 4:4:4 YUV in [0, 1] at twice the input size. `fine` is
    /// `spec.fine_channels()` planes at output size; `coarse` is the hidden
    /// state at input size (empty for a per-frame model).
    ///
    /// # Errors
    ///
    /// [`Error::OutOfOrder`] without a prepared frame;
    /// [`Error::FrameSizeMismatch`] if a tensor has the wrong length.
    pub fn finish(&mut self, fine: &[f32], coarse: &[f32]) -> Result<&Planes> {
        if !self.awaiting_network {
            return Err(Error::OutOfOrder("finish called without a prepared frame"));
        }
        let spec = self.spec;
        let fine_plane = self.output.plane_len();
        check_len(fine.len(), spec.fine_channels() * fine_plane)?;
        check_len(
            coarse.len(),
            spec.hidden_channels() * self.input.plane_len(),
        )?;
        // The network reconstructs the leading planes; the rest are the conventional upscale.
        let reconstructed = spec.planes * fine_plane;
        self.output.data_mut()[reconstructed..].copy_from_slice(&self.base.data()[reconstructed..]);
        let blended = spec.temporal == Temporal::Blend;
        let logit = &fine[reconstructed..];
        for channel in 0..spec.planes {
            let detail = &fine[channel * fine_plane..(channel + 1) * fine_plane];
            let (base, target) = (self.base.plane(channel), self.output.plane_mut(channel));
            if blended && self.valid {
                blend(base, detail, self.previous.plane(channel), logit, target);
            } else {
                add_detail(base, detail, target);
            }
        }
        if blended {
            let output = self.output.data();
            self.history.store(output, coarse, HiddenForm::Raw);
        } else if spec.temporal == Temporal::Recurrent {
            self.keep_recurrent_history(coarse);
        }
        self.awaiting_network = false;
        Ok(&self.output)
    }
}
