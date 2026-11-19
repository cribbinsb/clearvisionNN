//! What a temporal model carries between frames, and bringing it onto the
//! current frame's grid.

use cvnn_common::FrameFeatures;

use crate::SR_SCALE;
use crate::fastmath::tanh;
use crate::flow::Flow;
use crate::planes::Planes;
use crate::resample::Resampler;

/// The form in which a network hands over its hidden state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HiddenForm {
    /// Unbounded: squashed to (-1, 1) here, as in training.
    Raw,
    /// Already squashed inside the network.
    Bounded,
}

/// The previous frame's reconstructed planes and hidden state.
#[derive(Debug)]
pub(crate) struct History {
    /// (planes, 2h, 2w): the previous output's reconstructed planes.
    output: Planes,
    /// (state channels, h, w).
    hidden: Planes,
    /// The previous input luma, for models that measure how well motion
    /// predicts the frame; empty until one is stored.
    luma: Vec<f32>,
    /// Whether a frame has been stored since the stream started.
    stored: bool,
    flow: Flow,
}

impl History {
    /// Empty history for `height` x `width` input frames.
    pub(crate) fn new(planes: usize, state_channels: usize, height: usize, width: usize) -> Self {
        Self {
            output: Planes::zeros(planes, SR_SCALE * height, SR_SCALE * width),
            hidden: Planes::zeros(state_channels, height, width),
            luma: Vec::new(),
            stored: false,
            flow: Flow::default(),
        }
    }

    /// Whether there is a previous frame to use.
    pub(crate) fn is_stored(&self) -> bool {
        self.stored
    }

    /// Forget the previous frame.
    pub(crate) fn clear(&mut self) {
        self.stored = false;
    }

    /// Keep a finished frame: the leading planes of `output`, and the
    /// network's hidden state, squashed to (-1, 1) if the network has not.
    pub(crate) fn store(&mut self, output: &[f32], hidden: &[f32], form: HiddenForm) {
        let len = self.output.data().len();
        self.output.data_mut().copy_from_slice(&output[..len]);
        match form {
            HiddenForm::Bounded => self.hidden.data_mut().copy_from_slice(hidden),
            HiddenForm::Raw => {
                for (target, &value) in self.hidden.data_mut().iter_mut().zip(hidden) {
                    *target = tanh(value);
                }
            }
        }
        self.stored = true;
    }

    /// Keep the finished frame's input luma beside it.
    pub(crate) fn store_luma(&mut self, luma: &[f32]) {
        self.luma.clear();
        self.luma.extend_from_slice(luma);
    }

    /// Write the stored input luma onto the current frame's grid. With
    /// `align` it is warped by the motion [`History::aligned`] was last given,
    /// so that must have been called for this frame.
    pub(crate) fn carried_luma(&self, align: bool, out: &mut [f32]) {
        if align {
            self.flow.warp_coarse(&self.luma, out);
        } else {
            out.copy_from_slice(&self.luma);
        }
    }

    /// Write this history onto the current frame's grid: warped by the motion
    /// field when `align`, copied otherwise. `hidden` is the state channels'
    /// part of the network's input.
    pub(crate) fn aligned(
        &mut self,
        features: &FrameFeatures,
        align: bool,
        resampler: &mut Resampler,
        previous: &mut Planes,
        hidden: &mut [f32],
    ) {
        if !align {
            previous.data_mut().copy_from_slice(self.output.data());
            hidden.copy_from_slice(self.hidden.data());
            return;
        }
        let size = (self.hidden.height(), self.hidden.width());
        self.flow.set(features, size, resampler);
        for channel in 0..previous.channels() {
            let (source, target) = (self.output.plane(channel), previous.plane_mut(channel));
            self.flow.warp_fine(source, target);
        }
        let planes = hidden.chunks_exact_mut(self.hidden.plane_len());
        for (channel, target) in planes.enumerate() {
            self.flow.warp_coarse(self.hidden.plane(channel), target);
        }
    }
}
