//! The luma-only models' half of a session: a lean network input (luma, and
//! for the recurrent design its history and how far that history is to be
//! trusted), with chroma upscaled conventionally beside it.

use cvnn_common::FrameFeatures;

use super::{MAX_SAMPLE, Session};
use crate::SR_SCALE;
use crate::coding::{FrameCoding, coding_planes};
use crate::compose::pixel_unshuffle;
use crate::confidence::Confidence;
use crate::history::HiddenForm;
use crate::planes::Planes;
use crate::spec::{Input, ModelSpec, Temporal};

/// The chroma planes of a picture.
const CHROMA_PLANES: usize = 2;

/// What a luma-only model needs beyond the shared session state.
#[derive(Debug)]
pub(super) struct LumaState {
    /// Chroma at luma size: never a network input here, only upscaled.
    chroma: Planes,
    /// The previous input luma brought onto the current frame.
    carried: Vec<f32>,
    confidence: Confidence,
    /// How the frames being prepared were coded, if the caller knows.
    coding: Option<FrameCoding>,
}

impl LumaState {
    /// State for `height` x `width` input frames; empty for other models.
    pub(super) fn new(spec: ModelSpec, height: usize, width: usize) -> Self {
        let luma_only = matches!(spec.input, Input::Luma { .. });
        let planes = if luma_only { CHROMA_PLANES } else { 0 };
        Self {
            chroma: Planes::zeros(planes, height, width),
            carried: Vec::new(),
            confidence: Confidence::default(),
            coding: None,
        }
    }
}

impl Session {
    /// Say how the frames prepared from now on were coded (`None`: unknown,
    /// or never coded). Only a model with coding planes uses it.
    pub fn set_coding(&mut self, coding: Option<FrameCoding>) {
        self.luma.coding = coding;
    }

    /// Fill the network input and the conventional upscale for a luma-only
    /// model. Input channels, in order: luma; confidence (if the model takes
    /// it); hidden state and the previous output folded to input resolution
    /// (recurrent); the coding planes (if the model takes them).
    pub(super) fn prepare_luma(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        features: &FrameFeatures,
        coding: bool,
        with_confidence: bool,
    ) {
        let (spec, size) = (self.spec, (self.height, self.width));
        let to_unit = |sample: &u8| f32::from(*sample) / MAX_SAMPLE;
        let plane = self.input.plane_len();
        let (luma, mut rest) = self.input.data_mut().split_at_mut(plane);
        for (target, sample) in luma.iter_mut().zip(y) {
            *target = to_unit(sample);
        }
        self.resampler.bicubic2x(luma, size, self.base.plane_mut(0));
        let half = (self.height / 2, self.width / 2);
        for (channel, samples) in [u, v].into_iter().enumerate() {
            self.chroma.clear();
            self.chroma.extend(samples.iter().map(to_unit));
            let wide = self.luma.chroma.plane_mut(channel);
            self.resampler.bilinear2x(&self.chroma, half, wide);
            self.resampler
                .bicubic2x(wide, size, self.base.plane_mut(1 + channel));
        }
        if spec.temporal == Temporal::Recurrent {
            let confident = usize::from(with_confidence) * plane;
            let (confidence, history) = rest.split_at_mut(confident);
            let (hidden, history) = history.split_at_mut(spec.state_channels * plane);
            let (previous, after) = history.split_at_mut(SR_SCALE * SR_SCALE * plane);
            rest = after;
            self.valid = !features.reset && self.history.is_stored();
            if self.valid {
                let resampler = &mut self.resampler;
                self.history
                    .aligned(features, spec.align, resampler, &mut self.previous, hidden);
                if with_confidence {
                    self.luma.carried.resize(plane, 0.0);
                    self.history
                        .carried_luma(spec.align, &mut self.luma.carried);
                    let carried = &self.luma.carried;
                    self.luma.confidence.write(luma, carried, size, confidence);
                }
            } else {
                // No usable history: the conventional upscale stands in, and
                // there is no prediction to distrust.
                self.previous.data_mut().copy_from_slice(self.base.plane(0));
                hidden.fill(0.0);
                confidence.fill(1.0);
            }
            pixel_unshuffle(self.previous.plane(0), size, previous);
        }
        if coding {
            let values = coding_planes(self.luma.coding);
            for (target, value) in rest.chunks_exact_mut(plane).zip(values) {
                target.fill(value);
            }
        }
    }

    /// Keep the finished frame for the next: its output luma, the hidden
    /// state the network returned (already bounded), and its input luma.
    pub(super) fn keep_recurrent_history(&mut self, hidden: &[f32]) {
        let output = self.output.data();
        self.history.store(output, hidden, HiddenForm::Bounded);
        self.history.store_luma(self.input.plane(0));
    }
}
