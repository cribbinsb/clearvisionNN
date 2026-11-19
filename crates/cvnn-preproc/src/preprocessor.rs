//! The per-stream pre-process: frames in, [`FrameFeatures`] out.

use cvnn_common::{BlockGrid, Error, FrameFeatures, MotionVector, Result};
use nc_analysis::{Analyzer, FrameAnalysis};
use nc_common::PlaneIndex;
use nc_dsp::Dsp;

use crate::config::PreprocConfig;
use crate::luma::LumaFrame;
use crate::subpel::parabola_offset;

/// Analyses the luma of one video stream, frame by frame.
#[derive(Debug)]
pub struct Preprocessor {
    width: usize,
    height: usize,
    config: PreprocConfig,
    analyzer: Analyzer,
    current: LumaFrame,
    previous: LumaFrame,
    /// Mean luma of the previous frame; `None` before the first frame and
    /// after [`Preprocessor::restart`].
    previous_mean_luma: Option<f32>,
}

impl Preprocessor {
    /// A pre-process for `width` x `height` luma frames.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDimensions`] if the analysis cannot work at this size.
    pub fn new(width: usize, height: usize, config: PreprocConfig) -> Result<Self> {
        Ok(Self {
            width,
            height,
            config,
            analyzer: new_analyzer(width, height)?,
            current: LumaFrame::new(width, height),
            previous: LumaFrame::new(width, height),
            previous_mean_luma: None,
        })
    }

    /// Forget every earlier frame: the next one is treated as the first of a
    /// new stream.
    ///
    /// # Errors
    ///
    /// As [`Preprocessor::new`].
    pub fn restart(&mut self) -> Result<()> {
        self.analyzer = new_analyzer(self.width, self.height)?;
        self.previous_mean_luma = None;
        Ok(())
    }

    /// Analyse the next frame, given its luma samples row-major.
    ///
    /// # Errors
    ///
    /// [`Error::FrameSizeMismatch`] if `luma` is not `width * height` long.
    pub fn push(&mut self, luma: &[u8]) -> Result<FrameFeatures> {
        let expected = self.width * self.height;
        if luma.len() != expected {
            return Err(Error::FrameSizeMismatch {
                expected,
                actual: luma.len(),
            });
        }
        core::mem::swap(&mut self.current, &mut self.previous);
        self.current.copy_from(luma);
        self.write_to_analyzer();
        let analysis = self.analyzer.analyze_written().clone();
        let features = self.features(&analysis);
        self.previous_mean_luma = Some(features.mean_luma);
        Ok(features)
    }

    fn write_to_analyzer(&mut self) {
        let width = self.width;
        let picture = self.analyzer.next_frame_mut();
        let mut view = picture.plane_mut(PlaneIndex::Y).view_mut();
        for (y, row) in self.current.samples().chunks_exact(width).enumerate() {
            view.row_mut(y)[..width].copy_from_slice(row);
        }
    }

    fn features(&self, analysis: &FrameAnalysis) -> FrameFeatures {
        let mean_luma = analysis.pre.mean_luma;
        let mut features = FrameFeatures {
            motion: BlockGrid::for_frame(self.width, self.height, MotionVector::default()),
            match_error: BlockGrid::for_frame(self.width, self.height, 0.0),
            still_run: BlockGrid::for_frame(self.width, self.height, 0),
            noise_sigma: analysis.noise_sigma,
            mean_luma,
            reset: true,
        };
        features
            .still_run
            .cells_mut()
            .copy_from_slice(analysis.still.runs.cells());
        let (Some(motion), Some(previous_mean)) = (&analysis.motion, self.previous_mean_luma)
        else {
            return features;
        };
        let positions = features.motion.positions().collect::<Vec<_>>();
        for (index, position) in positions.into_iter().enumerate() {
            let whole = motion.fine.cells()[index].mv;
            let (vector, error) = self.refine(position, (whole.x.into(), whole.y.into()));
            features.motion.cells_mut()[index] = vector;
            features.match_error.cells_mut()[index] = error;
        }
        features.reset =
            self.config
                .reset
                .is_reset(previous_mean, mean_luma, features.match_error.cells());
        features
    }

    /// The block's vector at sub-sample precision, and its match error at
    /// the whole-sample vector.
    fn refine(&self, block: (usize, usize), (x, y): (isize, isize)) -> (MotionVector, f32) {
        let diff = |dx, dy| {
            self.current
                .block_mean_abs_diff(&self.previous, block, (dx, dy))
        };
        let centre = diff(x, y);
        let mut vector = MotionVector {
            dx: x as f32,
            dy: y as f32,
        };
        if self.config.subpel {
            vector.dx += parabola_offset(diff(x - 1, y), centre, diff(x + 1, y));
            vector.dy += parabola_offset(diff(x, y - 1), centre, diff(x, y + 1));
        }
        (vector, centre)
    }
}

fn new_analyzer(width: usize, height: usize) -> Result<Analyzer> {
    Analyzer::new(width, height, Dsp::new()).map_err(|e| Error::InvalidDimensions {
        width,
        height,
        reason: e.to_string(),
    })
}
