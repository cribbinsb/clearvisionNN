//! Whole-sequence analysis: the form training's data loading uses.

use cvnn_common::FrameFeatures;
use cvnn_preproc::{PLANES, PreprocConfig, Preprocessor, SIDE_CHANNELS, write_side_planes};
use numpy::ndarray::{Array1, Array4, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyArray4, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The per-frame features of a sequence, stacked along a leading time axis.
struct SequenceFeatures {
    motion: Array4<f32>,
    side: Array4<f32>,
    reset: Array1<bool>,
}

impl SequenceFeatures {
    fn new(frames: usize, rows: usize, cols: usize) -> Self {
        Self {
            motion: Array4::zeros((frames, 2, rows, cols)),
            side: Array4::zeros((frames, SIDE_CHANNELS, rows, cols)),
            reset: Array1::from_elem(frames, false),
        }
    }

    fn store(&mut self, t: usize, features: &FrameFeatures, withheld: [bool; PLANES]) {
        let cols = features.motion.cols();
        for (index, vector) in features.motion.cells().iter().enumerate() {
            let (row, col) = (index / cols, index % cols);
            self.motion[[t, 0, row, col]] = vector.dx;
            self.motion[[t, 1, row, col]] = vector.dy;
        }
        let mut frame = self.side.index_axis_mut(numpy::ndarray::Axis(0), t);
        let side = frame
            .as_slice_mut()
            .expect("freshly allocated arrays are contiguous");
        write_side_planes(features, withheld, side);
        self.reset[t] = features.reset;
    }
}

fn analyze(
    luma: ArrayView3<'_, u8>,
    subpel: bool,
    withheld: [bool; PLANES],
) -> cvnn_common::Result<SequenceFeatures> {
    let (frames, height, width) = luma.dim();
    let config = PreprocConfig {
        subpel,
        ..PreprocConfig::default()
    };
    let mut preprocessor = Preprocessor::new(width, height, config)?;
    let block = cvnn_common::BLOCK_SIZE;
    let mut out = SequenceFeatures::new(frames, height.div_ceil(block), width.div_ceil(block));
    let mut buffer = vec![0u8; width * height];
    for (t, frame) in luma.outer_iter().enumerate() {
        for (target, &sample) in buffer.iter_mut().zip(frame.iter()) {
            *target = sample;
        }
        out.store(t, &preprocessor.push(&buffer)?, withheld);
    }
    Ok(out)
}

type SequenceArrays<'py> = (
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray4<f32>>,
    Bound<'py, PyArray1<bool>>,
);

/// Analyse a luma sequence of shape (frames, height, width), dtype uint8.
///
/// Returns `(motion, side, reset)`: motion is (frames, 2, rows, cols) as
/// (dx, dy) in luma samples towards the previous frame; side is (frames,
/// `SIDE_CHANNELS`, rows, cols), the network's side planes; reset is per
/// frame. `withheld` blanks the flagged side planes. The GIL is released
/// while analysing.
#[pyfunction]
#[pyo3(signature = (luma, subpel = true, withheld = [false; PLANES]))]
#[allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts arguments by value"
)]
pub(crate) fn analyze_sequence<'py>(
    py: Python<'py>,
    luma: PyReadonlyArray3<'py, u8>,
    subpel: bool,
    withheld: [bool; PLANES],
) -> PyResult<SequenceArrays<'py>> {
    let view = luma.as_array();
    let features = py
        .allow_threads(|| analyze(view, subpel, withheld))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        features.motion.into_pyarray(py),
        features.side.into_pyarray(py),
        features.reset.into_pyarray(py),
    ))
}
