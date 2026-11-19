//! The integer runtime exposed to Python: a whole frame in fixed point, for
//! tests and for scoring it against the float definitions it replaces.

use cvnn_common::MotionVector;
use cvnn_runtime::coding::FrameCoding;
use cvnn_runtime::fixed::{FrameInput, IntegerRuntime, PICTURE_MAX, vector_steps};
use cvnn_runtime::integer::IntegerModel;
use cvnn_runtime::{Input, ModelSpec, SR_SCALE};
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[allow(
    clippy::needless_pass_by_value,
    reason = "the signature `map_err` calls"
)]
fn to_py_error(error: cvnn_common::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn contiguous<T: numpy::Element>(slice: Option<&[T]>) -> PyResult<&[T]> {
    slice.ok_or_else(|| PyValueError::new_err("arrays must be C-contiguous"))
}

/// One stream upscaled entirely in integers (`cvnn_runtime::fixed::IntegerRuntime`).
#[pyclass]
pub(crate) struct IntegerRuntimeSession {
    runtime: IntegerRuntime,
    width: usize,
    height: usize,
}

#[pymethods]
impl IntegerRuntimeSession {
    /// A session for `width` x `height` frames of the luma-only model in
    /// `words` (a model file), whose contract the other arguments give.
    #[new]
    #[allow(
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools,
        clippy::needless_pass_by_value,
        reason = "one argument per field of the contract; PyO3 extracts arguments by value"
    )]
    fn new(
        words: PyReadonlyArray1<'_, i32>,
        width: usize,
        height: usize,
        temporal: &str,
        align: bool,
        state_channels: usize,
        coding_planes: bool,
        confidence_plane: bool,
    ) -> PyResult<Self> {
        let model = IntegerModel::from_words(contiguous(words.as_slice().ok())?);
        let spec = ModelSpec {
            planes: 1,
            temporal: temporal.parse().map_err(to_py_error)?,
            align,
            state_channels,
            input: Input::Luma {
                coding: coding_planes,
                confidence: confidence_plane,
            },
        };
        let runtime = IntegerRuntime::new(model.map_err(to_py_error)?, spec, width, height);
        Ok(Self {
            runtime: runtime.map_err(to_py_error)?,
            width,
            height,
        })
    }

    /// Forget earlier frames; the next frame starts a new stream.
    fn restart(&mut self) {
        self.runtime.restart();
    }

    /// Upscale one 4:2:0 frame. `motion` is the block field as (2, rows,
    /// cols): dx then dy, in input samples. Returns the output luma
    /// (2 height, 2 width) in [0, 1].
    #[pyo3(signature = (y, u, v, motion, reset, quantiser=None, key_frame=false, tile=0))]
    #[allow(
        clippy::too_many_arguments,
        clippy::needless_pass_by_value,
        reason = "one argument per fact about the frame; PyO3 extracts arguments by value"
    )]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray2<'py, u8>,
        u: PyReadonlyArray2<'py, u8>,
        v: PyReadonlyArray2<'py, u8>,
        motion: PyReadonlyArray3<'py, f32>,
        reset: bool,
        quantiser: Option<f32>,
        key_frame: bool,
        tile: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let field = contiguous(motion.as_slice().ok())?;
        let (dx, dy) = field.split_at(field.len() / 2);
        let vectors: Vec<_> = dx
            .iter()
            .zip(dy)
            .map(|(&dx, &dy)| vector_steps(MotionVector { dx, dy }))
            .collect();
        let frame = FrameInput {
            y: contiguous(y.as_slice().ok())?,
            u: contiguous(u.as_slice().ok())?,
            v: contiguous(v.as_slice().ok())?,
            motion: &vectors,
            reset,
            coding: quantiser.map(|quantiser| FrameCoding {
                quantiser,
                key_frame,
            }),
        };
        let runtime = &mut self.runtime;
        py.allow_threads(|| runtime.step(&frame, tile))
            .map_err(to_py_error)?;
        let luma = self
            .runtime
            .luma()
            .iter()
            .map(|&steps| steps as f32 / PICTURE_MAX as f32);
        let shape = (SR_SCALE * self.height, SR_SCALE * self.width);
        let array =
            Array2::from_shape_vec(shape, luma.collect()).expect("length matches the shape");
        Ok(array.into_pyarray(py))
    }

    /// The last frame's output as RGBA, (2 height, 2 width, 4).
    fn picture<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        let bytes: Vec<u8> = self
            .runtime
            .picture()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let shape = (SR_SCALE * self.height, SR_SCALE * self.width, 4);
        Array3::from_shape_vec(shape, bytes)
            .expect("length matches the shape")
            .into_pyarray(py)
    }

    /// The network's packed input for the last frame.
    fn packed_input<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.runtime.packed_input().to_vec().into_pyarray(py)
    }

    /// The network's packed outputs for the last frame.
    fn outputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.runtime.outputs().to_vec().into_pyarray(py)
    }
}
