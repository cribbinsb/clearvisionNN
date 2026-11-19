//! The integer network exposed to Python, so that tests can check the
//! exporter and the reference arithmetic against an independent calculation.

use cvnn_runtime::integer::{IntegerModel, PICTURE_OUTPUTS};
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
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

/// An exported integer model (`cvnn_runtime::integer::IntegerModel`).
#[pyclass]
pub(crate) struct IntegerNetwork {
    model: IntegerModel,
}

#[pymethods]
impl IntegerNetwork {
    /// The model in the words of a model file; raises if the file is not valid.
    #[new]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts arguments by value"
    )]
    fn new(words: PyReadonlyArray1<'_, i32>) -> PyResult<Self> {
        let model = IntegerModel::from_words(contiguous(words.as_slice().ok())?);
        Ok(Self {
            model: model.map_err(to_py_error)?,
        })
    }

    #[getter]
    fn input_words(&self) -> usize {
        self.model.input_words()
    }

    #[getter]
    fn output_words(&self) -> usize {
        self.model.output_words()
    }

    #[getter]
    fn tile(&self) -> usize {
        self.model.tile()
    }

    #[getter]
    fn halo(&self) -> usize {
        self.model.halo()
    }

    /// The float network input (channels, height, width) as packed planes:
    /// `input_words` words per pixel, row by row.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts arguments by value"
    )]
    fn pack_input<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let features = contiguous(features.as_slice().ok())?;
        let pixels = features.len() / self.model.input_channels();
        let mut packed = vec![0; pixels * self.model.input_words()];
        self.model
            .pack_input(features, &mut packed)
            .map_err(to_py_error)?;
        Ok(packed.into_pyarray(py))
    }

    /// The network's packed outputs for packed input planes; `tile` 0 runs
    /// the whole frame as one tile.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts arguments by value"
    )]
    fn infer<'py>(
        &self,
        py: Python<'py>,
        packed: PyReadonlyArray1<'py, u32>,
        width: usize,
        height: usize,
        tile: usize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let input = contiguous(packed.as_slice().ok())?;
        let output = py.allow_threads(|| self.model.infer(input, width, height, tile));
        Ok(output.map_err(to_py_error)?.into_pyarray(py))
    }

    /// The float outputs a session finishes from: the residual
    /// (1, 2 height, 2 width) and the state (channels, height, width).
    #[allow(
        clippy::needless_pass_by_value,
        clippy::type_complexity,
        reason = "PyO3 extracts arguments by value and returns a pair of arrays"
    )]
    fn unpack_output<'py>(
        &self,
        py: Python<'py>,
        packed: PyReadonlyArray1<'py, u32>,
        width: usize,
        height: usize,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
        let state = self.model.state_outputs();
        let mut fine = vec![0.0; PICTURE_OUTPUTS * width * height];
        let mut coarse = vec![0.0; state * width * height];
        self.model
            .unpack_output(
                contiguous(packed.as_slice().ok())?,
                width,
                height,
                &mut fine,
                &mut coarse,
            )
            .map_err(to_py_error)?;
        let shaped = |values, shape| {
            Array3::from_shape_vec(shape, values).expect("length matches the shape")
        };
        Ok((
            shaped(fine, (1, 2 * height, 2 * width)).into_pyarray(py),
            shaped(coarse, (state, height, width)).into_pyarray(py),
        ))
    }
}
