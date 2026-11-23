//! The runtime session exposed to Python, so that tests can check it against
//! the training model frame for frame.

use cvnn_runtime::{Input, ModelSpec, Session};
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
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

/// One stream being upscaled around an external network (`cvnn_runtime::Session`).
#[pyclass]
pub(crate) struct RuntimeSession {
    session: Session,
    width: usize,
    height: usize,
}

#[pymethods]
impl RuntimeSession {
    #[new]
    #[pyo3(signature = (width, height, planes, temporal, align, state_channels, side_planes,
                        luma_only=false, confidence_plane=false))]
    #[allow(
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools,
        reason = "one argument per field of the contract"
    )]
    fn new(
        width: usize,
        height: usize,
        planes: usize,
        temporal: &str,
        align: bool,
        state_channels: usize,
        side_planes: bool,
        luma_only: bool,
        confidence_plane: bool,
    ) -> PyResult<Self> {
        let spec = ModelSpec {
            planes,
            temporal: temporal.parse().map_err(to_py_error)?,
            align,
            state_channels,
            input: Input::from_contract(luma_only, side_planes, confidence_plane)
                .map_err(to_py_error)?,
        };
        let session = Session::new(spec, width, height).map_err(to_py_error)?;
        Ok(Self {
            session,
            width,
            height,
        })
    }

    /// The grain on the luma of the pictures finished from now on, as the
    /// name of a profile (`GRAIN_PROFILES`).
    fn set_grain(&mut self, profile: &str) -> PyResult<()> {
        self.session
            .set_grain(profile.parse().map_err(to_py_error)?);
        Ok(())
    }

    /// Forget earlier frames; the next frame starts a new stream.
    fn restart(&mut self) -> PyResult<()> {
        self.session.restart().map_err(to_py_error)
    }

    /// The network input for the next frame: (channels, height, width) float32.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts arguments by value"
    )]
    fn prepare<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray2<'py, u8>,
        u: PyReadonlyArray2<'py, u8>,
        v: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let (y, u, v) = (
            contiguous(y.as_slice().ok())?,
            contiguous(u.as_slice().ok())?,
            contiguous(v.as_slice().ok())?,
        );
        let input = self.session.prepare(y, u, v).map_err(to_py_error)?.to_vec();
        let shape = (
            input.len() / (self.width * self.height),
            self.height,
            self.width,
        );
        let array = Array3::from_shape_vec(shape, input).expect("length matches the shape");
        Ok(array.into_pyarray(py))
    }

    /// The output picture for the prepared frame: (3, 2 * height, 2 * width) YUV in [0, 1].
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 extracts arguments by value"
    )]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        fine: PyReadonlyArray3<'py, f32>,
        coarse: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let (fine, coarse) = (
            contiguous(fine.as_slice().ok())?,
            contiguous(coarse.as_slice().ok())?,
        );
        let output = self.session.finish(fine, coarse).map_err(to_py_error)?;
        let shape = (output.channels(), output.height(), output.width());
        let array = Array3::from_shape_vec(shape, output.data().to_vec())
            .expect("length matches the shape");
        Ok(array.into_pyarray(py))
    }
}
