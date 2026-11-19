//! The sender's 2x reduction exposed to Python, so that tests can check it
//! against the reduction the training data was made with.

use cvnn_runtime::SR_SCALE;
use cvnn_runtime::kernel::Kernel;
use cvnn_runtime::reduce::Reducer;
use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const RGBA_BYTES: usize = 4;

/// The half-size picture of `rgba`, a (height, width, 4) array of bytes with
/// even height and width.
#[pyfunction]
#[allow(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts arguments by value"
)]
pub(crate) fn reduce_rgba<'py>(
    py: Python<'py>,
    rgba: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let (height, width, bytes) = rgba.as_array().dim();
    if bytes != RGBA_BYTES || height % SR_SCALE != 0 || width % SR_SCALE != 0 {
        return Err(PyValueError::new_err(
            "expected an even-sized (height, width, 4) array",
        ));
    }
    let pixels = rgba
        .as_slice()
        .map_err(|_| PyValueError::new_err("arrays must be C-contiguous"))?;
    let shape = (height / SR_SCALE, width / SR_SCALE, RGBA_BYTES);
    let mut reduced = vec![0; shape.0 * shape.1 * RGBA_BYTES];
    Reducer::new(width, height, Kernel::Bicubic).reduce(pixels, &mut reduced);
    let array =
        Array3::from_shape_vec(shape, reduced).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}
