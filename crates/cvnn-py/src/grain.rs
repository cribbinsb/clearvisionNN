//! The runtime's grain exposed to Python, so that training and scoring can
//! add exactly the grain a deployed model is shown with.

use cvnn_runtime::SR_SCALE;
use cvnn_runtime::grain::{Grain, GrainProfile};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Every grain profile's name with its grey levels of grain.
pub(crate) fn profiles() -> Vec<(&'static str, f32)> {
    let named = GrainProfile::NAMED.iter();
    named
        .map(|&(name, profile)| (name, profile.level()))
        .collect()
}

/// The grain of frame `frame` of a stream of `height` x `width` input frames
/// under the profile named `profile`: (2 height, 2 width) int32 in picture
/// steps (1/32 of a grey level), to add to the output luma.
#[pyfunction]
pub(crate) fn grain_steps<'py>(
    py: Python<'py>,
    frame: u32,
    height: usize,
    width: usize,
    profile: &str,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let profile: GrainProfile = profile
        .parse()
        .map_err(|error: cvnn_common::Error| PyValueError::new_err(error.to_string()))?;
    let mut grain = Grain::new(height, width);
    grain.set(profile);
    grain.seek(frame);
    let shape = (SR_SCALE * height, SR_SCALE * width);
    let steps = grain.next_frame().map(<[i32]>::to_vec);
    let steps = steps.unwrap_or_else(|| vec![0; shape.0 * shape.1]);
    let array = Array2::from_shape_vec(shape, steps).expect("length matches the shape");
    Ok(array.into_pyarray(py))
}
