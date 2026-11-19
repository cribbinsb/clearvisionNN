//! Python bindings to the pre-process (`clearvisionnn_native`), so that training
//! computes motion and side planes with the code deployment runs, and tests
//! can check the runtime against the training code.
//!
//! Bindings only: no analysis or runtime logic lives here.

mod fixed;
mod integer;
mod runtime;
mod sender;
mod sequence;

use pyo3::prelude::*;

/// The `clearvisionnn_native` extension module.
#[pymodule]
fn clearvisionnn_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sequence::analyze_sequence, module)?)?;
    module.add_class::<runtime::RuntimeSession>()?;
    module.add_class::<integer::IntegerNetwork>()?;
    module.add_class::<fixed::IntegerRuntimeSession>()?;
    module.add_function(wrap_pyfunction!(sender::reduce_rgba, module)?)?;
    module.add("BLOCK_SIZE", cvnn_common::BLOCK_SIZE)?;
    module.add("SIDE_CHANNELS", cvnn_preproc::SIDE_CHANNELS)?;
    module.add("SIDE_PLANES", cvnn_preproc::PLANES)?;
    module.add("CODING_CHANNELS", cvnn_runtime::coding::CODING_CHANNELS)?;
    module.add("MAX_QP", cvnn_runtime::coding::MAX_QP)?;
    module.add("TRUSTED_ERROR", cvnn_runtime::confidence::TRUSTED_ERROR)?;
    module.add("UNTRUSTED_ERROR", cvnn_runtime::confidence::UNTRUSTED_ERROR)?;
    module.add("INTEGER_MAGIC", cvnn_runtime::integer::MAGIC)?;
    module.add(
        "INTEGER_SCHEMA_VERSION",
        cvnn_runtime::integer::SCHEMA_VERSION,
    )?;
    module.add("INTEGER_HEADER_WORDS", cvnn_runtime::integer::HEADER_WORDS)?;
    module.add(
        "INTEGER_MULTIPLIER_BITS",
        cvnn_runtime::integer::MULTIPLIER_BITS,
    )?;
    module.add(
        "INTEGER_RESIDUAL_STEPS",
        cvnn_runtime::integer::RESIDUAL_STEPS,
    )?;
    module.add("INTEGER_TANH_STEPS", cvnn_runtime::integer::TANH_STEPS)?;
    module.add("INTEGER_STATE_BITS", cvnn_runtime::integer::STATE_BITS)?;
    Ok(())
}
