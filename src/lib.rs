#![allow(unsafe_op_in_unsafe_fn)]

pub mod core;

use crate::core::{
    DenseHistogram as CoreDenseHistogram, Histogram as CoreHistogram,
    HistogramRecorder as CoreHistogramRecorder, HistogramSnapshot as CoreHistogramSnapshot,
    OutOfRangeMode, SnapshotDiff as CoreSnapshotDiff, SparseHistogram as CoreSparseHistogram,
};
use numpy::PyReadonlyArrayDyn;
use pyo3::FromPyObject;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

impl From<crate::core::HistogramError> for PyErr {
    fn from(err: crate::core::HistogramError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

impl<'source, 'py> FromPyObject<'source, 'py> for OutOfRangeMode {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, 'py, PyAny>) -> PyResult<Self> {
        let text = obj.extract::<&str>()?.to_ascii_lowercase();
        match text.as_str() {
            "clip" => Ok(Self::Clip),
            "ignore" => Ok(Self::Ignore),
            "error" => Ok(Self::Error),
            _ => Err(PyValueError::new_err(
                "out_of_range must be one of 'Clip', 'Ignore', or 'Error'",
            )),
        }
    }
}

fn update_from_vec<H: CoreHistogram>(hist: &mut H, values: Vec<f64>) -> PyResult<()> {
    hist.update(values).map_err(PyErr::from)
}

fn update_from_pyarray<'py, H: CoreHistogram>(
    hist: &mut H,
    values: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<()> {
    hist.update(values.as_array().iter().copied())
        .map_err(PyErr::from)
}

fn update_from_pyarray_mask<'py, H: CoreHistogram>(
    hist: &mut H,
    values: PyReadonlyArrayDyn<'py, f64>,
    mask: PyReadonlyArrayDyn<'py, bool>,
) -> PyResult<()> {
    let values_view = values.as_array();
    let mask_view = mask.as_array();

    if values_view.shape() != mask_view.shape() {
        return Err(PyValueError::new_err(
            "mask must have the same shape as values",
        ));
    }

    for (value, should_use) in values_view.iter().zip(mask_view.iter()) {
        if *should_use {
            hist.ingest_value(*value).map_err(PyErr::from)?;
        }
    }

    Ok(())
}

#[pyclass(name = "SparseHistogram")]
pub struct PySparseHistogram {
    inner: CoreSparseHistogram,
}

#[pymethods]
impl PySparseHistogram {
    #[new]
    fn new(bin_width: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CoreSparseHistogram::new(bin_width)?,
        })
    }

    pub fn update(&mut self, values: Vec<f64>) -> PyResult<()> {
        update_from_vec(&mut self.inner, values)
    }

    pub fn update_np<'py>(&mut self, values: PyReadonlyArrayDyn<'py, f64>) -> PyResult<()> {
        update_from_pyarray(&mut self.inner, values)
    }

    pub fn update_np_mask<'py>(
        &mut self,
        values: PyReadonlyArrayDyn<'py, f64>,
        mask: PyReadonlyArrayDyn<'py, bool>,
    ) -> PyResult<()> {
        update_from_pyarray_mask(&mut self.inner, values, mask)
    }

    pub fn get(&self) -> Vec<((f64, f64), u64)> {
        self.inner.get()
    }
}

#[pyclass(name = "DenseHistogram")]
pub struct PyDenseHistogram {
    inner: CoreDenseHistogram,
}

#[pymethods]
impl PyDenseHistogram {
    #[new]
    fn new(range: (f64, f64), num_bins: usize, out_of_range: OutOfRangeMode) -> PyResult<Self> {
        Ok(Self {
            inner: CoreDenseHistogram::new(range, num_bins, out_of_range)?,
        })
    }

    pub fn update(&mut self, values: Vec<f64>) -> PyResult<()> {
        update_from_vec(&mut self.inner, values)
    }

    pub fn update_np<'py>(&mut self, values: PyReadonlyArrayDyn<'py, f64>) -> PyResult<()> {
        update_from_pyarray(&mut self.inner, values)
    }

    pub fn get(&self) -> Vec<((f64, f64), u64)> {
        self.inner.get()
    }
}

#[pyclass(name = "HistogramSnapshot")]
#[derive(Clone)]
pub struct PyHistogramSnapshot {
    inner: CoreHistogramSnapshot,
}

impl From<CoreHistogramSnapshot> for PyHistogramSnapshot {
    fn from(inner: CoreHistogramSnapshot) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHistogramSnapshot {
    #[getter]
    fn index(&self) -> u64 {
        self.inner.index()
    }

    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label()
    }

    pub fn bins(&self) -> Vec<((f64, f64), u64)> {
        self.inner.bins()
    }

    pub fn total(&self) -> u64 {
        self.inner.total()
    }
}

#[pyclass(name = "SnapshotDiff")]
#[derive(Clone)]
pub struct PySnapshotDiff {
    inner: CoreSnapshotDiff,
}

impl From<CoreSnapshotDiff> for PySnapshotDiff {
    fn from(inner: CoreSnapshotDiff) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySnapshotDiff {
    #[getter]
    fn start_index(&self) -> u64 {
        self.inner.start_index()
    }

    #[getter]
    fn end_index(&self) -> u64 {
        self.inner.end_index()
    }

    #[getter]
    fn start_label(&self) -> Option<String> {
        self.inner.start_label()
    }

    #[getter]
    fn end_label(&self) -> Option<String> {
        self.inner.end_label()
    }

    pub fn bins(&self) -> Vec<((f64, f64), u64)> {
        self.inner.bins()
    }

    pub fn total(&self) -> u64 {
        self.inner.total()
    }
}

#[pyclass(name = "HistogramRecorder")]
pub struct PyHistogramRecorder {
    inner: CoreHistogramRecorder,
}

#[pymethods]
impl PyHistogramRecorder {
    #[staticmethod]
    #[pyo3(signature = (bin_width, max_snapshots=None))]
    fn from_sparse(bin_width: f64, max_snapshots: Option<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: CoreHistogramRecorder::from_sparse(bin_width, max_snapshots)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (range, num_bins, out_of_range, max_snapshots=None))]
    fn from_dense(
        range: (f64, f64),
        num_bins: usize,
        out_of_range: OutOfRangeMode,
        max_snapshots: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreHistogramRecorder::from_dense(range, num_bins, out_of_range, max_snapshots)?,
        })
    }

    pub fn update(&mut self, values: Vec<f64>) -> PyResult<()> {
        update_from_vec(&mut self.inner, values)
    }

    pub fn update_np<'py>(&mut self, values: PyReadonlyArrayDyn<'py, f64>) -> PyResult<()> {
        update_from_pyarray(&mut self.inner, values)
    }

    pub fn update_np_mask<'py>(
        &mut self,
        values: PyReadonlyArrayDyn<'py, f64>,
        mask: PyReadonlyArrayDyn<'py, bool>,
    ) -> PyResult<()> {
        update_from_pyarray_mask(&mut self.inner, values, mask)
    }

    pub fn get(&self) -> Vec<((f64, f64), u64)> {
        self.inner.get()
    }

    pub fn snapshot(&mut self, label: Option<String>) -> PyHistogramSnapshot {
        PyHistogramSnapshot::from(self.inner.snapshot(label))
    }

    pub fn latest(&self) -> Option<PyHistogramSnapshot> {
        self.inner.latest().map(PyHistogramSnapshot::from)
    }

    pub fn get_snapshot(&self, index: u64) -> PyResult<PyHistogramSnapshot> {
        Ok(PyHistogramSnapshot::from(self.inner.get_snapshot(index)?))
    }

    pub fn diff(&self, later_index: u64, earlier_index: u64) -> PyResult<PySnapshotDiff> {
        Ok(PySnapshotDiff::from(
            self.inner.diff(later_index, earlier_index)?,
        ))
    }

    pub fn drain_snapshots(&mut self, end_index: u64) -> usize {
        self.inner.drain_snapshots(end_index)
    }

    pub fn clear_snapshots(&mut self) {
        self.inner.clear_snapshots();
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySparseHistogram>()?;
    module.add_class::<PyDenseHistogram>()?;
    module.add_class::<PyHistogramRecorder>()?;
    module.add_class::<PyHistogramSnapshot>()?;
    module.add_class::<PySnapshotDiff>()?;
    Ok(())
}
