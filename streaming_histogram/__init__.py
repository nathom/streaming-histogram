from __future__ import annotations

from importlib import import_module as _import_module

try:
    _native = _import_module("._native", __name__)
except ModuleNotFoundError as exc:  # pragma: no cover - guidance for dev installs
    raise ModuleNotFoundError(
        "streaming_histogram extension not built; run `maturin develop` first"
    ) from exc

DenseHistogram = _native.DenseHistogram
HistogramRecorder = _native.HistogramRecorder
HistogramSnapshot = _native.HistogramSnapshot
SparseHistogram = _native.SparseHistogram

from .wrapper import Bucket, Histogram, Snapshot, SnapshotDiff

__all__ = [
    "Bucket",
    "DenseHistogram",
    "Histogram",
    "HistogramRecorder",
    "HistogramSnapshot",
    "SparseHistogram",
    "Snapshot",
    "SnapshotDiff",
]
