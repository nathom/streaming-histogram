from __future__ import annotations

import os
from typing import Iterable, Sequence, overload

import numpy as np
import numpy.typing as npt

from ._native import (
    DenseHistogram,
    HistogramRecorder,
    HistogramSnapshot,
    SparseHistogram,
    OutOfRangeMode,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
ValueInput = float | Sequence[float] | Iterable[float] | FloatArray

class Bucket:
    start: float
    end: float
    count: int

class DensityBucket:
    start: float
    end: float
    density: float

class Snapshot:
    index: int
    label: str | None
    buckets: tuple[Bucket, ...]
    total: int

class SnapshotDiff:
    start_index: int
    end_index: int
    start_label: str | None
    end_label: str | None
    buckets: tuple[Bucket, ...]
    total: int

class Histogram:
    def __init__(
        self,
        range: tuple[float, float] | None,
        bins: int | None,
        *,
        bin_width: float | None = None,
        out_of_range: OutOfRangeMode = ...,
        max_snapshots: int | None = None,
        save_path: os.PathLike[str] | str | None = ...,
        resume_from_path: bool = ...,
    ) -> None: ...
    def feed(
        self,
        values: ValueInput,
        mask: BoolArray | Sequence[bool] | Iterable[bool] | None = ...,
    ) -> None: ...
    def buckets(self) -> list[Bucket]: ...
    def density(self) -> list[DensityBucket]: ...
    @property
    def total(self) -> int: ...
    def snapshot(self, label: str | None = ...) -> Snapshot: ...
    def diff(
        self, later: int | None = ..., earlier: int | None = ...
    ) -> SnapshotDiff: ...
    def drain_snapshots(self, end_index: int) -> int: ...
    def clear_snapshots(self) -> None: ...
    @property
    def last_snapshot(self) -> Snapshot | None: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> Snapshot: ...
    @overload
    def __getitem__(self, index: slice) -> SnapshotDiff: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, payload: str) -> Histogram: ...

__all__ = [
    "Bucket",
    "DensityBucket",
    "DenseHistogram",
    "Histogram",
    "HistogramRecorder",
    "HistogramSnapshot",
    "SparseHistogram",
    "Snapshot",
    "SnapshotDiff",
]
