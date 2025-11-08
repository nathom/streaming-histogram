from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Iterable, Iterator, Literal, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt

from ._native import HistogramRecorder

OutOfRangeMode = Literal["Clip", "Ignore", "Error"]
ValueInput = Union[float, Sequence[float], Iterable[float], npt.NDArray[np.float64]]
MaskInput = Union[npt.NDArray[np.bool_], Sequence[bool], Iterable[bool]]


@dataclass(frozen=True)
class Bucket:
    start: float
    end: float
    count: int


@dataclass(frozen=True)
class Snapshot:
    index: int
    label: Optional[str]
    buckets: tuple[Bucket, ...]
    total: int


@dataclass(frozen=True)
class SnapshotDiff:
    start_index: int
    end_index: int
    start_label: Optional[str]
    end_label: Optional[str]
    buckets: tuple[Bucket, ...]
    total: int


class Histogram:
    """Minimal front-end for the Rust histogram recorder with built-in snapshotting."""

    def __init__(
        self,
        range: Optional[tuple[float, float]],
        bins: Optional[int],
        *,
        bin_width: Optional[float] = None,
        out_of_range: OutOfRangeMode = "Clip",
        max_snapshots: Optional[int] = None,
    ) -> None:
        if (range is None) != (bins is None):
            raise ValueError(
                "range and bins must either both be concrete values or both be None"
            )

        if range is None:
            if bin_width is None:
                raise ValueError(
                    "bin_width must be provided when range and bins are both None"
                )
            self._impl = HistogramRecorder.from_sparse(bin_width, max_snapshots)
            return

        if bin_width is not None:
            raise ValueError(
                "bin_width cannot be combined with explicit range/bins; set bin_width to None"
            )

        assert bins is not None  # for type checkers; validated above
        self._impl = HistogramRecorder.from_dense(
            range, bins, out_of_range, max_snapshots
        )

    def feed(self, values: ValueInput, mask: Optional[MaskInput] = None) -> None:
        if mask is not None:
            values_arr = np.asarray(values, dtype=np.float64)
            mask_arr = np.asarray(mask, dtype=bool)
            self._impl.update_np_mask(values_arr, mask_arr)
            return

        if isinstance(values, np.ndarray):
            self._impl.update_np(np.asarray(values, dtype=np.float64))
            return

        if isinstance(values, numbers.Real):
            self._impl.update([float(values)])
            return

        try:
            iterator_input = cast(Iterable[float], values)
            iterator: Iterator[float] = iter(iterator_input)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                "values must be a float, iterable of floats, or numpy array"
            ) from exc

        numeric_values = [float(v) for v in iterator]
        if numeric_values:
            self._impl.update(numeric_values)

    def buckets(self) -> list[Bucket]:
        return [Bucket(start, end, count) for (start, end), count in self._impl.get()]

    @property
    def total(self) -> int:
        return sum(bucket.count for bucket in self.buckets())

    def snapshot(self, label: Optional[str] = None) -> Snapshot:
        snap = self._impl.snapshot(label)
        buckets = tuple(
            Bucket(start, end, count) for (start, end), count in snap.bins()
        )
        return Snapshot(
            index=snap.index, label=snap.label, buckets=buckets, total=snap.total()
        )

    def diff(
        self, later: Optional[int] = None, earlier: Optional[int] = None
    ) -> SnapshotDiff:
        recorder = self._impl
        later_index = later
        if later_index is None:
            latest = recorder.latest()
            if latest is None:
                raise RuntimeError("no snapshots available")
            later_index = latest.index

        earlier_index = earlier
        if earlier_index is None:
            earlier_index = later_index - 1
            if earlier_index < 0:
                raise RuntimeError("need an earlier snapshot index")

        diff = recorder.diff(later_index, earlier_index)
        buckets = tuple(
            Bucket(start, end, count) for (start, end), count in diff.bins()
        )
        return SnapshotDiff(
            start_index=diff.start_index,
            end_index=diff.end_index,
            start_label=diff.start_label,
            end_label=diff.end_label,
            buckets=buckets,
            total=diff.total(),
        )

    def drain_snapshots(self, end_index: int) -> int:
        if end_index < 0:
            raise ValueError("end_index must be non-negative")
        return int(self._impl.drain_snapshots(end_index))

    def clear_snapshots(self) -> None:
        self._impl.clear_snapshots()

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, index: object) -> Union[Snapshot, SnapshotDiff]:
        if isinstance(index, slice):
            slice_index = cast(
                "slice[Optional[int], Optional[int], Optional[int]]", index
            )
            step = slice_index.step
            if step not in (None, 1):
                raise ValueError("slice step must be 1")

            size = len(self)
            if size == 0:
                raise IndexError("no snapshots available")

            first_id, last_id = self._snapshot_id_bounds()

            def resolve(bound: Optional[int], default: int) -> int:
                if bound is None:
                    return default
                if bound < 0:
                    raise IndexError("snapshot slice bounds must be non-negative IDs")
                return bound

            raw_start = slice_index.start
            raw_stop = slice_index.stop

            start_id = resolve(raw_start, first_id)
            stop_id = resolve(raw_stop, last_id + 1)

            if start_id < first_id or start_id > last_id:
                raise IndexError("snapshot slice start out of range")

            if stop_id < first_id + 1 or stop_id > last_id + 1:
                raise IndexError("snapshot slice stop out of range")

            if stop_id - start_id < 2:
                raise IndexError("slice must span at least two snapshots")

            diff = self._impl.diff(stop_id - 1, start_id)
            buckets = tuple(
                Bucket(start, end, count) for (start, end), count in diff.bins()
            )
            return SnapshotDiff(
                start_index=diff.start_index,
                end_index=diff.end_index,
                start_label=diff.start_label,
                end_label=diff.end_label,
                buckets=buckets,
                total=diff.total(),
            )

        if isinstance(index, int):
            target_id = index
            if target_id < 0:
                # Negative indices count from the newest snapshot.
                first_id, last_id = self._snapshot_id_bounds()
                target_id = last_id + 1 + target_id
                if target_id < first_id:
                    raise IndexError("snapshot index out of range")

            if target_id < 0:
                raise IndexError("snapshot index must be non-negative")

            try:
                snap = self._impl.get_snapshot(target_id)
            except ValueError as exc:  # drained or missing id
                raise IndexError(str(exc)) from exc

            buckets = tuple(
                Bucket(start, end, count) for (start, end), count in snap.bins()
            )
            return Snapshot(
                index=snap.index, label=snap.label, buckets=buckets, total=snap.total()
            )

        raise TypeError("histogram indices must be integers or slices")

    @property
    def last_snapshot(self) -> Optional[Snapshot]:
        latest = self._impl.latest()
        if latest is None:
            return None
        buckets = tuple(
            Bucket(start, end, count) for (start, end), count in latest.bins()
        )
        return Snapshot(
            index=latest.index,
            label=latest.label,
            buckets=buckets,
            total=latest.total(),
        )

    def _snapshot_id_bounds(self) -> tuple[int, int]:
        size = len(self)
        if size == 0:
            raise IndexError("no snapshots available")
        latest = self._impl.latest()
        if latest is None:
            raise IndexError("no snapshots available")
        last_id = int(latest.index)
        first_id = last_id - size + 1
        return first_id, last_id


__all__ = ["Bucket", "Histogram", "Snapshot", "SnapshotDiff"]
