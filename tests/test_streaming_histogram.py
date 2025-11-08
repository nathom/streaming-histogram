from __future__ import annotations

from collections.abc import Callable, Iterable
import math

import numpy as np
import pytest

from streaming_histogram import (
    DenseHistogram,
    Histogram,
    HistogramRecorder,
    SparseHistogram,
)

Bucket = tuple[tuple[float, float], int]
CanonicalValue = float | str
CanonicalBucket = tuple[tuple[CanonicalValue, CanonicalValue], int]


def _canonicalize(entries: Iterable[Bucket]) -> list[CanonicalBucket]:
    normalized: list[CanonicalBucket] = []
    for (start, end), count in entries:
        if math.isnan(start) and math.isnan(end):
            key = ("nan", "nan")
        elif math.isinf(start) and math.isinf(end):
            key = ("-inf" if start < 0 else "inf", "-inf" if end < 0 else "inf")
        else:
            key = (start, end)
        normalized.append((key, count))
    return normalized


def _build_histogram_with_snapshots(count: int) -> Histogram:
    hist = Histogram(None, None, bin_width=0.25)
    for i in range(count):
        hist.feed([i * 0.1, i * 0.1 + 0.05])
        hist.snapshot(f"phase_{i}")
    return hist


def test_histogram_counts_round_trip():
    hist = SparseHistogram(2.5)
    hist.update([-4.9, -0.1, 0.0, 2.49, 2.5])
    assert hist.get() == [
        ((-5.0, -2.5), 1),
        ((-2.5, 0.0), 1),
        ((0.0, 2.5), 2),
        ((2.5, 5.0), 1),
    ]


def test_tracks_special_buckets():
    hist = SparseHistogram(1.0)
    hist.update([float("nan"), float("nan"), float("inf"), float("-inf"), 0.5])

    buckets: list[Bucket] = hist.get()

    def find_bucket(predicate: Callable[[float, float], bool]) -> int:
        for (start, end), count in buckets:
            if predicate(start, end):
                return count
        return 0

    assert find_bucket(lambda s, e: math.isinf(s) and math.isinf(e) and s < 0) == 1
    assert find_bucket(lambda s, e: math.isinf(s) and math.isinf(e) and s > 0) == 1
    assert find_bucket(lambda s, e: math.isnan(s) and math.isnan(e)) == 2
    assert find_bucket(lambda s, e: (s, e) == (0.0, 1.0)) == 1


def test_rejects_invalid_bin_width():
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            SparseHistogram(invalid)


def test_update_np_matches_update():
    arr = np.array([[0.2, 1.1, float("nan")], [float("inf"), -0.3, float("-inf")]])

    list_hist = SparseHistogram(0.5)
    list_hist.update(arr.ravel().tolist())

    np_hist = SparseHistogram(0.5)
    np_hist.update_np(arr)

    assert _canonicalize(np_hist.get()) == _canonicalize(list_hist.get())


def test_update_np_mask_applies_mask():
    values = np.array([[0.2, 1.1, -1.3], [0.49, float("nan"), -0.3]])
    mask = np.array([[True, False, True], [False, True, False]])

    expected = SparseHistogram(0.5)
    expected.update([v for v, flag in zip(values.ravel(), mask.ravel()) if flag])

    masked = SparseHistogram(0.5)
    masked.update_np_mask(values, mask)

    assert _canonicalize(masked.get()) == _canonicalize(expected.get())


def test_update_np_mask_shape_mismatch():
    values = np.zeros((2, 2))
    mask = np.zeros((2, 1), dtype=bool)

    hist = SparseHistogram(1.0)
    with pytest.raises(ValueError):
        hist.update_np_mask(values, mask)



def test_dense_histogram_basic_bins():
    hist = DenseHistogram((0.0, 1.0), 4, "Clip")
    hist.update([0.05, 0.24, 0.25, 0.75, 0.99])

    assert hist.get() == [
        ((0.0, 0.25), 2),
        ((0.25, 0.5), 1),
        ((0.5, 0.75), 0),
        ((0.75, 1.0), 2),
    ]


def test_dense_histogram_clip_behavior():
    hist = DenseHistogram((0.0, 1.0), 2, "Clip")
    hist.update([-0.2, 0.1, 1.5, float("nan")])

    assert hist.get() == [((0.0, 0.5), 2), ((0.5, 1.0), 1)]


def test_dense_histogram_ignore_behavior():
    hist = DenseHistogram((0.0, 1.0), 2, "Ignore")
    hist.update([-0.2, 0.1, 1.5])

    assert hist.get() == [((0.0, 0.5), 1), ((0.5, 1.0), 0)]


def test_dense_histogram_error_behavior():
    hist = DenseHistogram((0.0, 1.0), 2, "Error")

    with pytest.raises(ValueError):
        hist.update([-0.1])
    with pytest.raises(ValueError):
        hist.update([float("nan")])


def test_dense_histogram_update_np():
    hist = DenseHistogram((0.0, 1.0), 2, "Ignore")
    arr = np.array([[0.1, 0.6], [1.2, -0.3]])
    hist.update_np(arr)

    assert hist.get() == [((0.0, 0.5), 1), ((0.5, 1.0), 1)]


def test_recorder_behaves_like_histogram():
    recorder = HistogramRecorder.from_sparse(0.5)
    recorder.update([-0.2, 0.1, 0.5, 0.9])

    sparse = SparseHistogram(0.5)
    sparse.update([-0.2, 0.1, 0.5, 0.9])

    assert _canonicalize(recorder.get()) == _canonicalize(sparse.get())


def test_recorder_snapshot_and_diff():
    recorder = HistogramRecorder.from_sparse(0.5)
    recorder.update([0.1, 0.6, -0.4])
    snap_a = recorder.snapshot(label="phase_a")
    assert snap_a.index == 0
    assert snap_a.total() == 3

    recorder.update([0.2, 1.1])
    snap_b = recorder.snapshot(label=None)
    assert snap_b.index == 1
    assert len(recorder) == 2

    diff = recorder.diff(1, 0)
    diff_bins: list[CanonicalBucket] = _canonicalize(diff.bins())
    assert diff.start_index == 0
    assert diff.end_index == 1
    assert diff.start_label == "phase_a"
    assert diff.end_label is None
    assert diff_bins == [((0.0, 0.5), 1), ((1.0, 1.5), 1)]


def test_recorder_diff_requires_monotonic_ids():
    recorder = HistogramRecorder.from_sparse(1.0)
    recorder.update([0.1])
    recorder.snapshot(label=None)
    recorder.update([1.1])
    recorder.snapshot(label=None)

    with pytest.raises(ValueError, match="later snapshot index must be greater"):
        recorder.diff(0, 1)


def test_recorder_diff_missing_snapshot():
    recorder = HistogramRecorder.from_sparse(1.0)
    recorder.update([0.1])
    recorder.snapshot(label=None)

    with pytest.raises(ValueError, match="snapshot index 9 not found"):
        recorder.diff(9, 0)


def test_recorder_diff_handles_special_buckets():
    recorder = HistogramRecorder.from_sparse(1.0)
    recorder.update([float("nan"), float("inf"), float("-inf")])
    recorder.snapshot(label=None)

    recorder.update([float("nan"), float("inf")])
    recorder.snapshot(label=None)
    diff = recorder.diff(1, 0)
    diff_bins: list[CanonicalBucket] = _canonicalize(diff.bins())

    assert diff_bins == [(("inf", "inf"), 1), (("nan", "nan"), 1)]


def test_recorder_diff_no_change_returns_empty_histogram():
    recorder = HistogramRecorder.from_sparse(0.5)
    recorder.update([0.1])
    recorder.snapshot(label=None)
    snapshot = recorder.snapshot(label=None)

    later_index = snapshot.index
    diff = recorder.diff(later_index, later_index - 1)
    assert diff.bins() == []
    assert diff.total() == 0


def test_recorder_diff_matches_increment_sparse():
    bin_width = 0.25
    recorder = HistogramRecorder.from_sparse(bin_width)
    recorder.update([-0.6, -0.4, 0.0])
    first = recorder.snapshot(label=None)

    new_batch = [-0.1, 0.1, 0.37, 0.61, 1.02]
    recorder.update(new_batch)
    second = recorder.snapshot(label=None)

    second_index = second.index
    first_index = first.index
    diff = recorder.diff(second_index, first_index)
    diff_bins: list[CanonicalBucket] = _canonicalize(diff.bins())

    expected = SparseHistogram(bin_width)
    expected.update(new_batch)
    assert diff_bins == _canonicalize(expected.get())


def test_recorder_diff_matches_increment_dense():
    recorder = HistogramRecorder.from_dense((0.0, 1.0), 4, "Clip")
    recorder.update([0.05, 0.2])
    first = recorder.snapshot(label=None)

    new_batch = [0.1, 0.26, 0.74, 1.2, -0.4]
    recorder.update(new_batch)
    second = recorder.snapshot(label=None)

    second_index = second.index
    first_index = first.index
    diff = recorder.diff(second_index, first_index)

    expected = DenseHistogram((0.0, 1.0), 4, "Clip")
    expected.update(new_batch)
    assert diff.bins() == expected.get()


def test_histogram_wrapper_matches_sparse_behavior():
    wrapper = Histogram(None, None, bin_width=0.5)
    wrapper.feed(-0.25)
    wrapper.feed([0.1, 0.49])

    values = np.array([0.5, 0.75, float("nan"), float("inf"), float("-inf")])
    mask = np.array([True, False, True, True, True])
    wrapper.feed(values, mask=mask)

    sparse = SparseHistogram(0.5)
    sparse.update([-0.25])
    sparse.update([0.1, 0.49])
    sparse.update_np_mask(values, mask)

    wrapped_bins = [((b.start, b.end), b.count) for b in wrapper.buckets()]
    assert _canonicalize(wrapped_bins) == _canonicalize(sparse.get())


def test_histogram_wrapper_snapshots_and_diff_defaults():
    hist = Histogram(None, None, bin_width=0.25)
    hist.feed([0.1, 0.2])
    first = hist.snapshot("phase_a")
    assert first.index == 0

    hist.feed([0.4, 0.6, 0.9])
    second = hist.snapshot()
    assert second.index == 1
    assert hist.last_snapshot == second

    diff = hist.diff()
    assert diff.start_index == 0
    assert diff.end_index == 1
    assert diff.start_label == "phase_a"
    assert diff.buckets
    assert diff.total == 3

    hist2 = Histogram((0.0, 1.0), 2)
    hist2.feed([0.25])
    assert hist2.snapshot().total == 1


def test_histogram_len_and_indexing():
    hist = _build_histogram_with_snapshots(3)
    assert len(hist) == 3
    assert hist[0].index == 0
    assert hist[-1].index == 2

    with pytest.raises(IndexError):
        _ = hist[3]
    with pytest.raises(IndexError):
        _ = hist[-4]
    with pytest.raises(TypeError):
        _ = hist[1.0]  # type: ignore[index]


def test_histogram_slicing_returns_diff():
    hist = _build_histogram_with_snapshots(5)

    mid = hist[1:3]
    assert mid.start_index == 1
    assert mid.end_index == 2

    last_id = hist[-1].index
    tail = hist[last_id - 2 : last_id + 1]
    assert tail.start_index == len(hist) - 3
    assert tail.end_index == len(hist) - 1

    head = hist[:4]
    assert head.start_index == 0
    assert head.end_index == 3

    with pytest.raises(IndexError):
        _ = hist[-3:]
    with pytest.raises(IndexError):
        _ = hist[1:2]
    with pytest.raises(ValueError):
        _ = hist[::2]


def test_histogram_slice_negative_bounds_rejected():
    hist = _build_histogram_with_snapshots(3)
    with pytest.raises(IndexError):
        _ = hist[-1:2]
    with pytest.raises(IndexError):
        _ = hist[0:-1]


def test_histogram_slice_errors_for_drained_ids():
    hist = _build_histogram_with_snapshots(4)
    drained = hist.drain_snapshots(2)
    assert drained == 2

    with pytest.raises(IndexError):
        _ = hist[0:2]

    window = hist[2:4]
    assert window.start_index == 2
    assert window.end_index == 3


def test_recorder_drain_and_clear_snapshots():
    recorder = HistogramRecorder.from_sparse(0.5)
    for value in range(4):
        recorder.update([float(value)])
        recorder.snapshot(label=None)

    assert len(recorder) == 4
    removed = recorder.drain_snapshots(2)
    assert removed == 2
    assert len(recorder) == 2

    with pytest.raises(ValueError, match="snapshot index 0 not found"):
        recorder.get_snapshot(0)

    recorder.clear_snapshots()
    assert len(recorder) == 0

    recorder.update([1.0])
    snap = recorder.snapshot(label=None)
    # IDs remain monotonic even after clearing.
    assert snap.index == 4


def test_recorder_max_snapshots_auto_drains():
    recorder = HistogramRecorder.from_sparse(0.5, max_snapshots=2)
    for value in range(3):
        recorder.update([float(value)])
        recorder.snapshot(label=None)

    assert len(recorder) == 2
    with pytest.raises(ValueError, match="snapshot index 0 not found"):
        recorder.get_snapshot(0)

    latest = recorder.latest()
    assert latest is not None
    assert latest.index == 2


def test_recorder_drain_snapshots_exclusive_upper_bound():
    recorder = HistogramRecorder.from_sparse(0.5)
    for value in range(5):
        recorder.update([float(value)])
        recorder.snapshot(label=f"phase_{value}")

    drained = recorder.drain_snapshots(3)
    assert drained == 3
    assert len(recorder) == 2

    remaining = [recorder.get_snapshot(idx).index for idx in (3, 4)]
    assert remaining == [3, 4]

    with pytest.raises(ValueError, match="snapshot index 2 not found"):
        recorder.get_snapshot(2)


def test_recorder_drain_snapshots_idempotent_and_beyond_end():
    recorder = HistogramRecorder.from_sparse(0.5)
    assert recorder.drain_snapshots(10) == 0

    for value in range(3):
        recorder.update([float(value)])
        recorder.snapshot(label=None)

    assert recorder.drain_snapshots(1) == 1
    assert len(recorder) == 2

    drained_all = recorder.drain_snapshots(100)
    assert drained_all == 2
    assert len(recorder) == 0
    assert recorder.latest() is None

    recorder.update([42.0])
    snap = recorder.snapshot(label=None)
    assert snap.index == 3


def test_histogram_drain_and_clear_round_trip():
    hist = Histogram(None, None, bin_width=0.25)
    for value in range(4):
        hist.feed([value * 0.1])
        hist.snapshot(f"phase_{value}")

    drained = hist.drain_snapshots(2)
    assert drained == 2
    assert len(hist) == 2

    with pytest.raises(IndexError):
        _ = hist[0]

    kept = hist[2]
    assert kept.index == 2

    window = hist[2:4]
    assert window.start_index == 2
    assert window.end_index == 3

    assert hist[-1].index == 3

    hist.clear_snapshots()
    assert len(hist) == 0
    with pytest.raises(IndexError):
        _ = hist[2:4]


def test_histogram_slice_full_range_after_partial_drain():
    hist = _build_histogram_with_snapshots(5)
    hist.drain_snapshots(3)

    full = hist[:]
    assert full.start_index == 3
    assert full.end_index == 4

    assert hist[3].index == 3
    assert hist[-1].index == 4


def test_histogram_negative_index_after_drain_enforces_bounds():
    hist = _build_histogram_with_snapshots(4)
    hist.drain_snapshots(2)

    assert hist[-1].index == 3
    assert hist[-2].index == 2

    with pytest.raises(IndexError):
        _ = hist[-3]


def test_recorder_diff_after_auto_drain_window():
    recorder = HistogramRecorder.from_sparse(0.5, max_snapshots=3)
    for i in range(6):
        recorder.update([float(i)])
        recorder.snapshot(label=None)

    assert len(recorder) == 3

    diff = recorder.diff(5, 4)
    assert diff.start_index == 4
    assert diff.end_index == 5

    with pytest.raises(ValueError, match="snapshot index 2 not found"):
        recorder.diff(5, 2)


def test_recorder_drain_snapshots_noop_when_before_window():
    recorder = HistogramRecorder.from_sparse(0.5)
    for i in range(3):
        recorder.update([float(i)])
        recorder.snapshot(label=None)

    hist_ids = [snap.index for snap in (recorder.get_snapshot(0), recorder.get_snapshot(1), recorder.get_snapshot(2))]
    assert hist_ids == [0, 1, 2]

    removed = recorder.drain_snapshots(0)
    assert removed == 0

    recorder.drain_snapshots(2)
    assert len(recorder) == 1

    # Second call with same bound should remove nothing.
    removed_again = recorder.drain_snapshots(2)
    assert removed_again == 0
def test_histogram_sparse_requires_bin_width():
    with pytest.raises(ValueError, match="bin_width must be provided"):
        Histogram(None, None)


def test_histogram_partial_range_bins_rejected():
    with pytest.raises(ValueError, match="range and bins must either both be concrete values or both be None"):
        Histogram(range=(0.0, 1.0), bins=None)


def test_histogram_dense_args_cannot_include_bin_width():
    with pytest.raises(ValueError, match="bin_width cannot be combined"):
        Histogram(range=(0.0, 1.0), bins=2, bin_width=0.25)


def test_histogram_sparse_accepts_explicit_nones():
    hist = Histogram(None, None, bin_width=0.25)
    hist.feed([0.1, 0.2])
    assert hist.total == 2
