"""Benchmark accumulating chunked data with numpy.histogram vs streaming_histogram."""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import threading
import time
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from streaming_histogram import Histogram

VERY_WIDE_RANGE_SCALE = 1_000_000.0

try:
    import psutil
except ImportError:  # pragma: no cover - optional dep for profiling
    psutil = None  # type: ignore[assignment]

Mode = Literal["sparse", "dense", "both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a series of large numpy arrays, then compare the cost of building "
            "a histogram by storing them for numpy.histogram versus updating a "
            "streaming_histogram incrementally."
        )
    )
    parser.add_argument("--chunk-size", type=int, default=500_000, help="values per chunk")
    parser.add_argument("--chunks", type=int, default=10, help="number of chunks to generate")
    parser.add_argument("--bins", type=int, default=256, help="bin count used for dense/numpy modes")
    parser.add_argument("--mode", choices=("sparse", "dense", "both"), default="both", help="which histogram backend(s) to time")
    parser.add_argument("--distribution", choices=("normal", "uniform"), default="normal", help="value distribution")
    parser.add_argument("--range", type=float, nargs=2, metavar=("LO", "HI"), default=(-6.0, 6.0), help="histogram range for dense mode and numpy")
    parser.add_argument("--seed", type=int, default=1234, help="seed for RNG")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations before timing")
    parser.add_argument("--repeat", type=int, default=8, help="timed iterations")
    parser.add_argument(
        "--wide-range",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            f"scale generated samples and the histogram range by {VERY_WIDE_RANGE_SCALE:.0f}× "
            "to simulate extremely wide value distributions (default: off)"
        ),
    )
    parser.add_argument(
        "--profile-memory",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="track per-run peak RSS increase using psutil (default: on)",
    )
    return parser.parse_args()


def make_chunk(
    rng: np.random.Generator, chunk_size: int, distribution: str, *, wide_range: bool
) -> np.ndarray:
    if distribution == "normal":
        chunk = rng.normal(size=chunk_size).astype(np.float64, copy=False)
    elif distribution == "uniform":
        chunk = rng.uniform(-1.0, 1.0, size=chunk_size).astype(np.float64, copy=False)
    else:
        raise ValueError(f"unsupported distribution {distribution}")

    if wide_range:
        chunk *= VERY_WIDE_RANGE_SCALE

    return chunk


def benchmark(
    func: Callable[[], None], repeat: int, warmup: int, *, profile_memory: bool
) -> tuple[list[float], list[float] | None]:
    for _ in range(warmup):
        gc.collect()
        func()
        gc.collect()
    samples: list[float] = []
    peaks: list[float] = []
    for _ in range(repeat):
        duration, peak = run_once(func, profile_memory)
        samples.append(duration)
        if peak is not None:
            peaks.append(peak)
    return samples, (peaks if peaks else None)


def run_once(func: Callable[[], None], profile_memory: bool) -> tuple[float, float | None]:
    gc.collect()
    if profile_memory:
        if psutil is None:
            raise RuntimeError("psutil is required for memory profiling; install the dev extras or rerun with --no-profile-memory")
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss
        peak = baseline
        stop_event = threading.Event()

        def monitor() -> None:
            nonlocal peak
            while not stop_event.is_set():
                rss = process.memory_info().rss
                if rss > peak:
                    peak = rss
                time.sleep(0.002)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        try:
            start = time.perf_counter()
            func()
            duration = (time.perf_counter() - start) * 1_000.0
        finally:
            stop_event.set()
            monitor_thread.join()
        gc.collect()
        peak_diff_mb = max(0.0, peak - baseline) / (1024 * 1024)
        return duration, peak_diff_mb

    start = time.perf_counter()
    func()
    duration = (time.perf_counter() - start) * 1_000.0
    gc.collect()
    return duration, None


def format_stats(name: str, values: Iterable[float]) -> str:
    data = list(values)
    mean = statistics.fmean(data)
    spread = statistics.pstdev(data)
    return f"{name:>35}: {mean:8.3f} ms ± {spread:6.3f} ms  (min={min(data):7.3f}, max={max(data):7.3f})"


def format_memory(name: str, values: Iterable[float]) -> str:
    data = list(values)
    mean = statistics.fmean(data)
    spread = statistics.pstdev(data) if len(data) > 1 else 0.0
    return f"{name:>35}: {mean:8.3f} MiB ± {spread:6.3f}  (min={min(data):7.3f}, max={max(data):7.3f})"


def numpy_callable(
    *,
    chunk_size: int,
    chunks: int,
    bins: int,
    hist_range: Sequence[float],
    distribution: str,
    seed: int,
    wide_range: bool,
) -> Callable[[], None]:
    def _runner() -> None:
        rng = np.random.default_rng(seed)
        stored: list[np.ndarray] = []
        for _ in range(chunks):
            stored.append(make_chunk(rng, chunk_size, distribution, wide_range=wide_range))
        merged = np.concatenate(stored, axis=0)
        np.histogram(merged, bins=bins, range=hist_range)

    return _runner


def streaming_callable(
    *,
    chunk_size: int,
    chunks: int,
    bins: int,
    hist_range: Sequence[float],
    distribution: str,
    seed: int,
    mode: Literal["sparse", "dense"],
    wide_range: bool,
) -> Callable[[], None]:
    width = (hist_range[1] - hist_range[0]) / bins

    def _runner_dense() -> None:
        rng = np.random.default_rng(seed)
        hist = Histogram(range=(hist_range[0], hist_range[1]), bins=bins, out_of_range="Clip")
        for _ in range(chunks):
            hist.feed(make_chunk(rng, chunk_size, distribution, wide_range=wide_range))
        hist.buckets()

    def _runner_sparse() -> None:
        rng = np.random.default_rng(seed)
        hist = Histogram(range=None, bins=None, bin_width=width)
        for _ in range(chunks):
            hist.feed(make_chunk(rng, chunk_size, distribution, wide_range=wide_range))
        hist.buckets()

    return _runner_dense if mode == "dense" else _runner_sparse


def run_mode(
    mode: Literal["sparse", "dense"],
    *,
    chunk_size: int,
    chunks: int,
    bins: int,
    hist_range: Sequence[float],
    distribution: str,
    seed: int,
    warmup: int,
    repeat: int,
    profile_memory: bool,
    wide_range: bool,
) -> None:
    streaming_times, streaming_mem = benchmark(
        streaming_callable(
            chunk_size=chunk_size,
            chunks=chunks,
            bins=bins,
            hist_range=hist_range,
            distribution=distribution,
            seed=seed,
            mode=mode,
            wide_range=wide_range,
        ),
        repeat=repeat,
        warmup=warmup,
        profile_memory=profile_memory,
    )
    numpy_times, numpy_mem = benchmark(
        numpy_callable(
            chunk_size=chunk_size,
            chunks=chunks,
            bins=bins,
            hist_range=hist_range,
            distribution=distribution,
            seed=seed,
            wide_range=wide_range,
        ),
        repeat=repeat,
        warmup=warmup,
        profile_memory=profile_memory,
    )
    print(f"\nMode: {mode}")
    print(format_stats("streaming_histogram (update)", streaming_times))
    if streaming_mem is not None:
        print(format_memory("streaming_histogram peak RSS", streaming_mem))
    print(format_stats("numpy.histogram (store+concat)", numpy_times))
    if numpy_mem is not None:
        print(format_memory("numpy peak RSS", numpy_mem))
    ratio = statistics.fmean(numpy_times) / statistics.fmean(streaming_times)
    if ratio >= 1:
        print(f"streaming_histogram is {ratio:.2f}x faster on average")
    else:
        print(f"numpy.histogram is {1/ratio:.2f}x faster on average")


def main() -> None:
    args = parse_args()
    total = args.chunk_size * args.chunks
    base_range = (float(args.range[0]), float(args.range[1]))
    if not base_range[0] < base_range[1]:
        raise ValueError("histogram range must have lo < hi")
    if args.wide_range:
        hist_range = (
            base_range[0] * VERY_WIDE_RANGE_SCALE,
            base_range[1] * VERY_WIDE_RANGE_SCALE,
        )
    else:
        hist_range = base_range

    print(f"chunks:         {args.chunks}")
    print(f"chunk size:     {args.chunk_size:,}")
    print(f"total samples:  {total:,}")
    print(f"bins:           {args.bins}")
    print(f"range:          {base_range[0]:.3f} .. {base_range[1]:.3f}")
    if args.wide_range:
        print(f"scaled range:   {hist_range[0]:.3f} .. {hist_range[1]:.3f}")
    print(f"distribution:   {args.distribution}")
    if args.wide_range:
        print(f"data pattern:   very wide range (values scaled by {VERY_WIDE_RANGE_SCALE:.0f}×)")
    print(f"runs:           warmup={args.warmup}, repeat={args.repeat}")
    if args.profile_memory:
        print("memory metric: peak RSS increase per run (MiB)")

    modes: Sequence[Literal["sparse", "dense"]]
    if args.mode == "both":
        modes = ("sparse", "dense")
    else:
        modes = (args.mode,)  # type: ignore[assignment]
    for mode in modes:
        run_mode(
            mode,
            chunk_size=args.chunk_size,
            chunks=args.chunks,
            bins=args.bins,
            hist_range=hist_range,
            distribution=args.distribution,
            seed=args.seed,
            warmup=args.warmup,
            repeat=args.repeat,
            profile_memory=args.profile_memory,
            wide_range=args.wide_range,
        )


if __name__ == "__main__":
    main()
