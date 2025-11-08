"""Benchmark streaming_histogram against numpy.histogram for single builds."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Iterable, Tuple

import numpy as np

from streaming_histogram import Histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare streaming_histogram.DenseHistogram to numpy.histogram "
            "for a single histogram build."
        )
    )
    parser.add_argument("--samples", type=int, default=1_000_000, help="number of values")
    parser.add_argument("--bins", type=int, default=256, help="bin count used for numpy and dense mode")
    parser.add_argument(
        "--mode",
        choices=("sparse", "dense"),
        default="sparse",
        help="Histogram mode to benchmark",
    )
    parser.add_argument(
        "--distribution",
        choices=("normal", "uniform"),
        default="normal",
        help="value distribution to sample",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=None,
        help="histogram range; defaults to min/max of generated data",
    )
    parser.add_argument("--seed", type=int, default=2024, help="seed for RNG")
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup runs")
    parser.add_argument("--repeat", type=int, default=10, help="number of timed runs")
    return parser.parse_args()


def make_data(samples: int, distribution: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        return rng.normal(size=samples).astype(np.float64, copy=False)
    if distribution == "uniform":
        return rng.uniform(-1.0, 1.0, size=samples).astype(np.float64, copy=False)
    raise ValueError(f"unknown distribution {distribution}")


def benchmark(func: Callable[[], None], repeat: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        func()
    timings: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        timings.append((end - start) * 1_000.0)  # milliseconds
    return timings


def format_stats(name: str, values: Iterable[float]) -> str:
    samples = list(values)
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    return f"{name:>25}: {mean:8.3f} ms ± {stdev:6.3f} ms  (min={min(samples):7.3f}, max={max(samples):7.3f})"


def streaming_callable(
    data: np.ndarray,
    bins: int,
    hist_range: Tuple[float, float],
    mode: str,
) -> Callable[[], None]:
    width = (hist_range[1] - hist_range[0]) / bins

    def _runner_dense() -> None:
        hist = Histogram(range=hist_range, bins=bins, out_of_range="Clip")
        hist.feed(data)

    def _runner_sparse() -> None:
        hist = Histogram(range=None, bins=None, bin_width=width)
        hist.feed(data)

    if mode == "dense":
        return _runner_dense
    return _runner_sparse


def numpy_callable(data: np.ndarray, bins: int, hist_range: Tuple[float, float]) -> Callable[[], None]:
    def _runner() -> None:
        np.histogram(data, bins=bins, range=hist_range)

    return _runner


def determine_range(data: np.ndarray, user_range: tuple[float, float] | None) -> tuple[float, float]:
    if user_range is not None:
        lo, hi = user_range
    else:
        lo = float(np.min(data))
        hi = float(np.max(data))
    if not lo < hi:
        raise ValueError("histogram range must have lo < hi")
    return (float(lo), float(hi))


def main() -> None:
    args = parse_args()
    data = make_data(args.samples, args.distribution, args.seed)
    hist_range = determine_range(data, None if args.range is None else (args.range[0], args.range[1]))

    streaming_times = benchmark(
        streaming_callable(data, args.bins, hist_range, args.mode), args.repeat, args.warmup
    )
    numpy_times = benchmark(numpy_callable(data, args.bins, hist_range), args.repeat, args.warmup)

    print(f"samples: {args.samples:,}")
    print(f"bins:    {args.bins}")
    if args.mode == "sparse":
        width = (hist_range[1] - hist_range[0]) / args.bins
        print(f"bin width (sparse): {width:.6f}")
    print(f"mode:    {args.mode}")
    print(f"range:   {hist_range[0]:.4f} .. {hist_range[1]:.4f}")
    print(f"runs:    warmup={args.warmup}, repeat={args.repeat}")
    print()
    print(format_stats("streaming_histogram", streaming_times))
    print(format_stats("numpy.histogram", numpy_times))
    ratio = statistics.fmean(numpy_times) / statistics.fmean(streaming_times)
    if ratio >= 1.0:
        print(f"\nstreaming_histogram is {ratio:.2f}x faster on average")
    else:
        print(f"\nnumpy.histogram is {1/ratio:.2f}x faster on average")


if __name__ == "__main__":
    main()
