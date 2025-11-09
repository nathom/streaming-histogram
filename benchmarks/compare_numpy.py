"""Benchmark streaming_histogram against numpy.histogram for single builds."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np

from streaming_histogram import Histogram


def ensure_native_extension_fresh(repo_root: Path) -> None:
    so_candidates = sorted((repo_root / "streaming_histogram").glob("_native*.so"))
    if not so_candidates:
        raise RuntimeError(
            "streaming_histogram/_native*.so not found; run `nix develop --command bash -lc '"
            "source .venv/bin/activate && maturin develop'` to rebuild the extension before benchmarking."
        )

    newest_so = max(so_candidates, key=lambda path: path.stat().st_mtime)
    source_paths = list((repo_root / "src").rglob("*.rs"))
    source_paths.extend(
        path
        for path in (
            repo_root / "Cargo.toml",
            repo_root / "Cargo.lock",
            repo_root / "pyproject.toml",
        )
        if path.exists()
    )
    if not source_paths:
        return

    newest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    if newest_so.stat().st_mtime + 1e-6 < newest_source_mtime:
        raise RuntimeError(
            "Detected stale streaming_histogram native build (Rust sources are newer than "
            "streaming_histogram/_native*.so). Run `nix develop --command bash -lc 'source .venv/bin/activate "
            "&& maturin develop'` and rerun the benchmark."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare streaming_histogram.DenseHistogram to numpy.histogram "
            "for a single histogram build."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[1_000_000, 5_000_000, 10_000_000, 100_000_000],
        help="one or more sample counts to benchmark",
    )
    parser.add_argument("--bins", type=int, default=256, help="bin count used for numpy and dense mode")
    parser.add_argument(
        "--mode",
        choices=("sparse", "dense"),
        default="dense",
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
    parser.add_argument(
        "--skip-build-check",
        action="store_true",
        help="skip validating that streaming_histogram's native extension is freshly built",
    )
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
    repo_root = Path(__file__).resolve().parents[1]
    if not args.skip_build_check:
        try:
            ensure_native_extension_fresh(repo_root)
        except RuntimeError as exc:
            sys.exit(str(exc))
    user_range = None if args.range is None else (args.range[0], args.range[1])

    print(f"bins:    {args.bins}")
    print(f"mode:    {args.mode}")
    print(f"runs:    warmup={args.warmup}, repeat={args.repeat}")
    print(f"dist:    {args.distribution}")

    for samples in args.samples:
        data = make_data(samples, args.distribution, args.seed)
        hist_range = determine_range(data, user_range)

        streaming_times = benchmark(
            streaming_callable(data, args.bins, hist_range, args.mode), args.repeat, args.warmup
        )
        numpy_times = benchmark(numpy_callable(data, args.bins, hist_range), args.repeat, args.warmup)

        print(f"\n=== {samples:,} samples ===")
        if args.mode == "sparse":
            width = (hist_range[1] - hist_range[0]) / args.bins
            print(f"bin width (sparse): {width:.6f}")
        print(f"range:   {hist_range[0]:.4f} .. {hist_range[1]:.4f}")
        print(format_stats("streaming_histogram", streaming_times))
        print(format_stats("numpy.histogram", numpy_times))
        ratio = statistics.fmean(numpy_times) / statistics.fmean(streaming_times)
        if ratio >= 1.0:
            print(f"streaming_histogram is {ratio:.2f}x faster on average")
        else:
            print(f"numpy.histogram is {1/ratio:.2f}x faster on average")


if __name__ == "__main__":
    main()
