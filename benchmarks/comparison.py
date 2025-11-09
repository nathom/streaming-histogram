"""Benchmark Dense/Sparse histogram parallel vs sequential vs NumPy scaling.

This script measures how the Rust-backed DenseHistogram and SparseHistogram
behave when processing increasingly large numpy arrays via either a single bulk
update that can leverage Rayon parallelism or a sequential path that stays on a
single thread. NumPy's ``histogram`` is included as a baseline so the tabular
output and plot show three columns/lines for quick comparison across input
sizes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np

from streaming_histogram import DenseHistogram, SparseHistogram

Distribution = Literal["normal", "uniform"]
ModeName = Literal["parallel", "sequential", "numpy"]
HistogramKind = Literal["dense", "sparse"]

# DenseHistogram switches to Rayon once a single update receives at least this
# many samples. The sequential path feeds batches that stay below the threshold.
PARALLEL_MIN_VALUES = 2_000_000
SEQUENTIAL_MAX_BATCH = PARALLEL_MIN_VALUES - 1
DEFAULT_SIZES: tuple[int, ...] = (
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
)


@dataclass
class ModeSummary:
    mean_ms: float
    spread_ms: float
    samples_ms: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_ms": self.mean_ms,
            "spread_ms": self.spread_ms,
            "samples_ms": list(self.samples_ms),
        }


@dataclass
class SizeResult:
    size: int
    sequential: ModeSummary
    parallel: ModeSummary
    numpy_runtime: ModeSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "sequential": self.sequential.to_dict(),
            "parallel": self.parallel.to_dict(),
            "numpy_runtime": self.numpy_runtime.to_dict(),
        }


@dataclass
class SuiteResult:
    kind: HistogramKind
    bins: int
    hist_range: tuple[float, float]
    distribution: Distribution
    warmup: int
    repeat: int
    figure_path: Path
    rows: list[SizeResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "bins": self.bins,
            "hist_range": [float(self.hist_range[0]), float(self.hist_range[1])],
            "distribution": self.distribution,
            "warmup": self.warmup,
            "repeat": self.repeat,
            "figure_path": str(self.figure_path),
            "rows": [row.to_dict() for row in self.rows],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DenseHistogram.parallel (single bulk update) with a forced "
            "sequential path (chunked updates) across a range of input sizes."
        )
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="value counts to benchmark (default: log-spaced from 5e4 to 1e8)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=512,
        help="bin count for the dense histogram (default: 512)",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(-6.0, 6.0),
        help="histogram min/max (default: -6 to 6)",
    )
    parser.add_argument(
        "--distribution",
        choices=("normal", "uniform"),
        default="normal",
        help="sample distribution (default: normal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20241108,
        help="RNG seed used for sample generation (default: 20241108)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of untimed warmup iterations per mode (default: 1)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="number of timed iterations per mode (default: 3)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("benchmarks") / "dense_parallel_scaling.png",
        help="destination for the generated plot (default: benchmarks/dense_parallel_scaling.png)",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="display the matplotlib window after saving the plot",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        metavar="PATH",
        help="write benchmark statistics to PATH as JSON (default: disabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = sorted({size for size in args.sizes if size > 0})
    if not sizes:
        raise SystemExit("provide at least one positive --size value")
    hist_range = (float(args.range[0]), float(args.range[1]))
    if not hist_range[0] < hist_range[1]:
        raise SystemExit("histogram range must satisfy LO < HI")

    rng = np.random.default_rng(args.seed)
    size_seed_pairs = [
        (size, int(rng.integers(0, np.iinfo(np.int32).max))) for size in sizes
    ]

    suite_results: list[SuiteResult] = []

    suite_results.append(
        run_histogram_suite(
            kind="dense",
            size_seed_pairs=size_seed_pairs,
            bins=args.bins,
            hist_range=hist_range,
            distribution=args.distribution,
            warmup=args.warmup,
            repeat=args.repeat,
            figure_path=args.figure,
        )
    )
    sparse_figure = derive_sparse_figure_path(args.figure)
    suite_results.append(
        run_histogram_suite(
            kind="sparse",
            size_seed_pairs=size_seed_pairs,
            bins=args.bins,
            hist_range=hist_range,
            distribution=args.distribution,
            warmup=args.warmup,
            repeat=args.repeat,
            figure_path=sparse_figure,
        )
    )

    if args.json_output:
        write_json_results(suite_results, args.json_output)

    if args.show:
        plt.show()


def run_histogram_suite(
    *,
    kind: HistogramKind,
    size_seed_pairs: list[tuple[int, int]],
    bins: int,
    hist_range: tuple[float, float],
    distribution: Distribution,
    warmup: int,
    repeat: int,
    figure_path: Path,
) -> SuiteResult:
    label = "Dense" if kind == "dense" else "Sparse"
    print(f"Benchmarking {label}Histogram scaling...\n")
    summaries_parallel: list[ModeSummary] = []
    summaries_sequential: list[ModeSummary] = []
    summaries_numpy: list[ModeSummary] = []
    rows: list[SizeResult] = []

    for size, seed in size_seed_pairs:
        samples = generate_samples(size, distribution, hist_range, seed)
        seq_summary = benchmark_mode(
            samples,
            mode="sequential",
            bins=bins,
            hist_range=hist_range,
            warmup=warmup,
            repeat=repeat,
            backend=kind,
        )
        par_summary = benchmark_mode(
            samples,
            mode="parallel",
            bins=bins,
            hist_range=hist_range,
            warmup=warmup,
            repeat=repeat,
            backend=kind,
        )
        numpy_summary = benchmark_mode(
            samples,
            mode="numpy",
            bins=bins,
            hist_range=hist_range,
            warmup=warmup,
            repeat=repeat,
            backend=kind,
        )
        summaries_sequential.append(seq_summary)
        summaries_parallel.append(par_summary)
        summaries_numpy.append(numpy_summary)
        rows.append(
            SizeResult(
                size=size,
                sequential=seq_summary,
                parallel=par_summary,
                numpy_runtime=numpy_summary,
            )
        )
        print(
            format_row(
                size,
                sequential=seq_summary,
                parallel=par_summary,
                numpy_runtime=numpy_summary,
            )
        )

    print()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{label}Histogram vs NumPy update scaling"
    plot_results(
        sizes=[size for size, _ in size_seed_pairs],
        sequential=summaries_sequential,
        parallel=summaries_parallel,
        numpy=summaries_numpy,
        destination=figure_path,
        title=title,
    )
    print(f"Wrote plot to {figure_path}")
    return SuiteResult(
        kind=kind,
        bins=bins,
        hist_range=hist_range,
        distribution=distribution,
        warmup=warmup,
        repeat=repeat,
        figure_path=figure_path,
        rows=rows,
    )


def write_json_results(results: list[SuiteResult], destination: str) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suites": [suite.to_dict() for suite in results],
    }
    serialized = json.dumps(payload, indent=2)
    if destination == "-":
        print(serialized)
        return
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(serialized + "\n", encoding="utf-8")


def generate_samples(
    size: int,
    distribution: Distribution,
    hist_range: tuple[float, float],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        lo, hi = hist_range
        data = rng.uniform(lo, hi, size=size)
    elif distribution == "normal":
        mean = sum(hist_range) / 2.0
        span = hist_range[1] - hist_range[0]
        # Use a std-dev that keeps most values within the configured range.
        stddev = span / 8.0
        data = rng.normal(loc=mean, scale=stddev, size=size)
    else:  # pragma: no cover - safeguarded by argparse choices
        raise ValueError(f"unsupported distribution: {distribution}")
    return np.asarray(data, dtype=np.float64)


def benchmark_mode(
    samples: np.ndarray,
    *,
    mode: ModeName,
    bins: int,
    hist_range: tuple[float, float],
    warmup: int,
    repeat: int,
    backend: HistogramKind = "dense",
) -> ModeSummary:
    runner = make_runner(
        samples=samples,
        mode=mode,
        bins=bins,
        hist_range=hist_range,
        backend=backend,
    )
    for _ in range(max(0, warmup)):
        runner()
    measurements = [runner() for _ in range(max(1, repeat))]
    mean = statistics.fmean(measurements)
    spread = statistics.pstdev(measurements) if len(measurements) > 1 else 0.0
    return ModeSummary(mean_ms=mean, spread_ms=spread, samples_ms=measurements)


def make_runner(
    *,
    samples: np.ndarray,
    mode: ModeName,
    bins: int,
    hist_range: tuple[float, float],
    backend: HistogramKind = "dense",
):
    def _run_once() -> float:
        if mode == "numpy":
            start = time.perf_counter()
            np.histogram(samples, bins=bins, range=hist_range)
        else:
            if backend == "dense":
                hist = DenseHistogram(hist_range, bins, "Clip")
            else:
                bin_width = (hist_range[1] - hist_range[0]) / float(bins)
                hist = SparseHistogram(bin_width)
            start = time.perf_counter()
            if mode == "parallel":
                hist.update_parallel(samples)
            else:
                hist.update_sequential(samples)
            _ = hist.get()
        return (time.perf_counter() - start) * 1_000.0

    return _run_once


def chunk_iter(values: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    total = values.shape[0]
    if total <= chunk_size:
        yield values
        return
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        yield values[start:stop]


def format_row(
    size: int,
    *,
    sequential: ModeSummary,
    parallel: ModeSummary,
    numpy_runtime: ModeSummary,
) -> str:
    seq_speedup = (
        sequential.mean_ms / parallel.mean_ms if parallel.mean_ms else float("inf")
    )
    par_vs_numpy = (
        numpy_runtime.mean_ms / parallel.mean_ms if parallel.mean_ms else float("inf")
    )
    best_runtime = min(sequential.mean_ms, parallel.mean_ms)
    best_vs_numpy = numpy_runtime.mean_ms / best_runtime

    return (
        f"{size:>12,} values | seq {sequential.mean_ms:8.2f} ms ± {sequential.spread_ms:5.2f} | "
        f"par {parallel.mean_ms:8.2f} ms ± {parallel.spread_ms:5.2f} | "
        f"numpy {numpy_runtime.mean_ms:8.2f} ms ± {numpy_runtime.spread_ms:5.2f} | "
        f"speedup seq→par ×{seq_speedup:5.2f} | par vs numpy ×{par_vs_numpy:5.2f} | best vs numpy ×{best_vs_numpy:5.2f}"
    )


def plot_results(
    *,
    sizes: list[int],
    sequential: list[ModeSummary],
    parallel: list[ModeSummary],
    numpy: list[ModeSummary],
    destination: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_means = [entry.mean_ms for entry in sequential]
    par_means = [entry.mean_ms for entry in parallel]
    np_means = [entry.mean_ms for entry in numpy]

    ax.plot(sizes, seq_means, marker="o", label="Sequential (chunked)")
    ax.plot(sizes, par_means, marker="s", label="Parallel (bulk)")
    ax.plot(sizes, np_means, marker="^", label="NumPy histogram")
    ax.set_xscale("log")
    ax.set_xlabel("Values ingested")
    ax.set_ylabel("Update time (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def derive_sparse_figure_path(base: Path) -> Path:
    stem = base.stem
    suffix = base.suffix
    if "dense" in stem:
        new_stem = stem.replace("dense", "sparse", 1)
    else:
        new_stem = f"{stem}_sparse"
    return base.with_name(new_stem + suffix)


if __name__ == "__main__":
    main()
