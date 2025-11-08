use _native::core::{
    DenseHistogram, Histogram, HistogramRecorder, OutOfRangeMode, SparseHistogram,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{f64::consts::TAU, rc::Rc};

const SPARSE_SAMPLES: usize = 1_000_000;
const DENSE_SAMPLES: usize = 1_500_000;
const RECORDER_BATCH_ONE: usize = 800_000;
const RECORDER_BATCH_TWO: usize = 400_000;
const DENSE_RANGE: (f64, f64) = (-500.0, 500.0);
const DENSE_BINS: usize = 1_000_000;
const DEFAULT_MIXTURE: &[(f64, f64)] = &[(-50.0, 10.0), (75.0, 20.0)];
const RNG_SEED: u64 = 0xfeed_cafe_d00d_beef;

fn gaussian_sample(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-12);
    let u2 = rng.random::<f64>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = TAU * u2;
    radius * theta.cos()
}

fn sample_gaussian_mixture(len: usize, components: &[(f64, f64)], seed: u64) -> Vec<f64> {
    assert!(
        !components.is_empty(),
        "mixture must have at least one component"
    );
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| {
            let choice = (rng.random::<f64>() * components.len() as f64) as usize;
            let (mu, sigma) = components[choice.min(components.len() - 1)];
            let z = gaussian_sample(&mut rng);
            mu + sigma * z
        })
        .collect()
}

fn sample_default(len: usize, salt: u64) -> Vec<f64> {
    sample_gaussian_mixture(len, DEFAULT_MIXTURE, RNG_SEED ^ salt)
}

fn bench_sparse_histogram(c: &mut Criterion) {
    let update_values = sample_default(SPARSE_SAMPLES, 0);
    c.bench_function("sparse_update", |b| {
        b.iter(|| {
            let mut hist = SparseHistogram::new(0.25).unwrap();
            hist.update(update_values.iter().copied()).unwrap();
            black_box(hist)
        });
    });

    let mut populated = SparseHistogram::new(0.25).unwrap();
    populated.update(update_values.iter().copied()).unwrap();
    c.bench_function("sparse_get", |b| {
        b.iter(|| black_box(populated.get()));
    });
}

fn bench_dense_histogram(c: &mut Criterion) {
    let update_values = sample_default(DENSE_SAMPLES, 0x1);
    c.bench_function("dense_update_clip", |b| {
        b.iter(|| {
            let mut hist =
                DenseHistogram::new(DENSE_RANGE, DENSE_BINS, OutOfRangeMode::Clip).unwrap();
            hist.update(update_values.iter().copied()).unwrap();
            black_box(hist)
        });
    });

    let mut populated = DenseHistogram::new(DENSE_RANGE, DENSE_BINS, OutOfRangeMode::Clip).unwrap();
    populated.update(update_values.iter().copied()).unwrap();
    c.bench_function("dense_get", |b| {
        b.iter(|| black_box(populated.get()));
    });
}

fn bench_histogram_recorder(c: &mut Criterion) {
    let batch_one = sample_default(RECORDER_BATCH_ONE, 0x3);
    let batch_two = sample_default(RECORDER_BATCH_TWO, 0x4);

    c.bench_function("recorder_update_snapshot", |b| {
        b.iter(|| {
            let mut recorder = HistogramRecorder::from_sparse(0.25, Some(64)).unwrap();
            recorder.update(batch_one.iter().copied()).unwrap();
            black_box(recorder.snapshot(Some(String::from("bench"))));
        });
    });

    let prepared_recorder = {
        let mut recorder = HistogramRecorder::from_sparse(0.25, Some(64)).unwrap();
        recorder.update(batch_one.iter().copied()).unwrap();
        recorder.snapshot(Some(String::from("baseline")));
        recorder.update(batch_two.iter().copied()).unwrap();
        recorder.snapshot(None);
        Rc::new(recorder)
    };

    c.bench_function("recorder_diff", {
        let recorder = prepared_recorder.clone();
        move |b| {
            let recorder = recorder.clone();
            b.iter(move || black_box(recorder.diff(1, 0).unwrap()));
        }
    });
}

criterion_group!(
    histogram_benches,
    bench_sparse_histogram,
    bench_dense_histogram,
    bench_histogram_recorder
);
criterion_main!(histogram_benches);
