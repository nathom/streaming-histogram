use ahash::RandomState;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;

const SERIALIZATION_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct HistogramError {
    message: String,
}

pub type Result<T> = std::result::Result<T, HistogramError>;

impl HistogramError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for HistogramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for HistogramError {}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum OutOfRangeMode {
    Clip,
    Ignore,
    Error,
}

pub trait Histogram {
    fn ingest_value(&mut self, value: f64) -> Result<()>;
    fn get(&self) -> Vec<((f64, f64), u64)>;
    fn update_sequential(&mut self, values: &[f64]) -> Result<()>;
    fn update_parallel(&mut self, values: &[f64]) -> Result<()>;

    fn parallel_threshold(&self) -> usize {
        usize::MAX
    }

    fn update(&mut self, values: &[f64]) -> Result<()> {
        if values.len() >= self.parallel_threshold() {
            self.update_parallel(values)
        } else {
            self.update_sequential(values)
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SparseHistogram {
    bin_width: f64,
    bins: HashMap<i64, u64, RandomState>,
    neg_inf_bucket: u64,
    pos_inf_bucket: u64,
    nan_bucket: u64,
}

impl SparseHistogram {
    const PARALLEL_MIN_VALUES: usize = 2_000_000;
    const PARALLEL_MIN_CHUNK_SIZE: usize = 4_096;

    pub fn new(bin_width: f64) -> Result<Self> {
        if !bin_width.is_finite() || bin_width <= 0.0 {
            return Err(HistogramError::new(
                "bin_width must be a positive, finite float",
            ));
        }

        Ok(Self {
            bin_width,
            bins: HashMap::with_hasher(RandomState::default()),
            neg_inf_bucket: 0,
            pos_inf_bucket: 0,
            nan_bucket: 0,
        })
    }

    fn ingest_raw(&mut self, value: f64) {
        if value.is_nan() {
            self.nan_bucket += 1;
            return;
        }
        if value.is_infinite() {
            if value.is_sign_negative() {
                self.neg_inf_bucket += 1;
            } else {
                self.pos_inf_bucket += 1;
            }
            return;
        }

        let bin_idx = (value / self.bin_width).floor() as i64;
        *self.bins.entry(bin_idx).or_insert(0) += 1;
    }

    fn parallel_chunk_size(total_values: usize) -> usize {
        let threads = rayon::current_num_threads().max(1);
        Self::PARALLEL_MIN_CHUNK_SIZE
            .max(total_values / threads)
            .max(1)
    }

    pub fn update_sequential(&mut self, values: &[f64]) -> Result<()> {
        for &value in values {
            self.ingest_raw(value);
        }
        Ok(())
    }

    pub fn update_parallel(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let bin_width = self.bin_width;
        let chunk_size = Self::parallel_chunk_size(values.len());

        let aggregated = values
            .par_iter()
            .with_min_len(chunk_size)
            .fold(
                || {
                    (
                        HashMap::<i64, u64, RandomState>::with_hasher(RandomState::default()),
                        0u64,
                        0u64,
                        0u64,
                    )
                },
                |mut acc, &value| {
                    let (bins, neg_inf, pos_inf, nan) = &mut acc;
                    if value.is_nan() {
                        *nan += 1;
                    } else if value.is_infinite() {
                        if value.is_sign_negative() {
                            *neg_inf += 1;
                        } else {
                            *pos_inf += 1;
                        }
                    } else {
                        let bin_idx = (value / bin_width).floor() as i64;
                        *bins.entry(bin_idx).or_insert(0) += 1;
                    }
                    acc
                },
            )
            .reduce(
                || {
                    (
                        HashMap::<i64, u64, RandomState>::with_hasher(RandomState::default()),
                        0u64,
                        0u64,
                        0u64,
                    )
                },
                |mut left, right| {
                    let (left_bins, left_neg_inf, left_pos_inf, left_nan) = &mut left;
                    let (right_bins, right_neg_inf, right_pos_inf, right_nan) = right;
                    for (idx, count) in right_bins {
                        *left_bins.entry(idx).or_insert(0) += count;
                    }
                    *left_neg_inf += right_neg_inf;
                    *left_pos_inf += right_pos_inf;
                    *left_nan += right_nan;
                    left
                },
            );

        let (bins, neg_inf, pos_inf, nan) = aggregated;

        for (idx, count) in bins {
            *self.bins.entry(idx).or_insert(0) += count;
        }
        self.neg_inf_bucket += neg_inf;
        self.pos_inf_bucket += pos_inf;
        self.nan_bucket += nan;

        Ok(())
    }
}

impl Histogram for SparseHistogram {
    fn ingest_value(&mut self, value: f64) -> Result<()> {
        self.ingest_raw(value);
        Ok(())
    }

    fn update_sequential(&mut self, values: &[f64]) -> Result<()> {
        Self::update_sequential(self, values)
    }

    fn update_parallel(&mut self, values: &[f64]) -> Result<()> {
        Self::update_parallel(self, values)
    }

    fn parallel_threshold(&self) -> usize {
        Self::PARALLEL_MIN_VALUES
    }

    fn get(&self) -> Vec<((f64, f64), u64)> {
        let mut entries: Vec<_> = self.bins.iter().collect();
        entries.sort_unstable_by_key(|(idx, _)| *idx);

        let mut result = Vec::with_capacity(entries.len() + 3);
        for (idx, count) in entries {
            let start = *idx as f64 * self.bin_width;
            let end = start + self.bin_width;
            result.push(((start, end), *count));
        }

        if self.neg_inf_bucket > 0 {
            result.push(((f64::NEG_INFINITY, f64::NEG_INFINITY), self.neg_inf_bucket));
        }
        if self.pos_inf_bucket > 0 {
            result.push(((f64::INFINITY, f64::INFINITY), self.pos_inf_bucket));
        }
        if self.nan_bucket > 0 {
            result.push(((f64::NAN, f64::NAN), self.nan_bucket));
        }

        result
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DenseHistogram {
    start: f64,
    end: f64,
    bin_width: f64,
    bins: Vec<u64>,
    out_of_range: OutOfRangeMode,
}

impl DenseHistogram {
    const PARALLEL_MIN_VALUES: usize = 1_000_000;
    const PARALLEL_MIN_CHUNK_SIZE: usize = 1_024;

    pub fn new(range: (f64, f64), num_bins: usize, out_of_range: OutOfRangeMode) -> Result<Self> {
        let (start, end) = range;
        if !start.is_finite() || !end.is_finite() {
            return Err(HistogramError::new("range bounds must be finite numbers"));
        }
        if !(start < end) {
            return Err(HistogramError::new("range must satisfy start < end"));
        }
        if num_bins == 0 {
            return Err(HistogramError::new("num_bins must be positive"));
        }

        let bin_width = (end - start) / num_bins as f64;
        Ok(Self {
            start,
            end,
            bin_width,
            bins: vec![0; num_bins],
            out_of_range,
        })
    }

    fn ingest_single(&mut self, value: f64) -> Result<()> {
        if value.is_nan() {
            return match self.out_of_range {
                OutOfRangeMode::Error => Err(HistogramError::new("NaN values are not allowed")),
                _ => Ok(()),
            };
        }

        if value < self.start {
            return self.handle_below();
        }
        if value >= self.end {
            return self.handle_above();
        }

        self.add_bin_for(value);
        Ok(())
    }

    fn handle_below(&mut self) -> Result<()> {
        match self.out_of_range {
            OutOfRangeMode::Clip => {
                self.bins[0] += 1;
                Ok(())
            }
            OutOfRangeMode::Ignore => Ok(()),
            OutOfRangeMode::Error => Err(HistogramError::new("value fell below histogram range")),
        }
    }

    fn handle_above(&mut self) -> Result<()> {
        match self.out_of_range {
            OutOfRangeMode::Clip => {
                let last = self.bins.len() - 1;
                self.bins[last] += 1;
                Ok(())
            }
            OutOfRangeMode::Ignore => Ok(()),
            OutOfRangeMode::Error => Err(HistogramError::new("value exceeded histogram range")),
        }
    }

    fn add_bin_for(&mut self, value: f64) {
        let idx = ((value - self.start) / self.bin_width).floor() as usize;
        let idx = idx.min(self.bins.len() - 1);
        self.bins[idx] += 1;
    }

    fn index_for_parallel(&self, value: f64) -> Result<Option<usize>> {
        if value.is_nan() {
            return match self.out_of_range {
                OutOfRangeMode::Error => Err(HistogramError::new("NaN values are not allowed")),
                _ => Ok(None),
            };
        }

        if value < self.start {
            return match self.out_of_range {
                OutOfRangeMode::Clip => Ok(Some(0)),
                OutOfRangeMode::Ignore => Ok(None),
                OutOfRangeMode::Error => {
                    Err(HistogramError::new("value fell below histogram range"))
                }
            };
        }
        if value >= self.end {
            return match self.out_of_range {
                OutOfRangeMode::Clip => Ok(Some(self.bins.len() - 1)),
                OutOfRangeMode::Ignore => Ok(None),
                OutOfRangeMode::Error => Err(HistogramError::new("value exceeded histogram range")),
            };
        }

        let idx = ((value - self.start) / self.bin_width).floor() as usize;
        Ok(Some(idx.min(self.bins.len() - 1)))
    }
    pub fn update_sequential(&mut self, values: &[f64]) -> Result<()> {
        for &v in values {
            self.ingest_value(v)?;
        }
        Ok(())
    }

    pub fn update_parallel(&mut self, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let bins_len = self.bins.len();
        let threads = rayon::current_num_threads().max(1);
        let chunk_size = Self::PARALLEL_MIN_CHUNK_SIZE
            .max(values.len() / threads)
            .max(1);

        let aggregated = values
            .par_iter()
            .with_min_len(chunk_size)
            .fold(
                || Ok(vec![0u64; bins_len]),
                |acc_res, &value| {
                    let mut acc = acc_res?;
                    if let Some(bin_idx) = self.index_for_parallel(value)? {
                        acc[bin_idx] += 1;
                    }
                    Ok(acc)
                },
            )
            .reduce(
                || Ok(vec![0u64; bins_len]),
                |left_res, right_res| match (left_res, right_res) {
                    (Ok(mut left), Ok(right)) => {
                        for (l, r) in left.iter_mut().zip(right) {
                            *l += r;
                        }
                        Ok(left)
                    }
                    (Err(err), _) => Err(err),
                    (_, Err(err)) => Err(err),
                },
            )?;

        for (bin, increment) in self.bins.iter_mut().zip(aggregated) {
            *bin += increment;
        }

        Ok(())
    }
}

impl Histogram for DenseHistogram {
    fn ingest_value(&mut self, value: f64) -> Result<()> {
        self.ingest_single(value)
    }

    fn update_sequential(&mut self, values: &[f64]) -> Result<()> {
        Self::update_sequential(self, values)
    }

    fn update_parallel(&mut self, values: &[f64]) -> Result<()> {
        Self::update_parallel(self, values)
    }

    fn parallel_threshold(&self) -> usize {
        Self::PARALLEL_MIN_VALUES
    }

    fn get(&self) -> Vec<((f64, f64), u64)> {
        self.bins
            .iter()
            .enumerate()
            .map(|(idx, count)| {
                let start = self.start + idx as f64 * self.bin_width;
                let end = start + self.bin_width;
                ((start, end), *count)
            })
            .collect()
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct SnapshotEntry {
    index: u64,
    label: Option<String>,
    bins: Vec<StoredBucket>,
}

impl SnapshotEntry {
    fn new(index: u64, label: Option<String>, bins: Vec<((f64, f64), u64)>) -> Self {
        Self {
            index,
            label,
            bins: bins.into_iter().map(StoredBucket::from).collect(),
        }
    }

    fn bins(&self) -> Vec<((f64, f64), u64)> {
        self.bins.iter().map(|bucket| bucket.into()).collect()
    }
}

#[derive(Clone)]
pub struct HistogramSnapshot {
    index: u64,
    label: Option<String>,
    bins: Vec<((f64, f64), u64)>,
}

impl HistogramSnapshot {
    fn from_entry(entry: &SnapshotEntry) -> Self {
        Self {
            index: entry.index,
            label: entry.label.clone(),
            bins: entry.bins(),
        }
    }

    pub fn index(&self) -> u64 {
        self.index
    }

    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }

    pub fn bins(&self) -> Vec<((f64, f64), u64)> {
        self.bins.clone()
    }

    pub fn total(&self) -> u64 {
        self.bins.iter().map(|(_, count)| *count).sum()
    }
}

#[derive(Clone)]
pub struct SnapshotDiff {
    start_index: u64,
    end_index: u64,
    start_label: Option<String>,
    end_label: Option<String>,
    bins: Vec<((f64, f64), u64)>,
}

#[derive(Clone, Serialize, Deserialize)]
struct StoredBucket {
    start: BucketBound,
    end: BucketBound,
    count: u64,
}

impl From<((f64, f64), u64)> for StoredBucket {
    fn from(((start, end), count): ((f64, f64), u64)) -> Self {
        Self {
            start: BucketBound::from(start),
            end: BucketBound::from(end),
            count,
        }
    }
}

impl From<StoredBucket> for ((f64, f64), u64) {
    fn from(bucket: StoredBucket) -> Self {
        (
            (f64::from(bucket.start), f64::from(bucket.end)),
            bucket.count,
        )
    }
}

impl From<&StoredBucket> for ((f64, f64), u64) {
    fn from(bucket: &StoredBucket) -> Self {
        (
            (f64::from(bucket.start), f64::from(bucket.end)),
            bucket.count,
        )
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum BucketBound {
    Finite(f64),
    NegInfinity,
    PosInfinity,
    Nan,
}

impl From<f64> for BucketBound {
    fn from(value: f64) -> Self {
        if value.is_nan() {
            Self::Nan
        } else if value.is_infinite() {
            if value.is_sign_negative() {
                Self::NegInfinity
            } else {
                Self::PosInfinity
            }
        } else {
            Self::Finite(value)
        }
    }
}

impl From<BucketBound> for f64 {
    fn from(bound: BucketBound) -> Self {
        match bound {
            BucketBound::Finite(value) => value,
            BucketBound::NegInfinity => f64::NEG_INFINITY,
            BucketBound::PosInfinity => f64::INFINITY,
            BucketBound::Nan => f64::NAN,
        }
    }
}

impl SnapshotDiff {
    fn new(
        start_index: u64,
        end_index: u64,
        start_label: Option<String>,
        end_label: Option<String>,
        bins: Vec<((f64, f64), u64)>,
    ) -> Self {
        Self {
            start_index,
            end_index,
            start_label,
            end_label,
            bins,
        }
    }

    pub fn start_index(&self) -> u64 {
        self.start_index
    }

    pub fn end_index(&self) -> u64 {
        self.end_index
    }

    pub fn start_label(&self) -> Option<String> {
        self.start_label.clone()
    }

    pub fn end_label(&self) -> Option<String> {
        self.end_label.clone()
    }

    pub fn bins(&self) -> Vec<((f64, f64), u64)> {
        self.bins.clone()
    }

    pub fn total(&self) -> u64 {
        self.bins.iter().map(|(_, count)| *count).sum()
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "state", rename_all = "lowercase")]
enum RecordedHistogram {
    Sparse(SparseHistogram),
    Dense(DenseHistogram),
}

impl RecordedHistogram {
    fn bins(&self) -> Vec<((f64, f64), u64)> {
        match self {
            Self::Sparse(hist) => <SparseHistogram as Histogram>::get(hist),
            Self::Dense(hist) => <DenseHistogram as Histogram>::get(hist),
        }
    }

    fn ingest_value(&mut self, value: f64) -> Result<()> {
        match self {
            Self::Sparse(hist) => <SparseHistogram as Histogram>::ingest_value(hist, value),
            Self::Dense(hist) => <DenseHistogram as Histogram>::ingest_value(hist, value),
        }
    }

    fn update(&mut self, values: &[f64]) -> Result<()> {
        match self {
            Self::Sparse(hist) => <SparseHistogram as Histogram>::update(hist, values),
            Self::Dense(hist) => <DenseHistogram as Histogram>::update(hist, values),
        }
    }

    fn update_sequential(&mut self, values: &[f64]) -> Result<()> {
        match self {
            Self::Sparse(hist) => hist.update_sequential(values),
            Self::Dense(hist) => hist.update_sequential(values),
        }
    }

    fn update_parallel(&mut self, values: &[f64]) -> Result<()> {
        match self {
            Self::Sparse(hist) => hist.update_parallel(values),
            Self::Dense(hist) => hist.update_parallel(values),
        }
    }

    fn parallel_threshold(&self) -> usize {
        match self {
            Self::Sparse(hist) => <SparseHistogram as Histogram>::parallel_threshold(hist),
            Self::Dense(hist) => <DenseHistogram as Histogram>::parallel_threshold(hist),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HistogramRecorder {
    inner: RecordedHistogram,
    snapshots: VecDeque<SnapshotEntry>,
    next_snapshot_index: u64,
    max_snapshots: Option<usize>,
}

#[derive(Serialize)]
struct RecorderStateRef<'a> {
    version: u32,
    recorder: &'a HistogramRecorder,
}

#[derive(Deserialize)]
struct RecorderStateOwned {
    version: u32,
    recorder: HistogramRecorder,
}

impl HistogramRecorder {
    fn from_inner(inner: RecordedHistogram, max_snapshots: Option<usize>) -> Result<Self> {
        if let Some(limit) = max_snapshots {
            if limit == 0 {
                return Err(HistogramError::new("max_snapshots must be positive"));
            }
        }

        Ok(Self {
            inner,
            snapshots: VecDeque::new(),
            next_snapshot_index: 0,
            max_snapshots,
        })
    }

    pub fn from_sparse(bin_width: f64, max_snapshots: Option<usize>) -> Result<Self> {
        Self::from_inner(
            RecordedHistogram::Sparse(SparseHistogram::new(bin_width)?),
            max_snapshots,
        )
    }

    pub fn from_dense(
        range: (f64, f64),
        num_bins: usize,
        out_of_range: OutOfRangeMode,
        max_snapshots: Option<usize>,
    ) -> Result<Self> {
        Self::from_inner(
            RecordedHistogram::Dense(DenseHistogram::new(range, num_bins, out_of_range)?),
            max_snapshots,
        )
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&RecorderStateRef {
            version: SERIALIZATION_VERSION,
            recorder: self,
        })
        .map_err(|err| HistogramError::new(format!("failed to serialize histogram: {err}")))
    }

    pub fn from_json(payload: &str) -> Result<Self> {
        let state: RecorderStateOwned = serde_json::from_str(payload).map_err(|err| {
            HistogramError::new(format!("failed to deserialize histogram: {err}"))
        })?;

        if state.version != SERIALIZATION_VERSION {
            return Err(HistogramError::new(format!(
                "unsupported histogram serialization version {}",
                state.version
            )));
        }

        let mut recorder = state.recorder;
        if let Some(limit) = recorder.max_snapshots {
            if limit == 0 {
                return Err(HistogramError::new("max_snapshots must be positive"));
            }
        }
        recorder.enforce_max_snapshots();
        Ok(recorder)
    }

    fn first_snapshot_id(&self) -> u64 {
        self.next_snapshot_index
            .saturating_sub(self.snapshots.len() as u64)
    }

    fn enforce_max_snapshots(&mut self) {
        if let Some(limit) = self.max_snapshots {
            while self.snapshots.len() > limit {
                self.snapshots.pop_front();
            }
        }
    }

    fn drain_before(&mut self, end_index: u64) -> usize {
        let mut removed = 0;
        while let Some(front) = self.snapshots.front() {
            if front.index < end_index {
                self.snapshots.pop_front();
                removed += 1;
            } else {
                break;
            }
        }
        removed
    }

    fn record_snapshot(&mut self, label: Option<String>) -> HistogramSnapshot {
        let bins = Histogram::get(self);
        let entry = SnapshotEntry::new(self.next_snapshot_index, label, bins);
        self.next_snapshot_index += 1;
        self.snapshots.push_back(entry.clone());
        self.enforce_max_snapshots();
        HistogramSnapshot::from_entry(&entry)
    }

    fn find_snapshot(&self, index: u64) -> Result<&SnapshotEntry> {
        if index >= self.next_snapshot_index {
            return Err(HistogramError::new(format!(
                "snapshot index {index} not found"
            )));
        }

        let first_id = self.first_snapshot_id();
        if index < first_id {
            return Err(HistogramError::new(format!(
                "snapshot index {index} not found"
            )));
        }

        let offset = (index - first_id) as usize;
        self.snapshots
            .get(offset)
            .ok_or_else(|| HistogramError::new(format!("snapshot index {index} not found")))
    }

    fn subtract_snapshots(
        &self,
        later: &SnapshotEntry,
        earlier: &SnapshotEntry,
    ) -> Result<SnapshotDiff> {
        if later.index <= earlier.index {
            return Err(HistogramError::new(
                "later snapshot index must be greater than earlier snapshot index",
            ));
        }

        #[derive(Hash, Eq, PartialEq, Clone)]
        enum BucketKey {
            Range(u64, u64),
            NegInf,
            PosInf,
            Nan,
        }

        impl BucketKey {
            fn from_bounds(start: f64, end: f64) -> Self {
                if start.is_nan() && end.is_nan() {
                    Self::Nan
                } else if start.is_infinite() && end.is_infinite() {
                    if start.is_sign_negative() {
                        Self::NegInf
                    } else {
                        Self::PosInf
                    }
                } else {
                    Self::Range(start.to_bits(), end.to_bits())
                }
            }
        }

        let earlier_bins = earlier.bins();
        let mut earlier_map = HashMap::with_hasher(RandomState::new());
        for ((start, end), count) in &earlier_bins {
            earlier_map.insert(BucketKey::from_bounds(*start, *end), *count);
        }

        let later_bins = later.bins();
        let mut later_map = HashMap::with_hasher(RandomState::new());
        let mut diff_bins = Vec::new();
        for ((start, end), count) in &later_bins {
            let key = BucketKey::from_bounds(*start, *end);
            later_map.insert(key.clone(), *count);
            let previous = earlier_map.get(&key).copied().unwrap_or(0);
            if *count < previous {
                return Err(HistogramError::new(
                    "later snapshot contains fewer samples than earlier snapshot",
                ));
            }
            let delta = *count - previous;
            if delta > 0 {
                diff_bins.push(((*start, *end), delta));
            }
        }

        for (key, earlier_count) in &earlier_map {
            if *earlier_count == 0 {
                continue;
            }
            let later_value = later_map.get(key).copied().unwrap_or(0);
            if later_value < *earlier_count {
                return Err(HistogramError::new(
                    "earlier snapshot has buckets missing from later snapshot",
                ));
            }
        }

        Ok(SnapshotDiff::new(
            earlier.index,
            later.index,
            earlier.label.clone(),
            later.label.clone(),
            diff_bins,
        ))
    }

    pub fn update(&mut self, values: &[f64]) -> Result<()> {
        self.inner.update(values)
    }

    pub fn get(&self) -> Vec<((f64, f64), u64)> {
        Histogram::get(self)
    }

    pub fn snapshot(&mut self, label: Option<String>) -> HistogramSnapshot {
        self.record_snapshot(label)
    }

    pub fn latest(&self) -> Option<HistogramSnapshot> {
        self.snapshots.back().map(HistogramSnapshot::from_entry)
    }

    pub fn get_snapshot(&self, index: u64) -> Result<HistogramSnapshot> {
        Ok(HistogramSnapshot::from_entry(self.find_snapshot(index)?))
    }

    pub fn diff(&self, later_index: u64, earlier_index: u64) -> Result<SnapshotDiff> {
        let later = self.find_snapshot(later_index)?;
        let earlier = self.find_snapshot(earlier_index)?;
        self.subtract_snapshots(later, earlier)
    }

    pub fn drain_snapshots(&mut self, end_index: u64) -> usize {
        self.drain_before(end_index)
    }

    pub fn clear_snapshots(&mut self) {
        self.snapshots.clear();
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }
}

impl Histogram for HistogramRecorder {
    fn ingest_value(&mut self, value: f64) -> Result<()> {
        self.inner.ingest_value(value)
    }

    fn get(&self) -> Vec<((f64, f64), u64)> {
        self.inner.bins()
    }

    fn update_sequential(&mut self, values: &[f64]) -> Result<()> {
        self.inner.update_sequential(values)
    }

    fn update_parallel(&mut self, values: &[f64]) -> Result<()> {
        self.inner.update_parallel(values)
    }

    fn parallel_threshold(&self) -> usize {
        self.inner.parallel_threshold()
    }
}
