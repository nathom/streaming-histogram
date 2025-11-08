use ahash::RandomState;
use std::collections::{HashMap, VecDeque};
use std::fmt;

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

#[derive(Clone, Copy)]
pub enum OutOfRangeMode {
    Clip,
    Ignore,
    Error,
}

pub trait Histogram {
    fn ingest_value(&mut self, value: f64) -> Result<()>;
    fn get(&self) -> Vec<((f64, f64), u64)>;

    fn update<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = f64>,
    {
        for value in values {
            self.ingest_value(value)?;
        }
        Ok(())
    }
}

pub struct SparseHistogram {
    bin_width: f64,
    bins: HashMap<i64, u64, RandomState>,
    neg_inf_bucket: u64,
    pos_inf_bucket: u64,
    nan_bucket: u64,
}

impl SparseHistogram {
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
}

impl Histogram for SparseHistogram {
    fn ingest_value(&mut self, value: f64) -> Result<()> {
        self.ingest_raw(value);
        Ok(())
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

pub struct DenseHistogram {
    start: f64,
    end: f64,
    bin_width: f64,
    bins: Vec<u64>,
    out_of_range: OutOfRangeMode,
}

impl DenseHistogram {
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
}

impl Histogram for DenseHistogram {
    fn ingest_value(&mut self, value: f64) -> Result<()> {
        self.ingest_single(value)
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

#[derive(Clone)]
struct SnapshotEntry {
    index: u64,
    label: Option<String>,
    bins: Vec<((f64, f64), u64)>,
}

impl SnapshotEntry {
    fn new(index: u64, label: Option<String>, bins: Vec<((f64, f64), u64)>) -> Self {
        Self { index, label, bins }
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
            bins: entry.bins.clone(),
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
}

pub struct HistogramRecorder {
    inner: RecordedHistogram,
    snapshots: VecDeque<SnapshotEntry>,
    next_snapshot_index: u64,
    max_snapshots: Option<usize>,
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

        let mut earlier_map = HashMap::with_hasher(RandomState::new());
        for ((start, end), count) in &earlier.bins {
            earlier_map.insert(BucketKey::from_bounds(*start, *end), *count);
        }

        let mut later_map = HashMap::with_hasher(RandomState::new());
        let mut diff_bins = Vec::new();
        for ((start, end), count) in &later.bins {
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

    pub fn update<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = f64>,
    {
        Histogram::update(self, values)
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
}
