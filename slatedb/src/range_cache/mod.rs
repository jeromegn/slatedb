//! Local key-range cache for point reads and ascending scans.
//!
//! A *cached range* covers a half-open key interval `[start, end)` completely, so point reads in
//! it are definite hits or definite misses and scans need no other source. Each range has one
//! layer set, newest first: an active write buffer, sealed buffers awaiting flush, up to
//! [`RangeCacheTuning::max_runs`] run files (indexed by one sorted delta view for scans and
//! Bloom filters for point reads), and a base file holding live rows at `base_seq`.
//!
//! Only the batch writer mutates the range index and layer sets; loads, flushes, merges, splits,
//! and removals apply at writer boundaries. A range's base is a sorted list of [`Chunks`] tiling
//! its span, each holding live rows at its own `base_seq`. Ranges larger than twice
//! [`RangeCacheTuning::range_target_bytes`] split without copying files (see `split`). Reads pin
//! layers without locking and verify `seq >= chunk.base_seq` for each touched chunk and
//! `seq <= layers.active.sealed_at`. Read and fallback counts accumulate per thread and fold
//! periodically into shared metrics and range hits.

mod auto;
mod buffer;
mod delta;
mod file;
mod filter;
mod hits;
mod maintain;
mod merge;
mod mixed;
mod scan;
mod split;
pub mod stats;
mod view;
mod writer;

use std::collections::BTreeMap;
use std::io;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::ops::RangeBounds;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use bytes::Bytes;
use fail_parallel::{fail_point, FailPointRegistry};
use log::warn;
use parking_lot::Mutex;
use rand::RngExt;
use slatedb_common::clock::SystemClock;
use slatedb_common::metrics::MetricsRecorderHelper;
use slatedb_common::DbRand;

use crate::bytes_range::{ByteRangeBounds, BytesRange};
use crate::config::{DurabilityLevel, ReadOptions, ScanOptions};
use crate::db::{Db, DbInner};
use crate::db_iter::DbIterator;
use crate::db_snapshot::DbSnapshot;
use crate::error::SlateDBError;
use crate::iter::{EmptyIterator, IterationOrder, RowEntryIterator};
use crate::merge_operator::MergeOperatorType;
use crate::oracle::Oracle;
use crate::read_trace::ReadTrace;
use crate::reader::ScanContext;
use crate::types::{KeyValue, ValueDeletable};

pub use auto::AutoPromoteOptions;
use auto::AutoSettings;
pub(crate) use auto::{Promoter, PromoterMessage, RANGE_CACHE_PROMOTER_TASK_NAME};
use buffer::Kind;
use delta::{Delta, DeltaIterator, DeltaMemory, DeltaShape};
pub use file::RangeCachePageSource;
use file::{
    FileBudget, FileRead, RangeFile, RangeFileCursor, RangeFileIterator, RangeFileWriter,
    FILE_EXTENSION, TEMP_FILE_EXTENSION,
};
use filter::{hash_key, Filter};
use hits::{HitStats, RangeHits};
pub(crate) use maintain::{Maintainer, RANGE_CACHE_MAINTENANCE_TASK_NAME};
use mixed::{GapSource, MixedScan, Part, PartKind};
use scan::CachedScan;
use stats::{
    AutoStats, Fallback, FallbackOp, PinOp, RangeCacheStats, ReadCount, RemoveReason, Unfit,
};
use view::{DeltaView, ViewMemory};
use writer::{Padded, Target, WriterQueue};

/// Max attempts to pin a range before falling back to the normal read path.
const PIN_ATTEMPTS: usize = 3;

/// Valid range for [`RangeCacheTuning::max_runs`] (one selector slot per run in a delta view).
pub(crate) const MAX_RUNS_RANGE: std::ops::RangeInclusive<usize> = 4..=view::VIEW_SLOTS;

/// Options for the range cache. See [`crate::DbBuilder::with_range_cache`].
///
/// The range cache is experimental and its API may change. Build options with
/// [`RangeCacheOptions::new`] and override fields as needed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RangeCacheOptions {
    /// Directory owned by the range cache. Stale cache files are removed on open.
    pub root_folder: PathBuf,
    /// Max disk bytes of range cache files. When exceeded, least recently read ranges are evicted.
    /// Loads larger than this budget are rejected.
    pub max_disk_bytes: u64,
    /// How the cache reads range cache files.
    pub page_source: RangeCachePageSource,
    /// Automatic promotion settings. `None` disables automatic promotion.
    pub auto_promote: Option<AutoPromoteOptions>,
    /// Internal tuning parameters. See [`RangeCacheTuning`].
    #[doc(hidden)]
    pub tuning: RangeCacheTuning,
}

impl RangeCacheOptions {
    /// Returns options with default values and the given directory.
    pub fn new(root_folder: impl Into<PathBuf>) -> Self {
        Self {
            root_folder: root_folder.into(),
            max_disk_bytes: 1024 * 1024 * 1024,
            page_source: RangeCachePageSource::default(),
            auto_promote: None,
            tuning: RangeCacheTuning::default(),
        }
    }
}

/// Internal tuning for the range cache, intended for tests and benchmarks.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RangeCacheTuning {
    /// File page size in bytes. Must be a power of two from 512 to 65536.
    pub page_size: usize,
    /// Minimum run bytes required before a major merge starts. `usize::MAX` disables major merges.
    pub min_rebuild_delta_bytes: usize,
    /// Starts a major merge when run bytes reach `major_merge_factor * base_bytes` (and
    /// `min_rebuild_delta_bytes`). Must be positive and finite.
    pub major_merge_factor: f64,
    /// Max run files per range, from 4 to 64. Flushes at the cap merge newest runs to stay within
    /// this limit.
    pub max_runs: usize,
    /// Chunk bytes at which the writer seals an active write buffer for flushing.
    pub range_seal_bytes: usize,
    /// Hard memory cap across all write buffers (chunks and filters). Writes exceeding this cap
    /// evict ranges instead of blocking.
    pub max_recent_bytes: u64,
    /// Minimum active buffer size eligible for early sealing under memory pressure.
    pub min_seal_bytes: usize,
    /// Chunk bytes threshold at which a write buffer allocates a Bloom filter. `usize::MAX`
    /// disables buffer filters.
    pub buffer_filter_bytes: usize,
    /// Target size per cached range. `cache_range` cuts larger requests at sorted-run block
    /// boundaries, bases exceeding twice this size split without copying, and adjacent quiet
    /// ranges below an eighth of this size merge without copying. `0` disables splits and quiet
    /// merges.
    pub range_target_bytes: u64,
    /// Heap metadata budget for fence indexes, filters, delta views, and layer sets. `None` uses
    /// `max(max_disk_bytes / 16, 64 MiB)`.
    pub max_meta_bytes: Option<u64>,
    /// Interval between promoter candidate evaluations.
    pub evaluate_interval: Duration,
    /// Max candidates formed or re-evaluated per evaluation pass.
    pub max_candidates_per_evaluation: usize,
    /// Sampled reads required to admit a candidate when the disk budget is at most half full.
    pub min_evidence: u32,
    /// Sampled reads required to admit a candidate when the disk budget is full.
    pub full_evidence: u32,
    /// Minimum time span across a candidate's samples to reject single-pass scans.
    pub min_reuse_span: Duration,
    /// Required read-density multiple over automatic eviction victims.
    pub admission_margin: f64,
    /// Retention duration for ghosts of recently removed automatic ranges.
    pub ghost_ttl: Duration,
    /// Retention duration for rejected candidates.
    pub candidate_ttl: Duration,
    /// Minimum wait before re-evaluating a remembered rejected candidate.
    pub candidate_recheck: Duration,
    /// Initial exponential backoff wait after a failed automatic promotion.
    pub backoff_base: Duration,
    /// Total sample capacity across the windowed reservoir.
    pub reservoir_capacity: usize,
    /// Capacity of the sample channel from reads to the promoter.
    pub queue_capacity: usize,
    /// Floor on per-sample read cost.
    pub min_read_cost: Duration,
    /// Estimated base rewrite cost per byte in nanoseconds for payback checks.
    pub rebuild_cost_ns_per_byte: f64,
    /// Target read-ahead size for object store fetches during range loads.
    pub load_fetch_bytes: usize,
    /// Max concurrent fetch tasks per SST during range loads.
    pub load_fetch_tasks: usize,
    /// Read events per thread before folding counters into shared metrics and range hits. `0`
    /// counts as `1`.
    pub stats_fold_events: u32,
    /// Elapsed time before a reading thread folds its counters into shared state.
    pub stats_fold_interval: Duration,
}

impl Default for RangeCacheTuning {
    fn default() -> Self {
        Self {
            page_size: 4096,
            min_rebuild_delta_bytes: 1024 * 1024,
            major_merge_factor: 0.25,
            max_runs: 64,
            range_seal_bytes: 4 * 1024 * 1024,
            max_recent_bytes: 256 * 1024 * 1024,
            min_seal_bytes: 64 * 1024,
            buffer_filter_bytes: 128 * 1024,
            range_target_bytes: 64 * 1024 * 1024,
            max_meta_bytes: None,
            evaluate_interval: Duration::from_millis(250),
            max_candidates_per_evaluation: 2,
            min_evidence: 8,
            full_evidence: 32,
            min_reuse_span: Duration::from_secs(2),
            admission_margin: 2.0,
            ghost_ttl: Duration::from_secs(60),
            candidate_ttl: Duration::from_secs(30),
            candidate_recheck: Duration::from_secs(1),
            backoff_base: Duration::from_secs(10),
            reservoir_capacity: 8192,
            queue_capacity: 4096,
            min_read_cost: Duration::from_micros(1),
            rebuild_cost_ns_per_byte: 5.0,
            load_fetch_bytes: 2 * 1024 * 1024,
            load_fetch_tasks: 4,
            stats_fold_events: 4096,
            stats_fold_interval: Duration::from_millis(10),
        }
    }
}

impl RangeCacheTuning {
    fn delta_shape(&self) -> DeltaShape {
        DeltaShape {
            seal_bytes: self.range_seal_bytes as u64,
            filter_from: u64::try_from(self.buffer_filter_bytes).unwrap_or(u64::MAX),
        }
    }

    /// Base size below which adjacent quiet ranges merge (`0` disables quiet merges).
    fn quiet_merge_below(&self) -> u64 {
        self.range_target_bytes / 8
    }

    fn meta_limit(&self, max_disk_bytes: u64) -> u64 {
        self.max_meta_bytes
            .unwrap_or((max_disk_bytes / 16).max(64 * 1024 * 1024))
    }

    /// Buffer memory threshold above which new loads are rejected.
    fn high_water(&self) -> u64 {
        self.max_recent_bytes / 8 * 7
    }

    /// Buffer memory target after overflow removals.
    fn low_water(&self) -> u64 {
        self.max_recent_bytes / 4 * 3
    }

    /// Buffer memory threshold above which maintenance seals large active buffers early.
    fn pressure(&self) -> u64 {
        self.max_recent_bytes / 2
    }

    /// Scan options for range loads. Skips populating the block cache.
    fn bulk_scan_options(&self) -> ScanOptions {
        ScanOptions {
            read_ahead_bytes: self.load_fetch_bytes,
            max_fetch_tasks: self.load_fetch_tasks,
            cache_blocks: false,
            ..ScanOptions::default()
        }
    }
}

/// Snapshot of one cached range's state. See [`Db::cached_ranges`].
///
/// Ranges created by a split share base files, runs, and sealed write buffers. Base and run sizes
/// report only this range's pages; buffer memory counts shared sealed buffers whole until flushed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CachedRangeInfo {
    /// Inclusive start key of the range.
    pub start: Bytes,
    /// Exclusive end key, or `None` if unbounded.
    pub end: Option<Bytes>,
    /// True while the initial load is still in progress.
    pub loading: bool,
    /// True while sealed buffers await flush or a major merge or split is due or running.
    /// Deferred unpaid automatic merges do not count until the range reaches its run limit.
    pub rebuilding: bool,
    /// Size of the range's base chunk pages on disk in bytes.
    pub file_bytes: u64,
    /// Row count across base chunks (estimated by page bytes when sharing a split base file).
    pub file_rows: u64,
    /// Number of base chunks (`0` while loading, `1` after).
    pub chunks: usize,
    /// Estimated size of unflushed writes in memory in bytes.
    pub delta_bytes: u64,
    /// True if automatic promotion cached the range.
    pub automatic: bool,
    /// Number of run files above the base.
    pub runs: usize,
    /// Disk bytes of the range's pages across its run files.
    pub run_bytes: u64,
    /// Number of sealed write buffers awaiting flush.
    pub sealed_buffers: usize,
    /// Heap memory of active and sealed write buffers in bytes, counting shared buffers whole.
    pub buffer_bytes: u64,
    /// Run bytes flushed since registration or split, including run-cap flush merges.
    pub flushed_bytes: u64,
    /// Base bytes written by major merges since registration or split.
    pub major_merge_bytes: u64,
    /// Major merges installed since registration or split.
    pub major_merges: u64,
    /// Point reads served since registration or split, as of the last thread fold.
    pub gets: u64,
    /// Scans starting in this range served since registration or split, as of the last fold.
    pub scans: u64,
}

/// One run of a range: a file of every version of the range with a sequence number in
/// `(lo, hi]`, its key bounds, and its filter on the heap.
pub(crate) struct Run {
    lo: u64,
    hi: u64,
    min_key: Bytes,
    max_key: Bytes,
    filter: Filter,
    file: Arc<RangeFile>,
}

impl Run {
    /// A run of the versions in `(lo, hi]` from a finished run file.
    fn new(mut file: RangeFile, lo: u64, hi: u64) -> io::Result<Self> {
        if !file.is_run() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "range cache run: not a run file",
            ));
        }
        let filter = file
            .take_filter()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "range cache run: filter"))?;
        let min_key = file.first_key().map(Bytes::copy_from_slice);
        let max_key = file.last_key()?;
        Ok(Self {
            lo,
            hi,
            min_key: min_key.unwrap_or_default(),
            max_key: max_key.unwrap_or_default(),
            filter,
            file: Arc::new(file),
        })
    }

    fn file_bytes(&self) -> u64 {
        self.file.file_len()
    }
}

/// One chunk of a range's base, holding live rows at `base_seq` for its key span.
#[derive(Clone)]
pub(crate) struct Chunk {
    file: Arc<RangeFile>,
    base_seq: u64,
}

impl Chunk {
    fn new(file: Arc<RangeFile>, base_seq: u64) -> Self {
        Self { file, base_seq }
    }

    /// Returns true if `run` is already absorbed into this chunk (`run.hi <= base_seq`).
    fn absorbed(&self, run: &Run) -> bool {
        run.hi <= self.base_seq
    }
}

/// Ordered chunks tiling a range's base. Single-chunk ranges store the chunk inline inside
/// [`Layers`] so hot reads avoid extra pointer indirections and key comparisons.
#[derive(Clone)]
pub(crate) enum Chunks {
    /// Range is loading and has no base yet.
    Loading,
    /// Single chunk covering the entire range.
    One(Chunk),
    /// Multiple chunks tiling the range in key order (created by quiet merges across distinct
    /// base files).
    Many(Arc<ChunkList>),
}

/// Multi-chunk base storage and split keys (`starts[i]` is the first key of chunk `i + 1`).
pub(crate) struct ChunkList {
    chunks: Box<[Chunk]>,
    starts: Box<[Bytes]>,
}

impl ChunkList {
    #[inline(never)]
    fn index_of(&self, key: &[u8]) -> usize {
        self.starts.partition_point(|s| s.as_ref() <= key)
    }

    #[inline(never)]
    fn newest_base(&self, start: Bound<&[u8]>, end: Bound<&[u8]>) -> u64 {
        let first = match start {
            Unbounded => 0,
            Included(key) | Excluded(key) => self.index_of(key),
        };
        let last = match end {
            Unbounded => self.chunks.len() - 1,
            Included(key) => self.index_of(key),
            Excluded(key) => self.starts.partition_point(|s| s.as_ref() < key),
        };
        self.chunks[first..=last.max(first)]
            .iter()
            .map(|c| c.base_seq)
            .max()
            .unwrap_or(0)
    }

    #[inline(never)]
    fn chunk_of(&self, key: &[u8]) -> Option<&Chunk> {
        self.chunks.get(self.index_of(key))
    }

    /// Creates a chunk list where chunk `i + 1` starts at `starts[i]`.
    fn new(chunks: Vec<Chunk>, starts: Vec<Bytes>) -> Self {
        assert_eq!(
            chunks.len(),
            starts.len() + 1,
            "a start for each chunk after the first"
        );
        Self {
            chunks: chunks.into(),
            starts: starts.into(),
        }
    }
}

impl std::ops::Deref for Chunks {
    type Target = [Chunk];

    fn deref(&self) -> &[Chunk] {
        match self {
            Chunks::Loading => &[],
            Chunks::One(chunk) => std::slice::from_ref(chunk),
            Chunks::Many(list) => &list.chunks,
        }
    }
}

impl Chunks {
    #[inline]
    fn loading(&self) -> bool {
        matches!(self, Chunks::Loading)
    }

    #[inline]
    fn start(&self, i: usize) -> Option<&Bytes> {
        match self {
            Chunks::Many(list) => i.checked_sub(1).and_then(|at| list.starts.get(at)),
            _ => None,
        }
    }

    #[inline]
    fn index_of(&self, key: &[u8]) -> Option<usize> {
        match self {
            Chunks::Loading => None,
            Chunks::One(_) => Some(0),
            Chunks::Many(list) => Some(list.index_of(key)),
        }
    }

    #[inline]
    fn of(&self, key: &[u8]) -> Option<&Chunk> {
        match self {
            Chunks::Loading => None,
            Chunks::One(chunk) => Some(chunk),
            Chunks::Many(list) => list.chunk_of(key),
        }
    }

    /// Returns the key span of chunk `i` within `[start, end)`.
    fn span(&self, i: usize, start: Option<&Bytes>, end: Option<&Bytes>) -> BytesRange {
        let from = match self.start(i).or(if i == 0 { start } else { None }) {
            Some(start) => Included(start.clone()),
            None => Unbounded,
        };
        let to = match self.start(i + 1).or(end) {
            Some(end) => Excluded(end.clone()),
            None => Unbounded,
        };
        BytesRange::new(from, to)
    }

    /// Returns the highest `base_seq` among chunks intersecting `[start, end)`.
    #[inline]
    fn newest_base(&self, start: Bound<&[u8]>, end: Bound<&[u8]>) -> Option<u64> {
        match self {
            Chunks::Loading => None,
            Chunks::One(chunk) => Some(chunk.base_seq),
            Chunks::Many(list) => Some(list.newest_base(start, end)),
        }
    }

    fn same(&self, other: &Chunks) -> bool {
        match (self, other) {
            (Chunks::One(a), Chunks::One(b)) => {
                Arc::ptr_eq(&a.file, &b.file) && a.base_seq == b.base_seq
            }
            (Chunks::Many(a), Chunks::Many(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    fn handle_bytes(&self) -> usize {
        match self {
            Chunks::Many(list) => {
                size_of::<ChunkList>()
                    + list.chunks.len() * size_of::<Chunk>()
                    + list.starts.len() * size_of::<Bytes>()
            }
            _ => 0,
        }
    }
}

fn one_chunk(file: Arc<RangeFile>, base_seq: u64) -> Chunks {
    Chunks::One(Chunk::new(file, base_seq))
}

/// Immutable layer set of a cached range, replaced atomically by the batch writer.
///
/// Cache-line aligned so `Arc` reference count updates on scans do not contend with read fields.
#[repr(align(128))]
pub(crate) struct Layers {
    /// The buffer that the writer appends to. Its `sealed_at` is the readers' validity word.
    active: Arc<Delta>,
    /// Sealed buffers that wait for their flush, newest first. Each one's `lo` is the seal of the
    /// next one, and the oldest one's `lo` is `covered_to`.
    sealed: Vec<Arc<Delta>>,
    /// Runs, newest first, at most [`RangeCacheTuning::max_runs`]: run `i` covers
    /// `(lo_i, hi_i]` with `hi_{i+1} = lo_i`, and the oldest one's `lo` is the oldest chunk's
    /// `base_seq` (see [`Layers::floor`]).
    runs: Vec<Arc<Run>>,
    /// Delta view over `runs`, omitting versions already absorbed by a base chunk. `None` when
    /// there are no runs.
    view: Option<Arc<DeltaView>>,
    /// The base: chunks that tile the range's span (see [`Chunks`]). None while the range loads.
    chunks: Chunks,
    /// The base and the runs hold every version of the range at or below this sequence number:
    /// the newest run's `hi`, or with no run the `base_seq` of every chunk (of the registration,
    /// while the range loads). The buffers hold every version above it.
    covered_to: u64,
    /// The end of the range, so that a scan can walk adjacent ranges with one reference to
    /// each.
    end: Option<Bytes>,
}

// Every field that a read loads, the one chunk's file and `base_seq` included, is in the layer
// set's one cache line, as before chunks.
const _: () = assert!(
    size_of::<Layers>() == 128,
    "the layer set outgrew its cache line"
);

impl Layers {
    /// The buffers, newest first: the active one, then the sealed ones.
    fn buffers(&self) -> impl Iterator<Item = &Arc<Delta>> {
        std::iter::once(&self.active).chain(self.sealed.iter())
    }

    /// Oldest sequence number covered by runs, or `covered_to` when no runs exist.
    fn floor(&self) -> u64 {
        self.runs.last().map_or(self.covered_to, |run| run.lo)
    }

    fn chunk_index(&self, key: &[u8]) -> Option<usize> {
        self.chunks.index_of(key)
    }

    fn chunk_of(&self, key: &[u8]) -> Option<&Chunk> {
        self.chunks.of(key)
    }

    fn chunk_span(&self, i: usize) -> BytesRange {
        self.chunks.span(i, None, self.end.as_ref())
    }

    /// Base bytes and estimated rows within this range's key span (exact unless sharing a split
    /// base file).
    fn base_share(&self, start: &Bytes) -> (u64, u64) {
        split::base_share(&self.chunks, start, self.end.as_ref())
    }

    /// Run file bytes within this range's key span. Searches the fence index only for runs shared
    /// across a split.
    fn run_share(&self, start: &[u8]) -> u64 {
        self.runs
            .iter()
            .map(|run| {
                let within = run.file.rows() == 0
                    || (run.min_key.as_ref() >= start
                        && self.end.as_ref().is_none_or(|end| run.max_key < end));
                if within {
                    run.file_bytes()
                } else {
                    run.file.span_share(Some(start), self.end.as_deref()).0
                }
            })
            .sum()
    }

    #[cfg(test)]
    fn only_chunk(&self) -> &Chunk {
        match &self.chunks {
            Chunks::One(chunk) => chunk,
            _ => panic!("a range without exactly one chunk"),
        }
    }

    /// Validates layer invariants in debug builds within the range's key span.
    fn check(&self, start: &[u8], max_runs: usize) {
        if !cfg!(debug_assertions) {
            return;
        }
        assert!(
            self.runs.len() <= max_runs,
            "a range holds {} runs",
            self.runs.len()
        );
        // I4c: the runs are contiguous and end at `covered_to`.
        let mut to = self.covered_to;
        for run in &self.runs {
            assert_eq!(run.hi, to, "runs are not contiguous");
            assert!(run.lo < run.hi || run.lo == run.hi && run.file.rows() == 0);
            to = run.lo;
        }
        // I3c: the chunks are sorted, disjoint and tile the range's span: chunk 0 starts at the
        // range's start, and each later one starts above the one before and below the range's end.
        if let Chunks::Many(list) = &self.chunks {
            assert_eq!(
                list.chunks.len(),
                list.starts.len() + 1,
                "a chunk list without a start for each chunk after the first"
            );
            let mut before: &[u8] = start;
            for later in list.starts.iter() {
                assert!(
                    before < later.as_ref(),
                    "the chunks are not sorted and disjoint"
                );
                before = later;
            }
            if let Some(end) = &self.end {
                assert!(
                    before < end.as_ref(),
                    "a chunk starts at or past the range's end"
                );
            }
        }
        // I4c and I7c: every chunk's versions above its `base_seq` are in runs that end above it,
        // and the oldest run is one that a chunk needs.
        if let Some(oldest) = self.chunks.iter().map(|c| c.base_seq).min() {
            assert_eq!(
                to, oldest,
                "the runs do not start at the oldest chunk's base_seq"
            );
        }
        for (i, chunk) in self.chunks.iter().enumerate() {
            assert!(
                chunk.base_seq <= self.covered_to,
                "a chunk above covered_to"
            );
            // I4a: a run that a chunk has not absorbed holds no version of its keys at or below
            // its `base_seq`. A run that starts at or above it holds none by its interval.
            for run in &self.runs {
                if chunk.absorbed(run) || run.lo >= chunk.base_seq {
                    continue;
                }
                let range_start = Bytes::copy_from_slice(start);
                let span = self.chunks.span(i, Some(&range_start), self.end.as_ref());
                let mut cursor = RangeFileCursor::new(&run.file, &span).expect("a run cursor");
                while let Some(seq) = cursor.seq() {
                    assert!(
                        seq > chunk.base_seq,
                        "a run holds a version at or below the base_seq of a chunk that needs it"
                    );
                    cursor.skip(&run.file).expect("a run cursor");
                }
            }
        }
        assert_eq!(
            self.view.is_some(),
            !self.runs.is_empty(),
            "a range with runs and no view, or a view and no runs"
        );
        if let Some(view) = &self.view {
            let in_view = view.runs().count();
            assert_eq!(in_view, self.runs.len(), "the view does not hold the runs");
            assert!(
                self.runs.iter().all(|r| view.holds(r)),
                "the view holds other runs"
            );
            // I12: the view lists each run's versions in the range's keys, in the chunks that have
            // not absorbed it.
            let rows: u64 = self
                .runs
                .iter()
                .map(|r| self.unabsorbed_rows(r, start))
                .sum();
            assert_eq!(
                view.len() as u64,
                rows,
                "the view does not list the runs' rows in the range that no chunk absorbed"
            );
        }
        let mut lo = self.covered_to;
        for buffer in self.sealed.iter().rev() {
            assert_eq!(buffer.lo(), lo, "sealed buffers are not contiguous");
            lo = buffer.sealed_at().load(Ordering::Acquire);
        }
        assert_eq!(
            self.active.lo(),
            lo,
            "the active buffer does not follow the sealed ones"
        );
    }

    /// The rows of `run` in the keys of the chunks that have not absorbed it, in a range that
    /// starts at `start`: all of them unless a chunk has, or the run holds keys outside the range
    /// (a run that a split shares). Counts the rows by key in those cases, for [`Layers::check`].
    fn unabsorbed_rows(&self, run: &Run, start: &[u8]) -> u64 {
        let within = run.file.rows() == 0
            || (run.min_key.as_ref() >= start
                && self.end.as_ref().is_none_or(|end| run.max_key < end));
        if within && !self.chunks.iter().any(|c| c.absorbed(run)) {
            return run.file.rows();
        }
        if self.chunks.loading() {
            // A loading range's runs: within its keys.
            let span = BytesRange::new(
                Included(Bytes::copy_from_slice(start)),
                self.end.clone().map_or(Unbounded, Excluded),
            );
            let mut cursor = RangeFileCursor::new(&run.file, &span).expect("a run cursor");
            let mut rows = 0;
            while cursor.key().is_some() {
                rows += 1;
                cursor.skip(&run.file).expect("a run cursor");
            }
            return rows;
        }
        let start = Bytes::copy_from_slice(start);
        let mut rows = 0;
        for (i, chunk) in self.chunks.iter().enumerate() {
            if chunk.absorbed(run) {
                continue;
            }
            let span = self.chunks.span(i, Some(&start), self.end.as_ref());
            let mut cursor = RangeFileCursor::new(&run.file, &span).expect("a run cursor");
            while cursor.key().is_some() {
                rows += 1;
                cursor.skip(&run.file).expect("a run cursor");
            }
        }
        rows
    }
}

/// Source that requested caching a range.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Origin {
    /// Application call to `cache_range`.
    Explicit,
    /// Automatic promotion.
    Auto {
        /// Estimated normal-path cost of one read in nanoseconds.
        read_cost_ns: u64,
        /// Estimated reads per second when admitted.
        read_rate: f64,
    },
}

/// Parent heat share inherited by a split child (by base bytes). See [`CachedRange::split_from`].
struct Inherited {
    share: f64,
    rate: f64,
    now_ms: i64,
}

/// Folded read count, access timestamp, and decayed read rate of a cached range.
struct Heat {
    hits: Arc<RangeHits>,
    rate: Mutex<HeatRate>,
}

struct HeatRate {
    reads: u64,
    at_ms: i64,
    rate: f64,
}

impl Heat {
    fn new(rate: f64, now_ms: i64) -> Self {
        Self {
            hits: Arc::new(RangeHits::default()),
            rate: Mutex::new(HeatRate {
                reads: 0,
                at_ms: now_ms,
                rate,
            }),
        }
    }

    /// The heat of a range that a split cut from this one: the read rate `rate` (its share of
    /// this range's), this range's last access, and no reads yet. No lock: the writer calls it.
    fn split(&self, rate: f64, now_ms: i64) -> Self {
        let hits = RangeHits::default();
        hits.set_last_access(self.hits.last_access());
        Self {
            hits: Arc::new(hits),
            rate: Mutex::new(HeatRate {
                reads: 0,
                at_ms: now_ms,
                rate,
            }),
        }
    }

    /// Combines heat for a quiet merge: summed `rate`, newer last access, zero new reads.
    fn merged(a: &Heat, b: &Heat, rate: f64, now_ms: i64) -> Self {
        let hits = RangeHits::default();
        hits.set_last_access(a.hits.last_access().max(b.hits.last_access()));
        Self {
            hits: Arc::new(hits),
            rate: Mutex::new(HeatRate {
                reads: 0,
                at_ms: now_ms,
                rate,
            }),
        }
    }

    /// The decayed read rate as of the last time anyone computed it.
    fn last_rate(&self) -> f64 {
        self.rate.lock().rate
    }

    fn reads(&self) -> u64 {
        self.hits.reads()
    }

    /// Returns exponentially decayed reads per second at `now_ms`.
    fn rate(&self, now_ms: i64, half_life_ms: f64) -> f64 {
        let mut state = self.rate.lock();
        let dt_ms = (now_ms - state.at_ms) as f64;
        if dt_ms < 50.0 {
            return state.rate;
        }
        let reads = self.reads();
        let instant = (reads - state.reads) as f64 * 1000.0 / dt_ms;
        let keep = (-dt_ms * std::f64::consts::LN_2 / half_life_ms.max(1.0)).exp();
        state.rate = state.rate * keep + instant * (1.0 - keep);
        state.reads = reads;
        state.at_ms = now_ms;
        state.rate
    }
}

struct CachedRange {
    start: Bytes,
    end: Option<Bytes>,
    layers: ArcSwap<Layers>,
    /// Reusable slot index used by thread-local hit tracking.
    slot: u32,
    /// True if automatic promotion cached this range.
    auto: bool,
    /// True if this range starts at a cache-chosen split key. Explicit ranges merge quietly only
    /// across cache-chosen boundaries.
    cut_start: bool,
    /// Folded read count, last access clock, and decayed read rate.
    heat: Heat,
    /// Estimated normal-path cost per read in nanoseconds (`0` for explicit ranges).
    read_cost_ns: u64,
    /// Share of parent base reads inherited on split for major-merge payback checks.
    reads_carried: u64,
    /// `heat.reads + reads_carried` when the current base was built, for payback checks.
    reads_at_build: AtomicU64,
    /// Sequence number of the last batch that wrote to this range.
    last_write_seq: Padded<AtomicU64>,
    /// Maintenance state for in-flight flush and major merge jobs.
    jobs: maintain::RangeJobs,
}

impl CachedRange {
    fn new(
        start: Bytes,
        end: Option<Bytes>,
        layers: Layers,
        origin: Origin,
        now_ms: i64,
        slot: u32,
    ) -> Self {
        let (auto, read_cost_ns, rate) = match origin {
            Origin::Explicit => (false, 0, 0.0),
            Origin::Auto {
                read_cost_ns,
                read_rate,
            } => (true, read_cost_ns, read_rate),
        };
        Self {
            start,
            end,
            layers: ArcSwap::from_pointee(layers),
            slot,
            auto,
            cut_start: false,
            heat: Heat::new(rate, now_ms),
            read_cost_ns,
            reads_carried: 0,
            reads_at_build: AtomicU64::new(0),
            last_write_seq: Padded(AtomicU64::new(0)),
            jobs: maintain::RangeJobs::default(),
        }
    }

    /// A range from `start` to `end` with `layers` that a split cut from `parent` and that holds
    /// `share` of its base: it keeps the parent's origin, read cost, last write and last access, a
    /// read rate of `rate`, `share` times the parent's rate (see [`Heat::split`]), `share` of the
    /// reads that the parent's base served since it was written (see
    /// [`CachedRange::reads_since_build`]), and its jobs' headroom. Its counts of reads, flushes
    /// and merges start at zero.
    fn split_from(
        parent: &CachedRange,
        start: Bytes,
        end: Option<Bytes>,
        layers: Layers,
        inherited: Inherited,
        slot: u32,
    ) -> Self {
        let reads_carried = (parent.reads_since_build() as f64 * inherited.share).round() as u64;
        let cut_start = start != parent.start || parent.cut_start;
        Self {
            start,
            end,
            layers: ArcSwap::from_pointee(layers),
            slot,
            auto: parent.auto,
            cut_start,
            heat: parent
                .heat
                .split(inherited.rate * inherited.share, inherited.now_ms),
            read_cost_ns: parent.read_cost_ns,
            reads_carried,
            reads_at_build: AtomicU64::new(0),
            last_write_seq: Padded(AtomicU64::new(
                parent.last_write_seq.0.load(Ordering::Acquire),
            )),
            jobs: maintain::RangeJobs::inherit(&parent.jobs),
        }
    }

    /// Creates a range from quietly merging adjacent ranges `left` and `right`, preserving their
    /// origin, combined read history, newer timestamps, and larger headroom.
    fn merged_from(
        left: &CachedRange,
        right: &CachedRange,
        layers: Layers,
        rate: f64,
        now_ms: i64,
        slot: u32,
    ) -> Self {
        let last_write = |r: &CachedRange| r.last_write_seq.0.load(Ordering::Acquire);
        Self {
            start: left.start.clone(),
            end: right.end.clone(),
            layers: ArcSwap::from_pointee(layers),
            slot,
            auto: left.auto,
            cut_start: left.cut_start,
            heat: Heat::merged(&left.heat, &right.heat, rate, now_ms),
            read_cost_ns: left.read_cost_ns.max(right.read_cost_ns),
            reads_carried: left.reads_since_build() + right.reads_since_build(),
            reads_at_build: AtomicU64::new(0),
            last_write_seq: Padded(AtomicU64::new(last_write(left).max(last_write(right)))),
            jobs: maintain::RangeJobs::merged(&left.jobs, &right.jobs),
        }
    }

    /// Returns true if two adjacent ranges may merge quietly: both automatic, or both explicit
    /// separated by a cache-chosen split boundary.
    fn may_merge(left: &CachedRange, right: &CachedRange) -> bool {
        left.auto == right.auto && (right.auto || right.cut_start)
    }

    /// The reads that the range's base served since it was written: since its load or its last
    /// major merge, or for a range that a split made before its first major merge, since the split
    /// with its share of the parent's (`reads_carried`). Exact as of the last fold of each reading
    /// thread.
    fn reads_since_build(&self) -> u64 {
        (self.heat.reads() + self.reads_carried)
            .saturating_sub(self.reads_at_build.load(Ordering::Relaxed))
    }

    /// Notes that a new base is being written now, for [`CachedRange::reads_since_build`].
    fn mark_build(&self) {
        self.reads_at_build
            .store(self.heat.reads() + self.reads_carried, Ordering::Relaxed);
    }

    fn contains(&self, key: &[u8]) -> bool {
        key >= self.start.as_ref() && self.end.as_ref().is_none_or(|end| key < end.as_ref())
    }

    /// Returns true if the range holds every key of `range`.
    fn contains_range(&self, range: &BytesRange) -> bool {
        match RangeBounds::end_bound(range) {
            Unbounded => self.end.is_none(),
            Included(k) => self.end.as_ref().is_none_or(|end| k < end),
            Excluded(k) => self.end.as_ref().is_none_or(|end| k <= end),
        }
    }

    fn bounds(&self) -> BytesRange {
        let end = match &self.end {
            Some(end) => Excluded(end.clone()),
            None => Unbounded,
        };
        BytesRange::new(Included(self.start.clone()), end)
    }

    fn info(&self, cache: &RangeCache) -> CachedRangeInfo {
        let tuning = &cache.options.tuning;
        let layers = self.layers.load();
        let (file_bytes, file_rows) = layers.base_share(&self.start);
        // An unpaid due merge on an automatic range waits until reads justify it or it hits the
        // run limit.
        let merge_due = maintain::major_due(self, &layers, tuning).is_some_and(|trigger| {
            cache.rebuild_pays(self, file_bytes) || maintain::at_run_limit(self, &layers, trigger)
        });
        CachedRangeInfo {
            start: self.start.clone(),
            end: self.end.clone(),
            loading: layers.chunks.loading(),
            rebuilding: !layers.sealed.is_empty()
                || self.jobs.merging()
                || self.jobs.splitting()
                || merge_due
                || maintain::split_cuts(self, &layers, tuning).is_some(),
            file_bytes,
            file_rows,
            chunks: layers.chunks.len(),
            delta_bytes: self.delta_bytes(&layers),
            automatic: self.auto,
            runs: layers.runs.len(),
            run_bytes: layers.run_share(&self.start),
            sealed_buffers: layers.sealed.len(),
            buffer_bytes: layers.buffers().map(|b| b.memory_bytes()).sum(),
            flushed_bytes: self.jobs.flushed_bytes(),
            major_merge_bytes: self.jobs.major_merge_bytes(),
            major_merges: self.jobs.major_merges(),
            gets: self.heat.hits.gets(),
            scans: self.heat.hits.scans(),
        }
    }

    /// The estimated bytes of the range's buffers.
    fn delta_bytes(&self, layers: &Layers) -> u64 {
        layers.buffers().map(|b| b.size() as u64).sum()
    }

    /// The memory of the range's buffers that no run holds yet.
    fn unflushed_bytes(&self) -> u64 {
        self.layers.load().buffers().map(|b| b.memory_bytes()).sum()
    }

    /// The bytes of the range's share of its base and of its runs (see [`Layers::base_share`] and
    /// [`Layers::run_share`]), which read densities divide by. The disk budget counts each file
    /// once instead (see [`Footprint`]).
    fn file_bytes(&self) -> u64 {
        let layers = self.layers.load();
        layers.base_share(&self.start).0 + layers.run_share(&self.start)
    }

    /// The heap metadata of the range, its files' and view's included, each counted whole: what
    /// it adds to the metadata budget when it shares nothing (see [`Footprint`]).
    #[cfg(test)]
    fn meta_bytes(&self) -> u64 {
        let layers = self.layers.load();
        let files = layers
            .chunks
            .iter()
            .map(|c| c.file.meta_bytes())
            .chain(layers.runs.iter().map(|r| r.file.meta_bytes()))
            .sum::<u64>();
        files + Self::handle_bytes(&layers) + layers.view.as_ref().map_or(0, |v| v.bytes())
    }

    /// The heap bytes of the range's own metadata: its `CachedRange`, its layer set, and its run
    /// and chunk handles, not the files and views that they point to.
    fn handle_bytes(layers: &Layers) -> u64 {
        (size_of::<CachedRange>()
            + size_of::<Layers>()
            + (layers.sealed.capacity() + layers.runs.capacity()) * size_of::<usize>()
            + layers.runs.len() * size_of::<Run>()
            + layers.chunks.handle_bytes()) as u64
    }
}

/// Disk and heap metadata footprint across a set of ranges, counting shared files and views once.
#[derive(Debug, Default, PartialEq, Eq)]
struct Footprint {
    /// The bytes of the base files and run files.
    disk: u64,
    /// Of `disk`, the bytes of the base files.
    base_disk: u64,
    /// The fence indexes and filters of those files, the views, and each range's own handles.
    meta: u64,
}

impl Footprint {
    fn of<'a>(ranges: impl Iterator<Item = &'a Arc<CachedRange>>) -> Self {
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut once = |ptr: usize| seen.insert(ptr);
        let mut total = Footprint::default();
        for range in ranges {
            let layers = range.layers.load();
            let files = layers
                .chunks
                .iter()
                .map(|c| (&c.file, true))
                .chain(layers.runs.iter().map(|r| (&r.file, false)));
            for (file, base) in files {
                if once(Arc::as_ptr(file).addr()) {
                    total.disk += file.file_len();
                    total.meta += file.meta_bytes();
                    if base {
                        total.base_disk += file.file_len();
                    }
                }
            }
            if let Some(view) = &layers.view {
                if once(Arc::as_ptr(view).addr()) {
                    total.meta += view.bytes();
                }
            }
            total.meta += CachedRange::handle_bytes(&layers);
        }
        total
    }
}

/// Cached ranges by start key. Ranges do not overlap.
type RangeIndex = BTreeMap<Bytes, Arc<CachedRange>>;

/// Returns the range that holds `key`.
fn covering<'a>(index: &'a RangeIndex, key: &[u8]) -> Option<&'a Arc<CachedRange>> {
    index
        .range::<[u8], _>((Unbounded, Included(key)))
        .next_back()
        .map(|(_, range)| range)
        .filter(|range| range.contains(key))
}

/// Returns the parts of `[start, end)` that no range in `index` holds.
fn uncovered(
    index: &RangeIndex,
    start: &Bytes,
    end: Option<&Bytes>,
) -> Vec<(Bytes, Option<Bytes>)> {
    let before_end = |key: &Bytes| end.is_none_or(|end| key < end);
    let mut parts = Vec::new();
    // `None` means that ranges hold every key after the cursor.
    let mut cursor = Some(start.clone());
    let first = index
        .range::<[u8], _>((Unbounded, Included(start.as_ref())))
        .next_back();
    let rest = index.range::<[u8], _>((Excluded(start.as_ref()), Unbounded));
    for (_, range) in first.into_iter().chain(rest) {
        let Some(pos) = cursor.clone() else {
            break;
        };
        if !before_end(&range.start) {
            break;
        }
        if range
            .end
            .as_ref()
            .is_some_and(|range_end| range_end <= &pos)
        {
            continue;
        }
        if range.start > pos {
            parts.push((pos, Some(range.start.clone())));
        }
        cursor = range.end.clone();
    }
    if let Some(pos) = cursor {
        if before_end(&pos) {
            parts.push((pos, end.cloned()));
        }
    }
    parts
}

/// The bound `bound` of a key range, on the key's bytes.
fn as_slices(bound: &Bound<Bytes>) -> Bound<&[u8]> {
    bound.as_ref().map(|key| key.as_ref())
}

/// Converts `range` to a half-open range `[start, end)`. Returns `None` if it holds no key.
fn half_open(range: &BytesRange) -> Option<(Bytes, Option<Bytes>)> {
    let after = |key: &Bytes| {
        let mut next = key.to_vec();
        next.push(0);
        Bytes::from(next)
    };
    let start = match RangeBounds::start_bound(range) {
        Unbounded => Bytes::new(),
        Included(k) => k.clone(),
        Excluded(k) => after(k),
    };
    let end = match RangeBounds::end_bound(range) {
        Unbounded => None,
        Included(k) => Some(after(k)),
        Excluded(k) => Some(k.clone()),
    };
    match &end {
        Some(end) if *end <= start => None,
        _ => Some((start, end)),
    }
}

/// State shared between the cache core and the automatic promotion task.
struct AutoShared {
    options: AutoSettings,
    tx: async_channel::Sender<PromoterMessage>,
    stats: AutoStats,
    /// Moving average normal-path cost of sampled reads in nanoseconds.
    avg_read_cost_ns: AtomicU64,
    /// Removal notifications from the batch writer to the promoter.
    removals: (
        async_channel::Sender<auto::Removal>,
        async_channel::Receiver<auto::Removal>,
    ),
}

/// Timing token for a sampled normal-path read.
pub(crate) struct ReadSample {
    start: chrono::DateTime<chrono::Utc>,
}

/// Range cache state for one database.
pub(crate) struct RangeCache {
    options: RangeCacheOptions,
    index: ArcSwap<RangeIndex>,
    writer: WriterQueue,
    files: Arc<FileBudget>,
    delta_memory: Arc<DeltaMemory>,
    views: Arc<ViewMemory>,
    maintenance: maintain::MaintenanceQueue,
    next_file_id: AtomicU64,
    hits: HitStats,
    stats: RangeCacheStats,
    auto: Option<AutoShared>,
    clock: Arc<dyn SystemClock>,
    rand: Arc<DbRand>,
    handle: tokio::runtime::Handle,
    merge_operator: Option<MergeOperatorType>,
    #[allow(dead_code)]
    fp_registry: Arc<FailPointRegistry>,
}

impl RangeCache {
    /// Creates the cache directory if missing and removes stale cache files.
    /// Returns the promoter queue receiver when automatic promotion is enabled.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open(
        options: RangeCacheOptions,
        recorder: &MetricsRecorderHelper,
        handle: tokio::runtime::Handle,
        merge_operator: Option<MergeOperatorType>,
        fp_registry: Arc<FailPointRegistry>,
        clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
    ) -> Result<(Arc<Self>, Option<async_channel::Receiver<PromoterMessage>>), SlateDBError> {
        if !file::valid_page_size(options.tuning.page_size) {
            return Err(SlateDBError::InvalidConfiguration(format!(
                "range cache page size {} is not a power of two in 512..=65536",
                options.tuning.page_size
            )));
        }
        if options.tuning.load_fetch_bytes == 0 || options.tuning.load_fetch_tasks == 0 {
            return Err(SlateDBError::InvalidConfiguration(
                "range cache load_fetch_bytes and load_fetch_tasks must be positive".to_string(),
            ));
        }
        if !MAX_RUNS_RANGE.contains(&options.tuning.max_runs) {
            return Err(SlateDBError::InvalidConfiguration(format!(
                "range cache max_runs {} is not in {}..={}",
                options.tuning.max_runs,
                MAX_RUNS_RANGE.start(),
                MAX_RUNS_RANGE.end()
            )));
        }
        let f = options.tuning.major_merge_factor;
        if !f.is_finite() || f <= 0.0 {
            return Err(SlateDBError::InvalidConfiguration(format!(
                "range cache major_merge_factor {f} is not positive and finite"
            )));
        }
        std::fs::create_dir_all(&options.root_folder)?;
        for entry in std::fs::read_dir(&options.root_folder)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let is_cache_file = [FILE_EXTENSION, TEMP_FILE_EXTENSION].iter().any(|ext| {
                name.strip_suffix(ext)
                    .and_then(|stem| stem.strip_suffix('.'))
                    .is_some_and(|stem| {
                        !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit())
                    })
            });
            if is_cache_file {
                std::fs::remove_file(&path)?;
            }
        }
        let (auto, rx) = match &options.auto_promote {
            Some(auto_options) => {
                let auto_options = AutoSettings::new(auto_options, &options.tuning);
                auto_options.validate()?;
                let (tx, rx) = async_channel::bounded(auto_options.queue_capacity.max(1));
                let auto = AutoShared {
                    tx,
                    stats: AutoStats::new(recorder),
                    avg_read_cost_ns: AtomicU64::new(auto_options.min_read_cost.as_nanos() as u64),
                    removals: async_channel::unbounded(),
                    options: auto_options,
                };
                (Some(auto), Some(rx))
            }
            None => (None, None),
        };
        let hits = HitStats::new(
            stats::read_counters(recorder),
            options.tuning.stats_fold_events,
            options.tuning.stats_fold_interval,
            clock.clone(),
        );
        let cache = Arc::new(Self {
            options,
            index: ArcSwap::from_pointee(RangeIndex::new()),
            writer: WriterQueue::new(),
            files: Arc::new(FileBudget::default()),
            delta_memory: DeltaMemory::new(recorder.gauge(stats::DELTA_ALLOCATED_BYTES).register()),
            views: Arc::new(ViewMemory::default()),
            maintenance: maintain::MaintenanceQueue::new(),
            next_file_id: AtomicU64::new(0),
            hits,
            stats: RangeCacheStats::new(recorder),
            auto,
            clock,
            rand,
            handle,
            merge_operator,
            fp_registry,
        });
        Ok((cache, rx))
    }

    fn now_ms(&self) -> i64 {
        self.clock.now().timestamp_millis()
    }

    /// Returns true if reads served since the current base was built (including a split child's
    /// inherited share) justify rewriting `file_bytes`.
    fn rebuild_pays(&self, range: &CachedRange, file_bytes: u64) -> bool {
        let Some(auto) = &self.auto else {
            return true;
        };
        if !range.auto || file_bytes < 4 * self.options.tuning.min_rebuild_delta_bytes as u64 {
            return true;
        }
        let reads = range.reads_since_build();
        let saved_ns = reads as f64 * range.read_cost_ns as f64;
        let cost_ns = file_bytes as f64 * auto.options.rebuild_cost_ns_per_byte;
        saved_ns >= cost_ns
    }

    fn new_buffer(&self, lo: u64) -> Arc<Delta> {
        Arc::new(Delta::new(
            &self.delta_memory,
            lo,
            self.options.tuning.delta_shape(),
        ))
    }

    fn new_file_name(&self) -> String {
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        format!("{file_id:020}")
    }

    fn is_indexed(&self, range: &Arc<CachedRange>) -> bool {
        self.index
            .load()
            .get(&range.start)
            .is_some_and(|r| Arc::ptr_eq(r, range))
    }

    /// Returns eviction candidates when budgets are exceeded, excluding `skip`. Evicts automatic
    /// ranges by lowest read density first when automatic promotion is on, then explicit ranges
    /// in LRU order.
    fn eviction_order(&self, skip: &[Arc<CachedRange>]) -> Vec<Arc<CachedRange>> {
        if self.within_budgets() {
            return Vec::new();
        }
        let now = self.now_ms();
        let candidates = self
            .index
            .load()
            .values()
            .filter(|r| !r.layers.load().chunks.loading())
            .filter(|r| !skip.iter().any(|skip| Arc::ptr_eq(skip, r)))
            .cloned()
            .collect::<Vec<_>>();
        match &self.auto {
            None => {
                let mut candidates = candidates;
                candidates.sort_by_key(|r| r.heat.hits.last_access());
                candidates
            }
            Some(_) => {
                let mut keyed: Vec<_> = candidates
                    .into_iter()
                    .map(|r| (self.eviction_key(&r, now), r))
                    .collect();
                keyed.sort_by(|(a, _), (b, _)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                keyed.into_iter().map(|(_, r)| r).collect()
            }
        }
    }

    fn eviction_key(&self, range: &CachedRange, now_ms: i64) -> (bool, f64, u64) {
        let density = if range.auto {
            self.density(range, now_ms)
        } else {
            0.0
        };
        (!range.auto, density, range.heat.hits.last_access())
    }

    /// Decayed normal-path cost saved per second per byte of `range`.
    fn density(&self, range: &CachedRange, now_ms: i64) -> f64 {
        let Some(auto) = &self.auto else {
            return 0.0;
        };
        let half_life_ms = auto.options.heat_half_life.as_millis() as f64;
        let read_cost_ns = if range.auto {
            range.read_cost_ns
        } else {
            auto.avg_read_cost_ns.load(Ordering::Relaxed)
        };
        range.heat.rate(now_ms, half_life_ms) * read_cost_ns as f64
            / range.file_bytes().max(1) as f64
    }

    /// Sets the gauges. After a split, `delta_bytes` counts a buffer that ranges share once,
    /// `file_bytes`, `meta_bytes` and `view_bytes` count each object that anything holds once (they
    /// follow allocations), and `range_count` counts each range of the split.
    fn update_gauges(&self) {
        let index = self.index.load();
        self.stats.ranges.set(index.len() as i64);
        // A sealed buffer that a split shares counts once. Only buffers, which few layer sets or
        // jobs share: the writer calls this.
        let mut shared: Vec<usize> = Vec::new();
        let mut delta_bytes = 0u64;
        for range in index.values() {
            for buffer in range.layers.load().buffers() {
                if Arc::strong_count(buffer) > 1 {
                    let ptr = Arc::as_ptr(buffer).addr();
                    if shared.contains(&ptr) {
                        continue;
                    }
                    shared.push(ptr);
                }
                delta_bytes += buffer.size() as u64;
            }
        }
        self.stats.delta_bytes.set(delta_bytes as i64);
        self.delta_memory.publish();
        self.stats.file_bytes.set(self.files.disk() as i64);
        // Files and views count while anything holds them; layer sets while they are indexed.
        let layers: u64 = index
            .values()
            .map(|r| CachedRange::handle_bytes(&r.layers.load()))
            .sum();
        let views = self.views.bytes();
        self.stats.view_bytes.set(views as i64);
        self.stats
            .meta_bytes
            .set((self.files.meta() + views + layers) as i64);
    }

    /// Sets the gauges of the base files and run files of the indexed ranges, each file once (see
    /// [`Footprint`]). It walks every file of every range, so only the maintenance task calls it,
    /// about once a second.
    fn update_file_gauges(&self) {
        let indexed = Footprint::of(self.index.load().values());
        self.stats.base_file_bytes.set(indexed.base_disk as i64);
        self.stats
            .run_file_bytes
            .set((indexed.disk - indexed.base_disk) as i64);
    }

    /// Returns true if the cached ranges fit the disk budget and the metadata budget. Only
    /// indexed ranges count: files that only a reader holds go when the reader drops them. A file,
    /// run or view that ranges share after a split counts once.
    fn within_budgets(&self) -> bool {
        let indexed = Footprint::of(self.index.load().values());
        indexed.disk <= self.options.max_disk_bytes
            && indexed.meta <= self.options.tuning.meta_limit(self.options.max_disk_bytes)
    }

    /// Checks whether a read at `seq` can use `layers` when the chunks that it needs hold the
    /// state at `base_seq` or older: the highest `base_seq` among them. See the sequence number
    /// rules. Callers treat a loading range, which has no chunk, as uncovered before they call
    /// this.
    fn usable_from(layers: &Layers, base_seq: u64, seq: u64) -> Result<(), Unfit> {
        if seq < base_seq {
            return Err(Unfit::TooOld);
        }
        if layers.active.sealed_at().load(Ordering::SeqCst) < seq {
            return Err(Unfit::Sealed);
        }
        Ok(())
    }

    /// Checks whether a read of `[start, end)` at `seq` can use `layers`.
    fn usable_for(
        layers: &Layers,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
        seq: u64,
    ) -> Result<(), Unfit> {
        match layers.chunks.newest_base(start, end) {
            Some(base_seq) => Self::usable_from(layers, base_seq, seq),
            None => Err(Unfit::TooOld),
        }
    }

    #[cfg(test)]
    fn usable(layers: &Layers, seq: u64) -> Result<(), Unfit> {
        Self::usable_for(layers, Unbounded, Unbounded, seq)
    }

    /// Records a normal-path fallback on the current thread.
    #[cold]
    #[inline(never)]
    fn count_fallback(&self, op: FallbackOp, reason: Fallback) {
        let count = ReadCount::fallback(op, reason);
        debug_assert!(count.is_some(), "a fallback of {op:?} for {reason:?}");
        if let Some(count) = count {
            self.hits.count(count);
        }
    }

    fn count_read(&self, range: &CachedRange, count: ReadCount) {
        self.hits.read(range.slot, &range.heat.hits, count);
    }

    /// Records an unfit pin attempt and returns true if the read should retry (sealed delta only).
    fn count_unfit(&self, op: PinOp, reason: Unfit, attempt: usize) -> bool {
        let retry = reason == Unfit::Sealed && attempt < PIN_ATTEMPTS;
        self.hits.count(if retry {
            ReadCount::retry(op, reason)
        } else {
            ReadCount::unusable(op, reason)
        });
        retry
    }

    /// Reads `key` at the last committed sequence number (or `at` for snapshots/transactions)
    /// without mutating shared cache lines on the fast path.
    pub(crate) fn get(&self, key: &[u8], oracle: &dyn Oracle, at: Option<u64>) -> CacheRead {
        for attempt in 1..=PIN_ATTEMPTS {
            let index = self.index.load();
            let Some(cached) = covering(&index, key) else {
                self.count_fallback(FallbackOp::Get, Fallback::Uncovered);
                return CacheRead::Uncovered;
            };
            let layers = cached.layers.load();
            let Some(chunk) = layers.chunk_of(key) else {
                self.count_fallback(FallbackOp::Get, Fallback::Loading);
                return CacheRead::Uncovered;
            };
            fail_point!(Arc::clone(&self.fp_registry), "range-cache-pin");
            let seq = read_seq(oracle, at);
            if let Err(reason) = Self::usable_from(&layers, chunk.base_seq, seq) {
                if self.count_unfit(PinOp::Get, reason, attempt) {
                    continue;
                }
                self.count_fallback(FallbackOp::Get, reason.into());
                return CacheRead::Unusable;
            }
            return match lookup(&layers, key, seq) {
                Ok(Found::Value(kv)) => {
                    self.count_read(cached, ReadCount::GetHit);
                    CacheRead::Answer(Some(kv))
                }
                Ok(Found::Absent) => {
                    self.count_read(cached, ReadCount::GetAbsent);
                    CacheRead::Answer(None)
                }
                Ok(Found::Merge) => {
                    self.count_read(cached, ReadCount::GetMerge);
                    CacheRead::Resolve(Pinned {
                        layers: arc_swap::Guard::into_inner(layers),
                        seq,
                    })
                }
                Err(e) => {
                    warn!("range cache read failed [error={:?}]", e);
                    self.request_removal(Target::Range(cached.clone()), RemoveReason::Error);
                    self.count_fallback(FallbackOp::Get, Fallback::Error);
                    CacheRead::Uncovered
                }
            };
        }
        // The last attempt returns: `count_unfit` never retries it.
        CacheRead::Unusable
    }

    /// Pins layers for a fully cached scan across one or more adjacent ranges at one sequence
    /// number.
    pub(crate) fn pin_scan(
        &self,
        range: &BytesRange,
        oracle: &dyn Oracle,
        at: Option<u64>,
    ) -> ScanPin {
        match self.try_pin_scan(range, oracle, at) {
            Some(Ok(pinned)) => ScanPin::Pinned(pinned),
            Some(Err(())) => ScanPin::Unusable,
            None => ScanPin::Uncovered,
        }
    }

    /// See [`RangeCache::pin_scan`]. Returns `None` if ready ranges do not hold all of `range`,
    /// and `Some(Err(()))` if they do but fail the sequence number rules, which it counts as the
    /// scan's fallback (the error carries no reason, so that the result keeps the plain layout of a
    /// `PinnedScan`, which the read copies out cheaply). A part is usable if
    /// the chunks that the scan meets in it are.
    fn try_pin_scan(
        &self,
        range: &BytesRange,
        oracle: &dyn Oracle,
        at: Option<u64>,
    ) -> Option<Result<PinnedScan, ()>> {
        if self.index.load().is_empty() {
            return None;
        }
        // The first key of the scan. `BytesRange` is never empty, so only an excluded start
        // needs a new key; the common cases borrow it, without an allocation.
        let excluded_start;
        let start: &[u8] = match RangeBounds::start_bound(range) {
            Unbounded => &[],
            Included(key) => key,
            Excluded(_) => {
                excluded_start = half_open(range)?.0;
                &excluded_start
            }
        };
        for attempt in 1..=PIN_ATTEMPTS {
            let index = self.index.load();
            let first = covering(&index, start)?;
            let mut parts = vec![first.layers.load_full()];
            let mut last = first;
            while !last.contains_range(range) {
                last = index.get(last.end.as_ref()?)?;
                parts.push(last.layers.load_full());
            }
            if parts.iter().any(|l| l.chunks.loading()) {
                return None;
            }
            fail_point!(Arc::clone(&self.fp_registry), "range-cache-pin");
            let seq = read_seq(oracle, at);
            // The scan meets its first part from its start, its last part up to its end, and the
            // parts between whole.
            let end = RangeBounds::end_bound(range).map(|key| key.as_ref());
            let last_part = parts.len() - 1;
            let usable = |i: usize, layers: &Layers| {
                let from = if i == 0 { Included(start) } else { Unbounded };
                let to = if i == last_part { end } else { Unbounded };
                Self::usable_for(layers, from, to, seq)
            };
            let unfit = parts
                .iter()
                .enumerate()
                .find_map(|(i, l)| usable(i, l).err());
            if let Some(first) = unfit {
                // One part that is too old makes every attempt fail.
                let too_old = first == Unfit::TooOld
                    || parts
                        .iter()
                        .enumerate()
                        .any(|(i, l)| usable(i, l) == Err(Unfit::TooOld));
                let reason = if too_old { Unfit::TooOld } else { first };
                if self.count_unfit(PinOp::Scan, reason, attempt) {
                    continue;
                }
                self.count_fallback(FallbackOp::Scan, reason.into());
                return Some(Err(()));
            }
            // Only the first range counts as read: most scans end in it.
            self.count_read(first, ReadCount::Scan);
            return Some(Ok(PinnedScan { parts, seq }));
        }
        // The last attempt returns: `count_unfit` never retries it.
        Some(Err(()))
    }

    /// Plans a mixed scan of cached parts and gaps at one sequence number. Returns the highest-
    /// priority fallback reason if no usable cached range intersects `range`.
    fn pin_mixed(
        &self,
        range: &BytesRange,
        oracle: &dyn Oracle,
        at: Option<u64>,
    ) -> Result<MixedPin, Fallback> {
        let (start, end) = half_open(range).ok_or(Fallback::Uncovered)?;
        let index = self.index.load();
        let intersecting: Vec<&Arc<CachedRange>> = covering(&index, &start)
            .into_iter()
            .chain(
                index
                    .range::<[u8], _>((Excluded(start.as_ref()), Unbounded))
                    .map(|(_, r)| r)
                    .take_while(|r| end.as_ref().is_none_or(|end| r.start < *end)),
            )
            .collect();
        if intersecting.is_empty() {
            return Err(Fallback::Uncovered);
        }
        let pinned: Vec<Arc<Layers>> = intersecting.iter().map(|r| r.layers.load_full()).collect();
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-pin");
        let seq = read_seq(oracle, at);

        let scan_end = RangeBounds::end_bound(range).cloned();
        let mut parts = Vec::with_capacity(2 * intersecting.len() + 1);
        let mut push = |kind, start: Bound<Bytes>, end: Bound<Bytes>| {
            if BytesRange::try_new(start.clone(), end.clone()).is_some() {
                parts.push(Part { kind, start, end });
            }
        };
        let mut cursor = Some(RangeBounds::start_bound(range).cloned());
        let mut first_cached: Option<&Arc<CachedRange>> = None;
        // Why no part is cached, if none is.
        let mut closest = Fallback::Uncovered;
        let mut gap = |reason: Fallback| {
            if reason.rank() < closest.rank() {
                closest = reason;
            }
            PartKind::Gap(reason)
        };
        for (cached, layers) in intersecting.iter().zip(pinned) {
            let Some(from) = cursor.take() else {
                break;
            };
            // The gap before this range, if any.
            let before = match &from {
                Unbounded => !cached.start.is_empty(),
                Included(key) | Excluded(key) => *key < cached.start,
            };
            let from = if before {
                push(
                    PartKind::Gap(Fallback::Uncovered),
                    from,
                    Excluded(cached.start.clone()),
                );
                Included(cached.start.clone())
            } else {
                from
            };
            let to = scan::min_end(&scan_end, &cached.end);
            let kind = if layers.chunks.loading() {
                gap(Fallback::Loading)
            } else {
                match Self::usable_for(&layers, as_slices(&from), as_slices(&to), seq) {
                    Ok(()) => {
                        first_cached.get_or_insert(*cached);
                        PartKind::Cached(layers)
                    }
                    Err(reason) => {
                        self.hits
                            .count(ReadCount::unusable(PinOp::ScanPart, reason));
                        gap(reason.into())
                    }
                }
            };
            push(kind, from, to);
            cursor = cached.end.clone().map(Included);
        }
        if let Some(from) = cursor {
            push(PartKind::Gap(Fallback::Uncovered), from, scan_end);
        }
        let first = first_cached.ok_or(closest)?.clone();
        Ok(MixedPin { parts, seq, first })
    }

    pub(crate) fn ranges(&self) -> Vec<CachedRangeInfo> {
        self.index.load().values().map(|r| r.info(self)).collect()
    }

    pub(crate) fn buffer_memory(&self) -> u64 {
        self.delta_memory.bytes()
    }

    /// Starts timing a normal-path read sample when automatic promotion is on.
    pub(crate) fn sample(&self) -> Option<ReadSample> {
        let auto = self.auto.as_ref()?;
        let one_in = auto.options.sample_one_in.max(1);
        if one_in > 1 && self.rand.rng().random_range(0..one_in) != 0 {
            return None;
        }
        Some(ReadSample {
            start: self.clock.now(),
        })
    }

    /// Records a sampled normal-path read of `key` and drops it if the promoter queue is full.
    pub(crate) fn record(&self, sample: ReadSample, key: &[u8]) {
        let Some(auto) = &self.auto else {
            return;
        };
        let now = self.clock.now();
        let elapsed_ns = (now - sample.start).num_nanoseconds().unwrap_or(i64::MAX);
        let cost_ns = (elapsed_ns.max(0) as u64).max(auto.options.min_read_cost.as_nanos() as u64);
        let message = PromoterMessage::Sample(auto::Sample {
            key: Bytes::copy_from_slice(key),
            cost_ns,
            at_ms: now.timestamp_millis(),
        });
        match auto.tx.try_send(message) {
            Ok(()) => auto.stats.samples_recorded.increment(1),
            Err(_) => auto.stats.samples_dropped.increment(1),
        }
    }

    /// Returns true if any ready or loading range covers `key`.
    fn covers(&self, key: &[u8]) -> bool {
        covering(&self.index.load(), key).is_some()
    }

    /// Returns the bounds of the uncovered gap around `key`.
    fn neighbours(&self, key: &[u8]) -> (Bytes, Option<Bytes>) {
        let index = self.index.load();
        let below = index
            .range::<[u8], _>((Unbounded, Included(key)))
            .next_back()
            .and_then(|(_, r)| r.end.clone())
            .unwrap_or_default();
        let above = index
            .range::<[u8], _>((Excluded(key), Unbounded))
            .next()
            .map(|(start, _)| start.clone());
        (below, above)
    }

    /// Total disk bytes of cached range files, counting shared files once.
    fn attached_bytes(&self) -> u64 {
        Footprint::of(self.index.load().values()).disk
    }

    /// Returns the highest read density among automatic ranges evicted to free `need` bytes,
    /// or `None` if automatic ranges alone cannot free enough space.
    fn victim_density(&self, need: u64) -> Option<f64> {
        let now = self.now_ms();
        let index = self.index.load();
        let mut ranges: Vec<(f64, u64)> = index
            .values()
            .filter(|r| r.auto)
            .map(|r| (self.density(r, now), r.file_bytes()))
            .filter(|(_, bytes)| *bytes > 0)
            .collect();
        ranges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut freed = 0;
        let mut densest: f64 = 0.0;
        for (density, bytes) in ranges {
            if freed >= need {
                break;
            }
            freed += bytes;
            densest = densest.max(density);
        }
        (freed >= need).then_some(densest)
    }

    /// Streams rows of a loading range into a base file, publishes it, and evicts if over budget.
    async fn load(
        &self,
        range: &Arc<CachedRange>,
        mut iter: DbIterator,
        file_seq: u64,
        check_open: &(dyn Fn() -> Result<(), SlateDBError> + Send + Sync),
    ) -> Result<(), SlateDBError> {
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-load");
        let mut writer = RangeFileWriter::create(
            &self.options.root_folder,
            &self.new_file_name(),
            self.options.tuning.page_size,
        )?;
        while let Some(entry) = iter.next_entry().await? {
            if entry.expire_ts.is_some() {
                return Err(SlateDBError::RangeCacheLoadRejected(
                    "the range holds rows with an expire time",
                ));
            }
            writer.push(&entry)?;
            if writer.rows().is_multiple_of(1024) {
                check_open()?;
            }
            if writer.estimated_len() > self.options.max_disk_bytes {
                return Err(SlateDBError::RangeCacheLoadRejected(
                    "the range is larger than the disk budget",
                ));
            }
        }
        let rows = writer.rows();
        let path = writer.finish()?;
        let file = Arc::new(RangeFile::open(
            path,
            self.options.page_source,
            self.files.clone(),
        )?);
        debug_assert_eq!(rows, file.rows());

        let split = self.attach_split(range, &file, file_seq).await?;
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-attach");
        let attached = self.attach_base(range, file, file_seq, split).await;
        if !attached.is_empty() {
            // Keep newly admitted automatic range(s) while evicting displaced ranges.
            let skip = if range.auto { attached } else { Vec::new() };
            self.evict(&skip).await;
        }
        Ok(())
    }

    /// Plans a zero-copy attach split if the loaded base exceeds twice `range_target_bytes`.
    async fn attach_split(
        &self,
        range: &Arc<CachedRange>,
        file: &Arc<RangeFile>,
        base_seq: u64,
    ) -> Result<Option<split::SplitPlan>, SlateDBError> {
        let target = self.options.tuning.range_target_bytes;
        if target == 0 {
            return Ok(None);
        }
        let chunks = one_chunk(file.clone(), base_seq);
        let (bytes, _) = split::base_share(&chunks, &range.start, range.end.as_ref());
        if bytes <= target.saturating_mul(2) {
            return Ok(None);
        }
        let pieces = split::attach_pieces(bytes, target);
        let cuts = split::cut_keys(&chunks, &range.start, range.end.as_ref(), pieces);
        if cuts.is_empty() {
            return Ok(None);
        }
        let layers = range.layers.load_full();
        let (start, end) = (range.start.clone(), range.end.clone());
        let rate = range.heat.last_rate();
        let plan = maintain::off_workers(move || {
            Ok(split::plan(&layers, &chunks, &start, end.as_ref(), &cuts)?)
        })
        .await?;
        Ok(plan.map(|plan| split::SplitPlan { rate, ..plan }))
    }
}

/// Returns the read sequence number, clamped to the last committed sequence number.
fn read_seq(oracle: &dyn Oracle, at: Option<u64>) -> u64 {
    let committed = oracle.last_committed_seq();
    at.map_or(committed, |at| at.min(committed))
}

/// Result of [`RangeCache::get`].
pub(crate) enum CacheRead {
    /// No ready range holds the key, or a cache read failed (`uncovered`, `loading`, or `error`).
    Uncovered,
    /// A ready range holds the key, but failed sequence checks (`too_old` or `sealed`).
    Unusable,
    /// Resolved value or definite miss at the read sequence number.
    Answer(Option<KeyValue>),
    /// Newest version is a merge operand; resolve through [`Pinned::get`].
    Resolve(Pinned),
}

/// The layers of one range, pinned at one sequence number.
pub(crate) struct Pinned {
    layers: Arc<Layers>,
    seq: u64,
}

impl Pinned {
    /// Reads `key` through the normal iterators. This resolves merge operands.
    pub(crate) async fn get(
        &self,
        key: &[u8],
        merge_operator: Option<MergeOperatorType>,
    ) -> Result<Option<KeyValue>, SlateDBError> {
        let key = Bytes::copy_from_slice(key);
        let point = BytesRange::new(Included(key.clone()), Included(key));
        let mut iter = stack_iterator(&self.layers, self.seq, point, merge_operator).await?;
        Ok(iter.next_entry().await?.map(KeyValue::from))
    }
}

/// The result of [`RangeCache::pin_scan`].
pub(crate) enum ScanPin {
    /// Pinned layers that serve the scan.
    Pinned(PinnedScan),
    /// Ready ranges do not hold all of the scan range.
    Uncovered,
    /// Ready ranges hold the scan range, but failed the sequence number rules, which the pin
    /// counted. See [`CacheRead::Unusable`].
    Unusable,
}

/// The parts of a mixed scan, with the layers of its cached parts pinned at one sequence
/// number. See [`RangeCache::pin_mixed`].
struct MixedPin {
    parts: Vec<Part>,
    seq: u64,
    /// The first cached part's range, which counts as read if the scan goes ahead.
    first: Arc<CachedRange>,
}

/// The layers of the adjacent cached ranges that hold a scan range, pinned at one sequence
/// number.
pub(crate) struct PinnedScan {
    parts: Vec<Arc<Layers>>,
    seq: u64,
}

impl PinnedScan {
    /// Returns an iterator over `range` at the pinned sequence number. See [`CachedScan`].
    pub(crate) async fn scan(
        self,
        range: BytesRange,
        merge_operator: Option<MergeOperatorType>,
    ) -> Result<DbIterator, SlateDBError> {
        let scan = CachedScan::new(self.parts, self.seq, &range, merge_operator)?;
        DbIterator::new_resolved(range, Box::new(scan), ReadTrace::none()).await
    }
}

/// What a point read found in the layers of a range at its sequence number.
enum Found {
    /// The newest version is a value.
    Value(KeyValue),
    /// The newest version is a tombstone, or the key has no version.
    Absent,
    /// The newest version is a merge operand.
    Merge,
}

/// Finds the newest version of `key` in `layers` at `seq`, checking buffers, unabsorbed runs,
/// and the chunk holding `key`.
fn lookup(layers: &Layers, key: &[u8], seq: u64) -> io::Result<Found> {
    let mut hash = None;
    let mut hash_of = || *hash.get_or_insert_with(|| hash_key(key));
    for delta in layers.buffers() {
        // Most reads of a range that takes few writes find empty buffers: one load each. A buffer
        // whose versions are all newer than `seq` holds nothing for the read.
        if delta.is_empty() || delta.lo() >= seq {
            continue;
        }
        if let Some(filter) = delta.filter() {
            if !filter.may_contain(hash_of()) {
                continue;
            }
        }
        if let Some(version) = delta.newest(key, seq) {
            debug_assert!(
                version.seq > layers.covered_to,
                "a buffered version is covered"
            );
            return Ok(match version.kind {
                Kind::Value => Found::Value(KeyValue::from(delta::to_row(&version))),
                Kind::Tombstone => Found::Absent,
                Kind::Merge => Found::Merge,
            });
        }
    }
    for run in &layers.runs {
        if run.lo >= seq || key < run.min_key.as_ref() || key > run.max_key.as_ref() {
            continue;
        }
        if !run.filter.may_contain(hash_of()) {
            continue;
        }
        // Runs are newest first and contiguous, so stop once the chunk already absorbed this run.
        if run.hi <= layers.chunk_of(key).map_or(0, |c| c.base_seq) {
            break;
        }
        if let Some(found) = run.file.read_version(key, seq)? {
            return Ok(match found {
                FileRead::Value(row) => Found::Value(KeyValue::from(row)),
                FileRead::Tombstone => Found::Absent,
                FileRead::Merge => Found::Merge,
            });
        }
    }
    let row = match layers.chunk_of(key) {
        Some(chunk) => chunk.file.get(key)?,
        None => None,
    };
    Ok(match row {
        Some(row) => match row.value {
            ValueDeletable::Value(_) => Found::Value(KeyValue::from(row)),
            ValueDeletable::Merge(_) => Found::Merge,
            ValueDeletable::Tombstone => Found::Absent,
        },
        None => Found::Absent,
    })
}

/// Opens a normal iterator stack over `layers` at `seq` for merge resolution, reading each
/// touched chunk and only the unabsorbed runs meeting it.
async fn stack_iterator(
    layers: &Layers,
    seq: u64,
    range: BytesRange,
    merge_operator: Option<MergeOperatorType>,
) -> Result<DbIterator, SlateDBError> {
    let mut sources: Vec<Box<dyn RowEntryIterator + 'static>> = layers
        .buffers()
        .map(|delta| {
            Box::new(DeltaIterator::new(delta.clone(), &range))
                as Box<dyn RowEntryIterator + 'static>
        })
        .collect();
    let hash = one_key(&range).map(hash_key);
    // The runs above `floor` that can hold a version of `part` at or below `seq`.
    let push_runs =
        |sources: &mut Vec<Box<dyn RowEntryIterator + 'static>>, floor: u64, part: &BytesRange| {
            for run in &layers.runs {
                if run.hi <= floor {
                    break;
                }
                if run.lo >= seq
                    || run.file.rows() == 0
                    || !run_meets(run, part)
                    || hash.is_some_and(|hash| !run.filter.may_contain(hash))
                {
                    continue;
                }
                sources.push(Box::new(RangeFileIterator::new(
                    run.file.clone(),
                    part.clone(),
                )));
            }
        };
    match &layers.chunks {
        // Layers without a base, which only tests read: every run counts.
        Chunks::Loading => push_runs(&mut sources, 0, &range),
        Chunks::One(only) => {
            push_runs(&mut sources, only.base_seq, &range);
            sources.push(Box::new(RangeFileIterator::new(
                only.file.clone(),
                range.clone(),
            )));
        }
        Chunks::Many(list) => {
            for (i, chunk) in list.chunks.iter().enumerate() {
                let Some(part) = range.intersect(&layers.chunk_span(i)) else {
                    continue;
                };
                push_runs(&mut sources, chunk.base_seq, &part);
                sources.push(Box::new(RangeFileIterator::new(chunk.file.clone(), part)));
            }
        }
    }
    DbIterator::new(
        range,
        None,
        sources,
        Box::new(EmptyIterator::new()),
        Some(seq),
        merge_operator,
        IterationOrder::Ascending,
        ReadTrace::none(),
    )
    .await
}

/// The key of `range` if it holds one key only: `[k, k]`, or `[k, k + 0x00)` as a scan's merge
/// resolution reads it.
fn one_key(range: &BytesRange) -> Option<&[u8]> {
    if let Some(key) = range.as_point() {
        return Some(key);
    }
    match (
        RangeBounds::start_bound(range),
        RangeBounds::end_bound(range),
    ) {
        (Included(start), Excluded(end))
            if end.len() == start.len() + 1 && end.starts_with(start) && end.ends_with(&[0]) =>
        {
            Some(start)
        }
        _ => None,
    }
}

/// Returns true if a non-empty run can hold a key of `range`, by its key bounds.
fn run_meets(run: &Run, range: &BytesRange) -> bool {
    let after_start = match RangeBounds::start_bound(range) {
        Unbounded => true,
        Included(start) => run.max_key >= start,
        Excluded(start) => run.max_key > start,
    };
    let before_end = match RangeBounds::end_bound(range) {
        Unbounded => true,
        Included(end) => run.min_key <= end,
        Excluded(end) => run.min_key < end,
    };
    after_start && before_end
}

/// Returns the first key a scan of `range` can return, for promotion sampling.
fn scan_start(range: &BytesRange) -> Bytes {
    match RangeBounds::start_bound(range) {
        Included(key) | Excluded(key) => key.clone(),
        Unbounded => Bytes::new(),
    }
}

/// Returns true if a point read with `options` can use the range cache.
///
/// Traced reads bypass the cache so tests can compare cached reads against the normal path.
pub(crate) fn can_serve_get(options: &ReadOptions) -> bool {
    get_options_fallback(options).is_none()
}

/// Returns the first option preventing a point read from using the range cache, if any.
fn get_options_fallback(options: &ReadOptions) -> Option<Fallback> {
    if !matches!(options.durability_filter, DurabilityLevel::Memory) {
        Some(Fallback::Remote)
    } else if options.dirty {
        Some(Fallback::Dirty)
    } else if options.filter_context.is_some() {
        Some(Fallback::Filter)
    } else if options.tracing_options.is_some() {
        Some(Fallback::Traced)
    } else {
        None
    }
}

/// Returns the first option preventing a scan from using the range cache, if any.
fn scan_options_fallback(options: &ScanOptions) -> Option<Fallback> {
    if !matches!(options.durability_filter, DurabilityLevel::Memory) {
        Some(Fallback::Remote)
    } else if options.dirty {
        Some(Fallback::Dirty)
    } else if !matches!(options.order, IterationOrder::Ascending) {
        Some(Fallback::Descending)
    } else if options.filter_context.is_some() {
        Some(Fallback::Filter)
    } else if options.tracing_options.is_some() {
        Some(Fallback::Traced)
    } else {
        None
    }
}

/// Normal-path read sampled for automatic promotion, paired with its key or scan start key.
pub(crate) struct SampledRead {
    sample: ReadSample,
    key: Bytes,
}

/// Routing result for [`DbInner::range_cache_get`].
pub(crate) enum GetRoute {
    /// Served by the range cache.
    Served(Result<Option<KeyValue>, SlateDBError>),
    /// Fall back to the normal read path, optionally recording a promotion sample.
    Normal(Option<SampledRead>),
}

/// Routing result for [`DbInner::range_cache_scan`].
pub(crate) enum ScanRoute {
    /// Served by the range cache (fully or via mixed scan).
    Served(Result<DbIterator, SlateDBError>),
    /// Fall back to the normal read path for this range.
    Normal(BytesRange, Option<SampledRead>),
}

impl DbInner {
    /// Serves a point read of `key` from the range cache if possible. `at` is the snapshot or
    /// transaction sequence number, or `None` for latest state.
    pub(crate) async fn range_cache_get(
        &self,
        key: &[u8],
        options: &ReadOptions,
        at: Option<u64>,
    ) -> GetRoute {
        let Some(cache) = &self.range_cache else {
            return GetRoute::Normal(None);
        };
        if let Some(reason) = get_options_fallback(options) {
            cache.count_fallback(FallbackOp::Get, reason);
            return GetRoute::Normal(None);
        }
        cache.hits.bind_db(&self.db_stats);
        match cache.get(key, self.oracle.as_ref(), at) {
            CacheRead::Uncovered => GetRoute::Normal(cache.sample().map(|sample| SampledRead {
                sample,
                key: Bytes::copy_from_slice(key),
            })),
            CacheRead::Unusable => GetRoute::Normal(None),
            CacheRead::Answer(kv) => GetRoute::Served(Ok(kv)),
            CacheRead::Resolve(pinned) => GetRoute::Served(
                pinned
                    .get(key, self.reader.read_merge_operator.clone())
                    .await,
            ),
        }
    }

    /// Serves a scan of `range` from the range cache if possible.
    pub(crate) async fn range_cache_scan(
        &self,
        range: BytesRange,
        options: &ScanOptions,
        prefix: Option<&Bytes>,
        at: Option<u64>,
    ) -> ScanRoute {
        let Some(cache) = &self.range_cache else {
            return ScanRoute::Normal(range, None);
        };
        if let Some(reason) = scan_options_fallback(options) {
            cache.count_fallback(FallbackOp::Scan, reason);
            return ScanRoute::Normal(range, None);
        }
        cache.hits.bind_db(&self.db_stats);
        match cache.pin_scan(&range, self.oracle.as_ref(), at) {
            ScanPin::Pinned(pinned) => ScanRoute::Served(
                pinned
                    .scan(range, self.reader.read_merge_operator.clone())
                    .await,
            ),
            ScanPin::Uncovered => {
                // Why the scan uses the normal read path, if no attempt plans a mixed scan.
                let mut reason = Fallback::Frozen;
                for _ in 0..PIN_ATTEMPTS {
                    // For latest-state scans, capture the state view before pinning so it retains
                    // all versions at or below the pinned sequence number unless a freeze occurs.
                    let latest_view = at.is_none().then(|| self.state.read().view());
                    let pin = match cache.pin_mixed(&range, self.oracle.as_ref(), at) {
                        Ok(pin) => pin,
                        Err(why) => {
                            reason = why;
                            break;
                        }
                    };
                    fail_point!(Arc::clone(&cache.fp_registry), "range-cache-mixed-pinned");
                    let view = match latest_view {
                        Some(view) => {
                            let frozen =
                                !Arc::ptr_eq(&view.memtable, self.state.read().memtable().table());
                            if frozen {
                                cache.hits.count(ReadCount::MixedScanRetry);
                                continue;
                            }
                            view
                        }
                        None => self.state.read().view(),
                    };
                    cache.count_read(&pin.first, ReadCount::MixedScan);
                    let gaps = GapSource {
                        reader: self.reader.clone(),
                        view,
                        options: options.clone(),
                        prefix: prefix.cloned(),
                        cache: cache.clone(),
                    };
                    let scan = MixedScan::new(
                        pin.parts,
                        pin.seq,
                        range.as_point().is_some(),
                        self.reader.read_merge_operator.clone(),
                        gaps,
                    );
                    return ScanRoute::Served(
                        DbIterator::new_resolved(range, Box::new(scan), ReadTrace::none()).await,
                    );
                }
                cache.count_fallback(FallbackOp::Scan, reason);
                let sample = cache.sample().map(|sample| SampledRead {
                    sample,
                    key: scan_start(&range),
                });
                ScanRoute::Normal(range, sample)
            }
            ScanPin::Unusable => ScanRoute::Normal(range, None),
        }
    }

    /// Records a sampled normal-path read after the read finishes.
    pub(crate) fn range_cache_record(&self, sampled: Option<SampledRead>) {
        if let (Some(cache), Some(sampled)) = (&self.range_cache, sampled) {
            cache.record(sampled.sample, &sampled.key);
        }
    }

    /// Loads uncovered parts of `range` into the range cache.
    async fn cache_range(self: &Arc<Self>, range: BytesRange) -> Result<(), SlateDBError> {
        self.cache_range_with(range, Origin::Explicit).await
    }

    /// Loads uncovered parts of `range` into the range cache for `origin`.
    pub(crate) async fn cache_range_with(
        self: &Arc<Self>,
        range: BytesRange,
        origin: Origin,
    ) -> Result<(), SlateDBError> {
        let cache = self
            .range_cache
            .clone()
            .ok_or(SlateDBError::RangeCacheNotConfigured)?;
        self.check_closed()?;
        let Some((start, end)) = half_open(&range) else {
            return Ok(());
        };
        cache.bind_writer(&self.write_notifier);
        let target = cache.options.tuning.range_target_bytes;
        let splits = if target > 0 {
            self.range_split_points(&start, end.as_ref(), target)
                .await?
        } else {
            Vec::new()
        };
        // Hold a retention snapshot before registering so flush and compaction preserve every
        // version at or below the registration boundary `W >= guard.seq()`.
        let guard = DbSnapshot::new(self.clone(), None);
        fail_point!(
            Arc::clone(&cache.fp_registry),
            "range-cache-register-publish"
        );
        let registered = cache.register(start, end, splits, origin).await?;
        debug_assert!(registered.seq >= guard.seq());
        fail_point!(
            Arc::clone(&cache.fp_registry),
            "range-cache-register-snapshot"
        );
        let options = cache.options.tuning.bulk_scan_options();
        let mut result = Ok(());
        for range in &registered.ranges {
            if result.is_err() {
                cache.remove_and_wait(range, RemoveReason::Error).await;
                continue;
            }
            let db_state = self.state.read().view();
            let iter = self
                .reader
                .scan_with_options(
                    range.bounds(),
                    &options,
                    ScanContext {
                        db_state: &db_state,
                        write_batch_iter: None,
                        max_seq: Some(registered.seq),
                        prefix: None,
                    },
                )
                .await;
            result = match iter {
                Ok(iter) => {
                    cache
                        .load(range, iter, registered.seq, &|| self.check_closed())
                        .await
                }
                Err(e) => Err(e),
            };
            if result.is_err() {
                cache.remove_and_wait(range, RemoveReason::Error).await;
            }
        }
        drop(guard);
        result
    }

    /// Computes split keys for `[start, end)` at roughly `target`-byte block boundaries of the
    /// largest sorted run. Returns empty when no sorted run exists or segments are enabled.
    async fn range_split_points(
        &self,
        start: &Bytes,
        end: Option<&Bytes>,
        target: u64,
    ) -> Result<Vec<Bytes>, SlateDBError> {
        let run = {
            let view = self.state.read().view();
            let core = view.state.core();
            if core.segment_extractor_name.is_some() {
                return Ok(Vec::new());
            }
            match core
                .tree
                .compacted
                .iter()
                .max_by_key(|sr| sr.estimate_size())
            {
                Some(run) => run.clone(),
                None => return Ok(Vec::new()),
            }
        };
        let views = run.sst_views();
        let inside = |k: &[u8]| k > start.as_ref() && end.is_none_or(|end| k < end.as_ref());
        let mut splits = Vec::new();
        let mut bytes = 0u64;
        for (i, view) in views.iter().enumerate() {
            // The SST covers keys from its start to the next one's start.
            let sst_start = view.compacted_effective_start_key();
            let below_end = end.is_none_or(|end| sst_start.as_ref() < end.as_ref());
            let above_start = views
                .get(i + 1)
                .is_none_or(|next| next.compacted_effective_start_key() > start);
            if !below_end || !above_start {
                continue;
            }
            let index = self
                .table_store
                .read_index(
                    &view.sst,
                    true,
                    Some(Bytes::new()),
                    &ReadTrace::none(),
                    None,
                )
                .await?;
            let metas = index.borrow().block_meta();
            let n = metas.len();
            let avg = view.sst.estimate_size() / (n.max(1) as u64);
            for b in 0..n {
                let first = metas.get(b).first_key().bytes();
                if bytes >= target && inside(first) {
                    splits.push(Bytes::copy_from_slice(first));
                    bytes = 0;
                }
                let next = (b + 1 < n).then(|| metas.get(b + 1));
                let in_range = next.is_none_or(|next| next.first_key().bytes() > start.as_ref())
                    && end.is_none_or(|end| first < end.as_ref());
                if in_range {
                    bytes += next.map_or(avg, |next| {
                        next.offset().saturating_sub(metas.get(b).offset())
                    });
                }
            }
        }
        Ok(splits)
    }

    fn range_cache_or_err(&self) -> Result<&Arc<RangeCache>, SlateDBError> {
        self.range_cache
            .as_ref()
            .ok_or(SlateDBError::RangeCacheNotConfigured)
    }
}

impl Db {
    /// Loads uncovered parts of `range` into the range cache.
    ///
    /// Concurrent writes go to the range's write buffer so no updates are lost. Committed point
    /// reads and ascending scans serve cached keys from local disk and memory; dirty reads and
    /// snapshots older than the base use the normal read path.
    ///
    /// ## Errors
    /// - [`crate::ErrorKind::Invalid`] if the range cache is disabled, the range contains expiring
    ///   rows, the range exceeds `max_disk_bytes`, or write buffers are near capacity.
    pub async fn cache_range<T>(&self, range: T) -> Result<(), crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        let start = range.start_bound().map(Bytes::copy_from_slice);
        let end = range.end_bound().map(Bytes::copy_from_slice);
        let Some(range) = BytesRange::try_new(start, end) else {
            return Ok(());
        };
        self.inner.cache_range(range).await.map_err(Into::into)
    }

    /// Queues removal of all cached ranges overlapping `range`.
    ///
    /// The batch writer applies the removal at the next batch or wake-up. Reads stay exact
    /// meanwhile, and subsequent [`Db::cache_range`] calls run after the removal.
    pub fn uncache_range<T>(&self, range: T) -> Result<(), crate::Error>
    where
        T: ByteRangeBounds,
    {
        let cache = self.inner.range_cache_or_err()?;
        let start = range.start_bound().map(Bytes::copy_from_slice);
        let end = range.end_bound().map(Bytes::copy_from_slice);
        if let Some(range) = BytesRange::try_new(start, end) {
            cache.bind_writer(&self.inner.write_notifier);
            cache.request_removal(Target::Overlapping(range), RemoveReason::Explicit);
        }
        Ok(())
    }

    /// Queues removal of all cached ranges. See [`Db::uncache_range`].
    pub fn clear_range_cache(&self) -> Result<(), crate::Error> {
        let cache = self.inner.range_cache_or_err()?;
        cache.bind_writer(&self.inner.write_notifier);
        cache.request_removal(Target::All, RemoveReason::Explicit);
        Ok(())
    }

    /// Waits until the batch writer applies all earlier range cache requests.
    #[doc(hidden)]
    pub async fn wait_for_range_cache(&self) -> Result<(), crate::Error> {
        let cache = self.inner.range_cache_or_err()?;
        cache.bind_writer(&self.inner.write_notifier);
        cache.barrier().await;
        Ok(())
    }

    /// Returns cached ranges in key order, or an empty vector if the range cache is disabled.
    pub fn cached_ranges(&self) -> Vec<CachedRangeInfo> {
        self.inner
            .range_cache
            .as_ref()
            .map(|cache| cache.ranges())
            .unwrap_or_default()
    }

    /// Returns exact memory bytes held by all live range cache write buffers, or `0` if disabled.
    #[doc(hidden)]
    pub fn range_cache_buffer_memory(&self) -> u64 {
        self.inner
            .range_cache
            .as_ref()
            .map_or(0, |cache| cache.buffer_memory())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound;

    fn b(s: &str) -> Bytes {
        Bytes::copy_from_slice(s.as_bytes())
    }

    fn index(ranges: &[(&str, Option<&str>)]) -> RangeIndex {
        ranges
            .iter()
            .enumerate()
            .map(|(slot, (start, end))| {
                (
                    b(start),
                    Arc::new(CachedRange::new(
                        b(start),
                        end.map(b),
                        Layers {
                            active: Arc::new(Delta::for_test(&DeltaMemory::detached(), 0)),
                            sealed: Vec::new(),
                            runs: Vec::new(),
                            view: None,
                            chunks: Chunks::Loading,
                            covered_to: 0,
                            end: end.map(b),
                        },
                        Origin::Explicit,
                        0,
                        slot as u32,
                    )),
                )
            })
            .collect()
    }

    fn parts(index: &RangeIndex, start: &str, end: Option<&str>) -> Vec<(Bytes, Option<Bytes>)> {
        uncovered(index, &b(start), end.map(b).as_ref())
    }

    #[test]
    fn should_find_uncovered_parts() {
        let empty = index(&[]);
        assert_eq!(parts(&empty, "a", Some("z")), vec![(b("a"), Some(b("z")))]);
        assert_eq!(parts(&empty, "", None), vec![(b(""), None)]);

        let idx = index(&[("c", Some("e")), ("g", Some("h")), ("x", None)]);
        assert_eq!(
            parts(&idx, "a", Some("z")),
            vec![
                (b("a"), Some(b("c"))),
                (b("e"), Some(b("g"))),
                (b("h"), Some(b("x"))),
            ]
        );
        // Starts inside a range.
        assert_eq!(parts(&idx, "d", Some("f")), vec![(b("e"), Some(b("f")))]);
        // Fully covered.
        assert!(parts(&idx, "c", Some("e")).is_empty());
        assert!(parts(&idx, "c", Some("d")).is_empty());
        assert!(parts(&idx, "y", None).is_empty());
        // Ends where a range starts.
        assert_eq!(parts(&idx, "e", Some("g")), vec![(b("e"), Some(b("g")))]);
        // A range that ends before the start does not count.
        assert_eq!(parts(&idx, "f", Some("g")), vec![(b("f"), Some(b("g")))]);
        // Unbounded end after the last bounded range.
        let idx = index(&[("c", Some("e"))]);
        assert_eq!(
            parts(&idx, "a", None),
            vec![(b("a"), Some(b("c"))), (b("e"), None)]
        );
    }

    #[test]
    fn should_find_the_covering_range() {
        let idx = index(&[("c", Some("e")), ("x", None)]);
        assert!(covering(&idx, b"b").is_none());
        assert_eq!(covering(&idx, b"c").unwrap().start, b("c"));
        assert_eq!(covering(&idx, b"d\xff").unwrap().start, b("c"));
        assert!(covering(&idx, b"e").is_none());
        assert_eq!(covering(&idx, b"zzz").unwrap().start, b("x"));
    }

    #[test]
    fn should_convert_ranges_to_half_open_ranges() {
        let r = |s: Bound<&str>, e: Bound<&str>| half_open(&BytesRange::new(s.map(b), e.map(b)));
        assert_eq!(r(Unbounded, Unbounded), Some((b(""), None)));
        assert_eq!(
            r(Included("a"), Excluded("c")),
            Some((b("a"), Some(b("c"))))
        );
        assert_eq!(
            r(Excluded("a"), Included("c")),
            Some((Bytes::from_static(b"a\0"), Some(Bytes::from_static(b"c\0"))))
        );
    }

    #[test]
    fn should_check_scan_containment() {
        let idx = index(&[("c", Some("e"))]);
        let range = &idx[&b("c")];
        let r = |s: Bound<&str>, e: Bound<&str>| BytesRange::new(s.map(b), e.map(b));
        assert!(range.contains_range(&r(Included("c"), Excluded("e"))));
        assert!(range.contains_range(&r(Included("c"), Included("d"))));
        assert!(!range.contains_range(&r(Included("c"), Included("e"))));
        assert!(!range.contains_range(&r(Included("c"), Unbounded)));
    }

    /// Reads through a database with a range cache, checked against the cache's counters.
    mod reads {
        use super::super::*;
        use crate::db_stats::REQUEST_COUNT;
        use crate::object_store::memory::InMemory;
        use slatedb_common::clock::MockSystemClock;
        use slatedb_common::metrics::{lookup_metric_with_labels, DefaultMetricsRecorder};

        struct TestCache {
            db: Db,
            recorder: Arc<DefaultMetricsRecorder>,
            _dir: tempfile::TempDir,
        }

        impl TestCache {
            fn cache(&self) -> &Arc<RangeCache> {
                self.db.inner.range_cache.as_ref().unwrap()
            }

            fn range(&self, start: &Bytes) -> Arc<CachedRange> {
                self.cache().index.load().get(start).unwrap().clone()
            }

            fn metric(&self, name: &str, labels: &[(&str, &str)]) -> i64 {
                lookup_metric_with_labels(&self.recorder, name, labels).unwrap_or(0)
            }

            /// Hits, absent reads, scans, and the database's get and scan requests.
            fn read_metrics(&self) -> [i64; 5] {
                [
                    self.metric(stats::GET_COUNT, &[("result", "hit")]),
                    self.metric(stats::GET_COUNT, &[("result", "absent")]),
                    self.metric(stats::SCAN_COUNT, &[]),
                    self.metric(REQUEST_COUNT, &[("op", "get")]),
                    self.metric(REQUEST_COUNT, &[("op", "scan")]),
                ]
            }

            fn unfit(&self, name: &str, op: &str, reason: &str) -> i64 {
                self.metric(name, &[("op", op), ("reason", reason)])
            }
        }

        async fn open(
            tune: impl FnOnce(&mut RangeCacheOptions),
            clock: Option<Arc<MockSystemClock>>,
        ) -> TestCache {
            let dir = tempfile::tempdir().unwrap();
            let mut options = RangeCacheOptions::new(dir.path());
            tune(&mut options);
            let recorder = Arc::new(DefaultMetricsRecorder::new());
            let mut builder = Db::builder("range_cache_reads", Arc::new(InMemory::new()))
                .with_range_cache(options)
                .with_metrics_recorder(recorder.clone());
            if let Some(clock) = clock {
                builder = builder.with_system_clock(clock);
            }
            TestCache {
                db: builder.build().await.unwrap(),
                recorder,
                _dir: dir,
            }
        }

        fn key(i: usize) -> Bytes {
            Bytes::from(format!("key{i:05}"))
        }

        /// Puts the even keys below `n`.
        async fn put_even(db: &Db, n: usize) {
            for i in (0..n).step_by(2) {
                db.put(key(i), format!("value-{i}")).await.unwrap();
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn should_count_reads_only_on_their_thread_until_a_fold() {
            let t = open(
                |o| {
                    o.tuning.stats_fold_events = 1_000_000;
                    o.tuning.stats_fold_interval = Duration::from_secs(3600);
                },
                None,
            )
            .await;
            put_even(&t.db, 200).await;
            t.db.cache_range(key(0)..key(200)).await.unwrap();
            let range = t.range(&key(0));
            let before = t.read_metrics();
            let last_access = range.heat.hits.last_access();
            let clock = t.cache().hits.access_clock().load(Ordering::Relaxed);
            assert!(last_access < clock);

            let handle = tokio::runtime::Handle::current();
            std::thread::scope(|s| {
                let (main, worker) = hits::Turn::pair();
                s.spawn(|| {
                    let worker = worker;
                    handle.block_on(async {
                        for i in 0..400 {
                            t.db.get(key(i % 200)).await.unwrap();
                        }
                        for _ in 0..20 {
                            t.db.scan(key(10)..key(20)).await.unwrap();
                        }
                    });
                    worker.go();
                    worker.wait();
                    t.cache().hits.fold_current_thread();
                    worker.go();
                });
                main.wait();
                assert_eq!(t.read_metrics(), before);
                assert_eq!(range.heat.reads(), 0);
                assert_eq!(range.heat.hits.last_access(), last_access);
                main.go();
                main.wait();
                let [hits, absent, scans, gets, scan_requests] = before;
                assert_eq!(
                    t.read_metrics(),
                    [
                        hits + 200,
                        absent + 200,
                        scans + 20,
                        gets + 400,
                        scan_requests + 20
                    ]
                );
                assert_eq!(range.heat.hits.last_access(), clock);
                // The range's own counts, gets and scans apart.
                let info = &t.db.cached_ranges()[0];
                assert_eq!((info.gets, info.scans), (400, 20));
            });
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn should_converge_to_the_true_counts_after_threads_exit() {
            let t = open(|_| {}, None).await;
            put_even(&t.db, 200).await;
            t.db.cache_range(key(0)..key(200)).await.unwrap();
            let range = t.range(&key(0));
            let before = t.read_metrics();
            let handle = tokio::runtime::Handle::current();
            std::thread::scope(|s| {
                let threads: Vec<_> = (0..4)
                    .map(|_| {
                        s.spawn(|| {
                            handle.block_on(async {
                                for i in 0..5_000 {
                                    t.db.get(key(i % 200)).await.unwrap();
                                }
                            })
                        })
                    })
                    .collect();
                for thread in threads {
                    thread.join().unwrap();
                }
            });
            let [hits, absent, scans, gets, scan_requests] = before;
            assert_eq!(
                t.read_metrics(),
                [
                    hits + 10_000,
                    absent + 10_000,
                    scans,
                    gets + 20_000,
                    scan_requests
                ]
            );
            assert_eq!(range.heat.reads(), 20_000);
        }

        #[tokio::test]
        async fn should_count_fallbacks_by_reason() {
            let t = open(|_| {}, None).await;
            put_even(&t.db, 200).await;
            let old = t.db.snapshot().unwrap();
            // A write after the snapshot, so that the load reads a newer state.
            t.db.put(key(300), "later").await.unwrap();
            t.db.cache_range(key(50)..key(150)).await.unwrap();

            // The snapshot is older than the range's file: a point read and a covered scan fall
            // back at once, since a retry cannot succeed, and a mixed scan makes the cached
            // part a gap.
            assert_eq!(
                old.get(key(60)).await.unwrap(),
                Some(Bytes::from("value-60"))
            );
            old.scan(key(60)..key(70)).await.unwrap();
            old.scan(key(40)..key(70)).await.unwrap();

            // A freeze or a removal that a reader races seals the active delta below the
            // reader's sequence number. Store the seal directly to make every attempt see it.
            let range = t.range(&key(50));
            range
                .layers
                .load()
                .active
                .sealed_at()
                .store(0, Ordering::SeqCst);
            assert_eq!(
                t.db.get(key(60)).await.unwrap(),
                Some(Bytes::from("value-60"))
            );
            t.db.scan(key(60)..key(70)).await.unwrap();
            t.db.scan(key(40)..key(70)).await.unwrap();

            t.cache().hits.fold_current_thread();
            for (reason, retries) in [("too_old", 0), ("sealed", 2)] {
                assert_eq!(
                    t.unfit(stats::PIN_RETRY_COUNT, "get", reason),
                    retries,
                    "{reason}"
                );
                assert_eq!(
                    t.unfit(stats::PIN_RETRY_COUNT, "scan", reason),
                    retries,
                    "{reason}"
                );
                assert_eq!(t.unfit(stats::UNUSABLE_COUNT, "get", reason), 1, "{reason}");
                assert_eq!(
                    t.unfit(stats::UNUSABLE_COUNT, "scan", reason),
                    1,
                    "{reason}"
                );
                assert_eq!(
                    t.unfit(stats::UNUSABLE_COUNT, "scan_part", reason),
                    1,
                    "{reason}"
                );
                // Each read used the normal path once for its reason: the mixed scan too, as no
                // range served a part of it.
                assert_eq!(t.unfit(stats::FALLBACK_COUNT, "get", reason), 1, "{reason}");
                assert_eq!(
                    t.unfit(stats::FALLBACK_COUNT, "scan", reason),
                    2,
                    "{reason}"
                );
                assert_eq!(t.unfit(stats::FALLBACK_COUNT, "gap", reason), 0, "{reason}");
            }
            // None of them counts as served.
            assert_eq!(t.read_metrics()[..3], [0, 0, 0]);
        }

        #[tokio::test]
        async fn should_evict_the_least_recently_read_range_by_folded_reads() {
            let t = open(
                |o| {
                    o.tuning.page_size = 4096;
                    o.max_disk_bytes = 9000;
                    o.tuning.stats_fold_interval = Duration::from_secs(3600);
                },
                None,
            )
            .await;
            for i in 0..300 {
                t.db.put(key(i), format!("value-{i}")).await.unwrap();
            }
            t.db.cache_range(key(0)..key(50)).await.unwrap();
            t.db.cache_range(key(100)..key(150)).await.unwrap();
            // Enough reads of the first range for a fold at the default fold size. Until a fold,
            // its last access is older than the second range's.
            let fold = RangeCacheTuning::default().stats_fold_events;
            let first = t.range(&key(0));
            for _ in 0..fold {
                t.db.get(key(10)).await.unwrap();
            }
            assert_eq!(first.heat.reads(), u64::from(fold));
            t.db.cache_range(key(200)..key(250)).await.unwrap();
            let starts: Vec<_> = t.db.cached_ranges().into_iter().map(|r| r.start).collect();
            assert_eq!(starts, vec![key(0), key(200)]);
        }

        #[tokio::test]
        async fn should_evict_automatic_ranges_by_folded_read_density() {
            let clock = Arc::new(MockSystemClock::new());
            let t = open(
                |o| {
                    o.tuning.page_size = 4096;
                    o.max_disk_bytes = 9000;
                    o.tuning.stats_fold_interval = Duration::from_secs(3600);
                    o.auto_promote = Some(AutoPromoteOptions {
                        heat_half_life: Duration::from_secs(1),
                        ..AutoPromoteOptions::default()
                    });
                },
                Some(clock.clone()),
            )
            .await;
            for i in 0..300 {
                t.db.put(key(i), format!("value-{i}")).await.unwrap();
            }
            let origin = Origin::Auto {
                read_cost_ns: 1000,
                read_rate: 0.0,
            };
            let load = |start: usize| {
                let range = BytesRange::new(Included(key(start)), Excluded(key(start + 50)));
                t.db.inner.cache_range_with(range, origin)
            };
            load(0).await.unwrap();
            load(100).await.unwrap();
            let fold = RangeCacheTuning::default().stats_fold_events;
            for _ in 0..fold {
                t.db.get(key(10)).await.unwrap();
            }
            clock.advance(Duration::from_secs(1)).await;
            let (first, second) = (t.range(&key(0)), t.range(&key(100)));
            let now = t.cache().now_ms();
            assert!(t.cache().density(&first, now) > 0.0);
            assert_eq!(t.cache().density(&second, now), 0.0);
            load(200).await.unwrap();
            let starts: Vec<_> = t.db.cached_ranges().into_iter().map(|r| r.start).collect();
            assert_eq!(starts, vec![key(0), key(200)]);
        }
    }
}
