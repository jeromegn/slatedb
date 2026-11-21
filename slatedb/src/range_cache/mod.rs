//! The range cache keeps a local copy of selected key ranges and serves reads from it.
//!
//! See RFC 0040 for the design. In short:
//!
//! - A *cached range* is a half-open key range `[start, end)`. The cache knows every live key
//!   in a cached range, so a point read in it is a definite hit or a definite miss, and a scan
//!   in it needs no other source.
//! - A cached range has one *layer set* (design 01 section 4.3), newest first: the *active
//!   buffer*, which the writer appends to; *sealed buffers*, which wait for their flush; up to
//!   [`MAX_RUNS`] *runs*, each a file of every version of the range in an interval of sequence
//!   numbers `(lo, hi]`; and a *base*, a file of the state of the range at `base_seq`. The runs
//!   cover `(base_seq, covered_to]` without a gap, and the buffers hold every version above
//!   `covered_to`. A buffer is a sorted map that only the writer appends to and that reads search
//!   without a shared write (see the `buffer` and `delta` modules).
//! - The batch writer applies each write to the active buffer after it puts the write in the
//!   memtable and before it advances the last committed sequence number. Reads at the latest
//!   committed state read the layers with that sequence number as the bound, so a write becomes
//!   visible in the cache and in the memtable at the same time. Reads in a snapshot or a
//!   transaction use the snapshot's sequence number as the bound instead: buffers and runs keep
//!   every version.
//! - The writer seals the active buffer when it reaches `range_seal_bytes`. The maintenance task
//!   (the `maintain` module) flushes the oldest sealed buffer of a range to a run, and merges
//!   the base with the runs into a new base when the runs grow (a *major merge*). The memory of
//!   all buffers has a hard cap: a write that does not fit removes its range instead of waiting.
//! - Only the batch writer changes the set of ranges and the layers of a range. Loads, flushes,
//!   merges and removals send it requests, which it applies between batches (see the `writer`
//!   module). Neither reads nor writes take a lock.
//!
//! # Sequence number rules
//!
//! A read loads the layers of a range, then the last committed sequence number. Its sequence
//! number `seq` is that number, or for a read in a snapshot or a transaction, the smaller of
//! that number and the snapshot's. It uses the layers only if both rules hold:
//!
//! - `seq >= layers.base_seq`. The base holds the state at `base_seq`, without older versions.
//! - `seq <= layers.active.sealed_at`. When the cache stops applying writes to the active buffer
//!   (a seal or a removal), it seals the buffer at the last sequence number that it applied.
//!   The buffer then holds every version above its `lo` up to that number and none after it.
//!   The word is on the buffer, not on the layers, so a read that holds an earlier set of layers
//!   with the same active buffer sees the seal too.
//!
//! If the first rule fails, the read is in a snapshot older than the base and uses the normal
//! read path. If the second fails, the read tries again, a few times, then uses the normal read
//! path. The first rule keeps versions that the base folded out of reach. The second rule makes
//! the answer exactly the state at `seq`, which is the answer of the normal read path at `seq`.
//!
//! # Automatic promotion
//!
//! With [`RangeCacheOptions::auto_promote`], a background task chooses ranges from sampled
//! normal-path reads (see the `auto` module). This module keeps the parts that the cache core
//! needs: the read heat of each range, eviction by read density, and the payback check of
//! rebuilds. The promoter keeps the memory of removed ranges and failed promotions.
//!
//! # Read bookkeeping
//!
//! A read that the cache serves writes no memory that other reads write. It counts its result
//! and the range that it used on its own thread, and the thread adds its counts to the metrics
//! and to the ranges' read counts and last accesses every few thousand reads or about every
//! 10 ms (see the `hits` module). Eviction and automatic promotion use those folded values.

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
pub mod stats;
mod writer;

use std::collections::BTreeMap;
use std::io;
use std::ops::Bound::{Excluded, Included, Unbounded};
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
    FileBudget, FileRead, RangeFile, RangeFileIterator, RangeFileWriter, FILE_EXTENSION,
    TEMP_FILE_EXTENSION,
};
use filter::{hash_key, Filter};
use hits::{HitStats, RangeHits};
pub(crate) use maintain::{Maintainer, RANGE_CACHE_MAINTENANCE_TASK_NAME};
use mixed::{GapSource, MixedScan, Part, PartKind};
use scan::CachedScan;
use stats::{AutoStats, PinOp, RangeCacheStats, ReadCount, RemoveReason, Unfit};
use writer::{Padded, Target, WriterQueue};

/// The number of times a read tries to pin a range before it uses the normal read path.
const PIN_ATTEMPTS: usize = 3;

/// The most runs that a range holds (design 01's `D_hard`). While a range holds this many, its
/// flushes wait for a major merge.
pub(crate) const MAX_RUNS: usize = 4;

/// Options for the range cache. See [`crate::DbBuilder::with_range_cache`].
///
/// The range cache is experimental and its API may change. Create the options with
/// [`RangeCacheOptions::new`] and set the fields to change.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RangeCacheOptions {
    /// The directory for range cache files. The cache owns this directory. It removes old
    /// range cache files from it when the database opens.
    pub root_folder: PathBuf,
    /// The maximum size of the range cache files of cached ranges. When the files would grow
    /// past it, the cache removes ranges until they fit: the least recently read ranges, or,
    /// with automatic promotion, the automatic ranges with the lowest read density first and
    /// explicitly cached ranges last. The cache rejects a load that is larger than this budget.
    pub max_disk_bytes: u64,
    /// How the cache reads range cache files.
    pub page_source: RangeCachePageSource,
    /// Automatic promotion. `None`, the default, turns it off: the cache then holds only the
    /// ranges that the application loads.
    pub auto_promote: Option<AutoPromoteOptions>,
    /// Internal tuning. See [`RangeCacheTuning`].
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

/// Internal tuning of the range cache, for tests and benchmarks.
///
/// This is not part of the stable API. Its fields can change or disappear in any release.
/// The promotion fields apply only when [`RangeCacheOptions::auto_promote`] is set.
#[doc(hidden)]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RangeCacheTuning {
    /// The page size of range cache files. It must be a power of two from 512 to 65536.
    pub page_size: usize,
    /// The smallest run bytes that cause a major merge (a rebuild of the base). A major merge
    /// starts when the run files of a range are larger than this value and larger than `f` times
    /// its base, where `f = sqrt(2r / B)` for the range's average flush `r` and base `B`, between
    /// 0.05 and 1 (design 01 sections 7.5 and 7.9).
    pub min_rebuild_delta_bytes: usize,
    /// The chunk bytes at which the writer seals a range's active write buffer, so that the
    /// maintenance task can flush it to a run.
    pub range_seal_bytes: usize,
    /// The hard cap on the memory of all write buffers: their chunks and filters, from their
    /// allocation until their last holder drops them. A write that would pass it removes its
    /// range from the cache; the write itself never waits. Above seven eighths of the cap the
    /// cache rejects loads, and when it removes a range at the cap, it also removes the ranges
    /// with the largest buffers until the memory would fall below three quarters of it. Above
    /// half of it, the maintenance task seals the largest active buffers early.
    pub max_recent_bytes: u64,
    /// The smallest active buffer that the maintenance task seals early when the memory of the
    /// buffers passes half of `max_recent_bytes`.
    pub min_seal_bytes: usize,
    /// The chunk bytes from which a write buffer has a filter of its keys, so that a read of a
    /// key that the buffer does not hold skips it after one probe. `usize::MAX` turns buffer
    /// filters off.
    pub buffer_filter_bytes: usize,
    /// The size of a range that `cache_range` registers: it splits a larger request at block
    /// boundaries of the largest sorted run, so that a major merge rewrites at most about this
    /// much. 0 turns splitting off.
    pub range_target_bytes: u64,
    /// The limit on the heap metadata of the cached ranges: the fence indexes and filters of
    /// their files and their layer sets. When they pass it, the cache evicts ranges as it does
    /// for the disk budget. `None`, the default, means a sixteenth of `max_disk_bytes`, and at
    /// least 64 MiB. Fence indexes and filters take about 2% of the files' bytes (inferred from
    /// 4 KiB pages and 10 bits per key), so the default rarely binds before the disk budget.
    pub max_meta_bytes: Option<u64>,
    /// How often the promoter looks for candidates.
    pub evaluate_interval: Duration,
    /// The most candidates that one evaluation forms or re-evaluates. Each new candidate costs
    /// an SST index read, or a bounded scan of about `target_range_bytes`.
    pub max_candidates_per_evaluation: usize,
    /// The sampled reads that a candidate needs while the disk budget is at most half full.
    pub min_evidence: u32,
    /// The sampled reads that a candidate needs when the disk budget is full. The threshold rises
    /// linearly from `min_evidence` as the budget fills from half to full.
    pub full_evidence: u32,
    /// The sampled reads of a candidate must span at least this long. A range that one pass
    /// reads once never qualifies, because caching it would not pay.
    pub min_reuse_span: Duration,
    /// A candidate that needs eviction must have this many times the read density (normal-path
    /// cost per second per byte) of each automatic range it would evict.
    pub admission_margin: f64,
    /// How long the cache remembers a removed automatic range. A candidate that overlaps it
    /// counts only samples taken after the removal.
    pub ghost_ttl: Duration,
    /// How long the promoter remembers a rejected candidate.
    pub candidate_ttl: Duration,
    /// How often a remembered candidate can be evaluated again.
    pub candidate_recheck: Duration,
    /// The wait after a failed promotion (a range removed because its reads did not pay for a
    /// rebuild, or by an expiring write or an error). It doubles with each failure.
    pub backoff_base: Duration,
    /// The samples that the reservoir keeps. The reservoir keeps four heat half-lives of
    /// samples.
    pub reservoir_capacity: usize,
    /// The capacity of the queue from the read path to the promoter. The read path drops samples
    /// when it is full.
    pub queue_capacity: usize,
    /// The smallest cost of a sampled read. It keeps fast reads from counting as free.
    pub min_read_cost: Duration,
    /// The estimated cost of rewriting one byte of a range file in a rebuild, in nanoseconds.
    pub rebuild_cost_ns_per_byte: f64,
    /// The target size of one object store read when the cache loads a range, and when the
    /// promoter scans to form a candidate. Each read fetches the fewest consecutive blocks of an
    /// SST that hold at least this many bytes. See
    /// [`crate::config::ScanOptions::read_ahead_bytes`].
    pub load_fetch_bytes: usize,
    /// The most reads in flight for each SST that a load reads. A load reads all the sorted
    /// runs and L0 SSTs that overlap its range at the same time, so the bytes in flight are at
    /// most `load_fetch_bytes * load_fetch_tasks` for each of them. See
    /// [`crate::config::ScanOptions::max_fetch_tasks`].
    pub load_fetch_tasks: usize,
    /// A thread adds the counts of the reads that the cache served to the shared counters after
    /// this many counted events. The shared counters are the read-path metrics, the database's
    /// request counters, and the read counts and last accesses of ranges, which eviction and
    /// automatic promotion use. 1 updates them at each read. 0 counts as 1.
    pub stats_fold_events: u32,
    /// A thread also adds its counts when this long has passed since it last did. It checks the
    /// time at the 1st, 2nd, 4th, 8th and so on counted event after it last added them, so a
    /// thread that keeps reading adds its counts within about twice this time.
    pub stats_fold_interval: Duration,
}

impl Default for RangeCacheTuning {
    fn default() -> Self {
        Self {
            page_size: 4096,
            min_rebuild_delta_bytes: 1024 * 1024,
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
            // The values of the compaction worker, which reads SSTs in bulk for the same
            // reason. See `CompactionWorkerOptions::bytes_to_fetch`.
            load_fetch_bytes: 2 * 1024 * 1024,
            load_fetch_tasks: 4,
            stats_fold_events: 4096,
            stats_fold_interval: Duration::from_millis(10),
        }
    }
}

impl RangeCacheTuning {
    /// How the cache's write buffers size themselves.
    fn delta_shape(&self) -> DeltaShape {
        DeltaShape {
            seal_bytes: self.range_seal_bytes as u64,
            filter_from: u64::try_from(self.buffer_filter_bytes).unwrap_or(u64::MAX),
        }
    }

    /// The limit on the heap metadata of the cached ranges.
    fn meta_limit(&self, max_disk_bytes: u64) -> u64 {
        self.max_meta_bytes
            .unwrap_or((max_disk_bytes / 16).max(64 * 1024 * 1024))
    }

    /// The memory of the write buffers above which the cache rejects loads.
    fn high_water(&self) -> u64 {
        self.max_recent_bytes / 8 * 7
    }

    /// The memory of the write buffers to which removals at the cap bring it back.
    fn low_water(&self) -> u64 {
        self.max_recent_bytes / 4 * 3
    }

    /// The memory of the write buffers above which the maintenance task seals the largest active
    /// buffers early.
    fn pressure(&self) -> u64 {
        self.max_recent_bytes / 2
    }

    /// The scan options of the bulk reads of the cache: loads and candidate scans. They do not
    /// put blocks in the block cache.
    fn bulk_scan_options(&self) -> ScanOptions {
        ScanOptions {
            read_ahead_bytes: self.load_fetch_bytes,
            max_fetch_tasks: self.load_fetch_tasks,
            cache_blocks: false,
            ..ScanOptions::default()
        }
    }
}

/// The state of one cached range. See [`Db::cached_ranges`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CachedRangeInfo {
    /// The first key of the range.
    pub start: Bytes,
    /// The key after the range, or `None` if the range has no end.
    pub end: Option<Bytes>,
    /// True while the cache loads the range. A loading range serves no reads.
    pub loading: bool,
    /// True while the cache writes the range's recent writes to disk: a sealed write buffer waits
    /// for its flush, or a merge, minor or major, is due or runs.
    pub rebuilding: bool,
    /// The size of the range's base file on disk.
    pub file_bytes: u64,
    /// The number of rows in the range's base file on disk.
    pub file_rows: u64,
    /// The estimated size of the recent writes to the range that the cache holds in memory and
    /// has not yet written to a run.
    pub delta_bytes: u64,
    /// True if automatic promotion cached the range.
    pub automatic: bool,
    /// The number of runs of the range: files of its versions above the base.
    pub runs: usize,
    /// The size of the range's runs on disk.
    pub run_bytes: u64,
    /// The number of sealed write buffers that wait for their flush to a run.
    pub sealed_buffers: usize,
    /// The memory of the range's write buffers, active and sealed: their chunks and filters.
    pub buffer_bytes: u64,
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

/// The layer set of a cached range (design 01 section 4.3). The cache replaces this value as a
/// whole, and only the writer stores it.
///
/// Every scan increments and decrements the reference count of the `Arc` that holds the layers
/// (see [`RangeCache::pin_scan`]). The alignment keeps that count off the cache line of the fields
/// that every read loads, so concurrent reads of the range do not miss on it. Apple silicon has
/// 128-byte lines.
#[repr(align(128))]
pub(crate) struct Layers {
    /// The buffer that the writer appends to. Its `sealed_at` is the readers' validity word.
    active: Arc<Delta>,
    /// Sealed buffers that wait for their flush, newest first. Each one's `lo` is the seal of the
    /// next one, and the oldest one's `lo` is `covered_to`.
    sealed: Vec<Arc<Delta>>,
    /// Runs, newest first, at most [`MAX_RUNS`]: run `i` covers `(lo_i, hi_i]` with
    /// `hi_{i+1} = lo_i`, and the oldest one's `lo` is `base_seq`.
    runs: Vec<Arc<Run>>,
    /// `None` while the range loads.
    base: Option<Arc<RangeFile>>,
    /// The base holds the state of the range at this sequence number.
    base_seq: u64,
    /// The base and the runs hold every version of the range at or below this sequence number:
    /// the newest run's `hi`, or `base_seq`. The buffers hold every version above it.
    covered_to: u64,
    /// The end of the range, so that a scan can walk adjacent ranges with one reference to
    /// each.
    end: Option<Bytes>,
}

impl Layers {
    /// The buffers, newest first: the active one, then the sealed ones.
    fn buffers(&self) -> impl Iterator<Item = &Arc<Delta>> {
        std::iter::once(&self.active).chain(self.sealed.iter())
    }

    /// Checks the layer set's invariants (design 01 section 10.1, I4, I5 and I7) in debug builds.
    fn check(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        assert!(
            self.runs.len() <= MAX_RUNS,
            "a range holds {} runs",
            self.runs.len()
        );
        let mut to = self.covered_to;
        for run in &self.runs {
            assert_eq!(run.hi, to, "runs are not contiguous");
            assert!(run.lo < run.hi || run.lo == run.hi && run.file.rows() == 0);
            to = run.lo;
        }
        assert_eq!(to, self.base_seq, "the runs do not reach the base");
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
}

/// Who asked for a range.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Origin {
    /// The application, through `cache_range`.
    Explicit,
    /// Automatic promotion.
    Auto {
        /// The estimated normal-path cost of one read of the range, in nanoseconds.
        read_cost_ns: u64,
        /// The estimated reads per second of the range when the promoter admitted it.
        read_rate: f64,
    },
}

/// The reads and the read rate of a cached range, for eviction and automatic promotion.
struct Heat {
    /// The reads that the range served, up to the last fold of each thread, and its last
    /// access. See the `hits` module.
    hits: Arc<RangeHits>,
    rate: Mutex<HeatRate>,
}

struct HeatRate {
    reads: u64,
    at_ms: i64,
    /// Reads per second, decayed exponentially.
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

    fn reads(&self) -> u64 {
        self.hits.reads()
    }

    /// Returns the decayed reads per second at `now_ms`, with half-life `half_life_ms`.
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
    /// A small number that no other indexed range has. A removed range's slot goes to a later
    /// range. Thread blocks of the `hits` module keep the counts of ranges by slot.
    slot: u32,
    /// True if automatic promotion cached the range.
    auto: bool,
    /// The reads of the range and its last access (the value of the cache's access clock at the
    /// last fold that counted a read of it), and its read rate.
    heat: Heat,
    /// The estimated normal-path cost of one read of the range, in nanoseconds. Zero for an
    /// explicit range, which uses the average cost of sampled reads.
    read_cost_ns: u64,
    /// `heat.reads` when the current base was written, for the payback check of major merges.
    reads_at_build: AtomicU64,
    /// The sequence number of the last batch that wrote to the range. Only the writer stores
    /// it. Nothing reads it yet: persistence will (design 01 section 15, R3).
    last_write_seq: Padded<AtomicU64>,
    /// The maintenance task's jobs on the range: a flush and a merge, at most one of each at a
    /// time, and its notifications. Only the maintenance task, its jobs and the writer's
    /// notifications store them; no read loads them.
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
            heat: Heat::new(rate, now_ms),
            read_cost_ns,
            reads_at_build: AtomicU64::new(0),
            last_write_seq: Padded(AtomicU64::new(0)),
            jobs: maintain::RangeJobs::default(),
        }
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

    fn info(&self, tuning: &RangeCacheTuning) -> CachedRangeInfo {
        let layers = self.layers.load();
        CachedRangeInfo {
            start: self.start.clone(),
            end: self.end.clone(),
            loading: layers.base.is_none(),
            rebuilding: !layers.sealed.is_empty()
                || self.jobs.merging()
                || maintain::merge_due(self, &layers, tuning).is_some(),
            file_bytes: layers.base.as_ref().map_or(0, |f| f.file_len()),
            file_rows: layers.base.as_ref().map_or(0, |f| f.rows()),
            delta_bytes: self.delta_bytes(&layers),
            automatic: self.auto,
            runs: layers.runs.len(),
            run_bytes: layers.runs.iter().map(|r| r.file_bytes()).sum(),
            sealed_buffers: layers.sealed.len(),
            buffer_bytes: layers.buffers().map(|b| b.memory_bytes()).sum(),
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

    /// The bytes of the range's base and runs, which count against the disk budget.
    fn file_bytes(&self) -> u64 {
        let layers = self.layers.load();
        layers.base.as_ref().map_or(0, |f| f.file_len())
            + layers.runs.iter().map(|r| r.file_bytes()).sum::<u64>()
    }

    /// The heap metadata of the range's files: the fence indexes and filters of its base and
    /// runs.
    fn files_meta_bytes(&self) -> u64 {
        let layers = self.layers.load();
        layers.base.as_ref().map_or(0, |f| f.meta_bytes())
            + layers.runs.iter().map(|r| r.file.meta_bytes()).sum::<u64>()
    }

    /// The heap metadata of the range, which counts against the metadata budget: the fence
    /// indexes and filters of its base and runs, and its layer set with its run handles.
    fn meta_bytes(&self) -> u64 {
        let files = self.files_meta_bytes();
        let layers = self.layers.load();
        let handles = size_of::<CachedRange>()
            + size_of::<Layers>()
            + (layers.sealed.capacity() + layers.runs.capacity()) * size_of::<usize>()
            + layers.runs.len() * size_of::<Run>();
        files + handles as u64
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

/// The parts of the cache that only automatic promotion uses.
struct AutoShared {
    options: AutoSettings,
    tx: async_channel::Sender<PromoterMessage>,
    stats: AutoStats,
    /// The average normal-path cost of sampled reads in nanoseconds, which the promoter
    /// updates. Explicit ranges use it as their read cost.
    avg_read_cost_ns: AtomicU64,
    /// Removals of automatic ranges, from the writer to the promoter.
    removals: (
        async_channel::Sender<auto::Removal>,
        async_channel::Receiver<auto::Removal>,
    ),
}

/// A normal-path read that automatic promotion samples. See [`RangeCache::sample`].
pub(crate) struct ReadSample {
    start: chrono::DateTime<chrono::Utc>,
}

/// The range cache of one database. See the module documentation.
pub(crate) struct RangeCache {
    options: RangeCacheOptions,
    /// Only the writer stores it. See the `writer` module.
    index: ArcSwap<RangeIndex>,
    /// The queue of requests to the writer, and the writer's own state.
    writer: WriterQueue,
    /// The disk and heap bytes of all open range cache files, including files that only a
    /// reader or a maintenance job keeps open.
    files: Arc<FileBudget>,
    /// The memory of all write buffers, including buffers that only a reader or a flush holds.
    delta_memory: Arc<DeltaMemory>,
    /// The queue to the maintenance task.
    maintenance: maintain::MaintenanceQueue,
    /// The number of the next range cache file. Loads, flushes and merges take one each.
    next_file_id: AtomicU64,
    /// The bookkeeping of the reads that the cache serves. Its access clock advances by two on
    /// each load and rebuild. Folds copy it to the ranges that reads used. A new range gets the
    /// value between the old and the new clock, so it counts as used after the reads before its
    /// load and before the reads after its load.
    hits: HitStats,
    stats: RangeCacheStats,
    /// Automatic promotion, if configured.
    auto: Option<AutoShared>,
    clock: Arc<dyn SystemClock>,
    rand: Arc<DbRand>,
    handle: tokio::runtime::Handle,
    merge_operator: Option<MergeOperatorType>,
    /// Only fail points use it, and they compile to nothing without the `failpoints` feature.
    #[allow(dead_code)]
    fp_registry: Arc<FailPointRegistry>,
}

impl RangeCache {
    /// Creates the cache directory if needed and removes old range cache files from it. With
    /// automatic promotion, it also returns the receiver of the promoter's queue.
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

    /// Returns true if a major merge of `range`, whose base has `file_bytes`, pays for itself: the
    /// reads that the range served since its base was written, at the normal-path cost of a
    /// read, must save at least the estimated cost of rewriting the base. Only automatic ranges
    /// pay this test. A small base always merges, since its merge is cheap. Writes alone are
    /// never a reason to remove a range that is read.
    fn rebuild_pays(&self, range: &CachedRange, file_bytes: u64) -> bool {
        let Some(auto) = &self.auto else {
            return true;
        };
        if !range.auto || file_bytes < 4 * self.options.tuning.min_rebuild_delta_bytes as u64 {
            return true;
        }
        let reads = range
            .heat
            .reads()
            .saturating_sub(range.reads_at_build.load(Ordering::Relaxed));
        let saved_ns = reads as f64 * range.read_cost_ns as f64;
        let cost_ns = file_bytes as f64 * auto.options.rebuild_cost_ns_per_byte;
        saved_ns >= cost_ns
    }

    /// A new empty write buffer for versions above `lo`.
    fn new_buffer(&self, lo: u64) -> Arc<Delta> {
        Arc::new(Delta::new(
            &self.delta_memory,
            lo,
            self.options.tuning.delta_shape(),
        ))
    }

    /// The name of a new range cache file.
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

    /// The ranges that eviction removes, in order, if the range files are larger than the
    /// budget, never `skip`. Without automatic promotion, the least recently read range comes
    /// first. With automatic promotion, automatic ranges come first by increasing read density,
    /// then explicit ranges, least recently read first. Returns nothing if the files fit.
    ///
    /// The caller computes the order, off the writer: read densities take each range's rate
    /// lock. The writer applies it (see [`writer::Request::Evict`]).
    fn eviction_order(&self, skip: Option<&Arc<CachedRange>>) -> Vec<Arc<CachedRange>> {
        if self.within_budgets() {
            return Vec::new();
        }
        let now = self.now_ms();
        let candidates = self
            .index
            .load()
            .values()
            .filter(|r| r.layers.load().base.is_some())
            .filter(|r| skip.is_none_or(|skip| !Arc::ptr_eq(skip, r)))
            .cloned()
            .collect::<Vec<_>>();
        // Stable sorts, so ranges with equal keys keep the index order, as the previous
        // one-at-a-time minimum did.
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

    /// The eviction order with automatic promotion: automatic ranges before explicit ones,
    /// then by read density, then least recently read first.
    fn eviction_key(&self, range: &CachedRange, now_ms: i64) -> (bool, f64, u64) {
        let density = if range.auto {
            self.density(range, now_ms)
        } else {
            0.0
        };
        (!range.auto, density, range.heat.hits.last_access())
    }

    /// The decayed normal-path cost per second per byte that `range` saves: its read rate
    /// times the cost of a read, over its file size.
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

    fn update_gauges(&self) {
        let index = self.index.load();
        self.stats.ranges.set(index.len() as i64);
        let delta_bytes: u64 = index
            .values()
            .map(|r| r.delta_bytes(&r.layers.load()))
            .sum();
        self.stats.delta_bytes.set(delta_bytes as i64);
        self.delta_memory.publish();
        self.stats.file_bytes.set(self.files.disk() as i64);
        let layers: u64 = index
            .values()
            .map(|r| r.meta_bytes().saturating_sub(r.files_meta_bytes()))
            .sum();
        self.stats
            .meta_bytes
            .set((self.files.meta() + layers) as i64);
    }

    /// Returns true if the cached ranges fit the disk budget and the metadata budget. Only
    /// indexed ranges count: files that only a reader holds go when the reader drops them.
    fn within_budgets(&self) -> bool {
        let index = self.index.load();
        let disk: u64 = index.values().map(|r| r.file_bytes()).sum();
        if disk > self.options.max_disk_bytes {
            return false;
        }
        let meta: u64 = index.values().map(|r| r.meta_bytes()).sum();
        meta <= self.options.tuning.meta_limit(self.options.max_disk_bytes)
    }

    /// Checks whether a read at `seq` can use `layers`. See the sequence number rules. Callers
    /// treat a loading range, which has no base, as uncovered before they call this.
    fn usable(layers: &Layers, seq: u64) -> Result<(), Unfit> {
        if layers.base.is_none() || seq < layers.base_seq {
            return Err(Unfit::TooOld);
        }
        if layers.active.sealed_at().load(Ordering::SeqCst) < seq {
            return Err(Unfit::Sealed);
        }
        Ok(())
    }

    /// Counts a read that used `range`, on this thread. See the `hits` module.
    fn count_read(&self, range: &CachedRange, count: ReadCount) {
        self.hits.read(range.slot, &range.heat.hits, count);
    }

    /// Counts an attempt `attempt` (from 1) of `op` that failed the sequence number rules, and
    /// returns true if the read tries again. A read whose sequence number is below the file's
    /// can only be in a snapshot or a transaction older than the file: its sequence number
    /// never grows, so it falls back at once. A sealed delta can be replaced by a newer one, so
    /// such a read tries again, up to [`PIN_ATTEMPTS`] attempts.
    fn count_unfit(&self, op: PinOp, reason: Unfit, attempt: usize) -> bool {
        let retry = reason == Unfit::Sealed && attempt < PIN_ATTEMPTS;
        self.hits.count(if retry {
            ReadCount::retry(op, reason)
        } else {
            ReadCount::unusable(op, reason)
        });
        retry
    }

    /// Reads `key` at the last committed sequence number, or at `at` for a read in a snapshot
    /// or a transaction (see [`read_seq`]).
    ///
    /// The common path holds the index and the layers through `arc_swap` guards and takes no
    /// reference count, and it counts the read on its own thread, so concurrent reads of one
    /// range do not write a shared cache line.
    pub(crate) fn get(&self, key: &[u8], oracle: &dyn Oracle, at: Option<u64>) -> CacheRead {
        for attempt in 1..=PIN_ATTEMPTS {
            let index = self.index.load();
            let Some(cached) = covering(&index, key) else {
                return CacheRead::Uncovered;
            };
            let layers = cached.layers.load();
            if layers.base.is_none() {
                return CacheRead::Uncovered;
            }
            // Checkers stretch the window between the pin and the load of the sequence number,
            // in which seals and installs happen (design 01, X23).
            fail_point!(Arc::clone(&self.fp_registry), "range-cache-pin");
            let seq = read_seq(oracle, at);
            if let Err(reason) = Self::usable(&layers, seq) {
                if self.count_unfit(PinOp::Get, reason, attempt) {
                    continue;
                }
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
                    CacheRead::Uncovered
                }
            };
        }
        CacheRead::Unusable
    }

    /// Pins the layers for a scan of `range` at the last committed sequence number, or at `at`
    /// (see [`read_seq`]), if ready cached ranges hold all of `range`. The ranges can be
    /// adjacent ranges, each starting where the one before ends. All of them must follow the
    /// sequence number rules at the same sequence number.
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
    /// and `Some(Err(()))` if they do but fail the sequence number rules.
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
            if parts.iter().any(|l| l.base.is_none()) {
                return None;
            }
            fail_point!(Arc::clone(&self.fp_registry), "range-cache-pin");
            let seq = read_seq(oracle, at);
            if let Some(first) = parts.iter().find_map(|l| Self::usable(l, seq).err()) {
                // One part that is too old makes every attempt fail.
                let too_old = first == Unfit::TooOld
                    || parts
                        .iter()
                        .any(|l| Self::usable(l, seq) == Err(Unfit::TooOld));
                let reason = if too_old { Unfit::TooOld } else { first };
                if self.count_unfit(PinOp::Scan, reason, attempt) {
                    continue;
                }
                return Some(Err(()));
            }
            // Only the first range counts as read: most scans end in it.
            self.count_read(first, ReadCount::Scan);
            return Some(Ok(PinnedScan { parts, seq }));
        }
        Some(Err(()))
    }

    /// Plans a scan of `range` over cached ranges and gaps, if a cached range that follows the
    /// sequence number rules intersects it (see the `mixed` module).
    ///
    /// It loads the index once, pins the layers of every cached range that intersects `range`,
    /// then loads the read's sequence number, then checks the rules for each. A range that
    /// fails them, or still loads, becomes a gap. Returns `None` if no part is cached.
    fn pin_mixed(
        &self,
        range: &BytesRange,
        oracle: &dyn Oracle,
        at: Option<u64>,
    ) -> Option<MixedPin> {
        let (start, end) = half_open(range)?;
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
            return None;
        }
        let pinned: Vec<Arc<Layers>> = intersecting.iter().map(|r| r.layers.load_full()).collect();
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-pin");
        let seq = read_seq(oracle, at);

        let scan_end = RangeBounds::end_bound(range).cloned();
        let mut parts = Vec::with_capacity(2 * intersecting.len() + 1);
        let mut push = |kind, start: std::ops::Bound<Bytes>, end: std::ops::Bound<Bytes>| {
            if BytesRange::try_new(start.clone(), end.clone()).is_some() {
                parts.push(Part { kind, start, end });
            }
        };
        let mut cursor = Some(RangeBounds::start_bound(range).cloned());
        let mut first_cached: Option<&Arc<CachedRange>> = None;
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
                    PartKind::Gap { sample: true },
                    from,
                    Excluded(cached.start.clone()),
                );
                Included(cached.start.clone())
            } else {
                from
            };
            let to = scan::min_end(&scan_end, &cached.end);
            let kind = if layers.base.is_none() {
                // A loading range holds no rows yet.
                PartKind::Gap { sample: true }
            } else {
                match Self::usable(&layers, seq) {
                    Ok(()) => {
                        first_cached.get_or_insert(*cached);
                        PartKind::Cached(layers)
                    }
                    Err(reason) => {
                        self.hits
                            .count(ReadCount::unusable(PinOp::ScanPart, reason));
                        PartKind::Gap { sample: false }
                    }
                }
            };
            push(kind, from, to);
            cursor = cached.end.clone().map(Included);
        }
        if let Some(from) = cursor {
            push(PartKind::Gap { sample: true }, from, scan_end);
        }
        let first = first_cached?.clone();
        Some(MixedPin { parts, seq, first })
    }

    pub(crate) fn ranges(&self) -> Vec<CachedRangeInfo> {
        let tuning = &self.options.tuning;
        self.index.load().values().map(|r| r.info(tuning)).collect()
    }

    /// The memory of all write buffers: their chunks and filters.
    pub(crate) fn buffer_memory(&self) -> u64 {
        self.delta_memory.bytes()
    }

    /// Starts a sample of a normal-path read that the cache could have served, if automatic
    /// promotion is on and the read is sampled. Pass it to [`RangeCache::record`] after the
    /// read.
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

    /// Records a sampled normal-path read of `key`, a point read or the start of a scan, with
    /// the time that it took. It drops the sample if the promoter's queue is full.
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

    /// Returns true if a cached range, ready or loading, holds `key`.
    fn covers(&self, key: &[u8]) -> bool {
        covering(&self.index.load(), key).is_some()
    }

    /// Returns the end of the cached range before `key`, or an empty key if there is none, and
    /// the start of the cached range after `key`, if there is one. `key` must be uncovered.
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

    /// The bytes of the files of cached ranges.
    fn attached_bytes(&self) -> u64 {
        self.index.load().values().map(|r| r.file_bytes()).sum()
    }

    /// Returns the highest read density among the automatic ranges that eviction would remove,
    /// lowest density first, to free `need` bytes. Returns `None` if automatic ranges alone
    /// cannot free that much: promotion never displaces explicitly cached ranges.
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

    /// Writes the rows of a loading range to a file, has the writer publish it, then evicts
    /// ranges if the files no longer fit the budget. It stops with an error when `check_open`
    /// fails, which it calls every 1024 rows. A range that was removed meanwhile gets no file,
    /// and the file is deleted when it drops.
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

        if self.attach_base(range, file, file_seq).await {
            // An automatic range won admission against the ranges it displaces, so the budget
            // never evicts it right after its load.
            self.evict(range.auto.then_some(range)).await;
        }
        Ok(())
    }
}

/// The sequence number of a read: the last committed sequence number, or `at` for a read in a
/// snapshot or a transaction. The caller loads it after it pins the layers.
///
/// Like the normal read path (see `Reader::prepare_max_seq`), a read at `at` never reads above
/// the last committed sequence number. `Db::snapshot` and `Db::begin` take `at` from the last
/// committed sequence number, so `at` is already committed for them. Only the snapshot of a
/// load can be above it, and a load reads through the normal path. The bound keeps the rules
/// of the RFC true for any `at`: every write at or below the read's sequence number is in a
/// delta or a file.
fn read_seq(oracle: &dyn Oracle, at: Option<u64>) -> u64 {
    let committed = oracle.last_committed_seq();
    at.map_or(committed, |at| at.min(committed))
}

/// The result of [`RangeCache::get`].
pub(crate) enum CacheRead {
    /// No ready range holds the key, or the cache cannot answer now.
    Uncovered,
    /// A ready range holds the key, but its layers failed the sequence number rules at each
    /// attempt. For a read in a snapshot, the file can be newer than the snapshot. Use the
    /// normal read path, and do not sample the read: a cached range holds it.
    Unusable,
    /// The value of the key at the read's sequence number.
    Answer(Option<KeyValue>),
    /// The newest version of the key is a merge operand. Use [`Pinned::get`] to resolve it.
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
    /// Ready ranges hold the scan range, but failed the sequence number rules. See
    /// [`CacheRead::Unusable`].
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

/// Returns the newest version of `key` in `layers` at `seq` (design 01 section 5.1). The
/// sources are disjoint in sequence numbers and visited newest first: buffers, then runs, then
/// the base. So the first version found is the newest at or below `seq`.
///
/// A buffer is searched in place, by the borrowed key: only a value is copied out, with its key,
/// in one allocation. The key's hash is computed once, for the first filter that the read
/// probes, and reused for the others. Nothing here writes shared memory.
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
        if let Some(found) = run.file.read_version(key, seq)? {
            return Ok(match found {
                FileRead::Value(row) => Found::Value(KeyValue::from(row)),
                FileRead::Tombstone => Found::Absent,
                FileRead::Merge => Found::Merge,
            });
        }
    }
    let row = match &layers.base {
        Some(file) => file.get(key)?,
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

/// Returns an iterator over `range` of `layers` at `seq`, through the iterator stack of the
/// normal read path. So merge operands resolve the same way as in the normal path.
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
    for run in &layers.runs {
        sources.push(Box::new(RangeFileIterator::new(
            run.file.clone(),
            range.clone(),
        )));
    }
    if let Some(file) = &layers.base {
        sources.push(Box::new(RangeFileIterator::new(
            file.clone(),
            range.clone(),
        )));
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

/// Returns the first key that a scan of `range` can return, for sampling.
fn scan_start(range: &BytesRange) -> Bytes {
    match RangeBounds::start_bound(range) {
        Included(key) | Excluded(key) => key.clone(),
        Unbounded => Bytes::new(),
    }
}

/// Returns true if a point read with `options` reads committed state, at the latest committed
/// sequence number or at a snapshot's, which the cache can serve.
///
/// The cache never serves a read with a tracing option. Tests rely on this to force the normal
/// read path at a snapshot's sequence number, so keep it so or give tests another way.
pub(crate) fn can_serve_get(options: &ReadOptions) -> bool {
    matches!(options.durability_filter, DurabilityLevel::Memory)
        && !options.dirty
        && options.filter_context.is_none()
        && options.tracing_options.is_none()
}

/// Returns true if a scan with `options` reads committed state in ascending order, which the
/// cache can serve. See [`can_serve_get`].
pub(crate) fn can_serve_scan(options: &ScanOptions) -> bool {
    matches!(options.durability_filter, DurabilityLevel::Memory)
        && !options.dirty
        && matches!(options.order, IterationOrder::Ascending)
        && options.filter_context.is_none()
        && options.tracing_options.is_none()
}

/// A normal-path read that automatic promotion sampled, with its key: the key of a point read
/// or the first key of a scan. Pass it to [`DbInner::range_cache_record`] after the read.
pub(crate) struct SampledRead {
    sample: ReadSample,
    key: Bytes,
}

/// Where a point read goes. See [`DbInner::range_cache_get`].
pub(crate) enum GetRoute {
    /// The range cache served the read.
    Served(Result<Option<KeyValue>, SlateDBError>),
    /// Use the normal read path.
    Normal(Option<SampledRead>),
}

/// Where a scan goes. See [`DbInner::range_cache_scan`].
pub(crate) enum ScanRoute {
    /// The range cache serves the scan.
    Served(Result<DbIterator, SlateDBError>),
    /// Use the normal read path for this range.
    Normal(BytesRange, Option<SampledRead>),
}

impl DbInner {
    /// Serves a point read of `key` from the range cache if it can. `at` is the sequence number
    /// of a snapshot or a transaction, or `None` for a read at the latest committed state. A
    /// read that no cached range holds may be sampled for automatic promotion.
    pub(crate) async fn range_cache_get(
        &self,
        key: &[u8],
        options: &ReadOptions,
        at: Option<u64>,
    ) -> GetRoute {
        let Some(cache) = &self.range_cache else {
            return GetRoute::Normal(None);
        };
        if !can_serve_get(options) {
            return GetRoute::Normal(None);
        }
        // A served read counts in `get_requests` through the cache's bookkeeping, at its folds.
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

    /// Serves a scan of `range` from the range cache if it can. See
    /// [`DbInner::range_cache_get`].
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
        if !can_serve_scan(options) {
            return ScanRoute::Normal(range, None);
        }
        // A served scan counts in `scan_requests` through the cache's bookkeeping, at its folds.
        cache.hits.bind_db(&self.db_stats);
        match cache.pin_scan(&range, self.oracle.as_ref(), at) {
            ScanPin::Pinned(pinned) => ScanRoute::Served(
                pinned
                    .scan(range, self.reader.read_merge_operator.clone())
                    .await,
            ),
            ScanPin::Uncovered => {
                for _ in 0..PIN_ATTEMPTS {
                    // The gaps read through the normal path from a state view `V` with
                    // `max_seq = S`, the scan's sequence number. `V` must hold every version
                    // visible at `S`. Flush and compaction keep only the versions that the
                    // latest state and registered snapshots need, so a view captured after `S`
                    // was loaded can lack a version that `S` sees, when a newer one exists.
                    //
                    // At the latest state nothing is registered at `S`. So capture `V` first,
                    // then load `S`, then check that the active memtable is still the one in
                    // `V`. The SSTs and immutable memtables of `V` never change, so every
                    // version written before the capture is in `V`, whatever flush and
                    // compaction do later. A write at or below `S` after the capture went into
                    // the active memtable, which `V` shares, unless a freeze happened. So if no
                    // freeze happened, `V` holds every version at or below `S`. Else try again.
                    //
                    // A snapshot or a transaction is registered at `S`, so flush and compaction
                    // keep its versions, and any view captured after `S` was loaded holds them.
                    let latest_view = at.is_none().then(|| self.state.read().view());
                    let Some(pin) = cache.pin_mixed(&range, self.oracle.as_ref(), at) else {
                        break;
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
                    // As for a fully cached scan, the first cached range counts as read.
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
                let sample = cache.sample().map(|sample| SampledRead {
                    sample,
                    key: scan_start(&range),
                });
                ScanRoute::Normal(range, sample)
            }
            ScanPin::Unusable => ScanRoute::Normal(range, None),
        }
    }

    /// Records a sampled normal-path read, after the read.
    pub(crate) fn range_cache_record(&self, sampled: Option<SampledRead>) {
        if let (Some(cache), Some(sampled)) = (&self.range_cache, sampled) {
            cache.record(sampled.sample, &sampled.key);
        }
    }

    /// Loads the parts of `range` that no cached range holds. See [`Db::cache_range`].
    async fn cache_range(self: &Arc<Self>, range: BytesRange) -> Result<(), SlateDBError> {
        self.cache_range_with(range, Origin::Explicit).await
    }

    /// Loads the parts of `range` that no cached range holds, for `origin`.
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
        // Bounded range size (design 01 section 7.6): split points from the SST index, before
        // the guard and the registration, as the writer only cuts the parts at them.
        let target = cache.options.tuning.range_target_bytes;
        let splits = if target > 0 {
            self.range_split_points(&start, end.as_ref(), target)
                .await?
        } else {
            Vec::new()
        };
        // The retention guard comes first (design 01 section 6.3). The snapshot registers at the
        // last committed sequence number `S_L`, before the request. The writer registers the
        // ranges at its boundary `W >= S_L`. Flush and compaction keep every version above the
        // oldest registered snapshot and the newest at or below it, so the newest version at or
        // below `W` of each key stays readable. The snapshot manager's lock is taken here, on
        // this task, never on the writer.
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
        // `W` is committed, so the load reads no dirty state. Read in large requests, several at
        // a time, so that a load costs about one request per `load_fetch_bytes` and its time
        // follows the bytes, not the store's latency.
        let options = cache.options.tuning.bulk_scan_options();
        let mut result = Ok(());
        for range in &registered.ranges {
            if result.is_err() {
                cache.remove_and_wait(range, RemoveReason::Error).await;
                continue;
            }
            // Every batch at or below `W` committed before the registration, so this view, taken
            // after it, holds it.
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

    /// Keys that cut `[start, end)` into parts of about `target` bytes, at block boundaries of
    /// the largest sorted run, as the promoter forms its candidates. It reads only the indexes of
    /// the SSTs of that run that overlap the range. Without a sorted run, or in a segmented
    /// database, it returns no key: the range is registered whole.
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
    /// Loads `range` into the range cache.
    ///
    /// The call loads the parts of `range` that no cached range holds yet, from a snapshot of
    /// the database. Writes that happen during the load go to the new ranges, so no write is
    /// lost. After the call returns, point reads and ascending scans read cached keys from
    /// local disk and memory: reads at the latest committed state, and reads in snapshots and
    /// transactions that start after the load or the last rebuild of the range. Other reads,
    /// such as dirty reads and reads in older snapshots, use the normal read path.
    ///
    /// ## Errors
    /// - [`crate::ErrorKind::Invalid`] if the database has no range cache, if the range holds
    ///   rows with an expire time, or if the range is larger than the disk budget. The call then
    ///   caches no part of `range` that it did not load before the error.
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

    /// Removes every cached range that overlaps `range` from the range cache.
    ///
    /// The call returns before the removal takes effect: the batch writer removes the ranges
    /// at its next batch, or soon after when no write comes. Reads stay exact meanwhile, and
    /// [`Db::cached_ranges`] can still list the ranges. A later [`Db::cache_range`] call takes
    /// effect after the removal.
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

    /// Removes all ranges from the range cache. Like [`Db::uncache_range`], the call returns
    /// before the removal takes effect.
    pub fn clear_range_cache(&self) -> Result<(), crate::Error> {
        let cache = self.inner.range_cache_or_err()?;
        cache.bind_writer(&self.inner.write_notifier);
        cache.request_removal(Target::All, RemoveReason::Explicit);
        Ok(())
    }

    /// Waits until the range cache applied every change requested before the call: removals
    /// from [`Db::uncache_range`] and [`Db::clear_range_cache`], and the queued steps of loads
    /// and rebuilds. Returns at once if the cache is closed.
    ///
    /// For tests and tools that check [`Db::cached_ranges`] after a removal.
    #[doc(hidden)]
    pub async fn wait_for_range_cache(&self) -> Result<(), crate::Error> {
        let cache = self.inner.range_cache_or_err()?;
        cache.bind_writer(&self.inner.write_notifier);
        cache.barrier().await;
        Ok(())
    }

    /// Returns the cached ranges in key order. Returns an empty list if the database has no
    /// range cache.
    pub fn cached_ranges(&self) -> Vec<CachedRangeInfo> {
        self.inner
            .range_cache
            .as_ref()
            .map(|cache| cache.ranges())
            .unwrap_or_default()
    }

    /// Returns the memory of the range cache's write buffers: the chunks and filters of every
    /// buffer, including buffers that only a reader or a flush still holds. It never exceeds
    /// [`RangeCacheTuning::max_recent_bytes`]. Returns 0 if the database has no range cache.
    ///
    /// For tests and tools.
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
                            base: None,
                            base_seq: 0,
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
                // The reader thread served 400 gets and 20 scans from the cache, and none of
                // them wrote a shared counter, the range's read count or its last access.
                assert_eq!(t.read_metrics(), before);
                assert_eq!(range.heat.reads(), 0);
                assert_eq!(range.heat.hits.last_access(), last_access);
                main.go();
                main.wait();
                // Its fold adds all of them.
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
                // A thread folds the rest of its counts when it exits, before its join returns.
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
            }
            // None of them counts as served.
            assert_eq!(t.read_metrics()[..3], [0, 0, 0]);
        }

        #[tokio::test]
        async fn should_evict_the_least_recently_read_range_by_folded_reads() {
            // As in the integration test of eviction: the budget holds two ranges of 50 rows.
            let t = open(
                |o| {
                    o.tuning.page_size = 4096;
                    o.max_disk_bytes = 9000;
                    // Only the fold size makes a thread fold.
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
            // Reads of the first range, one fold's worth. The second range gets none.
            let fold = RangeCacheTuning::default().stats_fold_events;
            for _ in 0..fold {
                t.db.get(key(10)).await.unwrap();
            }
            clock.advance(Duration::from_secs(1)).await;
            let (first, second) = (t.range(&key(0)), t.range(&key(100)));
            let now = t.cache().now_ms();
            assert!(t.cache().density(&first, now) > 0.0);
            assert_eq!(t.cache().density(&second, now), 0.0);
            // A third automatic range evicts the one with the lowest read density. Without the
            // folded reads both would have none, and eviction would take the first one.
            load(200).await.unwrap();
            let starts: Vec<_> = t.db.cached_ranges().into_iter().map(|r| r.start).collect();
            assert_eq!(starts, vec![key(0), key(200)]);
        }
    }
}
