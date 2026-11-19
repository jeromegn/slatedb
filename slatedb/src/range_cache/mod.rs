//! The range cache keeps a local copy of selected key ranges and serves reads from it.
//!
//! See RFC 0040 for the design. In short:
//!
//! - A *cached range* is a half-open key range `[start, end)`. The cache knows every live key
//!   in a cached range, so a point read in it is a definite hit or a definite miss, and a scan
//!   in it needs no other source.
//! - A cached range has three layers, newest first: the *active delta*, at most one *frozen
//!   delta*, and one *range cache file*. A delta is a [`KVTable`] that holds every version that
//!   the writer applied to the range since the file was written. The file holds the rows of the
//!   range at one sequence number.
//! - The batch writer applies each write to the deltas after it puts the write in the memtable
//!   and before it advances the last committed sequence number. Reads at the latest committed
//!   state read the deltas with that sequence number as the bound, so a write becomes visible
//!   in the cache and in the memtable at the same time.
//! - All changes to the set of ranges and to the layers of a range happen under one mutex. The
//!   batch writer takes the same mutex when it applies a write to a non-empty cache. Reads take
//!   no lock.
//!
//! # Sequence number rules
//!
//! A read loads the layers of a range, then the last committed sequence number `seq`. It uses
//! the layers only if both rules hold:
//!
//! - `seq >= layers.file_seq`. The file holds the state at `file_seq`.
//! - `seq <= layers.active.sealed_at`. When the cache stops applying writes to a delta (a
//!   freeze or a removal), it seals the delta at the last sequence number that it applied.
//!   The delta then holds every version up to that number and none after it.
//!
//! If a rule fails, the read tries again or uses the normal read path. The first rule keeps
//! writes that are not committed invisible. The second rule makes the answer exactly the state
//! at `seq`, which is the answer of the normal read path at `seq`.
//!
//! # Automatic promotion
//!
//! With [`RangeCacheOptions::auto_promote`], a background task chooses ranges from sampled
//! normal-path reads (see the `auto` module). This module keeps the parts that the cache core
//! needs: the read heat of each range, eviction by read density, the payback check of rebuilds,
//! and the memory of removed ranges and failed promotions.

mod auto;
mod file;
mod scan;
pub mod stats;

use std::cell::Cell;
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
use crate::mem_table::KVTable;
use crate::merge_operator::MergeOperatorType;
use crate::oracle::Oracle;
use crate::read_trace::ReadTrace;
use crate::reader::ScanContext;
use crate::types::{KeyValue, RowEntry, ValueDeletable};
use crate::utils::spawn_bg_task;

pub use auto::AutoPromoteOptions;
use auto::AutoSettings;
pub(crate) use auto::{Promoter, PromoterMessage, RANGE_CACHE_PROMOTER_TASK_NAME};
pub use file::RangeCachePageSource;
use file::{RangeFile, RangeFileIterator, RangeFileWriter, FILE_EXTENSION, TEMP_FILE_EXTENSION};
use scan::CachedScan;
use stats::{AutoStats, RangeCacheStats, RemoveReason};

/// The number of times a read tries to pin a range before it uses the normal read path.
const PIN_ATTEMPTS: usize = 3;

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
    /// The smallest active delta that causes a rebuild. A rebuild starts when the active delta
    /// of a range is larger than this value and larger than a quarter of the range file. The
    /// local write amplification of rebuilds is thus at most about 4.
    pub min_rebuild_delta_bytes: usize,
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
}

impl Default for RangeCacheTuning {
    fn default() -> Self {
        Self {
            page_size: 4096,
            min_rebuild_delta_bytes: 1024 * 1024,
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
    /// True while the cache rewrites the range's file on disk to include recent writes.
    pub rebuilding: bool,
    /// The size of the range's file on disk.
    pub file_bytes: u64,
    /// The number of rows in the range's file on disk.
    pub file_rows: u64,
    /// The estimated size of the recent writes to the range that the cache holds in memory and
    /// has not yet written to the range's file.
    pub delta_bytes: u64,
    /// True if automatic promotion cached the range.
    pub automatic: bool,
}

/// The versions that the writer applied to a range since its file was written.
struct Delta {
    table: KVTable,
    /// The cache applies no write with a greater sequence number to this delta. It is
    /// `u64::MAX` while the delta is active.
    sealed_at: AtomicU64,
}

impl Delta {
    fn new() -> Self {
        Self {
            table: KVTable::new(),
            sealed_at: AtomicU64::new(u64::MAX),
        }
    }

    fn size(&self) -> usize {
        self.table.metadata().entries_size_in_bytes
    }
}

/// The layers of a cached range. The cache replaces this value as a whole.
struct Layers {
    active: Arc<Delta>,
    /// A delta that a rebuild merges into a new file.
    frozen: Option<Arc<Delta>>,
    /// `None` while the range loads.
    file: Option<Arc<RangeFile>>,
    /// The file holds the state of the range at this sequence number.
    file_seq: u64,
    /// The end of the range, so that a scan can walk adjacent ranges with one reference to
    /// each.
    end: Option<Bytes>,
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

/// Reads count in steps of this size on each thread, so that a read of a hot range does not
/// write its counter each time.
const HEAT_STEP: u64 = 16;

thread_local! {
    static HEAT_TICKS: Cell<u64> = const { Cell::new(0) };
}

/// The read rate of a cached range, for automatic promotion.
struct Heat {
    /// The reads that the range served, counted in steps of `HEAT_STEP` on each thread.
    reads: AtomicU64,
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
            reads: AtomicU64::new(0),
            rate: Mutex::new(HeatRate {
                reads: 0,
                at_ms: now_ms,
                rate,
            }),
        }
    }

    fn tick(&self) {
        let step = HEAT_TICKS.with(|ticks| {
            let next = ticks.get().wrapping_add(1);
            ticks.set(next);
            next.is_multiple_of(HEAT_STEP)
        });
        if step {
            self.reads.fetch_add(HEAT_STEP, Ordering::Relaxed);
        }
    }

    fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
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
    /// The value of the cache's access clock when a read last used the range.
    last_access: AtomicU64,
    /// True if automatic promotion cached the range.
    auto: bool,
    heat: Heat,
    /// The estimated normal-path cost of one read of the range, in nanoseconds. Zero for an
    /// explicit range, which uses the average cost of sampled reads.
    read_cost_ns: u64,
    /// `heat.reads` when the current file was written, for the payback check of rebuilds.
    reads_at_build: AtomicU64,
}

impl CachedRange {
    fn new(start: Bytes, end: Option<Bytes>, layers: Layers, origin: Origin, now_ms: i64) -> Self {
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
            last_access: AtomicU64::new(0),
            auto,
            heat: Heat::new(rate, now_ms),
            read_cost_ns,
            reads_at_build: AtomicU64::new(0),
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

    fn info(&self) -> CachedRangeInfo {
        let layers = self.layers.load();
        CachedRangeInfo {
            start: self.start.clone(),
            end: self.end.clone(),
            loading: layers.file.is_none(),
            rebuilding: layers.frozen.is_some(),
            file_bytes: layers.file.as_ref().map_or(0, |f| f.file_len()),
            file_rows: layers.file.as_ref().map_or(0, |f| f.rows()),
            delta_bytes: self.delta_bytes(&layers),
            automatic: self.auto,
        }
    }

    fn delta_bytes(&self, layers: &Layers) -> u64 {
        (layers.active.size() + layers.frozen.as_ref().map_or(0, |f| f.size())) as u64
    }

    fn file_bytes(&self) -> u64 {
        self.layers.load().file.as_ref().map_or(0, |f| f.file_len())
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

/// State that only changes under the cache mutex.
struct CacheState {
    /// The sequence number of the last write that the writer applied to the cache.
    last_applied_seq: u64,
    next_file_id: u64,
    closed: bool,
    /// Automatic ranges that the cache removed recently. Only automatic promotion uses it.
    ghosts: Vec<Ghost>,
    /// Ranges where a promotion failed recently, by start key.
    backoff: BTreeMap<Bytes, Backoff>,
}

/// An automatic range that the cache removed. Promoting it again within the ghost time needs
/// sampled reads after the removal.
struct Ghost {
    start: Bytes,
    end: Option<Bytes>,
    at_ms: i64,
}

/// A range where promotion waits after a failure. The wait doubles with each failure, and the
/// level drops by one every eight base periods.
struct Backoff {
    end: Option<Bytes>,
    level: u32,
    at_ms: i64,
}

const MAX_BACKOFF_LEVEL: u32 = 10;

/// How long promotion waits at backoff `level`.
fn backoff_window_ms(level: u32, base_ms: i64) -> i64 {
    base_ms.saturating_mul(1 << level.saturating_sub(1).min(20))
}

/// The backoff level of `backoff` at `now_ms`, after its decay.
fn decayed_level(backoff: &Backoff, now_ms: i64, base_ms: i64) -> u32 {
    let steps = (now_ms - backoff.at_ms) / (8 * base_ms).max(1);
    backoff
        .level
        .saturating_sub(steps.clamp(0, i64::from(u32::MAX)) as u32)
}

fn overlaps(
    start: &Bytes,
    end: &Option<Bytes>,
    other_start: &Bytes,
    other_end: &Option<Bytes>,
) -> bool {
    end.as_ref().is_none_or(|end| other_start < end)
        && other_end.as_ref().is_none_or(|other_end| start < other_end)
}

/// The parts of the cache that only automatic promotion uses.
struct AutoShared {
    options: AutoSettings,
    tx: async_channel::Sender<PromoterMessage>,
    stats: AutoStats,
    /// The average normal-path cost of sampled reads in nanoseconds, which the promoter
    /// updates. Explicit ranges use it as their read cost.
    avg_read_cost_ns: AtomicU64,
}

/// A normal-path read that automatic promotion samples. See [`RangeCache::sample`].
pub(crate) struct ReadSample {
    start: chrono::DateTime<chrono::Utc>,
}

/// The range cache of one database. See the module documentation.
pub(crate) struct RangeCache {
    options: RangeCacheOptions,
    index: ArcSwap<RangeIndex>,
    state: Mutex<CacheState>,
    /// The bytes of all open range cache files, including files that only a reader or a
    /// rebuild keeps open.
    disk_bytes: Arc<AtomicU64>,
    /// The sequence number of the last write that the writer put in the memtable. The writer
    /// advances it before it checks the index. See [`RangeCache::register`].
    inserted_seq: AtomicU64,
    /// Advances each time `register` publishes ranges. See [`RangeCache::register`].
    publish_epoch: AtomicU64,
    /// Advances by two on each load and rebuild. Reads copy it to the ranges they use. A new
    /// range gets the value between the old and the new clock, so it counts as used after
    /// the reads before its load and before the reads after its load.
    access_clock: AtomicU64,
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
                    options: auto_options,
                };
                (Some(auto), Some(rx))
            }
            None => (None, None),
        };
        let cache = Arc::new(Self {
            options,
            index: ArcSwap::from_pointee(RangeIndex::new()),
            state: Mutex::new(CacheState {
                last_applied_seq: 0,
                next_file_id: 0,
                closed: false,
                ghosts: Vec::new(),
                backoff: BTreeMap::new(),
            }),
            disk_bytes: Arc::new(AtomicU64::new(0)),
            inserted_seq: AtomicU64::new(0),
            publish_epoch: AtomicU64::new(0),
            access_clock: AtomicU64::new(0),
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

    /// Applies one write batch. The batch writer calls this after it puts the entries in the
    /// memtable and before it advances the last committed sequence number. All entries have
    /// the same sequence number.
    pub(crate) fn apply_write(self: &Arc<Self>, entries: &[RowEntry]) {
        let Some(seq) = entries.first().map(|e| e.seq) else {
            return;
        };
        // Check the index without the mutex first. This pairs with `register`: the writer
        // updates `inserted_seq` and then loads `publish_epoch`, and `register` updates
        // `publish_epoch` and then loads `inserted_seq`, all `SeqCst`. So either this check
        // finds the new range, or the snapshot of the range includes this write. See RFC 0040,
        // "Loading a range".
        self.inserted_seq.fetch_max(seq, Ordering::SeqCst);
        self.publish_epoch.load(Ordering::SeqCst);
        {
            let index = self.index.load();
            if index.is_empty() || !entries.iter().any(|e| covering(&index, &e.key).is_some()) {
                return;
            }
        }
        let mut state = self.state.lock();
        let index = self.index.load_full();
        let mut touched: Vec<Arc<CachedRange>> = Vec::new();
        let mut expiring: Vec<Arc<CachedRange>> = Vec::new();
        for entry in entries {
            let Some(range) = covering(&index, &entry.key) else {
                continue;
            };
            if entry.expire_ts.is_some() {
                // The cache does not hold rows that expire. See RFC 0040, "TTL".
                if !expiring.iter().any(|r| Arc::ptr_eq(r, range)) {
                    expiring.push(range.clone());
                }
                continue;
            }
            range.layers.load().active.table.put(entry.clone());
            if !touched.iter().any(|r| Arc::ptr_eq(r, range)) {
                touched.push(range.clone());
            }
        }
        state.last_applied_seq = seq;
        for range in &expiring {
            // The delta does not hold this write, so seal it before the write.
            self.remove_locked(&mut state, range, seq - 1, RemoveReason::ExpiringWrite);
        }
        for range in &touched {
            if !expiring.iter().any(|r| Arc::ptr_eq(r, range)) {
                self.maybe_freeze_locked(&mut state, range, seq);
            }
        }
    }

    /// Starts a rebuild if the active delta of `range` is large enough.
    fn maybe_freeze_locked(
        self: &Arc<Self>,
        state: &mut CacheState,
        range: &Arc<CachedRange>,
        seq: u64,
    ) {
        let layers = range.layers.load_full();
        let Some(file) = layers.file.clone() else {
            return;
        };
        if layers.frozen.is_some() {
            return;
        }
        let threshold = self
            .options
            .tuning
            .min_rebuild_delta_bytes
            .max((file.file_len() / 4) as usize);
        if layers.active.size() < threshold {
            return;
        }
        if !self.rebuild_pays(range, file.file_len()) {
            // The delta holds this write, so it is complete up to `seq`.
            self.remove_locked(state, range, seq, RemoveReason::NotWorthIt);
            return;
        }
        range
            .reads_at_build
            .store(range.heat.reads(), Ordering::Relaxed);
        let frozen = layers.active.clone();
        frozen.sealed_at.store(seq, Ordering::SeqCst);
        range.layers.store(Arc::new(Layers {
            active: Arc::new(Delta::new()),
            frozen: Some(frozen.clone()),
            file: Some(file.clone()),
            file_seq: layers.file_seq,
            end: range.end.clone(),
        }));
        let file_id = state.next_file_id;
        state.next_file_id += 1;
        let cache = self.clone();
        let failed_cache = self.clone();
        let failed_range = range.clone();
        let range = range.clone();
        spawn_bg_task(
            "range_cache_rebuild".to_string(),
            &self.handle,
            move |result: &Result<(), SlateDBError>| {
                if let Err(e) = result {
                    warn!("range cache rebuild failed [error={:?}]", e);
                    failed_cache.remove(&failed_range, RemoveReason::Error);
                }
            },
            async move { cache.rebuild(range, frozen, file, seq, file_id).await },
        );
    }

    /// Returns true if a rebuild of `range`, whose file has `file_bytes`, pays for itself: the
    /// reads that the range served since its file was written, at the normal-path cost of a
    /// read, must save at least the estimated cost of rewriting the file. Only automatic ranges
    /// pay this test. A small file always rebuilds, since its rebuild is cheap. Writes alone
    /// are never a reason to remove a range that is read.
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

    /// Merges a frozen delta into the range file and installs the new file.
    async fn rebuild(
        self: Arc<Self>,
        range: Arc<CachedRange>,
        frozen: Arc<Delta>,
        file: Arc<RangeFile>,
        frozen_seq: u64,
        file_id: u64,
    ) -> Result<(), SlateDBError> {
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-rebuild");
        let all = BytesRange::unbounded();
        let sources: Vec<Box<dyn RowEntryIterator + 'static>> = vec![
            Box::new(
                frozen
                    .table
                    .range(all.clone(), IterationOrder::Ascending, ReadTrace::none()),
            ),
            Box::new(RangeFileIterator::new(file, all.clone())),
        ];
        let mut iter = DbIterator::new(
            all,
            None,
            sources,
            Box::new(EmptyIterator::new()),
            None,
            self.merge_operator.clone(),
            IterationOrder::Ascending,
            ReadTrace::none(),
        )
        .await?;
        let mut writer = RangeFileWriter::create(
            &self.options.root_folder,
            &format!("{file_id:020}"),
            self.options.tuning.page_size,
        )?;
        while let Some(entry) = iter.next_entry().await? {
            writer.push(&entry)?;
        }
        let path = writer.finish()?;
        let new_file = Arc::new(RangeFile::open(
            path,
            self.options.page_source,
            self.disk_bytes.clone(),
        )?);
        self.install(&range, &frozen, new_file, frozen_seq);
        Ok(())
    }

    /// Replaces the file and the frozen delta of `range` with `file`.
    fn install(
        &self,
        range: &Arc<CachedRange>,
        frozen: &Arc<Delta>,
        file: Arc<RangeFile>,
        seq: u64,
    ) {
        let mut state = self.state.lock();
        if !self.is_indexed(range) {
            return;
        }
        let layers = range.layers.load_full();
        if !layers
            .frozen
            .as_ref()
            .is_some_and(|f| Arc::ptr_eq(f, frozen))
        {
            return;
        }
        self.stats.rebuilds.increment(1);
        self.stats.rebuild_bytes.increment(file.file_len());
        range.layers.store(Arc::new(Layers {
            active: layers.active.clone(),
            frozen: None,
            file: Some(file),
            file_seq: seq,
            end: range.end.clone(),
        }));
        self.access_clock.fetch_add(2, Ordering::Relaxed);
        self.enforce_budget_locked(&mut state, None);
    }

    fn is_indexed(&self, range: &Arc<CachedRange>) -> bool {
        self.index
            .load()
            .get(&range.start)
            .is_some_and(|r| Arc::ptr_eq(r, range))
    }

    /// Removes `range` and seals its active delta at `sealed_at`. The caller must hold the
    /// mutex.
    fn remove_locked(
        &self,
        state: &mut CacheState,
        range: &Arc<CachedRange>,
        sealed_at: u64,
        reason: RemoveReason,
    ) {
        if !self.is_indexed(range) {
            return;
        }
        range
            .layers
            .load()
            .active
            .sealed_at
            .store(sealed_at, Ordering::SeqCst);
        let mut index = RangeIndex::clone(&self.index.load());
        index.remove(&range.start);
        self.index.store(Arc::new(index));
        self.stats.removed(reason);
        self.update_gauges();
        self.remember_removal(state, range, reason);
    }

    /// Records the removal of an automatic range for later promotions.
    fn remember_removal(&self, state: &mut CacheState, range: &CachedRange, reason: RemoveReason) {
        let Some(auto) = &self.auto else {
            return;
        };
        if !range.auto || reason == RemoveReason::Closed {
            return;
        }
        let now = self.now_ms();
        let ghost_ms = auto.options.ghost_ttl.as_millis() as i64;
        state.ghosts.retain(|g| now - g.at_ms < ghost_ms);
        state.ghosts.push(Ghost {
            start: range.start.clone(),
            end: range.end.clone(),
            at_ms: now,
        });
        if reason.is_failure() {
            let base_ms = auto.options.backoff_base.as_millis() as i64;
            let level = state
                .backoff
                .get(&range.start)
                .map_or(0, |b| decayed_level(b, now, base_ms));
            state.backoff.insert(
                range.start.clone(),
                Backoff {
                    end: range.end.clone(),
                    level: (level + 1).min(MAX_BACKOFF_LEVEL),
                    at_ms: now,
                },
            );
            state
                .backoff
                .retain(|_, b| decayed_level(b, now, base_ms) > 0 || now - b.at_ms < base_ms);
        }
    }

    fn remove(&self, range: &Arc<CachedRange>, reason: RemoveReason) {
        let mut state = self.state.lock();
        let seq = state.last_applied_seq;
        self.remove_locked(&mut state, range, seq, reason);
    }

    /// Removes every range that overlaps `range`.
    pub(crate) fn remove_overlapping(&self, range: &BytesRange, reason: RemoveReason) {
        let mut state = self.state.lock();
        let ranges: Vec<_> = self
            .index
            .load()
            .values()
            .filter(|r| r.bounds().intersect(range).is_some())
            .cloned()
            .collect();
        let seq = state.last_applied_seq;
        for r in ranges {
            self.remove_locked(&mut state, &r, seq, reason);
        }
    }

    /// Removes all ranges.
    pub(crate) fn clear(&self, reason: RemoveReason) {
        let mut state = self.state.lock();
        self.clear_locked(&mut state, reason);
    }

    fn clear_locked(&self, state: &mut CacheState, reason: RemoveReason) {
        let ranges: Vec<_> = self.index.load().values().cloned().collect();
        let seq = state.last_applied_seq;
        for r in ranges {
            self.remove_locked(state, &r, seq, reason);
        }
    }

    /// Removes all ranges and rejects later loads. The batch writer calls this when it stops,
    /// on close and on errors such as fencing.
    pub(crate) fn close(&self) {
        let mut state = self.state.lock();
        self.clear_locked(&mut state, RemoveReason::Closed);
        state.closed = true;
    }

    /// Removes ranges until the range files fit the budget, never `skip`. Without automatic
    /// promotion, it removes the least recently read range first. With automatic promotion, it
    /// removes automatic ranges by increasing read density, and explicit ranges only after all
    /// automatic ones, least recently read first.
    fn enforce_budget_locked(&self, state: &mut CacheState, skip: Option<&Arc<CachedRange>>) {
        let now = self.now_ms();
        loop {
            let index = self.index.load_full();
            let attached: u64 = index.values().map(|r| r.file_bytes()).sum();
            if attached <= self.options.max_disk_bytes {
                break;
            }
            let candidates = index
                .values()
                .filter(|r| r.layers.load().file.is_some())
                .filter(|r| skip.is_none_or(|skip| !Arc::ptr_eq(skip, r)));
            let victim = match &self.auto {
                None => candidates
                    .min_by_key(|r| r.last_access.load(Ordering::Relaxed))
                    .cloned(),
                Some(_) => candidates
                    .map(|r| (r, self.eviction_key(r, now)))
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(r, _)| r.clone()),
            };
            let Some(victim) = victim else {
                break;
            };
            let seq = state.last_applied_seq;
            self.remove_locked(state, &victim, seq, RemoveReason::Evicted);
        }
        self.update_gauges();
    }

    /// The eviction order with automatic promotion: automatic ranges before explicit ones,
    /// then by read density, then least recently read first.
    fn eviction_key(&self, range: &CachedRange, now_ms: i64) -> (bool, f64, u64) {
        let density = if range.auto {
            self.density(range, now_ms)
        } else {
            0.0
        };
        (
            !range.auto,
            density,
            range.last_access.load(Ordering::Relaxed),
        )
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
        self.stats
            .file_bytes
            .set(self.disk_bytes.load(Ordering::Relaxed) as i64);
    }

    /// Returns true if a read at `seq` can use `layers`. See the sequence number rules.
    fn usable(layers: &Layers, seq: u64) -> bool {
        layers.file.is_some()
            && seq >= layers.file_seq
            && layers.active.sealed_at.load(Ordering::SeqCst) >= seq
    }

    /// Records that a read used `cached`.
    fn touch(&self, cached: &CachedRange) {
        let now = self.access_clock.load(Ordering::Relaxed);
        if cached.last_access.load(Ordering::Relaxed) != now {
            cached.last_access.store(now, Ordering::Relaxed);
        }
        if self.auto.is_some() {
            cached.heat.tick();
        }
    }

    /// Reads `key` at the last committed sequence number.
    ///
    /// The common path holds the index and the layers through `arc_swap` guards and takes no
    /// reference count, so concurrent reads of one range do not write a shared cache line.
    pub(crate) fn get(&self, key: &[u8], oracle: &dyn Oracle) -> CacheRead {
        for _ in 0..PIN_ATTEMPTS {
            let index = self.index.load();
            let Some(cached) = covering(&index, key) else {
                return CacheRead::Uncovered;
            };
            let layers = cached.layers.load();
            if layers.file.is_none() {
                return CacheRead::Uncovered;
            }
            let seq = oracle.last_committed_seq();
            if !Self::usable(&layers, seq) {
                continue;
            }
            self.touch(cached);
            return match lookup(&layers, key, seq) {
                Ok(Some(entry)) => match entry.value {
                    ValueDeletable::Value(_) => {
                        self.stats.get_hits.increment(1);
                        CacheRead::Answer(Some(KeyValue::from(entry)))
                    }
                    ValueDeletable::Tombstone => {
                        self.stats.get_absent.increment(1);
                        CacheRead::Answer(None)
                    }
                    ValueDeletable::Merge(_) => {
                        self.stats.get_merges.increment(1);
                        CacheRead::Resolve(Pinned {
                            layers: arc_swap::Guard::into_inner(layers),
                            seq,
                        })
                    }
                },
                Ok(None) => {
                    self.stats.get_absent.increment(1);
                    CacheRead::Answer(None)
                }
                Err(e) => {
                    warn!("range cache read failed [error={:?}]", e);
                    let cached = cached.clone();
                    drop(layers);
                    drop(index);
                    self.remove(&cached, RemoveReason::Error);
                    CacheRead::Uncovered
                }
            };
        }
        CacheRead::Uncovered
    }

    /// Pins the layers for a scan of `range` at the last committed sequence number, if ready
    /// cached ranges hold all of `range`. The ranges can be adjacent ranges, each starting where
    /// the one before ends. All of them must follow the sequence number rules at the same
    /// sequence number.
    pub(crate) fn pin_scan(&self, range: &BytesRange, oracle: &dyn Oracle) -> Option<PinnedScan> {
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
        for _ in 0..PIN_ATTEMPTS {
            let index = self.index.load();
            let first = covering(&index, start)?;
            let mut parts = vec![first.layers.load_full()];
            let mut last = first;
            while !last.contains_range(range) {
                last = index.get(last.end.as_ref()?)?;
                parts.push(last.layers.load_full());
            }
            if parts.iter().any(|l| l.file.is_none()) {
                return None;
            }
            let seq = oracle.last_committed_seq();
            if !parts.iter().all(|l| Self::usable(l, seq)) {
                continue;
            }
            // Only the first range counts as read: most scans end in it.
            self.touch(first);
            self.stats.scans.increment(1);
            return Some(PinnedScan { parts, seq });
        }
        None
    }

    pub(crate) fn ranges(&self) -> Vec<CachedRangeInfo> {
        self.index.load().values().map(|r| r.info()).collect()
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

    /// Returns the time of the latest removal of an automatic range that overlaps
    /// `[start, end)`, if it is within the ghost time.
    fn ghost_since(&self, start: &Bytes, end: &Option<Bytes>) -> Option<i64> {
        let auto = self.auto.as_ref()?;
        let now = self.now_ms();
        let ttl_ms = auto.options.ghost_ttl.as_millis() as i64;
        let state = self.state.lock();
        state
            .ghosts
            .iter()
            .filter(|g| now - g.at_ms < ttl_ms && overlaps(&g.start, &g.end, start, end))
            .map(|g| g.at_ms)
            .max()
    }

    /// Returns true if a recent failed promotion of a range that holds `key` asks to wait.
    fn backed_off(&self, key: &[u8]) -> bool {
        let Some(auto) = &self.auto else {
            return false;
        };
        let now = self.now_ms();
        let base_ms = auto.options.backoff_base.as_millis() as i64;
        let state = self.state.lock();
        state
            .backoff
            .range::<[u8], _>((Unbounded, Included(key)))
            .next_back()
            .is_some_and(|(_, b)| {
                b.end.as_ref().is_none_or(|end| key < end.as_ref())
                    && now - b.at_ms < backoff_window_ms(b.level, base_ms)
            })
    }

    /// Registers loading ranges for the parts of `[start, end)` that no range holds, and calls
    /// `snapshot` to pick the state that they load.
    ///
    /// The order matters. The new ranges go into the index first, and `publish_epoch`
    /// advances. Then the cache reads `inserted_seq`, and `snapshot` reads `last_seq`. The writer
    /// does the mirror image in `apply_write`: it advances `inserted_seq` after its memtable
    /// insert, and then loads `publish_epoch` before it checks the index. All four operations
    /// are `SeqCst`, so at least one side sees the other. If the writer misses the new epoch,
    /// this side reads its `inserted_seq` and synchronizes with its memtable insert, so the
    /// snapshot includes the write. If the writer sees the new epoch, it also sees the new
    /// index, finds the range, and waits for the mutex to apply the write to the delta.
    fn register(
        &self,
        start: &Bytes,
        end: Option<&Bytes>,
        origin: Origin,
        snapshot: impl FnOnce() -> Arc<DbSnapshot>,
    ) -> Result<(Vec<Arc<CachedRange>>, Arc<DbSnapshot>), SlateDBError> {
        let state = self.state.lock();
        if state.closed {
            return Err(SlateDBError::Closed);
        }
        let now = self.now_ms();
        let mut index = RangeIndex::clone(&self.index.load());
        let parts = uncovered(&index, start, end);
        let mut ranges = Vec::with_capacity(parts.len());
        for (part_start, part_end) in parts {
            let layers = Layers {
                active: Arc::new(Delta::new()),
                frozen: None,
                file: None,
                // A loading range serves no read. `load` sets the real value.
                file_seq: u64::MAX,
                end: part_end.clone(),
            };
            let range = Arc::new(CachedRange::new(
                part_start.clone(),
                part_end,
                layers,
                origin,
                now,
            ));
            index.insert(part_start, range.clone());
            ranges.push(range);
        }
        fail_point!(
            Arc::clone(&self.fp_registry),
            "range-cache-register-publish"
        );
        if !ranges.is_empty() {
            self.index.store(Arc::new(index));
            self.update_gauges();
        }
        self.publish_epoch.fetch_add(1, Ordering::SeqCst);
        fail_point!(
            Arc::clone(&self.fp_registry),
            "range-cache-register-snapshot"
        );
        // Synchronizes with the writer: every write up to `inserted` is in the memtable.
        let inserted = self.inserted_seq.load(Ordering::SeqCst);
        let snapshot = snapshot();
        debug_assert!(snapshot.seq() >= inserted);
        drop(state);
        Ok((ranges, snapshot))
    }

    /// Writes the rows of a loading range to a file and publishes the range. It stops with an
    /// error when `check_open` fails, which it calls every 1024 rows.
    async fn load(
        &self,
        range: &Arc<CachedRange>,
        mut iter: DbIterator,
        file_seq: u64,
        check_open: &(dyn Fn() -> Result<(), SlateDBError> + Send + Sync),
    ) -> Result<(), SlateDBError> {
        let file_id = {
            let mut state = self.state.lock();
            state.next_file_id += 1;
            state.next_file_id - 1
        };
        let mut writer = RangeFileWriter::create(
            &self.options.root_folder,
            &format!("{file_id:020}"),
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
            self.disk_bytes.clone(),
        )?);
        debug_assert_eq!(rows, file.rows());

        let mut state = self.state.lock();
        if !self.is_indexed(range) {
            // A removal or a close won the race. The file is deleted on drop.
            return Ok(());
        }
        let layers = range.layers.load_full();
        if layers.file.is_some() {
            return Ok(());
        }
        self.stats.loads.increment(1);
        self.stats.load_bytes.increment(file.file_len());
        let now = self.access_clock.fetch_add(2, Ordering::Relaxed) + 2;
        range.last_access.store(now - 1, Ordering::Relaxed);
        range.layers.store(Arc::new(Layers {
            active: layers.active.clone(),
            frozen: None,
            file: Some(file),
            file_seq,
            end: range.end.clone(),
        }));
        // An automatic range won admission against the ranges it displaces, so the budget never
        // evicts it right after its load.
        let skip = range.auto.then_some(range);
        self.enforce_budget_locked(&mut state, skip);
        Ok(())
    }
}

/// The result of [`RangeCache::get`].
pub(crate) enum CacheRead {
    /// No ready range holds the key, or the cache cannot answer now.
    Uncovered,
    /// The value of the key at the last committed sequence number.
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

/// Returns the newest version of `key` in `layers` at `seq`, or `None` if the key has no
/// version.
fn lookup(layers: &Layers, key: &[u8], seq: u64) -> io::Result<Option<RowEntry>> {
    let deltas = std::iter::once(&layers.active).chain(layers.frozen.as_ref());
    // A delta lookup needs the key as `Bytes`. Most reads of a range that takes few writes
    // find empty deltas, and skip the copy.
    let mut owned: Option<Bytes> = None;
    for delta in deltas {
        if delta.table.is_empty() {
            continue;
        }
        let key = owned.get_or_insert_with(|| Bytes::copy_from_slice(key));
        if let Some(entry) = delta.table.newest_version(key, seq) {
            return Ok(Some(entry));
        }
    }
    match &layers.file {
        Some(file) => file.get(key),
        None => Ok(None),
    }
}

/// Returns an iterator over `range` of `layers` at `seq`, through the iterator stack of the
/// normal read path. So merge operands resolve the same way as in the normal path.
async fn stack_iterator(
    layers: &Layers,
    seq: u64,
    range: BytesRange,
    merge_operator: Option<MergeOperatorType>,
) -> Result<DbIterator, SlateDBError> {
    let mut sources: Vec<Box<dyn RowEntryIterator + 'static>> = std::iter::once(&layers.active)
        .chain(layers.frozen.as_ref())
        .map(|delta| {
            Box::new(
                delta
                    .table
                    .range(range.clone(), IterationOrder::Ascending, ReadTrace::none()),
            ) as Box<dyn RowEntryIterator + 'static>
        })
        .collect();
    if let Some(file) = &layers.file {
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
pub(crate) fn scan_start(range: &BytesRange) -> Bytes {
    match RangeBounds::start_bound(range) {
        Included(key) | Excluded(key) => key.clone(),
        Unbounded => Bytes::new(),
    }
}

/// Returns true if a point read with `options` reads the latest committed state, which the
/// cache can serve.
pub(crate) fn can_serve_get(options: &ReadOptions) -> bool {
    matches!(options.durability_filter, DurabilityLevel::Memory)
        && !options.dirty
        && options.filter_context.is_none()
        && options.tracing_options.is_none()
}

/// Returns true if a scan with `options` reads the latest committed state in ascending order,
/// which the cache can serve.
pub(crate) fn can_serve_scan(options: &ScanOptions) -> bool {
    matches!(options.durability_filter, DurabilityLevel::Memory)
        && !options.dirty
        && matches!(options.order, IterationOrder::Ascending)
        && options.filter_context.is_none()
        && options.tracing_options.is_none()
}

impl DbInner {
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
        // `register` calls this after it publishes the new ranges. `last_seq` can be one more
        // than the last committed sequence number while a write is in flight. The snapshot then
        // includes that write if it is in the memtable already. See RFC 0040, "Loading a range".
        let (ranges, snapshot) = cache.register(&start, end.as_ref(), origin, || {
            DbSnapshot::new(self.clone(), Some(self.oracle.last_seq()))
        })?;
        let options = ScanOptions {
            // The snapshot sequence number can be above the last committed one. See above.
            dirty: true,
            cache_blocks: false,
            ..ScanOptions::default()
        };
        let mut result = Ok(());
        for range in &ranges {
            if result.is_err() {
                cache.remove(range, RemoveReason::Error);
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
                        max_seq: Some(snapshot.seq()),
                        prefix: None,
                    },
                )
                .await;
            result = match iter {
                Ok(iter) => {
                    cache
                        .load(range, iter, snapshot.seq(), &|| self.check_closed())
                        .await
                }
                Err(e) => Err(e),
            };
            if result.is_err() {
                cache.remove(range, RemoveReason::Error);
            }
        }
        drop(snapshot);
        result
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
    /// lost. After the call returns, point reads and ascending scans at the latest committed
    /// state read cached keys from local disk and memory. Other reads, such as reads in
    /// snapshots and transactions, use the normal read path.
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
    pub fn uncache_range<T>(&self, range: T) -> Result<(), crate::Error>
    where
        T: ByteRangeBounds,
    {
        let cache = self.inner.range_cache_or_err()?;
        let start = range.start_bound().map(Bytes::copy_from_slice);
        let end = range.end_bound().map(Bytes::copy_from_slice);
        if let Some(range) = BytesRange::try_new(start, end) {
            cache.remove_overlapping(&range, RemoveReason::Explicit);
        }
        Ok(())
    }

    /// Removes all ranges from the range cache.
    pub fn clear_range_cache(&self) -> Result<(), crate::Error> {
        self.inner
            .range_cache_or_err()?
            .clear(RemoveReason::Explicit);
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
            .map(|(start, end)| {
                (
                    b(start),
                    Arc::new(CachedRange::new(
                        b(start),
                        end.map(b),
                        Layers {
                            active: Arc::new(Delta::new()),
                            frozen: None,
                            file: None,
                            file_seq: 0,
                            end: end.map(b),
                        },
                        Origin::Explicit,
                        0,
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
}
