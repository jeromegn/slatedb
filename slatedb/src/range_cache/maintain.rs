//! Background maintenance for the range cache: flushing sealed buffers to runs, merging runs
//! into bases, and sealing active buffers early under memory pressure.
//!
//! - **Flush**: writes all sealed buffers of a range into one run file, builds the updated delta
//!   view incrementally, and sends `InstallRun`. When a range holds `max_runs`, the flush merges
//!   up to [`FLUSH_MERGE_RUNS`] newest free runs into the new run so flushes never block on a
//!   major merge. Flushes only write keys belonging to the range.
//! - **Major merge**: merges each chunk of a range's base with the oldest unabsorbed runs (at most
//!   `max_runs - 1`, none held by an in-flight flush) into a single new chunk when run bytes reach
//!   `min_rebuild_delta_bytes` and either `major_merge_factor * base` or `max_runs - headroom`
//!   runs. Adaptive headroom tracks flushes that land while a merge runs. Unpaid automatic merges
//!   defer until reads pay for them or the range reaches its run limit.
//! - **Split**: divides a range whose base exceeds `2 * range_target_bytes` in two without copying
//!   when no flush or major merge runs for it. Splits take priority over major merges.
//! - **Quiet merge**: joins two adjacent quiet ranges (no runs, no sealed buffers, empty active
//!   buffer) whose bases are both below `range_target_bytes / 8` without copying file data.
//! - **View races**: flushes and major merges run concurrently on blocking threads. If one
//!   changes the view before the other installs, the second rebuilds its view on the new state
//!   (up to [`VIEW_ATTEMPTS`] times) without rewriting its file.
//! - **Pressure**: above half of `max_recent_bytes`, seals the largest active buffers at or above
//!   `min_seal_bytes` so flushes free buffer memory.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fail_parallel::fail_point;
use futures::stream::BoxStream;
use log::{debug, warn};
use tokio::task::JoinHandle;

use super::delta::Delta;
use super::file::{RangeFile, RangeFileIterator, RangeFileWriter, RowLocators};
use super::merge::merge_versions_in;
use super::split;
use super::spread::{Measured, SliceTally, Spread};
use super::stats::RemoveReason;
use super::writer::{same_view, Target, ViewSwap};
use super::{
    BuildCost, CachedRange, Chunks, DeltaView, Layers, RangeCache, RangeCacheTuning, RangeIndex,
    Run, MAX_RUNS_RANGE,
};
use crate::bytes_range::BytesRange;
use crate::db::DbInner;
use crate::db_iter::DbIterator;
use crate::dispatcher::{MessageHandler, MessageTickerDef};
use crate::error::SlateDBError;
use crate::iter::{EmptyIterator, IterationOrder, RowEntryIterator};
use crate::read_trace::ReadTrace;
use crate::utils::spawn_bg_task;

/// The name of the maintenance task on the database's task executor.
pub(crate) const RANGE_CACHE_MAINTENANCE_TASK_NAME: &str = "range_cache_maintenance";

/// How often the task walks every range, in case it missed a notification.
const TICK: Duration = Duration::from_millis(20);
/// The most flushes in flight.
const MAX_FLUSHES: usize = 2;
/// The most major merges in flight.
const MAX_MERGES: usize = 2;
/// The most splits in flight. A split builds views and writes no file.
const MAX_SPLITS: usize = 2;
/// The ticks between two updates of the gauges of base and run files, which walk every file:
/// about a second.
const FILE_GAUGE_TICKS: u64 = 50;
/// Max quiet merges requested per tick walk.
const MAX_QUIET_MERGES: usize = 4;
/// The most early seals that one pass asks for.
const PRESSURE_SEALS: usize = 4;
/// Max newest runs merged into a flush when the range is already at `max_runs`.
pub(super) const FLUSH_MERGE_RUNS: usize = 2;
/// The headroom of the count trigger before a range's first major merge installs: a few flushes.
/// After it, the range's own merges set it (see the module documentation).
pub(super) const INITIAL_HEADROOM: usize = 4;
/// The most times that a job builds its view and sends its install, when the writer rejects it
/// because the range's view moved on.
pub(super) const VIEW_ATTEMPTS: usize = 8;
// A flush at the cap brings a range below `max_runs` with room to spare, and a major merge of at
// most `max_runs - 1` runs leaves it a run to merge.
const _: () = assert!(FLUSH_MERGE_RUNS < *MAX_RUNS_RANGE.start());

/// A reason for the task to look for work.
pub(crate) enum MaintenanceMessage {
    /// A seal, a base or the end of a job of the range. The task does not keep a removed range
    /// alive: its buffers must be freed as soon as readers drop them.
    Range(Notified),
    /// The ticker.
    Tick,
}

/// The range of a notification.
pub(crate) struct Notified(Weak<CachedRange>);

impl std::fmt::Debug for MaintenanceMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaintenanceMessage::Range(_) => f.write_str("Range"),
            MaintenanceMessage::Tick => f.write_str("Tick"),
        }
    }
}

/// The queue to the maintenance task, and its jobs in flight.
pub(super) struct MaintenanceQueue {
    tx: async_channel::Sender<MaintenanceMessage>,
    rx: async_channel::Receiver<MaintenanceMessage>,
    /// The flushes in flight. A job counts itself out before it notifies the task.
    flushes: AtomicUsize,
    /// The major merges in flight.
    merges: AtomicUsize,
    /// The splits in flight.
    splits: AtomicUsize,
}

impl MaintenanceQueue {
    pub(super) fn new() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self {
            tx,
            rx,
            flushes: AtomicUsize::new(0),
            merges: AtomicUsize::new(0),
            splits: AtomicUsize::new(0),
        }
    }

    /// Tells the task to look at `range`. One message per range in flight is enough: the task
    /// clears the range's flag before it looks at its layers.
    pub(super) fn notify(&self, range: &Arc<CachedRange>) {
        if !range.jobs.notified.swap(true, Ordering::AcqRel) {
            let _ = self
                .tx
                .try_send(MaintenanceMessage::Range(Notified(Arc::downgrade(range))));
        }
    }
}

/// The jobs of one range. See [`CachedRange::jobs`].
#[derive(Default)]
pub(super) struct RangeJobs {
    flushing: AtomicBool,
    merging: AtomicBool,
    /// True while a split of the range runs. No flush or major merge starts meanwhile.
    splitting: AtomicBool,
    /// True after requesting a quiet merge until rejected (merged ranges leave the index).
    pub(super) quiet_merge_asked: AtomicBool,
    /// True while a message about the range is on its way to the task.
    notified: AtomicBool,
    /// True while the range waits in the task's flush queue, or in its merge queue. Only the task
    /// loads and stores them.
    flush_queued: AtomicBool,
    merge_queued: AtomicBool,
    /// The newest runs that the flush in flight replaces, 0 for a plain flush. Only the task
    /// stores it, before it sets `flushing`, and it means nothing while `flushing` is false.
    flush_replaces: AtomicUsize,
    /// The `hi` of the newest run that the major merge in flight holds. Only the task stores it,
    /// before it sets `merging`, and it means nothing while `merging` is false.
    merge_hi: AtomicU64,
    /// The flushes of the range that installed. Only the writer stores it, before it stores the
    /// layers with the flush's run, so a look at those layers sees the count.
    flushes: AtomicU64,
    /// The bytes of the runs that flushes of the range installed, flushes at the run cap
    /// included. Only the writer stores it.
    flushed_bytes: AtomicU64,
    /// The major merges of the range that installed, and the bytes of their bases. Only the
    /// writer stores them.
    major_merges: AtomicU64,
    major_merge_bytes: AtomicU64,
    /// `flushes` when the task queued the range's major merge. Only the task stores it.
    merge_queued_at: AtomicU64,
    /// The headroom of the count trigger (see the module documentation), or 0 before the first
    /// major merge installs. Only the writer stores it, when it installs a major merge.
    headroom: AtomicUsize,
}

impl RangeJobs {
    /// The jobs of a range that a split cut from a range with `parent`: none in flight, no counts,
    /// and the parent's headroom.
    pub(super) fn inherit(parent: &RangeJobs) -> Self {
        let jobs = Self::default();
        jobs.headroom
            .store(parent.headroom.load(Ordering::Relaxed), Ordering::Relaxed);
        jobs
    }

    /// Job state for a quietly merged range: idle, zero counts, and the larger headroom.
    pub(super) fn merged(a: &RangeJobs, b: &RangeJobs) -> Self {
        let jobs = Self::default();
        let headroom = a
            .headroom
            .load(Ordering::Relaxed)
            .max(b.headroom.load(Ordering::Relaxed));
        jobs.headroom.store(headroom, Ordering::Relaxed);
        jobs
    }

    /// True while a major merge of the range runs.
    pub(super) fn merging(&self) -> bool {
        self.merging.load(Ordering::Acquire)
    }

    /// True while a flush of the range runs.
    pub(super) fn flushing(&self) -> bool {
        self.flushing.load(Ordering::Acquire)
    }

    /// True while a split of the range runs.
    pub(super) fn splitting(&self) -> bool {
        self.splitting.load(Ordering::Acquire)
    }

    /// The newest runs that the flush in flight replaces, which a major merge must leave alone.
    /// Until the flush's end clears `flushing`, it may count runs that the flush already
    /// replaced, which only leaves a merge fewer runs.
    fn flush_held(&self) -> usize {
        if self.flushing() {
            self.flush_replaces.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// The newest of `runs` (newest first) that no major merge in flight holds: all of them, or
    /// the ones above the newest run of the merge. Until the merge's end clears `merging`, a
    /// rejected merge's inputs count as held, which only leaves a flush fewer runs.
    fn free_newest(&self, runs: &[Arc<Run>]) -> usize {
        if !self.merging() {
            return runs.len();
        }
        let hi = self.merge_hi.load(Ordering::Relaxed);
        runs.iter().take_while(|r| r.hi > hi).count()
    }

    /// The headroom of the count trigger for a range of at most `max_runs` runs.
    pub(super) fn headroom(&self, max_runs: usize) -> usize {
        let h = match self.headroom.load(Ordering::Relaxed) {
            0 => INITIAL_HEADROOM,
            h => h,
        };
        h.clamp(1, (max_runs / 2).max(1))
    }

    /// Counts a flush that installs a run of `bytes`. The writer calls it.
    pub(super) fn record_flush(&self, bytes: u64) {
        self.flushes.fetch_add(1, Ordering::Relaxed);
        self.flushed_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// The bytes of the runs that the range's flushes installed.
    pub(super) fn flushed_bytes(&self) -> u64 {
        self.flushed_bytes.load(Ordering::Relaxed)
    }

    /// The major merges of the range that installed.
    pub(super) fn major_merges(&self) -> u64 {
        self.major_merges.load(Ordering::Relaxed)
    }

    /// The bytes of the bases that the range's major merges installed.
    pub(super) fn major_merge_bytes(&self) -> u64 {
        self.major_merge_bytes.load(Ordering::Relaxed)
    }

    /// Counts a major merge that installs a base of `bytes`, sets the headroom from the flushes
    /// that installed since the task queued it, and returns their number. The writer calls it.
    pub(super) fn record_major(&self, bytes: u64) -> u64 {
        self.major_merges.fetch_add(1, Ordering::Relaxed);
        self.major_merge_bytes.fetch_add(bytes, Ordering::Relaxed);
        let landed = self
            .flushes
            .load(Ordering::Relaxed)
            .saturating_sub(self.merge_queued_at.load(Ordering::Relaxed));
        let need = usize::try_from(landed)
            .unwrap_or(usize::MAX)
            .saturating_add(1)
            .min(*MAX_RUNS_RANGE.end());
        let old = match self.headroom.load(Ordering::Relaxed) {
            0 => INITIAL_HEADROOM,
            h => h,
        };
        self.headroom
            .store(need.max((old + need) / 2), Ordering::Relaxed);
        landed
    }
}

/// Runs file I/O or view builds on a blocking thread to keep async read workers free.
pub(super) async fn off_workers<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, SlateDBError> + Send + 'static,
) -> Result<T, SlateDBError> {
    #[allow(clippy::disallowed_methods)]
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| SlateDBError::BackgroundTaskPanic(format!("range cache file work: {e}")))?
}

/// Elapsed nanoseconds since `started` on `clock`.
fn elapsed_nanos(
    clock: &dyn slatedb_common::clock::SystemClock,
    started: chrono::DateTime<chrono::Utc>,
) -> u64 {
    (clock.now() - started)
        .num_nanoseconds()
        .unwrap_or(0)
        .max(0) as u64
}

/// True if `held` (newest first) are the newest of `runs` (newest first).
fn holds_newest(runs: &[Arc<Run>], held: &[Arc<Run>]) -> bool {
    held.len() <= runs.len() && runs.iter().zip(held).all(|(a, b)| Arc::ptr_eq(a, b))
}

/// True if `held` (newest first) are the oldest of `runs` (newest first).
fn holds_oldest(runs: &[Arc<Run>], held: &[Arc<Run>]) -> bool {
    runs.len()
        .checked_sub(held.len())
        .is_some_and(|at| holds_newest(&runs[at..], held))
}

/// Why a major merge of a range is due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Trigger {
    /// The run bytes reached `major_merge_factor` times the base.
    Bytes,
    /// The runs reached `max_runs` less the range's headroom.
    Runs,
}

/// The run bytes at which a major merge of a range with this base starts: `major_merge_factor`
/// times the base, and at least `min_rebuild_delta_bytes`.
pub(super) fn major_threshold(base_bytes: u64, tuning: &RangeCacheTuning) -> u64 {
    ((tuning.major_merge_factor * base_bytes as f64) as u64)
        .max(tuning.min_rebuild_delta_bytes as u64)
}

/// Returns whether `layers` call for a major merge, and why. Uses the range's own share of shared
/// base and run files so splitting does not trigger merges early.
pub(super) fn major_due(
    range: &CachedRange,
    layers: &Layers,
    tuning: &RangeCacheTuning,
) -> Option<Trigger> {
    if layers.chunks.loading() {
        return None;
    }
    let runs = layers.runs.len();
    let run_bytes = layers.run_share(&range.start);
    if runs == 0 || run_bytes < tuning.min_rebuild_delta_bytes as u64 {
        return None;
    }
    if run_bytes >= major_threshold(layers.base_share(&range.start).0, tuning) {
        return Some(Trigger::Bytes);
    }
    (runs + range.jobs.headroom(tuning.max_runs) >= tuning.max_runs).then_some(Trigger::Runs)
}

/// Returns true when a range's delta reaches its run limit (count trigger or runs as large as its
/// base). Unpaid automatic ranges are evicted at this point.
pub(super) fn at_run_limit(range: &CachedRange, layers: &Layers, trigger: Trigger) -> bool {
    trigger == Trigger::Runs || layers.run_share(&range.start) >= layers.base_share(&range.start).0
}

/// Returns `range`'s base chunks if it is idle, has no runs or buffered writes, and its base share
/// is below `small` bytes.
pub(super) fn quiet_base(range: &CachedRange, small: u64) -> Option<Chunks> {
    let jobs = &range.jobs;
    if jobs.flushing()
        || jobs.merging()
        || jobs.splitting()
        || jobs.quiet_merge_asked.load(Ordering::Relaxed)
    {
        return None;
    }
    let layers = range.layers.load();
    let quiet = !layers.chunks.loading()
        && layers.runs.is_empty()
        && layers.sealed.is_empty()
        && layers.active.is_empty();
    (quiet && layers.base_share(&range.start).0 < small).then(|| layers.chunks.clone())
}

/// Returns cut keys to split `range` in two when its base exceeds `2 * range_target_bytes`.
pub(super) fn split_cuts(
    range: &CachedRange,
    layers: &Layers,
    tuning: &RangeCacheTuning,
) -> Option<Vec<Bytes>> {
    let target = tuning.range_target_bytes;
    if target == 0 || layers.chunks.loading() {
        return None;
    }
    if layers.base_share(&range.start).0 <= target.saturating_mul(2) {
        return None;
    }
    let cuts = split::cut_keys(&layers.chunks, &range.start, range.end.as_ref(), 2);
    (!cuts.is_empty()).then_some(cuts)
}

/// First newest-first run index a major merge may absorb, leaving `flush_held` runs and at least
/// one run free when at `max_runs`.
pub(super) fn mergeable_from(runs: usize, flush_held: usize, max_runs: usize) -> usize {
    flush_held
        .max(runs.saturating_sub(max_runs.saturating_sub(1)))
        .min(runs)
}

/// Number of newest runs a flush merges into its output: `0` below `max_runs`, otherwise up to
/// [`FLUSH_MERGE_RUNS`] runs free of an in-flight major merge.
pub(super) fn flush_replaces(runs: usize, free: usize, max_runs: usize) -> Option<usize> {
    if runs < max_runs {
        return Some(0);
    }
    match free.min(FLUSH_MERGE_RUNS) {
        0 => None,
        replaced => Some(replaced),
    }
}

/// Returns the oldest runs (newest-first indices) to absorb and the trigger, if a major merge is
/// due.
pub(super) fn plan_major(
    range: &CachedRange,
    layers: &Layers,
    tuning: &RangeCacheTuning,
    flush_held: usize,
) -> Option<(Range<usize>, Trigger)> {
    let trigger = major_due(range, layers, tuning)?;
    let runs = layers.runs.len();
    let first = mergeable_from(runs, flush_held, tuning.max_runs);
    (first < runs).then_some((first..runs, trigger))
}

impl RangeCache {
    /// The receiver of the maintenance task's queue, for its registration.
    pub(crate) fn maintenance_receiver(&self) -> async_channel::Receiver<MaintenanceMessage> {
        self.maintenance.rx.clone()
    }

    /// Writes `buffers`, the sealed buffers of `range` (newest first), with the versions of
    /// `replaces`, its newest runs (newest first, none for a plain flush), to one run of
    /// `(lo, hi]`, and installs it in their place with the range's view. The task chose them (see
    /// [`Job::Flush`]).
    async fn flush(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        buffers: Vec<Arc<Delta>>,
        replaces: Vec<Arc<Run>>,
        lo: u64,
        hi: u64,
    ) -> Result<(), SlateDBError> {
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-flush");
        let (run, rows) = {
            let (cache, buffers, replaces) = (self.clone(), buffers.clone(), replaces.clone());
            let keys = range.bounds();
            off_workers(move || cache.write_run(&buffers, &replaces, lo, hi, &keys)).await?
        };
        // Checkers keep the sealed buffers pinned by this job across many batches.
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-install-run");
        let installed = self
            .install_run_with_view(range, Arc::new(run), Arc::new(rows), buffers, replaces)
            .await?;
        if installed {
            self.evict(&[]).await;
        }
        Ok(())
    }

    /// Builds the view of `range` without `replaces` and with `run`, which a flush wrote from
    /// `buffers` and `replaces` and whose rows are `rows`, on the view that the range holds, and
    /// installs both. Builds and sends again while the writer rejects the install because the
    /// view moved on (a major merge installed meanwhile).
    async fn install_run_with_view(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        run: Arc<Run>,
        rows: Arc<RowLocators>,
        buffers: Vec<Arc<Delta>>,
        replaces: Vec<Arc<Run>>,
    ) -> Result<bool, SlateDBError> {
        for attempt in 0..VIEW_ATTEMPTS {
            let was = {
                let layers = range.layers.load();
                if !holds_newest(&layers.runs, &replaces) {
                    return Ok(false);
                }
                layers.view.clone()
            };
            let view = {
                let (cache, run, rows, replaces, was) = (
                    self.clone(),
                    run.clone(),
                    rows.clone(),
                    replaces.clone(),
                    was.clone(),
                );
                off_workers(move || {
                    let mut cost = BuildCost::default();
                    let started = cache.clock.now();
                    let view = DeltaView::after_flush(
                        was.as_deref(),
                        &replaces,
                        &run,
                        &rows,
                        &cache.views,
                        &mut cost,
                    )?;
                    cost.nanos = elapsed_nanos(&*cache.clock, started);
                    cache.stats.view[0].built(&cost);
                    Ok(view)
                })
                .await?
            };
            if attempt > 0 {
                self.stats.view[0].retries.increment(1);
            }
            fail_point!(
                Arc::clone(&self.fp_registry),
                "range-cache-install-run-view"
            );
            let swap = ViewSwap {
                new: Some(Arc::new(view)),
                was: was.clone(),
            };
            if self
                .install_run(range, run.clone(), buffers.clone(), replaces.clone(), swap)
                .await
            {
                return Ok(true);
            }
            if same_view(&range.layers.load().view, &was) || !self.is_indexed(range) {
                return Ok(false);
            }
        }
        Ok(false)
    }

    /// Writes versions in `keys` from `buffers` and `runs` (both newest first) to a run of
    /// `(lo, hi]` and returns the run with its row locators. Restricting to `keys` drops versions
    /// for sibling ranges from buffers and runs shared after a split.
    pub(super) fn write_run(
        &self,
        buffers: &[Arc<Delta>],
        runs: &[Arc<Run>],
        lo: u64,
        hi: u64,
        keys: &BytesRange,
    ) -> Result<(Run, RowLocators), SlateDBError> {
        let mut writer = RangeFileWriter::create_run(
            &self.options.root_folder,
            &self.new_file_name(),
            self.options.tuning.page_size,
        )?
        .with_locators();
        let sealed: Vec<(&Delta, u64)> = buffers
            .iter()
            .map(|b| (&**b, b.sealed_at().load(Ordering::Acquire)))
            .collect();
        let runs: Vec<&Run> = runs.iter().map(|r| &**r).collect();
        debug_assert!(
            sealed.iter().all(|(b, _)| b.lo() >= lo)
                && runs.iter().all(|r| r.lo >= lo && r.hi <= hi),
            "a merge input outside ({lo}, {hi}]"
        );
        merge_versions_in(&mut writer, &sealed, &runs, keys)?;
        let (path, rows) = writer.finish_with_locators()?;
        let file = RangeFile::open(path, self.options.page_source, self.files.clone())?;
        Ok((Run::new(file, lo, hi)?, rows))
    }

    /// Merges `chunks` with `absorbed` (newest first) into a new single-chunk base at the newest
    /// absorbed run's `hi`, and installs it with the updated delta view.
    pub(super) async fn major_merge(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        chunks: Chunks,
        absorbed: Vec<Arc<Run>>,
        trigger: Trigger,
    ) -> Result<(), SlateDBError> {
        let Some(base_seq) = absorbed.first().map(|r| r.hi) else {
            return Ok(());
        };
        range.mark_build();
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-rebuild");
        let (new_base, spread) = {
            let (cache, absorbed, chunks) = (self.clone(), absorbed.clone(), chunks.clone());
            let (start, end) = (range.start.clone(), range.end.clone());
            let handle = tokio::runtime::Handle::current();
            let (file, spread) = off_workers(move || {
                let merge = cache.merge_files(&absorbed, &chunks, &start, end.as_ref(), base_seq);
                handle.block_on(merge)
            })
            .await?;
            (Arc::new(file), spread)
        };
        if self
            .install_major_with_view(range, &chunks, absorbed, new_base, base_seq)
            .await?
        {
            self.stats.spread(&spread);
            if trigger == Trigger::Runs {
                self.stats.rebuilds_by_runs.increment(1);
            }
            self.evict(&[]).await;
        }
        Ok(())
    }

    async fn install_major_with_view(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        chunks: &Chunks,
        absorbed: Vec<Arc<Run>>,
        new_base: Arc<RangeFile>,
        base_seq: u64,
    ) -> Result<bool, SlateDBError> {
        for attempt in 0..VIEW_ATTEMPTS {
            let was = {
                let layers = range.layers.load();
                if !holds_oldest(&layers.runs, &absorbed) {
                    return Ok(false);
                }
                layers.view.clone()
            };
            let Some(old) = was.clone() else {
                return Ok(false);
            };
            let view = {
                let (cache, absorbed) = (self.clone(), absorbed.clone());
                off_workers(move || {
                    let mut cost = BuildCost::default();
                    let started = cache.clock.now();
                    let view = old.without(&absorbed, &mut cost)?;
                    cost.nanos = elapsed_nanos(&*cache.clock, started);
                    cache.stats.view[1].built(&cost);
                    Ok(view)
                })
                .await?
            };
            if attempt > 0 {
                self.stats.view[1].retries.increment(1);
            }
            fail_point!(
                Arc::clone(&self.fp_registry),
                "range-cache-install-major-view"
            );
            let swap = ViewSwap {
                new: view.map(Arc::new),
                was: was.clone(),
            };
            if self
                .install_major(
                    range,
                    chunks,
                    absorbed.clone(),
                    new_base.clone(),
                    base_seq,
                    swap,
                )
                .await
            {
                return Ok(true);
            }
            if same_view(&range.layers.load().view, &was) || !self.is_indexed(range) {
                return Ok(false);
            }
        }
        Ok(false)
    }

    /// Plans a zero-copy split of `range` at `cuts` on a blocking thread and asks the writer to
    /// install it. Runs only when no flush or major merge is active for `range`.
    pub(super) async fn split(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        cuts: Vec<Bytes>,
    ) -> Result<bool, SlateDBError> {
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-split");
        let layers = range.layers.load_full();
        let plan = {
            let cache = self.clone();
            let (start, end) = (range.start.clone(), range.end.clone());
            off_workers(move || {
                let mut cost = BuildCost::default();
                let started = cache.clock.now();
                let plan = split::plan(
                    &layers,
                    &layers.chunks,
                    &start,
                    end.as_ref(),
                    &cuts,
                    &mut cost,
                )?;
                cost.nanos = elapsed_nanos(&*cache.clock, started);
                cache.stats.view[2].built(&cost);
                Ok(plan)
            })
            .await?
        };
        let Some(mut plan) = plan else {
            return Ok(false);
        };
        plan.rate = range.heat.last_rate();
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-split-install");
        Ok(self.install_split(range, plan).await)
    }

    /// Writes the state at `base_seq` of `chunks` and `runs` (newest first) into a new base file
    /// and measures where absorbed delta landed across virtual base slices.
    pub(super) async fn merge_files(
        &self,
        runs: &[Arc<Run>],
        chunks: &Chunks,
        start: &Bytes,
        end: Option<&Bytes>,
        base_seq: u64,
    ) -> Result<(RangeFile, Spread), SlateDBError> {
        let mut writer = RangeFileWriter::create(
            &self.options.root_folder,
            &self.new_file_name(),
            self.options.tuning.page_size,
        )?;
        let mut spread = Spread::default();
        for (i, chunk) in chunks.iter().enumerate() {
            debug_assert!(chunk.base_seq <= base_seq, "a chunk newer than its merge");
            let span = chunks.span(i, Some(start), end);
            let tally = SliceTally::new(
                chunk.file.clone(),
                split::span_start(&span),
                split::span_end(&span),
            )?;
            let mut sources: Vec<Box<dyn RowEntryIterator + 'static>> = runs
                .iter()
                .filter(|run| !chunk.absorbed(run))
                .map(|run| {
                    let source = RangeFileIterator::new(run.file.clone(), span.clone());
                    Box::new(Measured::new(source, tally.clone()))
                        as Box<dyn RowEntryIterator + 'static>
                })
                .collect();
            sources.push(Box::new(RangeFileIterator::new(
                chunk.file.clone(),
                span.clone(),
            )));
            let mut iter = DbIterator::new(
                span,
                None,
                sources,
                Box::new(EmptyIterator::new()),
                Some(base_seq),
                self.merge_operator.clone(),
                IterationOrder::Ascending,
                ReadTrace::none(),
            )
            .await?;
            while let Some(entry) = iter.next_entry().await? {
                writer.push(&entry)?;
            }
            spread.add(&tally.spread());
        }
        let path = writer.finish()?;
        let file = RangeFile::open(path, self.options.page_source, self.files.clone())?;
        Ok((file, spread))
    }
}

/// The maintenance task. See the module documentation.
pub(crate) struct Maintainer {
    db: Arc<DbInner>,
    cache: Arc<RangeCache>,
    /// Jobs in flight, for the cleanup.
    jobs: Vec<JoinHandle<Result<(), SlateDBError>>>,
    /// Ranges whose sealed buffers wait for a flush slot, in the order the task found them: so
    /// the oldest sealed buffers flush first, and a range that seals again keeps its place.
    flush_queue: VecDeque<Weak<CachedRange>>,
    /// Ranges whose major merge waits for a merge slot.
    merge_queue: VecDeque<Weak<CachedRange>>,
    /// The ticks handled, for the gauges that the task sets every [`FILE_GAUGE_TICKS`] ticks.
    ticks: u64,
}

/// A job of the maintenance task, with the inputs that the task chose when it started it (see
/// "Which runs a job takes" in the module documentation).
enum Job {
    /// Write `buffers`, every sealed buffer of the range (newest first), and the versions of
    /// `replaces`, its newest runs (newest first), to one run of `(lo, hi]`.
    Flush {
        buffers: Vec<Arc<Delta>>,
        replaces: Vec<Arc<Run>>,
        lo: u64,
        hi: u64,
    },
    /// Merge `chunks`, the base, with `absorbed`, the oldest runs (newest first), into a new base.
    Major {
        chunks: Chunks,
        absorbed: Vec<Arc<Run>>,
        trigger: Trigger,
    },
    /// Split the range at `cuts` (see [`split_cuts`]).
    Split { cuts: Vec<Bytes> },
}

/// The kind of a job: which flag of its range and which count of jobs in flight it holds.
#[derive(Clone, Copy)]
enum JobKind {
    Flush,
    Major,
    Split,
}

impl JobKind {
    /// The range's flag of this kind of job, and the count of these jobs in flight.
    fn slots<'a>(
        self,
        range: &'a CachedRange,
        queue: &'a MaintenanceQueue,
    ) -> (&'a AtomicBool, &'a AtomicUsize) {
        match self {
            JobKind::Flush => (&range.jobs.flushing, &queue.flushes),
            JobKind::Major => (&range.jobs.merging, &queue.merges),
            JobKind::Split => (&range.jobs.splitting, &queue.splits),
        }
    }
}

impl Job {
    fn name(&self) -> &'static str {
        match self {
            Job::Flush { .. } => "range_cache_flush",
            Job::Major { .. } => "range_cache_major_merge",
            Job::Split { .. } => "range_cache_split",
        }
    }
}

impl Maintainer {
    pub(crate) fn new(db: Arc<DbInner>, cache: Arc<RangeCache>) -> Self {
        // The task sends requests to the writer, which must be able to wake it.
        cache.bind_writer(&db.write_notifier);
        Self {
            db,
            cache,
            jobs: Vec::new(),
            flush_queue: VecDeque::new(),
            merge_queue: VecDeque::new(),
            ticks: 0,
        }
    }

    /// Handles one message: looks at the notified range, or at every range on a tick, then
    /// starts queued jobs while slots are free and seals large buffers early under pressure.
    fn pass(&mut self, message: MaintenanceMessage) {
        if self.db.check_closed().is_err() || self.cache.writer.is_closed() {
            return;
        }
        self.jobs.retain(|job| !job.is_finished());
        let index = self.cache.index.load_full();
        match message {
            MaintenanceMessage::Range(Notified(range)) => {
                if let Some(range) = range.upgrade() {
                    // Clear the flag first: a notification after this point sends its own
                    // message, and the swap makes the state before one that sent none visible
                    // to the look below.
                    range.jobs.notified.swap(false, Ordering::AcqRel);
                    if self.cache.is_indexed(&range) {
                        self.look(&range);
                    }
                }
            }
            MaintenanceMessage::Tick => {
                for range in index.values() {
                    self.look(range);
                }
                self.merge_quiet(&index);
                // The file and metadata gauges change with every install; the writer updates
                // them only when the index changes.
                self.cache.update_gauges();
                if self.ticks.is_multiple_of(FILE_GAUGE_TICKS) {
                    self.cache.update_file_gauges();
                }
                self.ticks += 1;
            }
        }
        self.start_queued();
        self.seal_under_pressure(&index);
    }

    /// Queues the jobs that `range`, an indexed range, calls for, unless they run or wait. A split
    /// that is due starts at once if no job runs for the range and a split slot is free, and goes
    /// before a major merge, which would rewrite the whole range: until it is done, the range's
    /// major merge waits (a tick looks again), and only a flush can start.
    fn look(&mut self, range: &Arc<CachedRange>) {
        let layers = range.layers.load();
        let jobs = &range.jobs;
        if jobs.splitting() {
            return;
        }
        let tuning = &self.cache.options.tuning;
        if let Some(cuts) = split_cuts(range, &layers, tuning) {
            if !jobs.flushing()
                && !jobs.merging()
                && self.cache.maintenance.splits.load(Ordering::Acquire) < MAX_SPLITS
            {
                drop(layers);
                self.start(range, Job::Split { cuts });
                return;
            }
            if !layers.sealed.is_empty()
                && !jobs.flushing()
                && !jobs.flush_queued.swap(true, Ordering::Relaxed)
            {
                self.flush_queue.push_back(Arc::downgrade(range));
            }
            return;
        }
        if !layers.sealed.is_empty()
            && !jobs.flushing()
            && !jobs.flush_queued.swap(true, Ordering::Relaxed)
        {
            self.flush_queue.push_back(Arc::downgrade(range));
        }
        if jobs.merging() || jobs.merge_queued.load(Ordering::Relaxed) {
            return;
        }
        // A flush in flight may defer the merge: its end notifies the task.
        if plan_major(range, &layers, tuning, jobs.flush_held()).is_none() {
            return;
        }
        jobs.merge_queued_at
            .store(jobs.flushes.load(Ordering::Relaxed), Ordering::Relaxed);
        jobs.merge_queued.store(true, Ordering::Relaxed);
        self.merge_queue.push_back(Arc::downgrade(range));
    }

    /// Requests quiet merges for up to [`MAX_QUIET_MERGES`] adjacent pairs whose bases are both
    /// below `range_target_bytes / 8`. Merged ranges stay below `range_target_bytes / 4`, well
    /// below the split threshold (`2 * range_target_bytes`).
    fn merge_quiet(&self, index: &RangeIndex) {
        let small = self.cache.options.tuning.quiet_merge_below();
        if small == 0 {
            return;
        }
        let mut asked = 0;
        let mut left: Option<(&Arc<CachedRange>, Chunks)> = None;
        for range in index.values() {
            let quiet = quiet_base(range, small);
            if let (Some((l, l_chunks)), Some(chunks)) = (&left, &quiet) {
                if l.end.as_ref() == Some(&range.start) && CachedRange::may_merge(l, range) {
                    l.jobs.quiet_merge_asked.store(true, Ordering::Relaxed);
                    range.jobs.quiet_merge_asked.store(true, Ordering::Relaxed);
                    let rate = l.heat.last_rate() + range.heat.last_rate();
                    self.cache
                        .ask_quiet_merge(l, range, [l_chunks.clone(), chunks.clone()], rate);
                    asked += 1;
                    if asked >= MAX_QUIET_MERGES {
                        return;
                    }
                    left = None;
                    continue;
                }
            }
            left = quiet.map(|chunks| (range, chunks));
        }
    }

    /// The flush that `range` calls for, with its inputs: every sealed buffer, and when the range
    /// holds `max_runs` runs, its newest runs that the major merge in flight does not hold, at most
    /// [`FLUSH_MERGE_RUNS`]. `None` if the range has no sealed buffer.
    fn plan_flush(&self, range: &CachedRange) -> Option<Job> {
        let layers = range.layers.load();
        let newest = layers.sealed.first()?;
        let hi = newest.sealed_at().load(Ordering::Acquire);
        let (runs, max_runs) = (layers.runs.len(), self.cache.options.tuning.max_runs);
        let free = range.jobs.free_newest(&layers.runs);
        debug_assert!(
            runs < max_runs || free > 0,
            "a major merge holds the newest run of a full range"
        );
        // Unreachable by rule 1. Were it not, the flush would wait for the merge, whose end
        // notifies the task, rather than take a run that the merge holds.
        let replaced = flush_replaces(runs, free, max_runs)?;
        let replaces = layers.runs[..replaced].to_vec();
        let lo = replaces.last().map_or(layers.covered_to, |r| r.lo);
        Some(Job::Flush {
            buffers: layers.sealed.clone(),
            replaces,
            lo,
            hi,
        })
    }

    /// Starts queued jobs while slots are free: flushes, then major merges. A job's range is
    /// checked again when it starts, as it may have changed while it waited, and the job's inputs
    /// are chosen then.
    fn start_queued(&mut self) {
        let cache = self.cache.clone();
        while cache.maintenance.flushes.load(Ordering::Acquire) < MAX_FLUSHES {
            let Some(range) = self.flush_queue.pop_front() else {
                break;
            };
            let Some(range) = range.upgrade() else {
                continue;
            };
            range.jobs.flush_queued.store(false, Ordering::Relaxed);
            // A split in flight leaves the flush to the ranges that it makes, or to the look
            // that its end makes if the writer rejects it.
            if range.jobs.flushing() || range.jobs.splitting() || !cache.is_indexed(&range) {
                continue;
            }
            if let Some(job) = self.plan_flush(&range) {
                self.start(&range, job);
            }
        }
        while cache.maintenance.merges.load(Ordering::Acquire) < MAX_MERGES {
            let Some(range) = self.merge_queue.pop_front() else {
                break;
            };
            let Some(range) = range.upgrade() else {
                continue;
            };
            range.jobs.merge_queued.store(false, Ordering::Relaxed);
            if range.jobs.merging() || range.jobs.splitting() || !cache.is_indexed(&range) {
                continue;
            }
            let layers = range.layers.load();
            let tuning = &cache.options.tuning;
            // A split that became due while the merge waited goes first.
            if split_cuts(&range, &layers, tuning).is_some() {
                continue;
            }
            let Some((runs, trigger)) =
                plan_major(&range, &layers, tuning, range.jobs.flush_held())
            else {
                continue;
            };
            if layers.chunks.loading() {
                continue;
            }
            if !cache.rebuild_pays(&range, layers.base_share(&range.start).0) {
                // Defer unpaid automatic merges until reads justify them or the range hits its run
                // limit, at which point it is evicted.
                if at_run_limit(&range, &layers, trigger) {
                    cache.request_removal(Target::Range(range.clone()), RemoveReason::NotWorthIt);
                }
                continue;
            }
            let job = Job::Major {
                chunks: layers.chunks.clone(),
                absorbed: layers.runs[runs].to_vec(),
                trigger,
            };
            drop(layers);
            self.start(&range, job);
        }
    }

    /// Seals up to [`PRESSURE_SEALS`] largest active buffers (`>= min_seal_bytes`) when memory
    /// exceeds the pressure mark and fewer than [`MAX_FLUSHES`] flushes are queued.
    fn seal_under_pressure(&self, index: &RangeIndex) {
        let cache = &self.cache;
        let tuning = &cache.options.tuning;
        if self.flush_queue.len() >= MAX_FLUSHES || cache.delta_memory.bytes() <= tuning.pressure()
        {
            return;
        }
        let min_seal = tuning.min_seal_bytes as u64;
        let mut to_seal: Vec<(u64, &Arc<CachedRange>, Arc<Delta>)> = Vec::new();
        for range in index.values() {
            let layers = range.layers.load();
            let bytes = layers.active.allocated_bytes();
            if bytes >= min_seal && !layers.active.is_empty() {
                to_seal.push((bytes, range, layers.active.clone()));
            }
        }
        to_seal.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, range, buffer) in to_seal.into_iter().take(PRESSURE_SEALS) {
            cache.request_seal(range, &buffer);
        }
    }

    /// Starts `job` on `range` as its own task, after it notes what the job holds.
    fn start(&mut self, range: &Arc<CachedRange>, job: Job) {
        let jobs = &range.jobs;
        let kind = match &job {
            Job::Flush { replaces, .. } => {
                jobs.flush_replaces.store(replaces.len(), Ordering::Relaxed);
                JobKind::Flush
            }
            Job::Major { absorbed, .. } => {
                jobs.merge_hi
                    .store(absorbed.first().map_or(0, |r| r.hi), Ordering::Relaxed);
                JobKind::Major
            }
            Job::Split { .. } => JobKind::Split,
        };
        let (flag, running) = kind.slots(range, &self.cache.maintenance);
        flag.store(true, Ordering::Release);
        running.fetch_add(1, Ordering::AcqRel);
        let (cache, range) = (self.cache.clone(), range.clone());
        let (done_cache, done_range) = (cache.clone(), range.clone());
        let name = job.name();
        let handle = spawn_bg_task(
            name.to_string(),
            &self.cache.handle,
            move |result: &Result<(), SlateDBError>| {
                // Runs after the job, also if it failed or panicked.
                let (flag, running) = kind.slots(&done_range, &done_cache.maintenance);
                flag.store(false, Ordering::Release);
                running.fetch_sub(1, Ordering::AcqRel);
                if let Err(e) = result {
                    warn!("range cache {name} failed [error={:?}]", e);
                    done_cache
                        .request_removal(Target::Range(done_range.clone()), RemoveReason::Error);
                }
                // The range may call for its next job, and the slot may start a queued one.
                done_cache.maintenance.notify(&done_range);
            },
            async move {
                match job {
                    Job::Flush {
                        buffers,
                        replaces,
                        lo,
                        hi,
                    } => cache.flush(&range, buffers, replaces, lo, hi).await,
                    Job::Major {
                        chunks,
                        absorbed,
                        trigger,
                    } => cache.major_merge(&range, chunks, absorbed, trigger).await,
                    Job::Split { cuts } => cache.split(&range, cuts).await.map(|_| ()),
                }
            },
        );
        self.jobs.push(handle);
    }
}

#[async_trait]
impl MessageHandler<MaintenanceMessage> for Maintainer {
    fn tickers(&mut self) -> Vec<MessageTickerDef<MaintenanceMessage>> {
        vec![MessageTickerDef::new(
            TICK,
            Box::new(|| MaintenanceMessage::Tick),
        )]
    }

    async fn handle(&mut self, message: MaintenanceMessage) -> Result<(), SlateDBError> {
        self.pass(message);
        // Job failures remove their range. They never fail the database.
        Ok(())
    }

    async fn cleanup(
        &mut self,
        _messages: BoxStream<'async_trait, MaintenanceMessage>,
        _result: Result<(), SlateDBError>,
    ) -> Result<(), SlateDBError> {
        // The database is closed now. The writer still runs (it shuts down last), so jobs that
        // wait for its answer end.
        for job in self.jobs.drain(..) {
            if let Err(e) = job.await {
                debug!("range cache maintenance job failed [error={:?}]", e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound::{self, Excluded, Included};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    use bytes::Bytes;
    use fail_parallel::FailPointRegistry;
    use slatedb_common::metrics::{lookup_metric_with_labels, DefaultMetricsRecorder};

    use crate::config::{ReadOptions, TracingOptions};
    use crate::db::Db;
    use crate::merge_operator::{MergeOperator, MergeOperatorError};
    use crate::object_store::memory::InMemory;
    use crate::oracle::Oracle;
    use crate::range_cache::filter::hash_key;
    use crate::range_cache::stats::{
        Unfit, FLUSH_BYTES, FLUSH_COUNT, FLUSH_MERGE_COUNT, INSTALL_REJECTED_COUNT,
        LOAD_REJECTED_COUNT, REBUILD_COUNT, REMOVE_COUNT, SEAL_COUNT, VIEW_BYTES,
    };
    use crate::range_cache::{CachedRangeInfo, RangeCacheOptions};

    /// How long a test waits for something that must happen promptly.
    const PROMPT: Duration = Duration::from_secs(20);

    /// Appends the operand to the value.
    struct Concat;

    impl MergeOperator for Concat {
        fn merge(
            &self,
            _key: &Bytes,
            existing_value: Option<Bytes>,
            operand: Bytes,
        ) -> Result<Bytes, MergeOperatorError> {
            let mut merged = existing_value.map(|v| v.to_vec()).unwrap_or_default();
            merged.extend_from_slice(&operand);
            Ok(Bytes::from(merged))
        }
    }

    struct TestDb {
        db: Db,
        recorder: Arc<DefaultMetricsRecorder>,
        fp_registry: Arc<FailPointRegistry>,
        dir: tempfile::TempDir,
    }

    impl TestDb {
        fn cache(&self) -> &Arc<RangeCache> {
            self.db.inner.range_cache.as_ref().unwrap()
        }

        fn range(&self, start: &Bytes) -> Arc<CachedRange> {
            self.cache().index.load().get(start).unwrap().clone()
        }

        fn info(&self, start: &Bytes) -> CachedRangeInfo {
            self.db
                .cached_ranges()
                .into_iter()
                .find(|r| r.start == *start)
                .unwrap()
        }

        fn committed(&self) -> u64 {
            self.db.inner.oracle.last_committed_seq()
        }

        fn metric(&self, name: &str, labels: &[(&str, &str)]) -> i64 {
            if labels.is_empty() {
                slatedb_common::metrics::lookup_metric(&self.recorder, name).unwrap_or(0)
            } else {
                lookup_metric_with_labels(&self.recorder, name, labels).unwrap_or(0)
            }
        }

        fn seals(&self) -> i64 {
            self.metric(SEAL_COUNT, &[("reason", "full")])
        }

        fn fail(&self, name: &str, action: &str) {
            fail_parallel::cfg(self.fp_registry.clone(), name, action).unwrap();
        }

        fn cache_files(&self) -> usize {
            std::fs::read_dir(self.dir.path())
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".rcf")
                })
                .count()
        }

        /// Waits until `done` holds, and fails the test if it does not within [`PROMPT`].
        async fn wait(&self, what: &str, done: impl Fn(&Self) -> bool) {
            let deadline = Instant::now() + PROMPT;
            while !done(self) {
                assert!(Instant::now() < deadline, "timed out waiting for {what}");
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }

        /// Writes rows into `[a, b)` round after round until `done` holds after a write.
        async fn write_until(&self, a: usize, b: usize, done: impl Fn(&Self) -> bool) {
            let mut i = 0;
            while !done(self) {
                let k = a + i % (b - a);
                self.db.put(key(k), value(k, i)).await.unwrap();
                i += 1;
                assert!(i < 1_000_000, "the writes never reached their goal");
            }
        }

        /// Asserts that cached reads of the keys in `[a, b)`, at the latest state and in
        /// `snapshot`, equal the normal read path at the same state.
        async fn assert_reads(&self, a: usize, b: usize, snapshot: Option<&crate::DbSnapshot>) {
            let now = self.db.snapshot().unwrap();
            for (reader, label) in [(now.as_ref(), "latest")]
                .into_iter()
                .chain(snapshot.map(|s| (s, "snapshot")))
            {
                for i in a..b {
                    let cached = reader.get(key(i)).await.unwrap();
                    let normal = reader.get_with_options(key(i), &normal()).await.unwrap();
                    assert_eq!(cached, normal, "{label} read of key {i}");
                }
                let mut cached = reader.scan(key(a)..key(b)).await.unwrap();
                let mut normal = reader
                    .scan_with_options(
                        key(a)..key(b),
                        &crate::config::ScanOptions::default()
                            .with_tracing_options(Some(TracingOptions::new("normal"))),
                    )
                    .await
                    .unwrap();
                loop {
                    let (c, n) = (cached.next().await.unwrap(), normal.next().await.unwrap());
                    assert_eq!(c, n, "{label} scan");
                    if c.is_none() {
                        break;
                    }
                }
            }
        }
    }

    async fn open(tune: impl FnOnce(&mut RangeCacheOptions)) -> TestDb {
        open_with(tune, None).await
    }

    async fn open_with(
        tune: impl FnOnce(&mut RangeCacheOptions),
        settings: Option<crate::config::Settings>,
    ) -> TestDb {
        let dir = tempfile::tempdir().unwrap();
        let mut options = RangeCacheOptions::new(dir.path());
        options.tuning.stats_fold_events = 1;
        // No major merge unless a test asks for one.
        options.tuning.min_rebuild_delta_bytes = usize::MAX;
        tune(&mut options);
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let fp_registry = Arc::new(FailPointRegistry::new());
        let mut builder = Db::builder("range_cache_maintain", Arc::new(InMemory::new()))
            .with_range_cache(options)
            .with_merge_operator(Arc::new(Concat))
            .with_metrics_recorder(recorder.clone())
            .with_fp_registry(fp_registry.clone());
        if let Some(settings) = settings {
            builder = builder.with_settings(settings);
        }
        let db = builder.build().await.unwrap();
        TestDb {
            db,
            recorder,
            fp_registry,
            dir,
        }
    }

    fn key(i: usize) -> Bytes {
        Bytes::from(format!("key{i:05}"))
    }

    /// A 200-byte value, so that a buffered version takes about 260 bytes.
    fn value(i: usize, round: usize) -> Bytes {
        let mut v = format!("value-{i}-{round}-").into_bytes();
        v.resize(200, b'.');
        Bytes::from(v)
    }

    fn span(a: usize, b: usize) -> (Bound<Bytes>, Bound<Bytes>) {
        (Included(key(a)), Excluded(key(b)))
    }

    /// Reads through the normal path: the cache never serves a traced read.
    fn normal() -> ReadOptions {
        ReadOptions::default().with_tracing_options(Some(TracingOptions::new("normal")))
    }

    /// Writes until the next seal, then waits until no sealed buffer is left: one flush per
    /// seal.
    async fn seal_and_flush(t: &TestDb, a: usize, b: usize) {
        let seals = t.seals();
        t.write_until(a, b, |t| t.seals() > seals).await;
        t.wait("the flush", |t| t.info(&key(a)).sealed_buffers == 0)
            .await;
    }

    async fn load(t: &TestDb, a: usize, b: usize) {
        for i in a..b {
            t.db.put(key(i), value(i, 0)).await.unwrap();
        }
        t.db.cache_range(span(a, b)).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_seal_at_the_seal_size_and_flush_to_a_run() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        // Hold the flush, so the sealed buffer stays in the layers.
        t.fail("range-cache-flush", "pause");
        let before = range.layers.load_full();
        t.write_until(0, 100, |t| t.seals() == 1).await;
        let layers = range.layers.load_full();
        assert_eq!(layers.sealed.len(), 1);
        let sealed = layers.sealed[0].clone();
        assert!(Arc::ptr_eq(&sealed, &before.active));
        // Sealed at the batch that took it past the size, which is the last one.
        let s = sealed.sealed_at().load(Ordering::SeqCst);
        assert_eq!(s, t.committed());
        assert!(sealed.allocated_bytes() >= 16 * 1024);
        assert_eq!(layers.active.lo(), s);
        assert!(layers.active.is_empty());
        assert_eq!(layers.covered_to, layers.only_chunk().base_seq);
        // A reader that pinned the earlier set sees the seal through the shared word.
        assert_eq!(RangeCache::usable(&before, s), Ok(()));
        assert_eq!(RangeCache::usable(&before, s + 1), Err(Unfit::Sealed));
        assert!(t.info(&key(0)).rebuilding);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        drop(layers);

        t.fail("range-cache-flush", "off");
        t.wait("the flush", |t| {
            let info = t.info(&key(0));
            info.runs == 1 && info.sealed_buffers == 0
        })
        .await;
        let layers = range.layers.load_full();
        let run = &layers.runs[0];
        assert_eq!(
            (run.lo, run.hi, layers.covered_to),
            (layers.only_chunk().base_seq, s, s)
        );
        assert_eq!(run.file.rows(), sealed.len());
        assert_eq!(t.metric(FLUSH_COUNT, &[]), 1);
        assert_eq!(t.metric(FLUSH_BYTES, &[]) as u64, run.file_bytes());
        assert_eq!(t.info(&key(0)).flushed_bytes, run.file_bytes());
        assert_eq!(t.cache_files(), 2);
        assert!(!t.info(&key(0)).rebuilding);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // A snapshot inside the run's interval reads it at an older version.
        let inside = t.db.snapshot().unwrap();
        t.write_until(0, 100, |t| t.committed() > s + 50).await;
        t.assert_reads(0, 100, Some(&inside)).await;
        // The flushed buffer's memory goes when its last holder drops it. A later seal's flush
        // can hold its own buffer for a moment.
        drop((before, layers, sealed));
        t.wait("the memory of the flushed buffers", |t| {
            let held: u64 = range
                .layers
                .load()
                .buffers()
                .map(|b| b.memory_bytes())
                .sum();
            t.cache().buffer_memory() == held
        })
        .await;
        t.db.close().await.unwrap();
    }

    /// A flush absorbs every buffer that its range sealed before it started.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_flush_all_sealed_buffers_in_one_run() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        // The first flush waits after it took the first buffer; three more seal meanwhile.
        t.fail("range-cache-flush", "1*pause");
        t.write_until(0, 100, |t| t.seals() == 4).await;
        assert_eq!(t.info(&key(0)).sealed_buffers, 4);
        let sealed = range.layers.load_full().sealed.clone();
        t.fail("range-cache-flush", "off");
        t.wait("the flushes", |t| t.info(&key(0)).sealed_buffers == 0)
            .await;
        // The first flush took one or two buffers before it waited; a later one took the rest.
        let flushes = t.metric(FLUSH_COUNT, &[]);
        assert!(flushes < 4);
        // Runs do not merge with each other: one run per flush, all in the view.
        let layers = range.layers.load_full();
        assert_eq!(assert_view(&range), flushes as usize);
        assert_eq!(
            (layers.runs.last().unwrap().lo, layers.runs[0].hi),
            (
                layers.only_chunk().base_seq,
                sealed[0].sealed_at().load(Ordering::SeqCst)
            )
        );
        let versions: u64 = sealed.iter().map(|b| b.len()).sum();
        let rows: u64 = layers.runs.iter().map(|r| r.file.rows()).sum();
        assert_eq!(rows, versions);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.db.close().await.unwrap();
    }

    /// With both flush slots taken, a third range's sealed buffer waits in the task's queue, which
    /// does not keep the range alive: once it is removed, it and its buffers are freed while the
    /// slots are still taken, and its flush never starts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 6)]
    async fn should_queue_flushes_without_keeping_removed_ranges() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        for r in 0..3 {
            load(&t, r * 100, r * 100 + 100).await;
        }
        t.fail("range-cache-flush", "2*pause");
        for r in 0..3 {
            let seals = t.seals();
            t.write_until(r * 100, r * 100 + 100, |t| t.seals() > seals)
                .await;
        }
        assert_eq!(t.info(&key(200)).sealed_buffers, 1);
        let memory = t.cache().delta_memory.bytes();
        let (waiting, freed) = {
            let range = t.range(&key(200));
            (Arc::downgrade(&range), range.unflushed_bytes())
        };
        // Two flushes hold the slots; the third range's flush waits.
        t.wait("the flushes", |t| {
            t.cache().maintenance.flushes.load(Ordering::Acquire) == 2
        })
        .await;
        tokio::time::sleep(TICK * 3).await;
        assert_eq!(t.cache().maintenance.flushes.load(Ordering::Acquire), 2);
        assert_eq!(t.info(&key(200)).sealed_buffers, 1);
        t.db.uncache_range(span(200, 300)).unwrap();
        t.wait("the removed range to be freed", |_| {
            waiting.upgrade().is_none()
        })
        .await;
        assert_eq!(t.cache().delta_memory.bytes(), memory - freed);
        assert_eq!(t.cache().maintenance.flushes.load(Ordering::Acquire), 2);
        t.fail("range-cache-flush", "off");
        t.wait("the flushes", |t| {
            t.db.cached_ranges().iter().all(|r| r.sealed_buffers == 0)
        })
        .await;
        assert_eq!(t.metric(FLUSH_COUNT, &[]), 2);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        t.assert_reads(0, 200, None).await;
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_keep_runs_flushed_during_a_major_merge() {
        // A run larger than f times the base starts a major merge: the base of 100 rows is
        // about 28 KiB, and f is 0.25.
        let t = open(|o| {
            o.tuning.range_seal_bytes = 96 * 1024;
            o.tuning.min_rebuild_delta_bytes = 1;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        // Every major merge waits after it took its inputs.
        t.fail("range-cache-rebuild", "sleep(1500)");
        seal_and_flush(&t, 0, 100).await;
        let one = range.layers.load_full();
        let cut = one.runs[0].hi;
        t.wait("the merge", |_| range.jobs.merging()).await;
        // A second run arrives during the merge.
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(assert_view(&range), 2);
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 0);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.wait("the install", |t| t.metric(REBUILD_COUNT, &[]) == 1)
            .await;
        // A later merge sleeps too, so this state holds while the test reads it.
        let layers = range.layers.load_full();
        assert_eq!(layers.only_chunk().base_seq, cut);
        assert_eq!(assert_view(&range), 1);
        assert_eq!(layers.runs[0].lo, cut);
        assert!(!Arc::ptr_eq(
            &layers.only_chunk().file,
            &one.only_chunk().file
        ));
        // The range's own counts: two flushes, one major merge.
        let info = t.info(&key(0));
        assert_eq!(
            (info.major_merges, info.major_merge_bytes),
            (1, layers.only_chunk().file.file_len())
        );
        assert_eq!(
            info.flushed_bytes,
            one.runs[0].file_bytes() + layers.runs[0].file_bytes()
        );
        assert_eq!(info.chunks, 1);
        // The snapshot is older than the new base: it uses the normal path, and stays exact.
        assert!(snapshot.seq() < cut);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.fail("range-cache-rebuild", "off");
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        drop((one, layers));
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    /// `cache_range` cuts a request larger than `range_target_bytes` into ranges at block
    /// boundaries of the largest sorted run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_split_a_large_range_at_registration() {
        use crate::config::{
            CompactionWorkerOptions, CompactorOptions, FlushOptions, FlushType, Settings,
            SizeTieredCompactionSchedulerOptions,
        };
        // One memtable flush writes one L0 SST with every row, so the compactor forms one sorted
        // run, of several SSTs: the split points come from the largest sorted run only, and rows
        // in a second run would stay in one part past its last block.
        let settings = Settings {
            l0_sst_size_bytes: 4 * 1024 * 1024,
            compactor_options: Some(CompactorOptions {
                poll_interval: Duration::from_millis(5),
                scheduler_options: SizeTieredCompactionSchedulerOptions {
                    min_compaction_sources: 1,
                    ..Default::default()
                }
                .into(),
                worker: Some(CompactionWorkerOptions {
                    compactions_poll_interval: Duration::from_millis(5),
                    max_sst_size: 128 * 1024,
                    ..Default::default()
                }),
                ..CompactorOptions::default()
            }),
            ..Settings::default()
        };
        let target = 64 * 1024;
        let t = open_with(|o| o.tuning.range_target_bytes = target, Some(settings)).await;
        for i in 0..4000 {
            t.db.put(key(i), value(i, 0)).await.unwrap();
        }
        t.db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        t.wait("a sorted run", |t| {
            let manifest = t.db.manifest();
            manifest.l0().is_empty() && !manifest.compacted().is_empty()
        })
        .await;
        t.db.cache_range(..).await.unwrap();
        let ranges = t.db.cached_ranges();
        // About 860 KB of rows in parts of about 64 KiB.
        assert!(ranges.len() >= 8, "{} ranges", ranges.len());
        assert!(ranges[0].start.is_empty() && ranges.last().unwrap().end.is_none());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end.as_ref(), Some(&pair[1].start));
        }
        for r in &ranges {
            assert!(r.file_bytes < 3 * target, "{r:?}");
        }
        let rows: u64 = ranges.iter().map(|r| r.file_rows).sum();
        assert_eq!(rows, 4000);
        // Scans walk the adjacent ranges.
        t.assert_reads(0, 4000, None).await;
        t.db.close().await.unwrap();
    }

    /// Past the metadata budget, the cache evicts ranges as it does for the disk budget.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_evict_ranges_past_the_metadata_budget() {
        let probe = open(|_| {}).await;
        load(&probe, 0, 50).await;
        let one = probe.range(&key(0)).meta_bytes();
        assert!(one > 0);
        probe.db.close().await.unwrap();
        let t = open(|o| o.tuning.max_meta_bytes = Some(one * 5 / 2)).await;
        load(&t, 0, 50).await;
        load(&t, 100, 150).await;
        assert_eq!(t.db.cached_ranges().len(), 2);
        // The first range is read, so the second is the least recently read.
        t.db.get(key(10)).await.unwrap();
        load(&t, 200, 250).await;
        let starts: Vec<_> = t.db.cached_ranges().into_iter().map(|r| r.start).collect();
        assert_eq!(starts, vec![key(0), key(200)]);
        assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]), 1);
        let gauge = slatedb_common::metrics::lookup_metric(
            &t.recorder,
            crate::range_cache::stats::META_BYTES,
        )
        .unwrap();
        assert!(gauge as u64 >= 2 * one, "{gauge} < 2 x {one}");
        t.db.close().await.unwrap();
    }

    /// The cap of 32 KiB, with no seal: buffers grow by chunks of 4, 8, 16 and 32 KiB.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_remove_ranges_at_the_cap_and_reject_loads_above_the_high_water_mark() {
        let t = open(|o| {
            o.tuning.max_recent_bytes = 32 * 1024;
            o.tuning.range_seal_bytes = 1024 * 1024;
            o.tuning.min_seal_bytes = 1024 * 1024;
            o.tuning.buffer_filter_bytes = usize::MAX;
        })
        .await;
        load(&t, 0, 100).await;
        load(&t, 100, 200).await;
        load(&t, 200, 300).await;
        let memory = |t: &TestDb| t.db.range_cache_buffer_memory();
        // B: 4 + 8 + 16 KiB. Then A: 4 KiB, which reaches the cap exactly.
        t.write_until(100, 200, |t| t.info(&key(100)).buffer_bytes == 28 * 1024)
            .await;
        t.write_until(0, 1, |t| t.info(&key(0)).buffer_bytes == 4 * 1024)
            .await;
        assert_eq!(memory(&t), 32 * 1024);
        // Above the high-water mark (28 KiB), loads are rejected.
        let err = t.db.cache_range(span(300, 400)).await.unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::Invalid);
        assert_eq!(t.metric(LOAD_REJECTED_COUNT, &[("reason", "memory")]), 1);
        // A's next chunk does not fit: A leaves at the batch before, and B, the largest, goes
        // too, so the memory falls under the low-water mark (24 KiB). C stays. The write itself
        // succeeds.
        let mut i = 1;
        while t.db.cached_ranges().len() == 3 {
            t.db.put(key(i), value(i, 1)).await.unwrap();
            assert!(memory(&t) <= 32 * 1024);
            i += 1;
        }
        let starts: Vec<_> = t.db.cached_ranges().into_iter().map(|r| r.start).collect();
        assert_eq!(starts, vec![key(200)]);
        assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "overflow")]), 2);
        assert_eq!(t.db.get(key(i - 1)).await.unwrap(), Some(value(i - 1, 1)));
        t.wait("the memory of the removed ranges", |t| memory(t) == 0)
            .await;
        // Below the mark again, loads work.
        t.db.cache_range(span(300, 400)).await.unwrap();
        t.assert_reads(0, 400, None).await;
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_seal_large_buffers_early_under_pressure() {
        let t = open(|o| {
            o.tuning.max_recent_bytes = 256 * 1024;
            o.tuning.range_seal_bytes = 1024 * 1024;
            o.tuning.min_seal_bytes = 16 * 1024;
        })
        .await;
        load(&t, 0, 100).await;
        // Past half of the cap, the maintenance task seals the buffer, and the flush frees it.
        t.write_until(0, 100, |t| t.db.range_cache_buffer_memory() > 128 * 1024)
            .await;
        t.wait("the early seal", |t| {
            t.metric(SEAL_COUNT, &[("reason", "pressure")]) >= 1
        })
        .await;
        t.wait("the flush", |t| t.info(&key(0)).runs == 1).await;
        t.wait("the memory", |t| {
            t.db.range_cache_buffer_memory() < 128 * 1024
        })
        .await;
        assert_eq!(t.seals(), 0);
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_filter_large_buffers() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 1024 * 1024;
            o.tuning.buffer_filter_bytes = 16 * 1024;
        })
        .await;
        load(&t, 0, 1000).await;
        let range = t.range(&key(0));
        // Small buffers have no filter.
        t.write_until(0, 20, |t| t.info(&key(0)).buffer_bytes >= 8 * 1024)
            .await;
        assert!(range.layers.load().active.filter().is_none());
        let early = t.db.snapshot().unwrap();
        t.write_until(0, 200, |_| range.layers.load().active.filter().is_some())
            .await;
        t.write_until(0, 300, |t| t.info(&key(0)).buffer_bytes >= 64 * 1024)
            .await;
        let layers = range.layers.load_full();
        let active = &layers.active;
        // The filter counts in the buffers' memory, and passes every buffered key.
        assert!(active.filter_bytes() > 0);
        assert_eq!(
            t.db.range_cache_buffer_memory(),
            active.allocated_bytes() + active.filter_bytes()
        );
        let filter = active.filter().unwrap();
        let buffer = active.buffer();
        let mut cursor = buffer.seek(Bound::Unbounded);
        while let Some(version) = buffer.version(&cursor) {
            assert!(filter.may_contain(hash_key(version.key)));
            buffer.advance(&mut cursor);
        }
        // Keys only in the base are mostly ruled out.
        let passed = (300..1000)
            .filter(|i| filter.may_contain(hash_key(&key(*i))))
            .count();
        assert!(passed < 50, "{passed} of 700 absent keys passed");
        t.assert_reads(0, 1000, Some(&early)).await;
        t.db.close().await.unwrap();
    }

    /// A flush and a major merge of one range race: each changes the view, so the one that
    /// installs second is rejected, builds its view again on the new one, and installs. No run is
    /// lost and no file is written twice.
    #[tokio::test(flavor = "multi_thread", worker_threads = 6)]
    async fn should_rebuild_the_view_of_a_job_that_lost_a_race() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.max_runs = 16;
            // A major merge is due at the first run.
            o.tuning.min_rebuild_delta_bytes = 0;
            o.tuning.major_merge_factor = 1e-6;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();

        // A major merge waits to install its view while flushes install theirs.
        t.fail("range-cache-install-major-view", "pause");
        seal_and_flush(&t, 0, 100).await;
        t.wait("the first major merge", |_| range.jobs.merging())
            .await;
        for _ in 0..3 {
            seal_and_flush(&t, 0, 100).await;
        }
        assert_eq!(t.metric(FLUSH_COUNT, &[]), 4);
        assert_eq!(assert_view(&range), 4);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The next major merge waits at its start.
        t.fail("range-cache-rebuild", "pause");
        t.fail("range-cache-install-major-view", "off");
        t.wait("the first install", |t| t.metric(REBUILD_COUNT, &[]) == 1)
            .await;
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 1);
        // The merge absorbed the one run it found; the three flushed meanwhile stay.
        assert_eq!(assert_view(&range), 3);
        t.assert_reads(0, 100, None).await;

        // The second major merge holds those three runs. A flush waits to install its view while
        // that merge absorbs every run.
        t.wait("the second major merge", |_| range.jobs.merging())
            .await;
        t.fail("range-cache-install-run-view", "pause");
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        t.wait("the flush", |_| range.jobs.flushing()).await;
        t.fail("range-cache-rebuild", "off");
        t.wait("the second install", |t| t.metric(REBUILD_COUNT, &[]) == 2)
            .await;
        assert_eq!(assert_view(&range), 0);
        // The next major merge waits again, so the flushed run stays to be looked at.
        t.fail("range-cache-rebuild", "pause");
        t.fail("range-cache-install-run-view", "off");
        t.wait("the flush", |t| t.info(&key(0)).sealed_buffers == 0)
            .await;
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 2);
        assert_eq!(assert_view(&range), 1);
        t.assert_reads(0, 100, None).await;
        t.fail("range-cache-rebuild", "off");
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert_view(&range);
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    /// A flush's install as its job sends it: the view of the range now, without `in_view` and
    /// with the run, and the view it was built on.
    fn swap_for(
        cache: &RangeCache,
        range: &CachedRange,
        run: &Arc<Run>,
        rows: &RowLocators,
        in_view: &[Arc<Run>],
    ) -> ViewSwap {
        let was = range.layers.load().view.clone();
        let view = DeltaView::after_flush(
            was.as_deref(),
            in_view,
            run,
            rows,
            &cache.views,
            &mut BuildCost::default(),
        )
        .unwrap();
        ViewSwap {
            new: Some(Arc::new(view)),
            was,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_reject_installs_whose_state_moved_on() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let cache = t.cache().clone();
        // The maintenance task's flush takes the sealed buffers and waits here.
        t.fail("range-cache-flush", "pause");
        t.write_until(0, 100, |t| t.seals() == 2).await;
        let layers = range.layers.load_full();
        let (newer, oldest) = (layers.sealed[0].clone(), layers.sealed[1].clone());
        let seal = |b: &Arc<Delta>| b.sealed_at().load(Ordering::SeqCst);
        let run = |buffers: &[Arc<Delta>], lo: u64, hi: u64| {
            let (run, rows) = cache
                .write_run(buffers, &[], lo, hi, &BytesRange::unbounded())
                .unwrap();
            (Arc::new(run), rows)
        };
        let both = vec![newer.clone(), oldest.clone()];
        let lo = layers.covered_to;
        let files = t.cache_files();
        async fn install_with(
            cache: &RangeCache,
            range: &Arc<CachedRange>,
            r: (Arc<Run>, RowLocators),
            buffers: Vec<Arc<Delta>>,
            replaces: Vec<Arc<Run>>,
        ) -> bool {
            let swap = swap_for(cache, range, &r.0, &r.1, &[]);
            cache.install_run(range, r.0, buffers, replaces, swap).await
        }
        let install = |r, buffers, replaces| install_with(&cache, &range, r, buffers, replaces);
        // Not the oldest sealed buffers.
        let r = run(std::slice::from_ref(&newer), newer.lo(), seal(&newer));
        assert!(!install(r, vec![newer.clone()], vec![]).await);
        // Not from `covered_to`, or not to the newest buffer's seal.
        let r = run(&both, lo - 1, seal(&newer));
        assert!(!install(r, both.clone(), vec![]).await);
        let r = run(&both, lo, seal(&newer) - 1);
        assert!(!install(r, both.clone(), vec![]).await);
        // Runs to replace that the range does not hold.
        let stray = run(std::slice::from_ref(&oldest), lo, seal(&oldest)).0;
        let r = run(&both, lo, seal(&newer));
        assert!(!install(r, both.clone(), vec![stray]).await);
        // A view built on a view that the range does not hold.
        let r = run(&both, lo, seal(&newer));
        let mut swap = swap_for(&cache, &range, &r.0, &r.1, &[]);
        let other = DeltaView::after_flush(
            None,
            &[],
            &r.0,
            &r.1,
            &cache.views,
            &mut BuildCost::default(),
        )
        .unwrap();
        swap.was = Some(Arc::new(other));
        assert!(
            !cache
                .install_run(&range, r.0, both.clone(), vec![], swap)
                .await
        );
        // A rejected run's file is deleted when it drops.
        assert_eq!(t.cache_files(), files);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 5);
        // Both buffers in one run: one store removes both and publishes the view.
        let (first, rows) = run(&both, lo, seal(&newer));
        assert!(install((first.clone(), rows), both.clone(), vec![]).await);
        let after = range.layers.load_full();
        assert_eq!(after.covered_to, seal(&newer));
        assert!(after.sealed.is_empty());
        assert!(Arc::ptr_eq(&after.runs[0], &first));
        assert_eq!(first.file.rows(), newer.len() + oldest.len());
        assert_eq!(assert_view(&range), 1);
        t.assert_reads(0, 100, None).await;
        // The held flush wrote a run of buffers that are gone: it is rejected.
        t.fail("range-cache-flush", "off");
        t.wait("the held flush's rejection", |t| {
            t.metric(INSTALL_REJECTED_COUNT, &[]) == 6
        })
        .await;

        // A major merge whose base, runs, cut or view moved on.
        let base = after.chunks.clone();
        let merged = |base_seq| {
            let cache = cache.clone();
            let (runs, base, end) = (vec![first.clone()], base.clone(), range.end.clone());
            async move {
                let (file, _) = cache
                    .merge_files(&runs, &base, &key(0), end.as_ref(), base_seq)
                    .await
                    .unwrap();
                Arc::new(file)
            }
        };
        let without = || ViewSwap {
            new: None,
            was: range.layers.load().view.clone(),
        };
        // A base that is not the range's: another file at the same sequence number.
        let other = super::super::one_chunk(merged(first.hi).await, base[0].base_seq);
        let absorbed = vec![first.clone()];
        for (base_was, absorbed, cut, swap) in [
            (&other, absorbed.clone(), first.hi, without()),
            (&base, vec![], first.hi, without()),
            (&base, absorbed.clone(), first.hi + 1, without()),
            (
                &base,
                absorbed.clone(),
                first.hi,
                ViewSwap {
                    new: None,
                    was: None,
                },
            ),
        ] {
            assert!(
                !cache
                    .install_major(
                        &range,
                        base_was,
                        absorbed,
                        merged(first.hi).await,
                        cut,
                        swap
                    )
                    .await
            );
        }
        assert!(
            cache
                .install_major(
                    &range,
                    &base,
                    absorbed,
                    merged(first.hi).await,
                    first.hi,
                    without()
                )
                .await
        );
        let after = range.layers.load_full();
        assert_eq!(
            (after.only_chunk().base_seq, after.runs.len()),
            (first.hi, 0)
        );
        assert!(after.view.is_none());
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 1);
        t.assert_reads(0, 100, None).await;

        // Removed, then registered again: the old identity's installs are rejected.
        drop(after);
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        let held = range.layers.load_full().sealed.clone();
        t.db.uncache_range(span(0, 100)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        if let Some(newest) = held.first() {
            let oldest = held.last().unwrap();
            let rejected = t.metric(INSTALL_REJECTED_COUNT, &[]);
            assert!(!install(run(&held, oldest.lo(), seal(newest)), held.clone(), vec![]).await);
            t.db.cache_range(span(0, 100)).await.unwrap();
            assert!(!Arc::ptr_eq(&t.range(&key(0)), &range));
            assert!(!install(run(&held, oldest.lo(), seal(newest)), held.clone(), vec![]).await);
            assert!(t.metric(INSTALL_REJECTED_COUNT, &[]) >= rejected + 2);
        }
        t.db.close().await.unwrap();
    }

    /// The range's view lists exactly its runs, every version once. Returns the number of runs.
    fn assert_view(range: &CachedRange) -> usize {
        let layers = range.layers.load_full();
        match &layers.view {
            None => assert!(layers.runs.is_empty(), "runs without a view"),
            Some(view) => view.check().unwrap(),
        }
        layers.check(&range.start, *MAX_RUNS_RANGE.end());
        layers.runs.len()
    }

    /// Runs do not merge with each other: a range holds up to `max_runs` runs, read through its
    /// view, and the flush of a full range merges its two newest runs into its run instead of
    /// waiting. Without major merges, every other flush then does.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_hold_up_to_max_runs_and_flush_merge_at_the_cap() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.max_runs = 12;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        for runs in 1..=12 {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(assert_view(&range), runs);
            if runs % 4 == 0 {
                t.assert_reads(0, 100, Some(&snapshot)).await;
            }
        }
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 0);
        let full = range.layers.load_full();
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        let merged = range.layers.load_full();
        assert_eq!(assert_view(&range), 12 - FLUSH_MERGE_RUNS + 1);
        let run = &merged.runs[0];
        assert_eq!((run.lo, run.hi), (full.runs[1].lo, merged.covered_to));
        let flushed = run.file.rows() - full.runs[0].file.rows() - full.runs[1].file.rows();
        assert!(flushed > 0);
        for i in 1..merged.runs.len() {
            assert!(Arc::ptr_eq(&merged.runs[i], &full.runs[i + 1]));
        }
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The next flush is a plain one, and the one after merges again.
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(assert_view(&range), 12);
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(assert_view(&range), 11);
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 2);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        drop((full, merged));
        // Every file is a run or the base: the replaced runs' files go when their jobs end.
        t.wait("the replaced runs' files", |t| t.cache_files() == 12)
            .await;
        t.db.close().await.unwrap();
    }

    /// A flush never waits for a major merge. With `max_runs` 4 the count trigger starts a major
    /// merge at two runs (the headroom is at most half of `max_runs`); while it holds those two,
    /// flushes fill the range, and the flush of the full range merges the two newest runs, which
    /// the merge does not hold. Each job's view changes the other's, so the second to install
    /// builds its view again: first the major merge, then the flush at the cap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 6)]
    async fn should_flush_at_the_run_cap_beside_a_major_merge() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.max_runs = 4;
            // Major merges start by run count only.
            o.tuning.min_rebuild_delta_bytes = 1;
            o.tuning.major_merge_factor = 1e9;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        assert_eq!(range.jobs.headroom(4), 2);
        // The major merge holds the two runs it found, and waits after it built its view.
        t.fail("range-cache-install-major-view", "pause");
        seal_and_flush(&t, 0, 100).await;
        seal_and_flush(&t, 0, 100).await;
        t.wait("the major merge's view", |_| {
            range
                .layers
                .load()
                .view
                .as_ref()
                .is_some_and(|v| Arc::strong_count(v) >= 2)
        })
        .await;
        let held = range.layers.load_full();
        seal_and_flush(&t, 0, 100).await;
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(assert_view(&range), 4);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The flush of the full range does not wait: it merges the two newest runs.
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        let after = range.layers.load_full();
        assert_eq!(assert_view(&range), 3);
        assert!(Arc::ptr_eq(&after.runs[1], &held.runs[0]));
        assert!(Arc::ptr_eq(&after.runs[2], &held.runs[1]));
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The major merge's view was built on a view that the flushes replaced: it builds again.
        t.fail("range-cache-install-major-view", "off");
        t.wait("the major merge", |t| t.metric(REBUILD_COUNT, &[]) == 1)
            .await;
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 1);
        assert_eq!(assert_view(&range), 1);
        assert_eq!(range.layers.load().only_chunk().base_seq, held.runs[0].hi);
        // Three flushes installed while it waited and ran: the headroom grows to four, held at
        // half of `max_runs`.
        assert_eq!(range.jobs.headroom.load(Ordering::Relaxed), 4);
        assert_eq!(range.jobs.headroom(4), 2);
        t.assert_reads(0, 100, None).await;
        drop((held, after));

        // Now the major merge installs first: it waits at its start while the range fills, and
        // the flush of the full range waits after it built its view.
        t.fail("range-cache-rebuild", "pause");
        seal_and_flush(&t, 0, 100).await;
        t.wait("the second major merge", |_| range.jobs.merging())
            .await;
        seal_and_flush(&t, 0, 100).await;
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(assert_view(&range), 4);
        t.fail("range-cache-install-run-view", "pause");
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        t.wait("the flush's view", |_| {
            range
                .layers
                .load()
                .view
                .as_ref()
                .is_some_and(|v| Arc::strong_count(v) >= 2)
        })
        .await;
        t.fail("range-cache-rebuild", "off");
        t.wait("the second major merge", |t| {
            t.metric(REBUILD_COUNT, &[]) == 2
        })
        .await;
        assert_eq!(assert_view(&range), 2);
        t.fail("range-cache-install-run-view", "off");
        t.wait("the flush at the cap", |t| {
            t.info(&key(0)).sealed_buffers == 0
        })
        .await;
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 2);
        assert_eq!(assert_view(&range), 1);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 2);
        // Two flushes landed during the second merge: the headroom falls to three.
        assert_eq!(range.jobs.headroom.load(Ordering::Relaxed), 3);
        t.assert_reads(0, 100, None).await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert_view(&range);
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    /// The count trigger leaves the headroom that a major merge needs: at 16 runs it starts the
    /// merge at 12, the flushes that land while it waits fill the range to 16 without a flush at
    /// the cap, and one more is a flush at the cap. The merge then sets the headroom from the
    /// flushes that landed. The view's memory counts as metadata.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_start_a_major_merge_by_run_count_with_headroom() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.max_runs = 16;
            o.tuning.min_rebuild_delta_bytes = 1;
            o.tuning.major_merge_factor = 1e9;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        assert_eq!(range.jobs.headroom(16), INITIAL_HEADROOM);
        t.fail("range-cache-rebuild", "pause");
        for runs in 1..=11 {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(assert_view(&range), runs);
            assert!(!range.jobs.merging());
        }
        seal_and_flush(&t, 0, 100).await;
        t.wait("the major merge", |_| range.jobs.merging()).await;
        for runs in 13..=16 {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(assert_view(&range), runs);
        }
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 0);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        assert_eq!(assert_view(&range), 15);
        t.fail("range-cache-rebuild", "off");
        t.wait("the major merge", |t| t.metric(REBUILD_COUNT, &[]) == 1)
            .await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        // The merge absorbed the 12 runs it found; the three runs flushed after them stay.
        assert_eq!(assert_view(&range), 3);
        // Five flushes landed between its queueing and its install: the headroom is six.
        assert_eq!(range.jobs.headroom(16), 6);
        t.assert_reads(0, 100, None).await;
        // The view's memory is metadata, about seven bytes per version and the anchors.
        t.cache().update_gauges();
        let views = t.metric(VIEW_BYTES, &[]);
        let versions: u64 = range.layers.load().runs.iter().map(|r| r.file.rows()).sum();
        assert!(
            views as u64 >= 7 * versions,
            "{views} bytes, {versions} versions"
        );
        assert!(t.metric(crate::range_cache::stats::META_BYTES, &[]) >= views);
        t.db.close().await.unwrap();
    }

    /// Writes a run of `rows` rows of 200-byte values, to give a run its bytes.
    fn test_run(dir: &std::path::Path, name: &str, rows: usize, lo: u64) -> Arc<Run> {
        let mut writer = RangeFileWriter::create_run(dir, name, 4096).unwrap();
        for i in 0..rows {
            let row = crate::types::RowEntry::new_value(&key(i), &value(i, 0), lo + 1);
            writer.push(&row).unwrap();
        }
        let file = RangeFile::open(
            writer.finish().unwrap(),
            super::super::RangeCachePageSource::Mmap,
            super::super::file::FileBudget::detached(),
        )
        .unwrap();
        Arc::new(Run::new(file, lo, lo + 1).unwrap())
    }

    /// A major merge starts by bytes at `f` times the base, or by count at `max_runs` less the
    /// headroom, never below `min_rebuild_delta_bytes`; the headroom follows the flushes that
    /// landed during the range's major merges.
    #[test]
    fn should_start_major_merges_by_bytes_and_by_run_count() {
        let mib = 1024 * 1024;
        let tuning = |f: f64, floor: usize, max_runs: usize| RangeCacheTuning {
            major_merge_factor: f,
            min_rebuild_delta_bytes: floor,
            max_runs,
            ..RangeCacheTuning::default()
        };
        assert_eq!(RangeCacheTuning::default().major_merge_factor, 0.25);
        assert_eq!(RangeCacheTuning::default().max_runs, 64);
        assert_eq!(major_threshold(64 * mib, &tuning(0.25, 0, 64)), 16 * mib);
        assert_eq!(major_threshold(64 * mib, &tuning(2.0, 0, 64)), 128 * mib);
        assert_eq!(
            major_threshold(1024, &tuning(0.25, 5 * mib as usize, 64)),
            5 * mib
        );

        let dir = tempfile::tempdir().unwrap();
        let base = test_run(dir.path(), "base", 400, 0).file.clone();
        // Runs of about 4 KiB each on a base of about 90 KiB, newest first.
        let runs: Vec<Arc<Run>> = (0..8)
            .rev()
            .map(|i| test_run(dir.path(), &format!("run{i}"), 10, 100 + i))
            .collect();
        let run_bytes = |n: usize| runs[8 - n..].iter().map(|r| r.file_bytes()).sum::<u64>();
        let range = CachedRange::new(
            Bytes::new(),
            None,
            Layers {
                active: Arc::new(Delta::for_test(
                    &super::super::delta::DeltaMemory::detached(),
                    0,
                )),
                sealed: Vec::new(),
                runs: Vec::new(),
                view: None,
                chunks: Chunks::Loading,
                covered_to: 0,
                end: None,
            },
            super::super::Origin::Explicit,
            0,
            0,
        );
        let layers = |n: usize, with_base: bool| Layers {
            active: Arc::new(Delta::for_test(
                &super::super::delta::DeltaMemory::detached(),
                0,
            )),
            sealed: Vec::new(),
            runs: runs[8 - n..].to_vec(),
            view: None,
            chunks: match with_base {
                true => super::super::one_chunk(base.clone(), 0),
                false => Chunks::Loading,
            },
            covered_to: 0,
            end: None,
        };
        let due = |n: usize, t: &RangeCacheTuning| major_due(&range, &layers(n, true), t);
        // By bytes: a factor between the bytes of two runs and of three.
        let f = (run_bytes(2) + run_bytes(3)) as f64 / 2.0 / base.file_len() as f64;
        assert_eq!(due(2, &tuning(f, 0, 64)), None);
        assert_eq!(due(3, &tuning(f, 0, 64)), Some(Trigger::Bytes));
        // Without a base, or below the floor, none is due.
        assert_eq!(
            major_due(&range, &layers(3, false), &tuning(f, 0, 64)),
            None
        );
        let floor = run_bytes(8) as usize + 1;
        assert_eq!(due(8, &tuning(f, floor, 64)), None);
        // By count: at `max_runs` less the headroom, 4 at first, at most half of `max_runs`.
        assert_eq!(range.jobs.headroom(16), 4);
        assert_eq!(range.jobs.headroom(4), 2);
        assert_eq!(range.jobs.headroom(64), 4);
        assert_eq!(due(3, &tuning(1e9, 0, 8)), None);
        assert_eq!(due(4, &tuning(1e9, 0, 8)), Some(Trigger::Runs));
        assert_eq!(due(1, &tuning(1e9, 0, 4)), None);
        assert_eq!(due(2, &tuning(1e9, 0, 4)), Some(Trigger::Runs));
        // The headroom is the flushes that landed since the merge was queued, plus one: at once
        // when more, halfway when fewer.
        let land = |flushes: u64| {
            range.jobs.merge_queued_at.store(
                range.jobs.flushes.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            for _ in 0..flushes {
                range.jobs.record_flush(0);
            }
            assert_eq!(range.jobs.record_major(0), flushes);
            range.jobs.headroom(64)
        };
        assert_eq!(land(9), 10);
        assert_eq!(land(1), 6);
        assert_eq!(land(1), 4);
        assert_eq!(land(0), 2);
        assert_eq!(land(0), 1);
        assert_eq!(land(40), 32, "at most half of max_runs");
        assert_eq!(range.jobs.headroom(8), 4);
        assert_eq!(land(0), 21);
    }

    /// Rules 1 and 2 of the module documentation as numbers.
    #[test]
    fn should_bound_the_runs_of_flushes_and_major_merges() {
        // Rule 1: a major merge leaves the flush's runs alone, and the newest run of a full range.
        assert_eq!(mergeable_from(0, 0, 4), 0);
        assert_eq!(mergeable_from(3, 0, 4), 0);
        assert_eq!(mergeable_from(4, 0, 4), 1);
        assert_eq!(mergeable_from(63, 0, 64), 0);
        assert_eq!(mergeable_from(64, 0, 64), 1);
        assert_eq!(mergeable_from(4, 2, 4), 2);
        assert_eq!(mergeable_from(3, 2, 4), 2);
        assert_eq!(mergeable_from(2, 2, 4), 2);
        // Rule 2: the flush of a full range merges its free newest runs, at most two.
        assert_eq!(flush_replaces(3, 0, 4), Some(0));
        assert_eq!(flush_replaces(4, 4, 4), Some(2));
        assert_eq!(flush_replaces(4, 2, 4), Some(2));
        assert_eq!(flush_replaces(4, 1, 4), Some(1));
        assert_eq!(flush_replaces(4, 0, 4), None);
        assert_eq!(flush_replaces(64, 3, 64), Some(2));
    }

    /// A model of one range under rules 1 and 2, with the task's planning functions. Flushes and
    /// major merges start, install and end as separate steps in a random order, as the task and
    /// the writer see them, so a job's flag stays set for a while after its install. Every flush
    /// of a full range finds a run to merge, no run is held by a flush and a merge at once, every
    /// install's precondition holds when it arrives, and no range passes `max_runs`.
    #[test]
    fn should_keep_flushes_and_major_merges_off_each_others_runs() {
        /// A run by its interval, which identifies it (intervals are never empty here).
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        struct R {
            lo: u64,
            hi: u64,
        }
        /// A job in flight: the runs it holds, newest first.
        struct InFlight {
            runs: Vec<R>,
            installed: bool,
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        // Flush-merges of one run and of two, and major merges.
        let mut seen = [0u64; 3];
        for case in 0..2_000 {
            let max_runs = [4, 5, 8, 16, 64][case % 5];
            let mut runs: Vec<R> = Vec::new();
            let (mut seq, mut covered_to) = (0u64, 0u64);
            let mut flush: Option<InFlight> = None;
            let mut merge: Option<InFlight> = None;
            for _ in 0..600 {
                match rand(6) {
                    // The task starts a flush.
                    0 if flush.is_none() => {
                        let free = match &merge {
                            Some(m) => runs.iter().take_while(|r| r.hi > m.runs[0].hi).count(),
                            None => runs.len(),
                        };
                        let replaced = flush_replaces(runs.len(), free, max_runs)
                            .expect("a flush of a full range found no run to merge");
                        let held = runs[..replaced].to_vec();
                        if let Some(m) = merge.as_ref().filter(|m| !m.installed) {
                            assert!(held.iter().all(|r| !m.runs.contains(r)));
                        }
                        flush = Some(InFlight {
                            runs: held,
                            installed: false,
                        });
                    }
                    // The writer installs it, with the checks of `install_run_at`.
                    1 => {
                        if let Some(f) = flush.as_mut().filter(|f| !f.installed) {
                            assert_eq!(&runs[..f.runs.len()], &f.runs[..], "a flush's runs moved");
                            assert!(runs.len() - f.runs.len() < max_runs);
                            let lo = f.runs.last().map_or(covered_to, |r| r.lo);
                            seq += 1 + rand(4);
                            runs.splice(..f.runs.len(), [R { lo, hi: seq }]);
                            covered_to = seq;
                            if let n @ 1..=2 = f.runs.len() {
                                seen[n - 1] += 1;
                            }
                            f.installed = true;
                        }
                    }
                    // The flush's end clears its flag.
                    2 => {
                        if flush.as_ref().is_some_and(|f| f.installed) {
                            flush = None;
                        }
                    }
                    // The task starts a major merge, when its trigger says so: here, at random
                    // once the range has a run.
                    3 if merge.is_none() && !runs.is_empty() && rand(3) == 0 => {
                        let flush_held = flush.as_ref().map_or(0, |f| f.runs.len());
                        let first = mergeable_from(runs.len(), flush_held, max_runs);
                        if first < runs.len() {
                            let taken = runs[first..].to_vec();
                            assert!(taken.len() < max_runs);
                            if let Some(f) = flush.as_ref().filter(|f| !f.installed) {
                                assert!(taken.iter().all(|r| !f.runs.contains(r)));
                            }
                            merge = Some(InFlight {
                                runs: taken,
                                installed: false,
                            });
                        }
                    }
                    // The writer installs it, with the checks of `install_major_at`.
                    4 => {
                        if let Some(m) = merge.as_mut().filter(|m| !m.installed) {
                            let at = runs.len() - m.runs.len();
                            assert_eq!(
                                &runs[at..],
                                &m.runs[..],
                                "absorbed runs are not the oldest"
                            );
                            runs.truncate(at);
                            seen[2] += 1;
                            m.installed = true;
                        }
                    }
                    // The merge's end clears its flag.
                    5 => {
                        if merge.as_ref().is_some_and(|m| m.installed) {
                            merge = None;
                        }
                    }
                    _ => {}
                }
                assert!(runs.len() <= max_runs);
                for pair in runs.windows(2) {
                    assert_eq!(pair[0].lo, pair[1].hi, "runs are not contiguous");
                }
            }
        }
        assert!(seen.iter().all(|n| *n > 0), "a path never ran: {seen:?}");
    }

    /// Run settings out of their bounds fail the build.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_reject_run_settings_out_of_bounds() {
        type Tune = fn(&mut RangeCacheTuning);
        let cases: [(&str, Tune); 6] = [
            ("max_runs 3", |t| t.max_runs = 3),
            ("max_runs 65", |t| t.max_runs = 65),
            ("factor 0", |t| t.major_merge_factor = 0.0),
            ("factor -1", |t| t.major_merge_factor = -1.0),
            ("factor infinite", |t| t.major_merge_factor = f64::INFINITY),
            ("factor NaN", |t| t.major_merge_factor = f64::NAN),
        ];
        let build = |tune: fn(&mut RangeCacheTuning)| async move {
            let dir = tempfile::tempdir().unwrap();
            let mut options = RangeCacheOptions::new(dir.path());
            tune(&mut options.tuning);
            let built = Db::builder("range_cache_settings", Arc::new(InMemory::new()))
                .with_range_cache(options)
                .build()
                .await;
            (built, dir)
        };
        for (name, tune) in cases {
            assert!(build(tune).await.0.is_err(), "{name} was accepted");
        }
        let bounds: [Tune; 3] = [
            |t| t.max_runs = 4,
            |t| t.max_runs = 64,
            |t| t.major_merge_factor = 2.0,
        ];
        for tune in bounds {
            let (db, _dir) = build(tune).await;
            db.unwrap().close().await.unwrap();
        }
    }
}
