//! The maintenance task of the range cache (design 01 section 7): it flushes sealed buffers to
//! runs, merges runs, merges runs into new bases, and seals large buffers early when the
//! buffers' memory is high. It is a handler on the database's task executor, like the promoter.
//!
//! - **Flush** (7.2). For a range with sealed buffers, a job writes all of them, as the range
//!   held them when the job started, to one run of `(covered_to, newest seal]`: every version,
//!   key ascending then sequence number descending, tombstones and merge operands kept, with a
//!   filter of its keys. So one flush absorbs a backlog of buffers. The job sends `InstallRun`
//!   to the writer, which puts the run first, raises `covered_to` and removes the buffers in one
//!   store, or rejects it if the state moved on; a rejected run's file is deleted when it drops.
//! - **Flush-merge** (7.3). When the range holds `max_runs` runs, the flush also merges its
//!   newest runs, at most [`FLUSH_MERGE_RUNS`], into the run it writes, and the install replaces
//!   them. A flush never waits for a merge.
//! - **Minor merge** (7.4). Adjacent runs of a range merge into one run of their union interval,
//!   every version kept, installed by `InstallMinor`. `minor_merge_ratio` picks the runs (see
//!   [`minor_group`]): by default the two oldest whenever the range holds two or more, the rule
//!   of stage 2c; with a finite ratio, runs of similar size, a whole cascade in one job.
//! - **Major merge** (7.5). When the run files of a range reach `f` times its base, and at least
//!   `min_rebuild_delta_bytes`, a job merges the base with its oldest runs into a new base at the
//!   newest absorbed run's `hi`, and sends `InstallMajor`. `f` is `major_merge_factor`, or by
//!   default `sqrt(2r / B)` for the range's average flush `r` and its base `B` (7.9), between
//!   [`F_MIN`] and [`F_MAX`]. The merge absorbs at most [`MERGE_RUNS`] runs with the default
//!   minor rule, and every run it may take with merges by similar size (see [`major_runs`]).
//!   Flushes go on meanwhile; their runs stay. An automatic range whose reads do not pay for the
//!   merge leaves the cache instead.
//! - **Pressure** (7.1, 9). Above half of `max_recent_bytes`, the task asks the writer to seal the
//!   largest active buffers that hold at least `min_seal_bytes`, so that flushes can free them.
//!
//! **Which runs a job takes.** The task chooses the inputs of each job when it starts it, and
//! notes what the job holds: for a flush, how many of the newest runs it replaces; for a merge,
//! the `hi` of its newest run. Each range has at most one flush and one merge in flight, and two
//! rules keep them off each other's runs:
//!
//! 1. A merge, minor or major, takes none of the runs that the flush in flight replaces, and its
//!    runs together with all older runs number at most `max_runs - 1` (see [`mergeable_from`]).
//! 2. A flush of a range that holds `max_runs` runs replaces the newest runs that are newer than
//!    the runs of the merge in flight, at most [`FLUSH_MERGE_RUNS`] (see [`flush_replaces`]).
//!
//! While a merge is in flight, only flushes change the runs of its range, and only runs newer
//! than the merge's. So the merge's runs and the older ones keep their number `c <= max_runs - 1`,
//! and when the range holds `max_runs` runs, `max_runs - c >= 1` of them are newer than the
//! merge's: a flush always finds a run to merge, never waits, and never takes a run that the
//! merge holds. A merge that starts while a flush-merge is in flight leaves the flush's runs
//! alone by rule 1, also after an earlier merge lowered the run count. So no install is rejected
//! because a flush and a merge took the same run, and no file is written for nothing. A major
//! merge absorbs the oldest runs, so rule 1 covers it as it covers a minor merge. With the
//! default minor rule a merge holds the two oldest runs (`c = 2`), so a flush-merge always takes
//! the two newest, as in stage 2c.
//!
//! Flushes and merges have their own limits on concurrent jobs, so a long merge never holds back
//! the flushes that free memory.
//!
//! **With the delta view** (`RangeCacheTuning::delta_view`, see the `view` module), runs never
//! merge with each other: a flush writes only its sealed buffers, and a major merge absorbs every
//! run that the range holds when it starts. A major merge is due when the run bytes reach the
//! threshold above, or when the runs reach half of `max_runs`; a range that holds `max_runs` runs
//! flushes only after its major merge. Each job builds its view on a blocking thread, after its
//! file: a flush merges its run into the range's view, a major merge drops the absorbed runs from
//! it. The writer installs the view with the file only if the range still holds the view that the
//! job built on. A flush and a major merge of one range can race, and each changes the view: the
//! one that installs second is rejected, builds its view again on the new one, and sends its
//! install again, at most [`VIEW_ATTEMPTS`] times. Its file stays good, so a race costs a view
//! build, not a file, and neither job's work is lost: a flush that lands during a major merge
//! keeps its run, and the merge drops only the runs it absorbed.
//!
//! The writer notifies the task of a range when it seals one of its buffers or attaches its
//! base, and a job notifies it of its range when it ends. The task looks only at the notified
//! range, and queues the jobs it calls for until a slot is free, in the order it found them: so
//! the cost of a notification does not grow with the number of ranges, and the oldest sealed
//! buffers flush first. A ticker walks the whole index every [`TICK`] in case anything was
//! missed. Under pressure, while fewer than [`MAX_FLUSHES`] flushes wait, a walk picks the
//! largest active buffers to seal. Jobs write their files on blocking threads, not on the async
//! workers that reads use.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use fail_parallel::fail_point;
use futures::stream::BoxStream;
use log::{debug, warn};
use tokio::task::JoinHandle;

use super::delta::Delta;
use super::file::{RangeFile, RangeFileIterator, RangeFileWriter};
use super::merge::merge_versions;
use super::stats::RemoveReason;
use super::writer::{same_view, Target, ViewSwap};
use super::{
    BuildCost, CachedRange, DeltaView, Layers, RangeCache, RangeCacheTuning, RangeIndex, Run,
    MAX_RUNS_RANGE,
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
/// The most merges, minor and major, in flight.
const MAX_MERGES: usize = 2;
/// The most early seals that one pass asks for.
const PRESSURE_SEALS: usize = 4;
/// With the default minor rule (an infinite `minor_merge_ratio`): the runs at which a range has
/// its oldest runs merged (design 01's `D_minor`), the runs that a minor merge takes, and the
/// most runs that a major merge absorbs.
pub(super) const MERGE_RUNS: usize = 2;
/// The most newest runs that a flush merges into its run when the range holds `max_runs` (design
/// 01's `j`). Two brings the range below the most, so the next flush is a plain one, and the
/// newest runs are the youngest and so the smallest, as older ones grew by merges. A flush takes
/// one when a merge in flight holds the second newest run.
pub(super) const FLUSH_MERGE_RUNS: usize = 2;
/// The bounds of the adaptive major-merge factor `f`. At most 1: the runs of a range never grow
/// past its base, whose rewrite is then cheap anyway. At least 0.05: a base of more than 800
/// flushes still merges after a twentieth of its size. A fixed `major_merge_factor` is not
/// bounded by them.
pub(super) const F_MIN: f64 = 0.05;
pub(super) const F_MAX: f64 = 1.0;
/// With the delta view, the most times that a job builds its view and sends its install, when the
/// writer rejects it because the range's view moved on.
pub(super) const VIEW_ATTEMPTS: usize = 8;
// With the default minor rule a merge holds the two oldest runs, so a flush-merge always finds
// the two newest free, as in stage 2c; and a merge of two runs always fits rule 1.
const _: () = assert!(MERGE_RUNS + FLUSH_MERGE_RUNS <= *MAX_RUNS_RANGE.start());

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
    /// The merges, minor and major, in flight.
    merges: AtomicUsize,
}

impl MaintenanceQueue {
    pub(super) fn new() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self {
            tx,
            rx,
            flushes: AtomicUsize::new(0),
            merges: AtomicUsize::new(0),
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
    /// The bytes that a flush of the range adds, averaged over its flushes (0 before the first):
    /// design 01's `r`.
    flush_bytes: AtomicU64,
    /// True while a message about the range is on its way to the task.
    notified: AtomicBool,
    /// True while the range waits in the task's flush queue, or in one of its merge queues.
    /// Only the task loads and stores them.
    flush_queued: AtomicBool,
    merge_queued: AtomicBool,
    /// The newest runs that the flush in flight replaces, 0 for a plain flush. Only the task
    /// stores it, before it sets `flushing`, and it means nothing while `flushing` is false.
    flush_replaces: AtomicUsize,
    /// The `hi` of the newest run that the merge in flight holds. Only the task stores it, before
    /// it sets `merging`, and it means nothing while `merging` is false.
    merge_hi: AtomicU64,
}

impl RangeJobs {
    /// True while a merge of the range, minor or major, runs.
    pub(super) fn merging(&self) -> bool {
        self.merging.load(Ordering::Acquire)
    }

    fn flushing(&self) -> bool {
        self.flushing.load(Ordering::Acquire)
    }

    /// The newest runs that the flush in flight replaces, which a merge must leave alone. Until
    /// the flush's end clears `flushing`, it may count runs that the flush already replaced, which
    /// only leaves a merge fewer runs.
    fn flush_held(&self) -> usize {
        if self.flushing() {
            self.flush_replaces.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// The newest of `runs` (newest first) that no merge in flight holds: all of them, or the ones
    /// above the newest run of the merge. Until the merge's end clears `merging`, its output or a
    /// rejected merge's inputs count as held, which only leaves a flush fewer runs.
    fn free_newest(&self, runs: &[Arc<Run>]) -> usize {
        if !self.merging() {
            return runs.len();
        }
        let hi = self.merge_hi.load(Ordering::Relaxed);
        runs.iter().take_while(|r| r.hi > hi).count()
    }

    /// The average bytes that a flush adds, or `default` before the first flush.
    fn average_flush(&self, default: u64) -> u64 {
        match self.flush_bytes.load(Ordering::Relaxed) {
            0 => default,
            bytes => bytes,
        }
    }

    /// Counts a flush that added `bytes`, in an average that gives the newest flush a quarter.
    fn record_flush(&self, bytes: u64) {
        let bytes = bytes.max(1);
        let average = match self.flush_bytes.load(Ordering::Relaxed) {
            0 => bytes,
            old => old - old / 4 + bytes / 4,
        };
        self.flush_bytes.store(average.max(1), Ordering::Relaxed);
    }
}

/// Runs `work`, the file I/O of a flush or a merge, on a blocking thread, so that it never holds
/// an async worker that reads need. With the I/O on the async workers, 16-task scans ran 6% slower
/// while merges ran (stage 2c measurement).
async fn off_workers<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, SlateDBError> + Send + 'static,
) -> Result<T, SlateDBError> {
    // Synchronous local-disk I/O, as the disk tier of the object store cache does
    // (`cached_object_store/storage_fs.rs`): one blocking task for a whole file.
    #[allow(clippy::disallowed_methods)]
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| SlateDBError::BackgroundTaskPanic(format!("range cache file work: {e}")))?
}

/// The nanoseconds since `started` on `clock`.
fn elapsed_nanos(
    clock: &dyn slatedb_common::clock::SystemClock,
    started: chrono::DateTime<chrono::Utc>,
) -> u64 {
    (clock.now() - started)
        .num_nanoseconds()
        .map_or(0, |n| n.max(0) as u64)
}

/// The merge that a range's layers call for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Merge {
    Minor,
    Major,
}

/// The run bytes at which a major merge of a range with this base starts: `f` times the base,
/// and at least `min_rebuild_delta_bytes`. `f` is `major_merge_factor`, or by default
/// `sqrt(2r / B)` between [`F_MIN`] and [`F_MAX`] (design 01 section 7.9).
pub(super) fn major_threshold(
    range: &CachedRange,
    base_bytes: u64,
    tuning: &RangeCacheTuning,
) -> u64 {
    let f = tuning.major_merge_factor.unwrap_or_else(|| {
        let r = range.jobs.average_flush(tuning.range_seal_bytes as u64) as f64;
        let b = base_bytes.max(1) as f64;
        (2.0 * r / b).sqrt().clamp(F_MIN, F_MAX)
    });
    ((f * base_bytes as f64) as u64).max(tuning.min_rebuild_delta_bytes as u64)
}

/// The first of a range's `runs` runs (positions newest first) that a merge may take, when the
/// flush in flight replaces the `flush_held` newest ones. A merge never takes those, and its runs
/// together with all older ones number at most `max_runs - 1`, so that a flush of a full range
/// always finds its newest run free (rule 1 of the module documentation).
pub(super) fn mergeable_from(runs: usize, flush_held: usize, max_runs: usize) -> usize {
    flush_held
        .max(runs.saturating_sub(max_runs.saturating_sub(1)))
        .min(runs)
}

/// The runs of a minor merge, as positions `a..b` of a range's runs (newest first, `sizes` in
/// bytes), none before `first` (see [`mergeable_from`]). `None` if no runs call for one.
///
/// With a finite `ratio`, by similar size: the newest two adjacent runs from `first` on where
/// the older one holds at most `ratio` times the bytes of the newer one start the group, and the
/// group takes each next older run while that run holds at most `ratio` times the bytes of the
/// group so far. So runs of 1, 1, 2 and 4 units, newest first, merge in one job.
///
/// With an infinite `ratio` every pair qualifies, and the rule is that of stage 2c: the two
/// oldest runs, once there are two. (Read literally, the similar-size rule would merge all runs
/// from the newest at once; any finite ratio, however large, does that.)
pub(super) fn minor_group(sizes: &[u64], first: usize, ratio: f64) -> Option<Range<usize>> {
    let n = sizes.len();
    if n < first + 2 {
        return None;
    }
    if ratio == f64::INFINITY {
        return Some(n - MERGE_RUNS..n);
    }
    let within = |older: u64, newer: u64| older as f64 <= ratio * newer as f64;
    let a = (first..n - 1).find(|&i| within(sizes[i + 1], sizes[i]))?;
    let mut group = sizes[a] + sizes[a + 1];
    let mut b = a + 2;
    while b < n && within(sizes[b], group) {
        group += sizes[b];
        b += 1;
    }
    Some(a..b)
}

/// The oldest runs that a major merge absorbs, of the `runs - first` that it may take (see
/// [`mergeable_from`]): at most [`MERGE_RUNS`] with the default minor rule, as in stage 2c, and
/// all of them with merges by similar size, whose runs are a few of growing size: what the merge
/// leaves would otherwise start the next cycle as runs to merge again.
pub(super) fn major_runs(runs: usize, first: usize, ratio: f64) -> usize {
    let may = runs.saturating_sub(first);
    if ratio == f64::INFINITY {
        may.min(MERGE_RUNS)
    } else {
        may
    }
}

/// The newest runs that a flush of a range with `runs` runs replaces, when `free` of its newest
/// runs are newer than the runs of the merge in flight: none below `max_runs`, else the newest
/// `free`, at most [`FLUSH_MERGE_RUNS`] (rule 2 of the module documentation). `None` if the range
/// is full and the merge holds its newest run, which rule 1 rules out.
pub(super) fn flush_replaces(runs: usize, free: usize, max_runs: usize) -> Option<usize> {
    if runs < max_runs {
        return Some(0);
    }
    match free.min(FLUSH_MERGE_RUNS) {
        0 => None,
        replaced => Some(replaced),
    }
}

/// A merge that a range's layers call for, with its runs as positions newest first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MergePlan {
    pub(super) merge: Merge,
    pub(super) runs: Range<usize>,
}

/// The merge that `layers`, the layers of `range`, call for, when the flush in flight replaces
/// the `flush_held` newest runs: a major merge of the oldest runs of [`major_runs`] when the range
/// has a base and its run bytes reach [`major_threshold`], else the minor merge of
/// [`minor_group`], if any.
pub(super) fn plan_merge(
    range: &CachedRange,
    layers: &Layers,
    tuning: &RangeCacheTuning,
    flush_held: usize,
) -> Option<MergePlan> {
    let n = layers.runs.len();
    if tuning.delta_view {
        // No minor merge: a major merge absorbs every run, when their bytes call for one or when
        // they reach half of the most runs.
        let base = layers.base.as_ref()?;
        let run_bytes: u64 = layers.runs.iter().map(|r| r.file_bytes()).sum();
        let due = n > 0
            && (run_bytes >= major_threshold(range, base.file_len(), tuning)
                || n >= view_major_runs(tuning));
        return due.then_some(MergePlan {
            merge: Merge::Major,
            runs: 0..n,
        });
    }
    let first = mergeable_from(n, flush_held, tuning.max_runs);
    if first == n {
        return None;
    }
    if let Some(base) = &layers.base {
        let run_bytes: u64 = layers.runs.iter().map(|r| r.file_bytes()).sum();
        if run_bytes >= major_threshold(range, base.file_len(), tuning) {
            let absorbed = major_runs(n, first, tuning.minor_merge_ratio);
            return Some(MergePlan {
                merge: Merge::Major,
                runs: n - absorbed..n,
            });
        }
    }
    let sizes: Vec<u64> = layers.runs.iter().map(|r| r.file_bytes()).collect();
    minor_group(&sizes, first, tuning.minor_merge_ratio).map(|runs| MergePlan {
        merge: Merge::Minor,
        runs,
    })
}

/// With the delta view, the runs at which a major merge of a range is due whatever their bytes:
/// half of `max_runs`, so that flushes have room while it runs.
pub(super) fn view_major_runs(tuning: &RangeCacheTuning) -> usize {
    (tuning.max_runs / 2).max(1)
}

/// The merge that `layers`, the layers of `range`, call for when no flush is in flight. A flush
/// in flight only defers it: its end notifies the task, which looks at the range again.
pub(super) fn merge_due(
    range: &CachedRange,
    layers: &Layers,
    tuning: &RangeCacheTuning,
) -> Option<Merge> {
    plan_merge(range, layers, tuning, 0).map(|plan| plan.merge)
}

impl RangeCache {
    /// The receiver of the maintenance task's queue, for its registration.
    pub(crate) fn maintenance_receiver(&self) -> async_channel::Receiver<MaintenanceMessage> {
        self.maintenance.rx.clone()
    }

    /// Writes `buffers`, the sealed buffers of `range` (newest first), with the versions of
    /// `replaces`, its newest runs (newest first, none for a plain flush), to one run of
    /// `(lo, hi]`, and installs it in their place. The task chose them (see [`Job::Flush`]).
    async fn flush(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        buffers: Vec<Arc<Delta>>,
        replaces: Vec<Arc<Run>>,
        lo: u64,
        hi: u64,
    ) -> Result<(), SlateDBError> {
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-flush");
        let run = {
            let (cache, buffers, replaces) = (self.clone(), buffers.clone(), replaces.clone());
            Arc::new(off_workers(move || cache.write_run(&buffers, &replaces, lo, hi)).await?)
        };
        let replaced: u64 = replaces.iter().map(|r| r.file_bytes()).sum();
        let added = run.file_bytes().saturating_sub(replaced);
        // Checkers keep the sealed buffers pinned by this job across many batches.
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-install-run");
        let installed = if self.options.tuning.delta_view {
            debug_assert!(replaces.is_empty(), "a flush-merge with the delta view");
            self.install_run_with_view(range, run, buffers).await?
        } else {
            self.install_run(range, run, buffers, replaces, None).await
        };
        if installed {
            range.jobs.record_flush(added);
            self.evict(None).await;
        }
        Ok(())
    }

    /// Builds the view of `range` with `run`, which a flush wrote from `buffers`, on the view that
    /// the range holds, and installs both. Builds and sends again while the writer rejects the
    /// install because the view moved on (a major merge installed meanwhile).
    async fn install_run_with_view(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        run: Arc<Run>,
        buffers: Vec<Arc<Delta>>,
    ) -> Result<bool, SlateDBError> {
        for attempt in 0..VIEW_ATTEMPTS {
            let (was, runs) = {
                let layers = range.layers.load();
                (layers.view.clone(), layers.runs.clone())
            };
            let view = {
                let (cache, run, was) = (self.clone(), run.clone(), was.clone());
                off_workers(move || {
                    let mut cost = BuildCost::default();
                    let started = cache.clock.now();
                    let view = match &was {
                        Some(view) => view.with_run(&run, &mut cost)?,
                        None => {
                            let mut all = vec![run];
                            all.extend(runs);
                            DeltaView::from_runs(&all, &cache.views, &mut cost)?
                        }
                    };
                    cost.nanos = elapsed_nanos(&*cache.clock, started);
                    cache.stats.view[0].built(&cost);
                    Ok(view)
                })
                .await?
            };
            if attempt > 0 {
                self.stats.view[0].retries.increment(1);
            }
            // Checkers stretch the window in which a major merge can change the view.
            fail_point!(
                Arc::clone(&self.fp_registry),
                "range-cache-install-run-view"
            );
            let swap = ViewSwap {
                new: Some(Arc::new(view)),
                was: was.clone(),
            };
            if self
                .install_run(range, run.clone(), buffers.clone(), Vec::new(), Some(swap))
                .await
            {
                return Ok(true);
            }
            // Only a view that moved on calls for another build: the writer rejects anything else
            // again.
            if same_view(&range.layers.load().view, &was) || !self.is_indexed(range) {
                return Ok(false);
            }
        }
        Ok(false)
    }

    /// Writes the versions of `buffers` (newest first, each up to its seal) and of `runs`
    /// (newest first) to a run of `(lo, hi]`.
    pub(super) fn write_run(
        &self,
        buffers: &[Arc<Delta>],
        runs: &[Arc<Run>],
        lo: u64,
        hi: u64,
    ) -> Result<Run, SlateDBError> {
        let mut writer = RangeFileWriter::create_run(
            &self.options.root_folder,
            &self.new_file_name(),
            self.options.tuning.page_size,
        )?;
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
        merge_versions(&mut writer, &sealed, &runs)?;
        let path = writer.finish()?;
        let file = RangeFile::open(path, self.options.page_source, self.files.clone())?;
        Ok(Run::new(file, lo, hi)?)
    }

    /// Merges `inputs`, adjacent runs of `range` (newest first), into one run of their union
    /// interval, every version kept, and installs it in their place, wherever they are by then.
    async fn minor_merge(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        inputs: Vec<Arc<Run>>,
    ) -> Result<(), SlateDBError> {
        let (Some(newest), Some(oldest)) = (inputs.first(), inputs.last()) else {
            return Ok(());
        };
        let (lo, hi) = (oldest.lo, newest.hi);
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-minor-merge");
        let output = {
            let (cache, inputs) = (self.clone(), inputs.clone());
            Arc::new(off_workers(move || cache.write_run(&[], &inputs, lo, hi)).await?)
        };
        // Checkers hold the merged run here while flushes put runs before its inputs.
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-install-minor");
        if self.install_minor(range, inputs, output).await {
            self.evict(None).await;
        }
        Ok(())
    }

    /// Merges `base`, the base of `range`, with `absorbed`, its oldest runs (newest first), into
    /// a new base at the newest absorbed run's `hi`, and installs it.
    async fn major_merge(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        base: Arc<RangeFile>,
        absorbed: Vec<Arc<Run>>,
    ) -> Result<(), SlateDBError> {
        let Some(base_seq) = absorbed.first().map(|r| r.hi) else {
            return Ok(());
        };
        range
            .reads_at_build
            .store(range.heat.reads(), Ordering::Relaxed);
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-rebuild");
        let new_base = {
            let (cache, absorbed, base) = (self.clone(), absorbed.clone(), base.clone());
            // The merge iterator is asynchronous, but reads only local files.
            let handle = tokio::runtime::Handle::current();
            Arc::new(
                off_workers(move || handle.block_on(cache.merge_files(&absorbed, &base, base_seq)))
                    .await?,
            )
        };
        let installed = if self.options.tuning.delta_view {
            self.install_major_with_view(range, &base, absorbed, new_base, base_seq)
                .await?
        } else {
            self.install_major(range, &base, absorbed, new_base, base_seq, None)
                .await
        };
        if installed {
            self.evict(None).await;
        }
        Ok(())
    }

    /// Builds the view of `range` without `absorbed`, on the view that the range holds, and
    /// installs it with `new_base`. Builds and sends again while the writer rejects the install
    /// because the view moved on (a flush installed meanwhile).
    async fn install_major_with_view(
        self: &Arc<Self>,
        range: &Arc<CachedRange>,
        base: &Arc<RangeFile>,
        absorbed: Vec<Arc<Run>>,
        new_base: Arc<RangeFile>,
        base_seq: u64,
    ) -> Result<bool, SlateDBError> {
        for attempt in 0..VIEW_ATTEMPTS {
            let was = range.layers.load().view.clone();
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
            // Checkers stretch the window in which a flush can change the view.
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
                    base,
                    absorbed.clone(),
                    new_base.clone(),
                    base_seq,
                    Some(swap),
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

    /// Writes the state at `base_seq` of `runs` (newest first) over `base` to a new base: newest
    /// versions, tombstones dropped, merge operands folded, base-less merge operands kept.
    pub(super) async fn merge_files(
        &self,
        runs: &[Arc<Run>],
        base: &Arc<RangeFile>,
        base_seq: u64,
    ) -> Result<RangeFile, SlateDBError> {
        let all = BytesRange::unbounded();
        let mut sources: Vec<Box<dyn RowEntryIterator + 'static>> = runs
            .iter()
            .map(|run| {
                Box::new(RangeFileIterator::new(run.file.clone(), all.clone()))
                    as Box<dyn RowEntryIterator + 'static>
            })
            .collect();
        sources.push(Box::new(RangeFileIterator::new(base.clone(), all.clone())));
        let mut iter = DbIterator::new(
            all,
            None,
            sources,
            Box::new(EmptyIterator::new()),
            Some(base_seq),
            self.merge_operator.clone(),
            IterationOrder::Ascending,
            ReadTrace::none(),
        )
        .await?;
        let mut writer = RangeFileWriter::create(
            &self.options.root_folder,
            &self.new_file_name(),
            self.options.tuning.page_size,
        )?;
        while let Some(entry) = iter.next_entry().await? {
            writer.push(&entry)?;
        }
        let path = writer.finish()?;
        Ok(RangeFile::open(
            path,
            self.options.page_source,
            self.files.clone(),
        )?)
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
    /// Ranges whose minor merge waits for a merge slot. Minor merges go first: they are small,
    /// and they keep ranges below the most runs.
    minor_queue: VecDeque<Weak<CachedRange>>,
    /// Ranges whose major merge waits for a merge slot.
    major_queue: VecDeque<Weak<CachedRange>>,
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
    /// Merge these adjacent runs (newest first) into one run.
    Minor(Vec<Arc<Run>>),
    /// Merge `base` with `absorbed`, the oldest runs (newest first), into a new base.
    Major {
        base: Arc<RangeFile>,
        absorbed: Vec<Arc<Run>>,
    },
}

impl Job {
    fn name(&self) -> &'static str {
        match self {
            Job::Flush { .. } => "range_cache_flush",
            Job::Minor(_) => "range_cache_minor_merge",
            Job::Major { .. } => "range_cache_major_merge",
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
            minor_queue: VecDeque::new(),
            major_queue: VecDeque::new(),
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
                // The file and metadata gauges change with every install; the writer updates
                // them only when the index changes.
                self.cache.update_gauges();
            }
        }
        self.start_queued();
        self.seal_under_pressure(&index);
    }

    /// Queues the jobs that `range`, an indexed range, calls for, unless they run or wait.
    fn look(&mut self, range: &Arc<CachedRange>) {
        let layers = range.layers.load();
        let jobs = &range.jobs;
        if !layers.sealed.is_empty()
            && !jobs.flushing()
            && !jobs.flush_queued.swap(true, Ordering::Relaxed)
        {
            self.flush_queue.push_back(Arc::downgrade(range));
        }
        if jobs.merging() || jobs.merge_queued.load(Ordering::Relaxed) {
            return;
        }
        let tuning = &self.cache.options.tuning;
        let queue = match plan_merge(range, &layers, tuning, jobs.flush_held()) {
            Some(MergePlan {
                merge: Merge::Minor,
                ..
            }) => &mut self.minor_queue,
            Some(MergePlan {
                merge: Merge::Major,
                ..
            }) => &mut self.major_queue,
            // A flush in flight may defer the merge: its end notifies the task.
            None => return,
        };
        jobs.merge_queued.store(true, Ordering::Relaxed);
        queue.push_back(Arc::downgrade(range));
    }

    /// The flush that `range` calls for, with its inputs: every sealed buffer, and when the range
    /// holds `max_runs` runs, its newest runs that the merge in flight does not hold, at most
    /// [`FLUSH_MERGE_RUNS`]. `None` if the range has no sealed buffer.
    fn plan_flush(&self, range: &CachedRange) -> Option<Job> {
        let layers = range.layers.load();
        let newest = layers.sealed.first()?;
        let hi = newest.sealed_at().load(Ordering::Acquire);
        let (runs, max_runs) = (layers.runs.len(), self.cache.options.tuning.max_runs);
        if self.cache.options.tuning.delta_view {
            // No flush-merge: a full range waits for its major merge, whose end notifies the task.
            if runs >= max_runs {
                return None;
            }
            return Some(Job::Flush {
                buffers: layers.sealed.clone(),
                replaces: Vec::new(),
                lo: layers.covered_to,
                hi,
            });
        }
        let free = range.jobs.free_newest(&layers.runs);
        debug_assert!(
            runs < max_runs || free > 0,
            "a merge holds the newest run of a full range"
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

    /// Starts queued jobs while slots are free: flushes, then minor merges, then major merges.
    /// A job's range is checked again when it starts, as it may have changed while it waited,
    /// and the job's inputs are chosen then.
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
            if range.jobs.flushing() || !cache.is_indexed(&range) {
                continue;
            }
            if let Some(job) = self.plan_flush(&range) {
                self.start(&range, job);
            }
        }
        while cache.maintenance.merges.load(Ordering::Acquire) < MAX_MERGES {
            let Some(range) = self
                .minor_queue
                .pop_front()
                .or_else(|| self.major_queue.pop_front())
            else {
                break;
            };
            let Some(range) = range.upgrade() else {
                continue;
            };
            range.jobs.merge_queued.store(false, Ordering::Relaxed);
            if range.jobs.merging() || !cache.is_indexed(&range) {
                continue;
            }
            let layers = range.layers.load();
            let tuning = &cache.options.tuning;
            let Some(plan) = plan_merge(&range, &layers, tuning, range.jobs.flush_held()) else {
                continue;
            };
            let runs = layers.runs[plan.runs].to_vec();
            let job = match (plan.merge, &layers.base) {
                (Merge::Minor, _) => Job::Minor(runs),
                (Merge::Major, Some(base)) => {
                    if !cache.rebuild_pays(&range, base.file_len()) {
                        // The range's buffers and runs hold its recent writes, so it stays exact
                        // until the writer removes it.
                        cache.request_removal(
                            Target::Range(range.clone()),
                            RemoveReason::NotWorthIt,
                        );
                        continue;
                    }
                    Job::Major {
                        base: base.clone(),
                        absorbed: runs,
                    }
                }
                // A major merge is due only with a base.
                (Merge::Major, None) => continue,
            };
            drop(layers);
            self.start(&range, job);
        }
    }

    /// Above the pressure mark, while fewer than [`MAX_FLUSHES`] flushes wait, asks the writer to
    /// seal the largest active buffers that hold at least `min_seal_bytes`, so that flushes can
    /// free them. With more flushes waiting, the seals could not free memory sooner, and the walk
    /// would cost a look at every range per notification.
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
        let flush = match &job {
            Job::Flush { replaces, .. } => {
                jobs.flush_replaces.store(replaces.len(), Ordering::Relaxed);
                true
            }
            Job::Minor(runs) | Job::Major { absorbed: runs, .. } => {
                jobs.merge_hi
                    .store(runs.first().map_or(0, |r| r.hi), Ordering::Relaxed);
                false
            }
        };
        let (flag, running) = if flush {
            (&jobs.flushing, &self.cache.maintenance.flushes)
        } else {
            (&jobs.merging, &self.cache.maintenance.merges)
        };
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
                let (flag, running) = if flush {
                    (&done_range.jobs.flushing, &done_cache.maintenance.flushes)
                } else {
                    (&done_range.jobs.merging, &done_cache.maintenance.merges)
                };
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
                    Job::Minor(inputs) => cache.minor_merge(&range, inputs).await,
                    Job::Major { base, absorbed } => {
                        cache.major_merge(&range, base, absorbed).await
                    }
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
        Unfit, FLUSHED_BUFFER_COUNT, FLUSH_BYTES, FLUSH_COUNT, FLUSH_MERGE_COUNT,
        FLUSH_MERGE_RUN_COUNT, INSTALL_REJECTED_COUNT, LOAD_REJECTED_COUNT, MINOR_MERGE_COUNT,
        MINOR_MERGE_RUNS_COUNT, MULTI_BUFFER_FLUSH_COUNT, REBUILD_COUNT, REMOVE_COUNT, SEAL_COUNT,
    };
    use crate::range_cache::stats::{VIEW_BUILD_COUNT, VIEW_BYTES, VIEW_RETRY_COUNT};
    use crate::range_cache::{CachedRangeInfo, RangeCacheOptions, MAX_VIEW_RUNS_RANGE};

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
        assert_eq!(layers.covered_to, layers.base_seq);
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
        assert_eq!((run.lo, run.hi, layers.covered_to), (layers.base_seq, s, s));
        assert_eq!(run.file.rows(), sealed.len());
        assert_eq!(t.metric(FLUSH_COUNT, &[]), 1);
        assert_eq!(t.metric(FLUSH_BYTES, &[]) as u64, run.file_bytes());
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
            Arc::new(cache.write_run(buffers, &[], lo, hi).unwrap())
        };
        let both = vec![newer.clone(), oldest.clone()];
        let lo = layers.covered_to;
        let files = t.cache_files();
        // Not the oldest sealed buffers.
        let r = run(std::slice::from_ref(&newer), newer.lo(), seal(&newer));
        assert!(
            !cache
                .install_run(&range, r, vec![newer.clone()], vec![], None)
                .await
        );
        // Not from `covered_to`, or not to the newest buffer's seal.
        let r = run(&both, lo - 1, seal(&newer));
        assert!(
            !cache
                .install_run(&range, r, both.clone(), vec![], None)
                .await
        );
        let r = run(&both, lo, seal(&newer) - 1);
        assert!(
            !cache
                .install_run(&range, r, both.clone(), vec![], None)
                .await
        );
        // Runs to replace that the range does not hold.
        let stray = run(std::slice::from_ref(&oldest), lo, seal(&oldest));
        let r = run(&both, lo, seal(&newer));
        assert!(
            !cache
                .install_run(&range, r, both.clone(), vec![stray], None)
                .await
        );
        // A rejected run's file is deleted when it drops.
        assert_eq!(t.cache_files(), files);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 4);
        // Both buffers in one run: one store removes both.
        let first = run(&both, lo, seal(&newer));
        assert!(
            cache
                .install_run(&range, first.clone(), both.clone(), vec![], None)
                .await
        );
        let after = range.layers.load_full();
        assert_eq!(after.covered_to, seal(&newer));
        assert!(after.sealed.is_empty());
        assert!(Arc::ptr_eq(&after.runs[0], &first));
        assert_eq!(first.file.rows(), newer.len() + oldest.len());
        assert_eq!(t.metric(MULTI_BUFFER_FLUSH_COUNT, &[]), 1);
        t.assert_reads(0, 100, None).await;
        // The held flush wrote a run of buffers that are gone: it is rejected.
        t.fail("range-cache-flush", "off");
        t.wait("the held flush's rejection", |t| {
            t.metric(INSTALL_REJECTED_COUNT, &[]) == 5
        })
        .await;

        // Minor merges whose runs moved on, or whose output does not cover them.
        let a = run(&[], lo, lo);
        assert!(
            !cache
                .install_minor(&range, vec![first.clone(), a], first.clone())
                .await
        );
        assert!(!cache.install_minor(&range, vec![], first.clone()).await);
        let short = Arc::new(
            cache
                .write_run(&[], std::slice::from_ref(&first), first.lo, first.hi)
                .unwrap(),
        );
        let wrong = Arc::new(
            Run::new(
                RangeFile::open(
                    {
                        let mut w = RangeFileWriter::create_run(
                            &cache.options.root_folder,
                            &cache.new_file_name(),
                            cache.options.tuning.page_size,
                        )
                        .unwrap();
                        w.push(&crate::types::RowEntry::new_value(&key(0), b"x", first.hi))
                            .unwrap();
                        w.finish().unwrap()
                    },
                    cache.options.page_source,
                    cache.files.clone(),
                )
                .unwrap(),
                first.lo + 1,
                first.hi,
            )
            .unwrap(),
        );
        assert!(
            !cache
                .install_minor(&range, vec![first.clone()], wrong)
                .await
        );
        // One run merged into itself is a valid, if useless, minor merge.
        assert!(
            cache
                .install_minor(&range, vec![first.clone()], short.clone())
                .await
        );
        assert!(Arc::ptr_eq(&range.layers.load().runs[0], &short));
        assert_eq!(t.metric(MINOR_MERGE_COUNT, &[]), 1);

        // A major merge whose base, runs or cut moved on.
        let base = after.base.clone().unwrap();
        let merged = |base_seq| {
            let cache = cache.clone();
            let (runs, base) = (vec![short.clone()], base.clone());
            async move { Arc::new(cache.merge_files(&runs, &base, base_seq).await.unwrap()) }
        };
        let other = merged(short.hi).await;
        let absorbed = vec![short.clone()];
        assert!(
            !cache
                .install_major(
                    &range,
                    &other,
                    absorbed.clone(),
                    merged(short.hi).await,
                    short.hi,
                    None
                )
                .await
        );
        assert!(
            !cache
                .install_major(
                    &range,
                    &base,
                    vec![],
                    merged(short.hi).await,
                    short.hi,
                    None
                )
                .await
        );
        assert!(
            !cache
                .install_major(
                    &range,
                    &base,
                    vec![first.clone()],
                    merged(short.hi).await,
                    short.hi,
                    None
                )
                .await
        );
        assert!(
            !cache
                .install_major(
                    &range,
                    &base,
                    absorbed.clone(),
                    merged(short.hi).await,
                    short.hi + 1,
                    None
                )
                .await
        );
        assert!(
            cache
                .install_major(
                    &range,
                    &base,
                    absorbed,
                    merged(short.hi).await,
                    short.hi,
                    None
                )
                .await
        );
        let after = range.layers.load_full();
        assert_eq!((after.base_seq, after.runs.len()), (short.hi, 0));
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
            let stale = run(&held, oldest.lo(), seal(newest));
            let rejected = t.metric(INSTALL_REJECTED_COUNT, &[]);
            assert!(
                !cache
                    .install_run(&range, stale.clone(), held.clone(), vec![], None)
                    .await
            );
            t.db.cache_range(span(0, 100)).await.unwrap();
            assert!(!Arc::ptr_eq(&t.range(&key(0)), &range));
            assert!(
                !cache
                    .install_run(&range, stale.clone(), held.clone(), vec![], None)
                    .await
            );
            assert!(
                !cache
                    .install_minor(&range, vec![short.clone()], stale)
                    .await
            );
            assert!(t.metric(INSTALL_REJECTED_COUNT, &[]) >= rejected + 3);
        }
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
        assert_eq!(t.metric(FLUSHED_BUFFER_COUNT, &[]), 4);
        assert!(t.metric(FLUSH_COUNT, &[]) < 4);
        assert!(t.metric(MULTI_BUFFER_FLUSH_COUNT, &[]) >= 1);
        t.wait("the minor merge", |t| t.info(&key(0)).runs == 1)
            .await;
        let layers = range.layers.load_full();
        let run = &layers.runs[0];
        assert_eq!(
            (run.lo, run.hi),
            (
                layers.base_seq,
                sealed[0].sealed_at().load(Ordering::SeqCst)
            )
        );
        let versions: u64 = sealed.iter().map(|b| b.len()).sum();
        assert_eq!(run.file.rows(), versions);
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
        assert_eq!(t.metric(FLUSHED_BUFFER_COUNT, &[]), 2);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        t.assert_reads(0, 200, None).await;
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_minor_merge_two_runs_keeping_every_version() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        t.fail("range-cache-minor-merge", "pause");
        seal_and_flush(&t, 0, 100).await;
        // A snapshot inside the first run's interval, so that it reads older versions.
        let inside = t.db.snapshot().unwrap();
        t.write_until(0, 100, |t| t.committed() > inside.seq() + 10)
            .await;
        seal_and_flush(&t, 0, 100).await;
        let two = range.layers.load_full();
        assert_eq!(two.runs.len(), 2);
        assert!(t.info(&key(0)).rebuilding, "a minor merge is due");
        t.assert_reads(0, 100, Some(&inside)).await;
        t.fail("range-cache-minor-merge", "off");
        t.wait("the minor merge", |t| t.metric(MINOR_MERGE_COUNT, &[]) == 1)
            .await;
        let one = range.layers.load_full();
        assert_eq!(one.runs.len(), 1);
        let merged = &one.runs[0];
        assert_eq!((merged.lo, merged.hi), (two.runs[1].lo, two.runs[0].hi));
        assert_eq!(
            merged.file.rows(),
            two.runs[0].file.rows() + two.runs[1].file.rows()
        );
        assert_eq!(one.covered_to, two.covered_to);
        t.assert_reads(0, 100, Some(&inside)).await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        drop((two, one));
        assert_eq!(t.cache_files(), 2, "the merged runs are deleted");
        t.db.close().await.unwrap();
    }

    /// While a minor merge holds the two oldest runs, flushes reach four runs, and the next one
    /// merges the two newest runs into its run instead of waiting.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_flush_merge_at_four_runs() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        let max_runs = RangeCacheTuning::default().max_runs;
        t.fail("range-cache-minor-merge", "pause");
        for runs in 1..=max_runs {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(t.info(&key(0)).runs, runs);
        }
        let four = range.layers.load_full();
        t.assert_reads(0, 100, Some(&snapshot)).await;
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        assert_eq!(
            t.metric(FLUSH_MERGE_RUN_COUNT, &[]),
            FLUSH_MERGE_RUNS as i64
        );
        let three = range.layers.load_full();
        assert_eq!(three.runs.len(), max_runs - FLUSH_MERGE_RUNS + 1);
        let merged = &three.runs[0];
        assert_eq!(merged.lo, four.runs[FLUSH_MERGE_RUNS - 1].lo);
        assert_eq!(merged.hi, three.covered_to);
        // The two oldest runs, which the minor merge holds, are still there.
        assert!(Arc::ptr_eq(&three.runs[1], &four.runs[2]));
        assert!(Arc::ptr_eq(&three.runs[2], &four.runs[3]));
        let flushed = merged.file.rows() - four.runs[0].file.rows() - four.runs[1].file.rows();
        assert!(flushed > 0);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The held minor merge still installs: its runs are a slice of the layers.
        t.fail("range-cache-minor-merge", "off");
        t.wait("the minor merge", |t| t.metric(MINOR_MERGE_COUNT, &[]) >= 1)
            .await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert_eq!(t.info(&key(0)).runs, 1);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_keep_runs_flushed_during_a_major_merge() {
        // A run larger than f times the base starts a major merge: the base of 100 rows is
        // about 28 KiB, and f is at most 1.
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
        assert_eq!(t.info(&key(0)).runs, 2);
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 0);
        assert_eq!(
            t.metric(MINOR_MERGE_COUNT, &[]),
            0,
            "one merge at a time per range"
        );
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.wait("the install", |t| t.metric(REBUILD_COUNT, &[]) == 1)
            .await;
        // A later merge sleeps too, so this state holds while the test reads it.
        let layers = range.layers.load_full();
        assert_eq!(layers.base_seq, cut);
        assert_eq!(layers.runs.len(), 1);
        assert_eq!(layers.runs[0].lo, cut);
        assert!(!Arc::ptr_eq(
            layers.base.as_ref().unwrap(),
            one.base.as_ref().unwrap()
        ));
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
    /// boundaries of the largest sorted run (design 01 section 7.6).
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

    /// The major-merge threshold follows design 01 section 7.9: `f = sqrt(2r / B)`, clamped.
    #[test]
    fn should_choose_the_major_merge_factor_from_the_run_size() {
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
                base: None,
                base_seq: 0,
                covered_to: 0,
                end: None,
            },
            super::super::Origin::Explicit,
            0,
            0,
        );
        let tuning = RangeCacheTuning {
            min_rebuild_delta_bytes: 0,
            ..RangeCacheTuning::default()
        };
        let mib = 1024 * 1024;
        // Before any flush, r is the seal size, 4 MiB: f = sqrt(8 / 64) for a 64 MiB base.
        let f = |t: u64, b: u64| t as f64 / b as f64;
        assert!((f(major_threshold(&range, 64 * mib, &tuning), 64 * mib) - 0.3536).abs() < 0.001);
        range.jobs.record_flush(16 * mib);
        // The average moves a quarter of the way to each new flush.
        assert_eq!(range.jobs.average_flush(0), 16 * mib);
        range.jobs.record_flush(0);
        assert_eq!(range.jobs.average_flush(0), 12 * mib);
        // Clamped: a tiny base merges at its size, a huge one at a twentieth.
        assert_eq!(major_threshold(&range, 1024, &tuning), 1024);
        assert_eq!(
            major_threshold(&range, 1 << 40, &tuning),
            (0.05 * (1u64 << 40) as f64) as u64
        );
        // And never below `min_rebuild_delta_bytes`.
        let floor = RangeCacheTuning {
            min_rebuild_delta_bytes: 5 * mib as usize,
            ..RangeCacheTuning::default()
        };
        assert_eq!(major_threshold(&range, 1024, &floor), 5 * mib);
        // A fixed factor ignores the run size and the adaptive bounds, but keeps the floor.
        let fixed = |f: f64, floor: usize| RangeCacheTuning {
            major_merge_factor: Some(f),
            min_rebuild_delta_bytes: floor,
            ..RangeCacheTuning::default()
        };
        assert_eq!(major_threshold(&range, 64 * mib, &fixed(0.5, 0)), 32 * mib);
        assert_eq!(major_threshold(&range, 64 * mib, &fixed(2.0, 0)), 128 * mib);
        assert_eq!(
            major_threshold(&range, 1 << 40, &fixed(0.01, 0)),
            (0.01 * (1u64 << 40) as f64) as u64
        );
        assert_eq!(
            major_threshold(&range, 1024, &fixed(1.0, 5 * mib as usize)),
            5 * mib
        );
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

    /// The minor rule of `minor_merge_ratio`: an infinite ratio takes the two oldest runs, as in
    /// stage 2c; a finite one takes runs of similar size from the newest end, a whole cascade at
    /// once, and never a run before `first`.
    #[test]
    fn should_pick_minor_merge_groups_by_size() {
        let inf = f64::INFINITY;
        // The stage 2c rule, whatever the sizes.
        assert_eq!(minor_group(&[], 0, inf), None);
        assert_eq!(minor_group(&[1], 0, inf), None);
        assert_eq!(minor_group(&[1, 1], 0, inf), Some(0..2));
        assert_eq!(minor_group(&[1, 100], 0, inf), Some(0..2));
        assert_eq!(minor_group(&[7, 1, 100, 3], 0, inf), Some(2..4));
        assert_eq!(minor_group(&[7, 1, 100, 3], 1, inf), Some(2..4));
        assert_eq!(minor_group(&[7, 1, 100], 2, inf), None);
        // By similar size at ratio 2: a cascade is one merge.
        assert_eq!(minor_group(&[1, 1, 2, 4], 0, 2.0), Some(0..4));
        assert_eq!(minor_group(&[1, 2, 5, 13], 0, 2.0), Some(0..4));
        assert_eq!(minor_group(&[1, 1, 3, 8], 0, 2.0), Some(0..4));
        // The group stops before a run of more than twice its bytes.
        assert_eq!(minor_group(&[1, 1, 5], 0, 2.0), Some(0..2));
        assert_eq!(minor_group(&[1, 1, 2, 9, 1], 0, 2.0), Some(0..3));
        // No pair qualifies: each older run holds more than twice the newer one.
        assert_eq!(minor_group(&[1, 3], 0, 2.0), None);
        assert_eq!(minor_group(&[1, 3, 8, 21], 0, 2.0), None);
        // The newest qualifying pair starts the group, and an older run may be the smaller.
        assert_eq!(minor_group(&[1, 3, 4, 5], 0, 2.0), Some(1..4));
        assert_eq!(minor_group(&[4, 1, 1], 0, 2.0), Some(0..3));
        // Runs before `first` are left alone.
        assert_eq!(minor_group(&[1, 1, 2, 4], 1, 2.0), Some(1..4));
        assert_eq!(minor_group(&[1, 1, 9], 1, 2.0), None);
        // Ratio 1 merges an older run only if it is no larger than the newer ones.
        assert_eq!(minor_group(&[2, 2], 0, 1.0), Some(0..2));
        assert_eq!(minor_group(&[2, 3], 0, 1.0), None);
        // Any finite ratio, however large, merges from the newest end; infinity does not.
        assert_eq!(minor_group(&[1, 1, 1], 0, 1e12), Some(0..3));
        assert_eq!(minor_group(&[1, 1, 1], 0, inf), Some(1..3));
    }

    /// Rules 1 and 2 of the module documentation, and the runs of major merges, as numbers.
    #[test]
    fn should_bound_the_runs_of_flushes_and_merges() {
        // Rule 1: a merge leaves the flush's runs alone, and the newest run of a full range.
        assert_eq!(mergeable_from(0, 0, 4), 0);
        assert_eq!(mergeable_from(3, 0, 4), 0);
        assert_eq!(mergeable_from(4, 0, 4), 1);
        assert_eq!(mergeable_from(7, 0, 8), 0);
        assert_eq!(mergeable_from(8, 0, 8), 1);
        assert_eq!(mergeable_from(4, 2, 4), 2);
        assert_eq!(mergeable_from(3, 2, 4), 2);
        assert_eq!(mergeable_from(2, 2, 4), 2);
        // Rule 2: the flush of a full range merges its free newest runs, at most two.
        assert_eq!(flush_replaces(3, 0, 4), Some(0));
        assert_eq!(flush_replaces(4, 4, 4), Some(2));
        assert_eq!(flush_replaces(4, 2, 4), Some(2));
        assert_eq!(flush_replaces(4, 1, 4), Some(1));
        assert_eq!(flush_replaces(4, 0, 4), None);
        assert_eq!(flush_replaces(8, 3, 8), Some(2));
        // Major merges: the two oldest runs with the default rule, every one they may take by
        // size.
        assert_eq!(major_runs(4, 1, f64::INFINITY), 2);
        assert_eq!(major_runs(1, 0, f64::INFINITY), 1);
        assert_eq!(major_runs(4, 1, 2.0), 3);
        assert_eq!(major_runs(3, 2, 2.0), 1);
    }

    /// A model of one range under rules 1 and 2, with the task's planning functions. Flushes and
    /// merges start, install and end as separate steps in a random order, as the task and the
    /// writer see them, so a job's flag stays set for a while after its install. Every flush of a
    /// full range finds a run to merge, no run is held by a flush and a merge at once, every
    /// install's precondition holds when it arrives, and no range passes `max_runs`.
    #[test]
    fn should_keep_flushes_and_merges_off_each_others_runs() {
        /// A run by its interval, which identifies it (intervals are never empty here), and its
        /// bytes.
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        struct R {
            lo: u64,
            hi: u64,
            bytes: u64,
        }
        /// A job in flight: the runs it holds, newest first.
        struct InFlight {
            runs: Vec<R>,
            installed: bool,
            major: bool,
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        let ratios = [f64::INFINITY, 1.0, 2.0, 4.0, 1e9];
        // Flush-merges of one run and of two, minor merges of more than two runs, major merges.
        let mut seen = [0u64; 4];
        for case in 0..2_000 {
            let max_runs = 4 + case % 5;
            let ratio = ratios[case / 5 % ratios.len()];
            let mut runs: Vec<R> = Vec::new();
            let (mut seq, mut covered_to) = (0u64, 0u64);
            let mut flush: Option<InFlight> = None;
            let mut merge: Option<InFlight> = None;
            for _ in 0..400 {
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
                            major: false,
                        });
                    }
                    // The writer installs it, with the checks of `install_run_at`.
                    1 => {
                        if let Some(f) = flush.as_mut().filter(|f| !f.installed) {
                            assert_eq!(&runs[..f.runs.len()], &f.runs[..], "a flush's runs moved");
                            assert!(runs.len() - f.runs.len() < max_runs);
                            let lo = f.runs.last().map_or(covered_to, |r| r.lo);
                            seq += 1 + rand(4);
                            let bytes = f.runs.iter().map(|r| r.bytes).sum::<u64>() + 1 + rand(8);
                            runs.splice(..f.runs.len(), [R { lo, hi: seq, bytes }]);
                            covered_to = seq;
                            match f.runs.len() {
                                1 => seen[0] += 1,
                                2 => seen[1] += 1,
                                _ => {}
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
                    // The task starts a merge, major or minor.
                    3 if merge.is_none() => {
                        let flush_held = flush.as_ref().map_or(0, |f| f.runs.len());
                        let n = runs.len();
                        let first = mergeable_from(n, flush_held, max_runs);
                        let major = rand(4) == 0;
                        let plan = if first == n {
                            None
                        } else if major {
                            Some(n - major_runs(n, first, ratio)..n)
                        } else {
                            let sizes: Vec<u64> = runs.iter().map(|r| r.bytes).collect();
                            minor_group(&sizes, first, ratio)
                        };
                        if let Some(at) = plan {
                            let taken = runs[at].to_vec();
                            if let Some(f) = flush.as_ref().filter(|f| !f.installed) {
                                assert!(taken.iter().all(|r| !f.runs.contains(r)));
                            }
                            merge = Some(InFlight {
                                runs: taken,
                                installed: false,
                                major,
                            });
                        }
                    }
                    // The writer installs it, with the checks of `install_minor_at` and
                    // `install_major_at`.
                    4 => {
                        if let Some(m) = merge.as_mut().filter(|m| !m.installed) {
                            let at = runs
                                .iter()
                                .position(|r| *r == m.runs[0])
                                .expect("a merge's newest run is gone");
                            let end = at + m.runs.len();
                            assert_eq!(&runs[at..end], &m.runs[..], "a merge's runs moved");
                            if m.major {
                                assert_eq!(end, runs.len(), "absorbed runs are not the oldest");
                                runs.truncate(at);
                                seen[3] += 1;
                            } else {
                                let lo = m.runs[m.runs.len() - 1].lo;
                                let bytes = m.runs.iter().map(|r| r.bytes).sum();
                                let hi = m.runs[0].hi;
                                runs.splice(at..end, [R { lo, hi, bytes }]);
                                if m.runs.len() > 2 {
                                    seen[2] += 1;
                                }
                            }
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

    /// With merges by similar size, a minor merge takes the two newest runs and leaves an older,
    /// larger one. Flushes then put runs before its runs, so it installs in the middle of the
    /// runs, and the flush of the full range merges only the newest run, the one that the merge
    /// does not hold. No install is rejected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_install_a_merge_by_size_in_the_middle_of_the_runs() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.minor_merge_ratio = 2.0;
        })
        .await;
        load(&t, 0, 1000).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        // A large run first: a batch of every key seals the buffer once, at its end.
        let mut batch = crate::WriteBatch::new();
        for i in 0..1000 {
            batch.put(key(i), value(i, 1));
        }
        let seals = t.seals();
        t.db.write(batch).await.unwrap();
        t.wait("the seal", |t| t.seals() == seals + 1).await;
        t.wait("the flush", |t| {
            let info = t.info(&key(0));
            info.sealed_buffers == 0 && info.runs == 1
        })
        .await;
        assert_eq!(t.seals(), seals + 1);
        // Then small runs. A small run on the large one is no pair; two small ones are, and
        // their merge waits before its install.
        t.fail("range-cache-install-minor", "pause");
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.info(&key(0)).runs, 2);
        assert!(!t.info(&key(0)).rebuilding);
        seal_and_flush(&t, 0, 100).await;
        let three = range.layers.load_full();
        assert_eq!(three.runs.len(), 3);
        t.wait("the minor merge", |_| range.jobs.merging()).await;
        // A plain flush puts a fourth run before the merge's runs.
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.info(&key(0)).runs, 4);
        // The range is full, and the merge holds its second newest run: the flush merges the
        // newest one only.
        seal_and_flush(&t, 0, 100).await;
        let four = range.layers.load_full();
        assert_eq!(four.runs.len(), 4);
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        assert_eq!(t.metric(FLUSH_MERGE_RUN_COUNT, &[]), 1);
        for i in 0..3 {
            assert!(Arc::ptr_eq(&four.runs[i + 1], &three.runs[i]));
        }
        t.assert_reads(0, 1000, Some(&snapshot)).await;
        // The merge installs in the middle. The next merge waits before it writes, so the
        // layers hold still.
        t.fail("range-cache-minor-merge", "pause");
        t.fail("range-cache-install-minor", "off");
        t.wait("the minor merge", |t| t.metric(MINOR_MERGE_COUNT, &[]) == 1)
            .await;
        let after = range.layers.load_full();
        assert_eq!(after.runs.len(), 3);
        assert!(Arc::ptr_eq(&after.runs[0], &four.runs[0]));
        let merged = &after.runs[1];
        assert_eq!((merged.lo, merged.hi), (three.runs[1].lo, three.runs[0].hi));
        assert_eq!(
            merged.file.rows(),
            three.runs[0].file.rows() + three.runs[1].file.rows()
        );
        assert!(Arc::ptr_eq(&after.runs[2], &three.runs[2]));
        assert_eq!(after.covered_to, four.covered_to);
        assert_eq!(t.metric(MINOR_MERGE_RUNS_COUNT, &[("runs", "2")]), 1);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        t.assert_reads(0, 1000, Some(&snapshot)).await;
        t.fail("range-cache-minor-merge", "off");
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        t.assert_reads(0, 1000, Some(&snapshot)).await;
        t.db.close().await.unwrap();
    }

    /// The writer installs a minor merge wherever its runs are, at the newest end and in the
    /// middle, and rejects runs that are not adjacent or not newest first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_install_minor_merges_anywhere_in_the_runs() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let cache = t.cache().clone();
        let snapshot = t.db.snapshot().unwrap();
        // The task's merge takes the two oldest runs and waits, while flushes reach four runs.
        t.fail("range-cache-minor-merge", "pause");
        for _ in 0..4 {
            seal_and_flush(&t, 0, 100).await;
        }
        let four = range.layers.load_full();
        assert_eq!(four.runs.len(), 4);
        let merged = |inputs: &[Arc<Run>], lo: u64, hi: u64| {
            Arc::new(cache.write_run(&[], inputs, lo, hi).unwrap())
        };
        let (r0, r1, r2) = (&four.runs[0], &four.runs[1], &four.runs[2]);
        // Not adjacent, and not newest first.
        let gap = vec![r0.clone(), r2.clone()];
        let out = merged(&gap, r2.lo, r0.hi);
        assert!(!cache.install_minor(&range, gap, out).await);
        let out = merged(&[r0.clone(), r1.clone()], r1.lo, r0.hi);
        assert!(
            !cache
                .install_minor(&range, vec![r1.clone(), r0.clone()], out)
                .await
        );
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 2);
        // At the newest end.
        let newest = vec![r0.clone(), r1.clone()];
        let out = merged(&newest, r1.lo, r0.hi);
        assert!(cache.install_minor(&range, newest, out.clone()).await);
        let three = range.layers.load_full();
        assert_eq!(three.runs.len(), 3);
        assert!(Arc::ptr_eq(&three.runs[0], &out));
        assert!(Arc::ptr_eq(&three.runs[1], &four.runs[2]));
        assert!(Arc::ptr_eq(&three.runs[2], &four.runs[3]));
        assert_eq!(three.covered_to, four.covered_to);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // A flush puts a run before it; then a merge of it and the next older run is in the
        // middle. That run is one that the task's merge holds.
        seal_and_flush(&t, 0, 100).await;
        let again = range.layers.load_full();
        assert_eq!(again.runs.len(), 4);
        let middle = vec![again.runs[1].clone(), again.runs[2].clone()];
        let out = merged(&middle, again.runs[2].lo, again.runs[1].hi);
        assert!(cache.install_minor(&range, middle, out.clone()).await);
        let after = range.layers.load_full();
        assert_eq!(after.runs.len(), 3);
        assert!(Arc::ptr_eq(&after.runs[0], &again.runs[0]));
        assert!(Arc::ptr_eq(&after.runs[1], &out));
        assert!(Arc::ptr_eq(&after.runs[2], &again.runs[3]));
        assert_eq!(after.covered_to, again.covered_to);
        assert_eq!(t.metric(MINOR_MERGE_RUNS_COUNT, &[("runs", "2")]), 2);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The task's merge lost a run to the middle merge: its install is rejected.
        let rejected = t.metric(INSTALL_REJECTED_COUNT, &[]);
        t.fail("range-cache-minor-merge", "off");
        t.wait("the held merge's rejection", |t| {
            t.metric(INSTALL_REJECTED_COUNT, &[]) == rejected + 1
        })
        .await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        t.assert_reads(0, 100, Some(&snapshot)).await;
        drop((four, three, again, after));
        t.db.close().await.unwrap();
    }

    /// `max_runs` raises the most runs of a range: with the oldest two held by a merge, flushes
    /// reach eight runs, and the next one merges the two newest.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_hold_up_to_max_runs() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.max_runs = 8;
        })
        .await;
        load(&t, 0, 100).await;
        let snapshot = t.db.snapshot().unwrap();
        t.fail("range-cache-minor-merge", "pause");
        for runs in 1..=8 {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(t.info(&key(0)).runs, runs);
        }
        t.assert_reads(0, 100, Some(&snapshot)).await;
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.info(&key(0)).runs, 7);
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        assert_eq!(t.metric(FLUSH_MERGE_RUN_COUNT, &[]), 2);
        t.fail("range-cache-minor-merge", "off");
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert_eq!(t.info(&key(0)).runs, 1);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.db.close().await.unwrap();
    }

    /// Views of the delta view tests: the range's view lists exactly its runs, every version once.
    fn assert_view(range: &CachedRange) -> usize {
        let layers = range.layers.load_full();
        match &layers.view {
            None => assert!(layers.runs.is_empty(), "runs without a view"),
            Some(view) => view.check().unwrap(),
        }
        layers.check(MAX_VIEW_RUNS_RANGE.end().to_owned());
        layers.runs.len()
    }

    fn view_counter(t: &TestDb, name: &str, job: &str) -> i64 {
        t.metric(name, &[("job", job)])
    }

    /// With the delta view, runs never merge with each other: a range holds many runs, up to
    /// `max_runs`, through one view, and reads stay exact. At `max_runs` flushes wait for the
    /// major merge, which absorbs every run it found and leaves the runs flushed meanwhile.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_hold_many_runs_through_the_delta_view() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.delta_view = true;
            o.tuning.max_runs = 12;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        // The major merge, due at 6 runs, waits.
        t.fail("range-cache-rebuild", "pause");
        for runs in 1..=12 {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(assert_view(&range), runs);
            if runs % 4 == 0 {
                t.assert_reads(0, 100, Some(&snapshot)).await;
            }
        }
        assert_eq!(t.metric(MINOR_MERGE_COUNT, &[]), 0);
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 0);
        assert_eq!(view_counter(&t, VIEW_BUILD_COUNT, "flush"), 12);
        // A full range does not flush: its sealed buffer waits.
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(t.info(&key(0)).runs, 12);
        assert_eq!(t.info(&key(0)).sealed_buffers, 1);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The major merge took the 6 runs it found; the 6 later ones stay, and the waiting flush
        // goes ahead.
        t.fail("range-cache-rebuild", "off");
        t.wait("the major merge and the flush", |t| {
            t.metric(REBUILD_COUNT, &[]) >= 1 && t.info(&key(0)).sealed_buffers == 0
        })
        .await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        let runs = assert_view(&range);
        assert!(runs < 12, "{runs} runs after the major merge");
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 0);
        assert!(view_counter(&t, VIEW_BUILD_COUNT, "major") >= 1);
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

    /// A flush and a major merge of one range race: each changes the view, so the one that
    /// installs second is rejected, builds its view again on the new one, and installs. No run is
    /// lost and no file is written twice.
    #[tokio::test(flavor = "multi_thread", worker_threads = 6)]
    async fn should_rebuild_the_view_of_a_job_that_lost_a_race() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.delta_view = true;
            o.tuning.max_runs = 16;
            // A major merge is due at the first run.
            o.tuning.min_rebuild_delta_bytes = 0;
            o.tuning.major_merge_factor = Some(1e-6);
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
        assert_eq!(view_counter(&t, VIEW_RETRY_COUNT, "major"), 1);
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
        assert_eq!(view_counter(&t, VIEW_RETRY_COUNT, "flush"), 1);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 2);
        assert_eq!(assert_view(&range), 1);
        t.assert_reads(0, 100, None).await;
        t.fail("range-cache-rebuild", "off");
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert_view(&range);
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    /// Run settings out of their bounds fail the build.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_reject_run_settings_out_of_bounds() {
        type Tune = fn(&mut RangeCacheTuning);
        let cases: [(&str, Tune); 10] = [
            ("max_runs 3", |t| t.max_runs = 3),
            ("max_runs 9", |t| t.max_runs = 9),
            ("max_runs 65 with the view", |t| {
                t.delta_view = true;
                t.max_runs = 65;
            }),
            ("max_runs 3 with the view", |t| {
                t.delta_view = true;
                t.max_runs = 3;
            }),
            ("ratio 0.5", |t| t.minor_merge_ratio = 0.5),
            ("ratio NaN", |t| t.minor_merge_ratio = f64::NAN),
            ("factor 0", |t| t.major_merge_factor = Some(0.0)),
            ("factor -1", |t| t.major_merge_factor = Some(-1.0)),
            ("factor infinite", |t| {
                t.major_merge_factor = Some(f64::INFINITY)
            }),
            ("factor NaN", |t| t.major_merge_factor = Some(f64::NAN)),
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
        let bounds: [Tune; 4] = [
            |t| t.max_runs = 4,
            |t| {
                t.max_runs = 8;
                t.minor_merge_ratio = 1.0;
            },
            |t| t.major_merge_factor = Some(2.0),
            |t| {
                t.delta_view = true;
                t.max_runs = 64;
            },
        ];
        for tune in bounds {
            let (db, _dir) = build(tune).await;
            db.unwrap().close().await.unwrap();
        }
    }
}
