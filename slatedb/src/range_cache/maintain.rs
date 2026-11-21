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
//! - **Flush-merge** (7.3). When the range holds [`MAX_RUNS`] runs, the flush also merges its
//!   [`FLUSH_MERGE_RUNS`] newest runs into the run it writes, and the install replaces them. A
//!   flush never waits for a merge.
//! - **Minor merge** (7.4). A range with [`MERGE_RUNS`] runs or more has its oldest two merged
//!   into one run of their union interval, every version kept, installed by `InstallMinor`.
//! - **Major merge** (7.5). When the run files of a range reach `f` times its base, with
//!   `f = sqrt(2r / B)` for its average flush `r` and its base `B` (7.9), between [`F_MIN`] and
//!   [`F_MAX`], and at least `min_rebuild_delta_bytes`, a job merges the base with its oldest
//!   runs, at most [`MERGE_RUNS`], into a new base at the newest absorbed run's `hi`, and sends
//!   `InstallMajor`. Flushes go on meanwhile; their runs stay. An automatic range whose reads do
//!   not pay for the merge leaves the cache instead.
//! - **Pressure** (7.1, 9). Above half of `max_recent_bytes`, the task asks the writer to seal the
//!   largest active buffers that hold at least `min_seal_bytes`, so that flushes can free them.
//!
//! A merge only ever takes a range's oldest runs, at most two, and a flush-merge its two newest
//! runs when the range holds four: so the two never take the same run, and a flush-merge never
//! waits for a merge. Each range has at most one flush and one merge in flight. Flushes and
//! merges have their own limits on concurrent jobs, so a long merge never holds back the
//! flushes that free memory.
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
use super::writer::Target;
use super::{CachedRange, Layers, RangeCache, RangeCacheTuning, RangeIndex, Run, MAX_RUNS};
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
/// The runs at which a range has its oldest runs merged (design 01's `D_minor`), and the most
/// runs that one merge takes.
pub(super) const MERGE_RUNS: usize = 2;
/// The newest runs that a flush merges into its run when the range holds [`MAX_RUNS`] (design
/// 01's `j`). Two brings the range back to three runs, so the next flush is a plain one; the
/// newest runs are the youngest and so the smallest, as older ones grew by merges; and merges
/// take only the two oldest runs, so these two are never in one.
pub(super) const FLUSH_MERGE_RUNS: usize = 2;
/// The bounds of the major-merge factor `f`. At most 1: the runs of a range never grow past its
/// base, whose rewrite is then cheap anyway. At least 0.05: a base of more than 800 flushes
/// still merges after a twentieth of its size.
pub(super) const F_MIN: f64 = 0.05;
pub(super) const F_MAX: f64 = 1.0;
const _: () = assert!(MERGE_RUNS + FLUSH_MERGE_RUNS <= MAX_RUNS);

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
}

impl RangeJobs {
    /// True while a merge of the range, minor or major, runs.
    pub(super) fn merging(&self) -> bool {
        self.merging.load(Ordering::Acquire)
    }

    fn flushing(&self) -> bool {
        self.flushing.load(Ordering::Acquire)
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

/// The merge that a range's layers call for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Merge {
    Minor,
    Major,
}

/// The run bytes at which a major merge of a range with this base starts: `f` times the base,
/// `f = sqrt(2r / B)` between [`F_MIN`] and [`F_MAX`] (design 01 section 7.9), and at least
/// `min_rebuild_delta_bytes`.
pub(super) fn major_threshold(
    range: &CachedRange,
    base_bytes: u64,
    tuning: &RangeCacheTuning,
) -> u64 {
    let r = range.jobs.average_flush(tuning.range_seal_bytes as u64) as f64;
    let b = base_bytes.max(1) as f64;
    let f = (2.0 * r / b).sqrt().clamp(F_MIN, F_MAX);
    ((f * base_bytes as f64) as u64).max(tuning.min_rebuild_delta_bytes as u64)
}

/// The merge that `layers`, the layers of `range`, call for: a major merge when the range has a
/// base and its run bytes reach [`major_threshold`], else a minor merge when it holds
/// [`MERGE_RUNS`] runs or more.
pub(super) fn merge_due(
    range: &CachedRange,
    layers: &Layers,
    tuning: &RangeCacheTuning,
) -> Option<Merge> {
    if layers.runs.is_empty() {
        return None;
    }
    if let Some(base) = &layers.base {
        let run_bytes: u64 = layers.runs.iter().map(|r| r.file_bytes()).sum();
        if run_bytes >= major_threshold(range, base.file_len(), tuning) {
            return Some(Merge::Major);
        }
    }
    (layers.runs.len() >= MERGE_RUNS).then_some(Merge::Minor)
}

impl RangeCache {
    /// The receiver of the maintenance task's queue, for its registration.
    pub(crate) fn maintenance_receiver(&self) -> async_channel::Receiver<MaintenanceMessage> {
        self.maintenance.rx.clone()
    }

    /// Writes every sealed buffer of `range` to one run and installs it. When the range holds
    /// [`MAX_RUNS`] runs, the run also takes the versions of its [`FLUSH_MERGE_RUNS`] newest runs,
    /// and replaces them. Does nothing if the range has no sealed buffer.
    async fn flush(self: &Arc<Self>, range: &Arc<CachedRange>) -> Result<(), SlateDBError> {
        let layers = range.layers.load_full();
        let buffers = layers.sealed.clone();
        let Some(newest) = buffers.first() else {
            return Ok(());
        };
        let hi = newest.sealed_at().load(Ordering::Acquire);
        let merged = if layers.runs.len() >= MAX_RUNS {
            FLUSH_MERGE_RUNS
        } else {
            0
        };
        let replaces = layers.runs[..merged].to_vec();
        let lo = replaces.last().map_or(layers.covered_to, |r| r.lo);
        drop(layers);
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-flush");
        let run = {
            let (cache, buffers, replaces) = (self.clone(), buffers.clone(), replaces.clone());
            Arc::new(off_workers(move || cache.write_run(&buffers, &replaces, lo, hi)).await?)
        };
        let replaced: u64 = replaces.iter().map(|r| r.file_bytes()).sum();
        let added = run.file_bytes().saturating_sub(replaced);
        // Checkers keep the sealed buffers pinned by this job across many batches.
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-install-run");
        if self.install_run(range, run, buffers, replaces).await {
            range.jobs.record_flush(added);
            self.evict(None).await;
        }
        Ok(())
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

    /// Merges the oldest [`MERGE_RUNS`] runs of `range` into one run, every version kept, and
    /// installs it.
    async fn minor_merge(self: &Arc<Self>, range: &Arc<CachedRange>) -> Result<(), SlateDBError> {
        let layers = range.layers.load_full();
        let Some(first) = layers.runs.len().checked_sub(MERGE_RUNS) else {
            return Ok(());
        };
        let inputs = layers.runs[first..].to_vec();
        drop(layers);
        let (lo, hi) = (inputs[inputs.len() - 1].lo, inputs[0].hi);
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-minor-merge");
        let output = {
            let (cache, inputs) = (self.clone(), inputs.clone());
            Arc::new(off_workers(move || cache.write_run(&[], &inputs, lo, hi)).await?)
        };
        if self.install_minor(range, inputs, output).await {
            self.evict(None).await;
        }
        Ok(())
    }

    /// Merges the base of `range` with its oldest runs, at most [`MERGE_RUNS`], into a new base
    /// at the newest absorbed run's `hi`, and installs it.
    async fn major_merge(self: &Arc<Self>, range: &Arc<CachedRange>) -> Result<(), SlateDBError> {
        let layers = range.layers.load_full();
        let Some(base) = layers.base.clone() else {
            return Ok(());
        };
        let first = layers.runs.len().saturating_sub(MERGE_RUNS);
        let absorbed = layers.runs[first..].to_vec();
        drop(layers);
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
        if self
            .install_major(range, &base, absorbed, new_base, base_seq)
            .await
        {
            self.evict(None).await;
        }
        Ok(())
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

/// A job of the maintenance task.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Job {
    Flush,
    Merge(Merge),
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
        let queue = match merge_due(range, &layers, &self.cache.options.tuning) {
            Some(Merge::Minor) => &mut self.minor_queue,
            Some(Merge::Major) => &mut self.major_queue,
            None => return,
        };
        jobs.merge_queued.store(true, Ordering::Relaxed);
        queue.push_back(Arc::downgrade(range));
    }

    /// Starts queued jobs while slots are free: flushes, then minor merges, then major merges.
    /// A job's range is checked again when it starts, as it may have changed while it waited.
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
            if !range.jobs.flushing()
                && !range.layers.load().sealed.is_empty()
                && cache.is_indexed(&range)
            {
                self.start(&range, Job::Flush);
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
            let Some(merge) = merge_due(&range, &layers, &cache.options.tuning) else {
                continue;
            };
            if merge == Merge::Major {
                let base_bytes = layers.base.as_ref().map_or(0, |b| b.file_len());
                if !cache.rebuild_pays(&range, base_bytes) {
                    // The range's buffers and runs hold its recent writes, so it stays exact
                    // until the writer removes it.
                    cache.request_removal(Target::Range(range.clone()), RemoveReason::NotWorthIt);
                    continue;
                }
            }
            drop(layers);
            self.start(&range, Job::Merge(merge));
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

    /// Starts `job` on `range` as its own task.
    fn start(&mut self, range: &Arc<CachedRange>, job: Job) {
        let (flag, running) = match job {
            Job::Flush => (&range.jobs.flushing, &self.cache.maintenance.flushes),
            Job::Merge(_) => (&range.jobs.merging, &self.cache.maintenance.merges),
        };
        flag.store(true, Ordering::Release);
        running.fetch_add(1, Ordering::AcqRel);
        let (cache, range) = (self.cache.clone(), range.clone());
        let (done_cache, done_range) = (cache.clone(), range.clone());
        let name = match job {
            Job::Flush => "range_cache_flush",
            Job::Merge(Merge::Minor) => "range_cache_minor_merge",
            Job::Merge(Merge::Major) => "range_cache_major_merge",
        };
        let handle = spawn_bg_task(
            name.to_string(),
            &self.cache.handle,
            move |result: &Result<(), SlateDBError>| {
                // Runs after the job, also if it failed or panicked.
                let (flag, running) = match job {
                    Job::Flush => (&done_range.jobs.flushing, &done_cache.maintenance.flushes),
                    Job::Merge(_) => (&done_range.jobs.merging, &done_cache.maintenance.merges),
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
                    Job::Flush => cache.flush(&range).await,
                    Job::Merge(Merge::Minor) => cache.minor_merge(&range).await,
                    Job::Merge(Merge::Major) => cache.major_merge(&range).await,
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
        INSTALL_REJECTED_COUNT, LOAD_REJECTED_COUNT, MINOR_MERGE_COUNT, MULTI_BUFFER_FLUSH_COUNT,
        REBUILD_COUNT, REMOVE_COUNT, SEAL_COUNT,
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
                .install_run(&range, r, vec![newer.clone()], vec![])
                .await
        );
        // Not from `covered_to`, or not to the newest buffer's seal.
        let r = run(&both, lo - 1, seal(&newer));
        assert!(!cache.install_run(&range, r, both.clone(), vec![]).await);
        let r = run(&both, lo, seal(&newer) - 1);
        assert!(!cache.install_run(&range, r, both.clone(), vec![]).await);
        // Runs to replace that the range does not hold.
        let stray = run(std::slice::from_ref(&oldest), lo, seal(&oldest));
        let r = run(&both, lo, seal(&newer));
        assert!(
            !cache
                .install_run(&range, r, both.clone(), vec![stray])
                .await
        );
        // A rejected run's file is deleted when it drops.
        assert_eq!(t.cache_files(), files);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 4);
        // Both buffers in one run: one store removes both.
        let first = run(&both, lo, seal(&newer));
        assert!(
            cache
                .install_run(&range, first.clone(), both.clone(), vec![])
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
                    short.hi
                )
                .await
        );
        assert!(
            !cache
                .install_major(&range, &base, vec![], merged(short.hi).await, short.hi)
                .await
        );
        assert!(
            !cache
                .install_major(
                    &range,
                    &base,
                    vec![first.clone()],
                    merged(short.hi).await,
                    short.hi
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
                    short.hi + 1
                )
                .await
        );
        assert!(
            cache
                .install_major(&range, &base, absorbed, merged(short.hi).await, short.hi)
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
                    .install_run(&range, stale.clone(), held.clone(), vec![])
                    .await
            );
            t.db.cache_range(span(0, 100)).await.unwrap();
            assert!(!Arc::ptr_eq(&t.range(&key(0)), &range));
            assert!(
                !cache
                    .install_run(&range, stale.clone(), held.clone(), vec![])
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
        t.fail("range-cache-minor-merge", "pause");
        for runs in 1..=MAX_RUNS {
            seal_and_flush(&t, 0, 100).await;
            assert_eq!(t.info(&key(0)).runs, runs);
        }
        let four = range.layers.load_full();
        t.assert_reads(0, 100, Some(&snapshot)).await;
        seal_and_flush(&t, 0, 100).await;
        assert_eq!(t.metric(FLUSH_MERGE_COUNT, &[]), 1);
        let three = range.layers.load_full();
        assert_eq!(three.runs.len(), MAX_RUNS - FLUSH_MERGE_RUNS + 1);
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
        let settings = Settings {
            l0_sst_size_bytes: 64 * 1024,
            compactor_options: Some(CompactorOptions {
                poll_interval: Duration::from_millis(5),
                scheduler_options: SizeTieredCompactionSchedulerOptions {
                    min_compaction_sources: 1,
                    ..Default::default()
                }
                .into(),
                worker: Some(CompactionWorkerOptions {
                    compactions_poll_interval: Duration::from_millis(5),
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
}
