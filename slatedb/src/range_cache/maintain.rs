//! The maintenance task of the range cache (design 01 section 7): it flushes sealed buffers to
//! runs, merges runs into new bases, and seals large buffers early when the buffers' memory is
//! high. It is a handler on the database's task executor, like the promoter.
//!
//! - **Flush** (7.2). For a range with a sealed buffer and fewer than [`MAX_RUNS`] runs, a job
//!   writes the oldest sealed buffer to a run file: every version, key ascending then sequence
//!   number descending, tombstones and merge operands kept, with a filter of its keys. The job
//!   sends `InstallRun` to the writer, which puts the run first, raises `covered_to` and removes
//!   the buffer in one store, or rejects it if the state moved on; a rejected run's file is
//!   deleted when it drops. While a range holds [`MAX_RUNS`] runs, its flushes wait: the buffers
//!   stay sealed and counted.
//! - **Major merge** (7.5, a stand-in until minor merges exist). When a range's run files reach
//!   `min_rebuild_delta_bytes` and a quarter of its base, or it holds `major_merge_runs` runs, a
//!   job merges the base with all its runs into a new base at the newest run's `hi`, and sends
//!   `InstallMajor`. Flushes go on meanwhile; their runs stay. An automatic range whose reads do
//!   not pay for the merge leaves the cache instead.
//! - **Pressure** (7.1, 9). Above half of `max_recent_bytes`, the task asks the writer to seal the
//!   largest active buffers that hold at least `min_seal_bytes`, so that flushes can free them.
//!
//! Each range has at most one flush and one merge in flight. Flushes and merges have their own
//! limits on concurrent jobs, so a long merge never holds back the flushes that free memory. The
//! writer wakes the task when it seals a buffer, jobs wake it when they end, and a ticker wakes
//! it every [`TICK`] in case anything was missed.

use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use fail_parallel::fail_point;
use futures::stream::BoxStream;
use log::{debug, warn};
use tokio::task::JoinHandle;

use super::delta::Delta;
use super::file::{RangeFile, RangeFileIterator, RangeFileWriter};
use super::stats::RemoveReason;
use super::writer::Target;
use super::{CachedRange, Layers, RangeCache, RangeCacheTuning, Run, MAX_RUNS};
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

/// How often the task looks for work without being woken.
const TICK: Duration = Duration::from_millis(20);
/// The most flushes in flight.
const MAX_FLUSHES: usize = 2;
/// The most major merges in flight.
const MAX_MERGES: usize = 2;
/// The most early seals that one pass asks for.
const PRESSURE_SEALS: usize = 4;

/// A reason for the task to look for work.
#[derive(Debug)]
pub(crate) enum MaintenanceMessage {
    /// A seal, an install or the end of a job.
    Work,
    /// The ticker.
    Tick,
}

/// The queue to the maintenance task.
pub(super) struct MaintenanceQueue {
    tx: async_channel::Sender<MaintenanceMessage>,
    rx: async_channel::Receiver<MaintenanceMessage>,
    /// True while a wake-up message is on its way to the task.
    pending: AtomicBool,
}

impl MaintenanceQueue {
    pub(super) fn new() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self {
            tx,
            rx,
            pending: AtomicBool::new(false),
        }
    }

    /// Wakes the task. One message in flight is enough: the task clears the flag before it
    /// looks for work.
    pub(super) fn wake(&self) {
        if !self.pending.swap(true, Ordering::AcqRel) {
            let _ = self.tx.try_send(MaintenanceMessage::Work);
        }
    }
}

/// The jobs of one range. See [`CachedRange::jobs`].
#[derive(Default)]
pub(super) struct RangeJobs {
    flushing: AtomicBool,
    merging: AtomicBool,
}

impl RangeJobs {
    /// True while a major merge of the range runs.
    pub(super) fn merging(&self) -> bool {
        self.merging.load(Ordering::Acquire)
    }
}

/// Returns true if `layers` call for a major merge: a base, and run files of at least
/// `min_rebuild_delta_bytes` and a quarter of the base, or `major_merge_runs` runs.
pub(super) fn merge_due(layers: &Layers, tuning: &RangeCacheTuning) -> bool {
    let Some(base) = &layers.base else {
        return false;
    };
    if layers.runs.is_empty() {
        return false;
    }
    if layers.runs.len() >= tuning.major_merge_runs {
        return true;
    }
    let run_bytes: u64 = layers.runs.iter().map(|r| r.file_bytes()).sum();
    run_bytes >= (tuning.min_rebuild_delta_bytes as u64).max(base.file_len() / 4)
}

impl RangeCache {
    /// The receiver of the maintenance task's queue, for its registration.
    pub(crate) fn maintenance_receiver(&self) -> async_channel::Receiver<MaintenanceMessage> {
        self.maintenance.rx.clone()
    }

    /// Writes the oldest sealed buffer of `range` to a run and installs it. Does nothing if the
    /// range has no sealed buffer, or holds [`MAX_RUNS`] runs.
    async fn flush(self: &Arc<Self>, range: &Arc<CachedRange>) -> Result<(), SlateDBError> {
        let layers = range.layers.load_full();
        let Some(buffer) = layers.sealed.last().cloned() else {
            return Ok(());
        };
        if layers.runs.len() >= MAX_RUNS {
            return Ok(());
        }
        let lo = layers.covered_to;
        let hi = buffer.sealed_at().load(Ordering::Acquire);
        drop(layers);
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-flush");
        let run = Arc::new(self.write_run(&buffer, lo, hi)?);
        // Checkers keep the sealed buffer pinned by this job across many batches.
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-install-run");
        if self.install_run(range, run, &buffer).await {
            self.evict(None).await;
        }
        Ok(())
    }

    /// Writes the versions of `buffer` at or below `hi` to a run of `(lo, hi]`.
    pub(super) fn write_run(&self, buffer: &Delta, lo: u64, hi: u64) -> Result<Run, SlateDBError> {
        let mut writer = RangeFileWriter::create_run(
            &self.options.root_folder,
            &self.new_file_name(),
            self.options.tuning.page_size,
        )?;
        let versions = buffer.buffer();
        let mut cursor = versions.seek(Bound::Unbounded);
        while let Some(v) = versions.version(&cursor) {
            debug_assert!(v.seq > lo, "a buffered version at {} below {lo}", v.seq);
            // A removed range's buffer can hold versions above its seal, which no read uses.
            if v.seq <= hi {
                writer.push_version(v.key, v.seq, v.kind, v.create_ts, v.value)?;
            }
            versions.advance(&mut cursor);
        }
        let path = writer.finish()?;
        let file = RangeFile::open(path, self.options.page_source, self.disk_bytes.clone())?;
        Ok(Run::new(file, lo, hi)?)
    }

    /// Merges the base of `range` with all its runs into a new base at the newest run's `hi`,
    /// and installs it.
    async fn major_merge(self: &Arc<Self>, range: &Arc<CachedRange>) -> Result<(), SlateDBError> {
        let layers = range.layers.load_full();
        let Some(base) = layers.base.clone() else {
            return Ok(());
        };
        let absorbed = layers.runs.clone();
        drop(layers);
        let Some(base_seq) = absorbed.first().map(|r| r.hi) else {
            return Ok(());
        };
        range
            .reads_at_build
            .store(range.heat.reads(), Ordering::Relaxed);
        fail_point!(Arc::clone(&self.fp_registry), "range-cache-rebuild");
        let new_base = Arc::new(self.merge_files(&absorbed, &base, base_seq).await?);
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
            self.disk_bytes.clone(),
        )?)
    }
}

/// The maintenance task. See the module documentation.
pub(crate) struct Maintainer {
    db: Arc<DbInner>,
    cache: Arc<RangeCache>,
    /// Jobs in flight, for the cleanup.
    jobs: Vec<JoinHandle<Result<(), SlateDBError>>>,
}

/// A job of the maintenance task.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Job {
    Flush,
    Merge,
}

impl Maintainer {
    pub(crate) fn new(db: Arc<DbInner>, cache: Arc<RangeCache>) -> Self {
        // The task sends requests to the writer, which must be able to wake it.
        cache.bind_writer(&db.write_notifier);
        Self {
            db,
            cache,
            jobs: Vec::new(),
        }
    }

    /// Looks for work and starts jobs: flushes first, then merges, then early seals.
    fn pass(&mut self) {
        let cache = self.cache.clone();
        // Clear the flag first: a wake-up after this point sends its own message, and one before
        // it is visible to the scan below (the swap reads the waker's flag).
        cache.maintenance.pending.swap(false, Ordering::AcqRel);
        if self.db.check_closed().is_err() || cache.writer.is_closed() {
            return;
        }
        self.jobs.retain(|job| !job.is_finished());
        let tuning = &cache.options.tuning;
        let index = cache.index.load_full();
        let running = |job: Job| {
            index
                .values()
                .filter(|r| match job {
                    Job::Flush => r.jobs.flushing.load(Ordering::Acquire),
                    Job::Merge => r.jobs.merging.load(Ordering::Acquire),
                })
                .count()
        };
        let mut flushes = running(Job::Flush);
        for range in index.values() {
            let layers = range.layers.load();
            if layers.sealed.is_empty() || range.jobs.flushing.load(Ordering::Acquire) {
                continue;
            }
            if layers.runs.len() >= MAX_RUNS {
                cache.stats.flush_waits.increment(1);
                continue;
            }
            if flushes < MAX_FLUSHES {
                flushes += 1;
                self.start(range, Job::Flush);
            }
        }
        let mut merges = running(Job::Merge);
        for range in index.values() {
            if merges >= MAX_MERGES {
                break;
            }
            let layers = range.layers.load();
            if range.jobs.merging() || !merge_due(&layers, tuning) {
                continue;
            }
            let base_bytes = layers.base.as_ref().map_or(0, |b| b.file_len());
            drop(layers);
            if !cache.rebuild_pays(range, base_bytes) {
                // The range's buffers and runs hold its recent writes, so it stays exact until
                // the writer removes it.
                cache.request_removal(Target::Range(range.clone()), RemoveReason::NotWorthIt);
                continue;
            }
            merges += 1;
            self.start(range, Job::Merge);
        }
        if cache.delta_memory.bytes() > tuning.pressure() {
            let min = tuning.min_seal_bytes as u64;
            let mut largest: Vec<(u64, &Arc<CachedRange>, Arc<Delta>)> = index
                .values()
                .filter_map(|r| {
                    let layers = r.layers.load();
                    let bytes = layers.active.allocated_bytes();
                    // A range at the most runs cannot flush until a merge: sealing it frees
                    // nothing soon.
                    (bytes >= min && !layers.active.is_empty() && layers.runs.len() < MAX_RUNS)
                        .then(|| (bytes, r, layers.active.clone()))
                })
                .collect();
            largest.sort_by(|a, b| b.0.cmp(&a.0));
            for (_, range, buffer) in largest.into_iter().take(PRESSURE_SEALS) {
                cache.request_seal(range, &buffer);
            }
        }
    }

    /// Starts `job` on `range` as its own task.
    fn start(&mut self, range: &Arc<CachedRange>, job: Job) {
        let flag = match job {
            Job::Flush => &range.jobs.flushing,
            Job::Merge => &range.jobs.merging,
        };
        flag.store(true, Ordering::Release);
        let (cache, range) = (self.cache.clone(), range.clone());
        let (done_cache, done_range) = (cache.clone(), range.clone());
        let name = match job {
            Job::Flush => "range_cache_flush",
            Job::Merge => "range_cache_major_merge",
        };
        let handle = spawn_bg_task(
            name.to_string(),
            &self.cache.handle,
            move |result: &Result<(), SlateDBError>| {
                // Runs after the job, also if it failed or panicked.
                match job {
                    Job::Flush => &done_range.jobs.flushing,
                    Job::Merge => &done_range.jobs.merging,
                }
                .store(false, Ordering::Release);
                if let Err(e) = result {
                    warn!("range cache {name} failed [error={:?}]", e);
                    done_cache.request_removal(Target::Range(done_range), RemoveReason::Error);
                }
                done_cache.maintenance.wake();
            },
            async move {
                match job {
                    Job::Flush => cache.flush(&range).await,
                    Job::Merge => cache.major_merge(&range).await,
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

    async fn handle(&mut self, _message: MaintenanceMessage) -> Result<(), SlateDBError> {
        self.pass();
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
    use std::ops::Bound::{Excluded, Included};
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
        Unfit, FLUSH_BYTES, FLUSH_COUNT, FLUSH_WAIT_COUNT, INSTALL_REJECTED_COUNT,
        LOAD_REJECTED_COUNT, REBUILD_COUNT, REMOVE_COUNT, SEAL_COUNT,
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
        let dir = tempfile::tempdir().unwrap();
        let mut options = RangeCacheOptions::new(dir.path());
        options.tuning.stats_fold_events = 1;
        // No major merge unless a test asks for one.
        options.tuning.min_rebuild_delta_bytes = usize::MAX;
        options.tuning.major_merge_runs = usize::MAX;
        tune(&mut options);
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let fp_registry = Arc::new(FailPointRegistry::new());
        let db = Db::builder("range_cache_maintain", Arc::new(InMemory::new()))
            .with_range_cache(options)
            .with_merge_operator(Arc::new(Concat))
            .with_metrics_recorder(recorder.clone())
            .with_fp_registry(fp_registry.clone())
            .build()
            .await
            .unwrap();
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
        // The maintenance task's flush takes the oldest buffer and waits here.
        t.fail("range-cache-flush", "pause");
        t.write_until(0, 100, |t| t.seals() == 2).await;
        let layers = range.layers.load_full();
        let (newer, oldest) = (layers.sealed[0].clone(), layers.sealed[1].clone());
        let seal = |b: &Arc<Delta>| b.sealed_at().load(Ordering::SeqCst);
        let run =
            |buffer: &Delta, lo: u64, hi: u64| Arc::new(cache.write_run(buffer, lo, hi).unwrap());
        let files = t.cache_files();
        // Not the oldest sealed buffer.
        assert!(
            !cache
                .install_run(&range, run(&newer, newer.lo(), seal(&newer)), &newer)
                .await
        );
        // Not from `covered_to`, or not to the buffer's seal.
        let lo = layers.covered_to;
        assert!(
            !cache
                .install_run(&range, run(&oldest, lo - 1, seal(&oldest)), &oldest)
                .await
        );
        assert!(
            !cache
                .install_run(&range, run(&oldest, lo, seal(&oldest) - 1), &oldest)
                .await
        );
        // A rejected run's file is deleted when it drops.
        assert_eq!(t.cache_files(), files);
        assert_eq!(t.metric(INSTALL_REJECTED_COUNT, &[]), 3);
        // The right one.
        let first = run(&oldest, lo, seal(&oldest));
        assert!(cache.install_run(&range, first.clone(), &oldest).await);
        let after = range.layers.load_full();
        assert_eq!(after.covered_to, seal(&oldest));
        assert_eq!(after.sealed.len(), 1);
        assert!(Arc::ptr_eq(&after.sealed[0], &newer));
        assert!(Arc::ptr_eq(&after.runs[0], &first));
        t.assert_reads(0, 100, None).await;

        // A major merge whose base, runs or cut moved on.
        let base = after.base.clone().unwrap();
        let merged = |base_seq| {
            let cache = cache.clone();
            let (runs, base) = (vec![first.clone()], base.clone());
            async move { Arc::new(cache.merge_files(&runs, &base, base_seq).await.unwrap()) }
        };
        let other = merged(first.hi).await;
        assert!(
            !cache
                .install_major(
                    &range,
                    &other,
                    vec![first.clone()],
                    merged(first.hi).await,
                    first.hi
                )
                .await
        );
        assert!(
            !cache
                .install_major(&range, &base, vec![], merged(first.hi).await, first.hi)
                .await
        );
        assert!(
            !cache
                .install_major(
                    &range,
                    &base,
                    vec![first.clone()],
                    merged(first.hi).await,
                    first.hi + 1
                )
                .await
        );
        assert!(
            cache
                .install_major(
                    &range,
                    &base,
                    vec![first.clone()],
                    merged(first.hi).await,
                    first.hi
                )
                .await
        );
        let after = range.layers.load_full();
        assert_eq!((after.base_seq, after.runs.len()), (first.hi, 0));
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 1);
        t.assert_reads(0, 100, None).await;

        // The held flush wrote a run of the oldest buffer, which is gone: it is rejected.
        let rejected = t.metric(INSTALL_REJECTED_COUNT, &[]);
        assert_eq!(rejected, 6);
        t.fail("range-cache-flush", "off");
        t.wait("the held flush's rejection", |t| {
            t.metric(INSTALL_REJECTED_COUNT, &[]) > rejected
        })
        .await;
        // Removed, then registered again: the old identity's runs are rejected.
        drop(after);
        t.wait("the flush of the newer buffer", |t| {
            t.info(&key(0)).sealed_buffers == 0
        })
        .await;
        t.db.uncache_range(span(0, 100)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        let stale = run(&newer, newer.lo(), seal(&newer));
        assert!(!cache.install_run(&range, stale.clone(), &newer).await);
        t.db.cache_range(span(0, 100)).await.unwrap();
        assert!(!Arc::ptr_eq(&t.range(&key(0)), &range));
        assert!(!cache.install_run(&range, stale, &newer).await);
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_keep_runs_flushed_during_a_major_merge() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.major_merge_runs = 2;
        })
        .await;
        load(&t, 0, 100).await;
        let range = t.range(&key(0));
        let snapshot = t.db.snapshot().unwrap();
        // The first major merge waits after it took its inputs.
        t.fail("range-cache-rebuild", "1*sleep(3000)");
        t.write_until(0, 100, |t| {
            t.info(&key(0)).runs + t.info(&key(0)).sealed_buffers >= 2
        })
        .await;
        t.wait("two runs", |t| t.info(&key(0)).runs == 2).await;
        let absorbed = range.layers.load_full();
        let cut = absorbed.runs[0].hi;
        t.wait("the merge", |_| range.jobs.merging()).await;
        // A third run arrives during the merge.
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        t.wait("the third run", |t| t.info(&key(0)).runs == 3).await;
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 0);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.wait("the install", |t| t.metric(REBUILD_COUNT, &[]) == 1)
            .await;
        let layers = range.layers.load_full();
        assert_eq!(layers.base_seq, cut);
        assert_eq!(layers.runs.len(), 1);
        assert_eq!(layers.runs[0].lo, cut);
        assert!(!Arc::ptr_eq(
            layers.base.as_ref().unwrap(),
            absorbed.base.as_ref().unwrap()
        ));
        // The snapshot is older than the new base: it uses the normal path, and stays exact.
        assert!(snapshot.seq() < cut);
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // One run is below the count, so no other merge follows.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 1);
        assert!(!t.info(&key(0)).rebuilding);
        drop((absorbed, layers));
        assert_eq!(
            t.cache_files(),
            2,
            "the old base and the absorbed runs are deleted"
        );
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_wait_to_flush_while_a_range_holds_four_runs() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        load(&t, 0, 100).await;
        let snapshot = t.db.snapshot().unwrap();
        t.write_until(0, 100, |t| t.info(&key(0)).runs == MAX_RUNS)
            .await;
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        t.wait("a flush that waits", |t| {
            t.metric(FLUSH_WAIT_COUNT, &[]) > 0
        })
        .await;
        let info = t.info(&key(0));
        assert_eq!((info.runs, info.sealed_buffers), (MAX_RUNS, 1));
        assert!(info.rebuilding);
        // Reads go through the active and the sealed buffer, four runs and the base.
        t.assert_reads(0, 100, Some(&snapshot)).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(t.info(&key(0)).runs, MAX_RUNS);
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_merge_at_four_runs_by_default() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.major_merge_runs = MAX_RUNS;
        })
        .await;
        load(&t, 0, 100).await;
        t.write_until(0, 100, |t| t.metric(REBUILD_COUNT, &[]) >= 2)
            .await;
        t.wait("quiet", |t| !t.info(&key(0)).rebuilding).await;
        assert!(t.info(&key(0)).runs < MAX_RUNS);
        t.assert_reads(0, 100, None).await;
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
}
