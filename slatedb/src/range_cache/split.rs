//! Zero-copy range splits.
//!
//! Splits divide a cached range whose base exceeds twice [`RangeCacheTuning::range_target_bytes`]
//! into adjacent child ranges without copying file data:
//! - Children share the parent's base files (as narrowed chunks), runs, and sealed buffers; each
//!   child receives a restricted delta view over its own key span.
//! - Budgets count shared files, runs, and views once.
//! - The batch writer seals the parent's active buffer at boundary `W` and atomically replaces the
//!   parent with its children in the index.
//! - Large loads split on attach; ranges grown by major merges split in two during maintenance.
//!
//! [`RangeCacheTuning::range_target_bytes`]: super::RangeCacheTuning::range_target_bytes

use std::io;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use bytes::Bytes;

use super::delta::Delta;
use super::view::DeltaView;
use super::{ChunkList, Chunks, Layers, Run};

/// One range that a split makes: its keys, its base, its runs and its view.
pub(super) struct SplitChild {
    pub(super) start: Bytes,
    pub(super) end: Option<Bytes>,
    pub(super) chunks: Chunks,
    /// Newest first: the parent's runs less the oldest ones that every chunk of the child absorbed.
    pub(super) runs: Vec<Arc<Run>>,
    /// The parent's view within the child's keys, without the runs that the child does not keep.
    pub(super) view: Option<Arc<DeltaView>>,
}

/// A split that a task prepared, and the base, runs and view of the range that it prepared it on.
/// The writer applies it only if the range still holds them.
pub(super) struct SplitPlan {
    /// The base that the split cut: the range's, or a load's for a split at attach (the range's
    /// own base is then [`Chunks::Loading`]).
    pub(super) chunks_was: Chunks,
    pub(super) runs_was: Vec<Arc<Run>>,
    pub(super) view_was: Option<Arc<DeltaView>>,
    pub(super) children: Vec<SplitChild>,
    /// The parent's read rate when the task planned the split, which the children share by the
    /// bytes of their bases. Read off the writer, which takes no lock.
    pub(super) rate: f64,
}

/// Builds child layer sets at writer boundary `w`, sealing `parent.active` at `w` first so
/// readers pinned to `parent` retry for sequence numbers above `w` and find the children.
/// Each child shares the parent's sealed buffers (including the sealed active buffer if non-empty)
/// and gets a fresh empty active buffer.
pub(super) fn children_layers(
    parent: &Layers,
    children: Vec<SplitChild>,
    w: u64,
    max_runs: usize,
    new_buffer: impl Fn(u64) -> Arc<Delta>,
) -> Vec<(Bytes, Option<Bytes>, Layers)> {
    let active = &parent.active;
    active.sealed_at().store(w, Ordering::SeqCst);
    let (sealed, lo) = if active.is_empty() {
        (parent.sealed.clone(), active.lo())
    } else {
        let mut sealed = Vec::with_capacity(parent.sealed.len() + 1);
        sealed.push(active.clone());
        sealed.extend(parent.sealed.iter().cloned());
        (sealed, w)
    };
    children
        .into_iter()
        .map(|child| {
            let layers = Layers {
                active: new_buffer(lo),
                sealed: sealed.clone(),
                runs: child.runs,
                view: child.view,
                chunks: child.chunks,
                covered_to: parent.covered_to,
                end: child.end.clone(),
            };
            layers.check(&child.start, max_runs);
            (child.start, child.end, layers)
        })
        .collect()
}

/// The data bytes of each chunk of `chunks`, a base of the range from `start` to `end`, within the
/// chunk's keys: the pages of its file that hold them (see `RangeFile::span_data`).
fn chunk_shares(chunks: &Chunks, start: &Bytes, end: Option<&Bytes>) -> Vec<u64> {
    (0..chunks.len())
        .map(|i| {
            let span = chunks.span(i, Some(start), end);
            let (from, to) = chunks[i].file.span_data(span_start(&span), span_end(&span));
            to - from
        })
        .collect()
}

/// The bytes and the estimated rows of the base `chunks` of the range from `start` to `end`, each
/// chunk counted within its keys (see `RangeFile::span_share`): exactly its files' bytes and rows
/// unless a chunk shares its file with another range.
pub(super) fn base_share(chunks: &Chunks, start: &Bytes, end: Option<&Bytes>) -> (u64, u64) {
    let mut total = (0, 0);
    for (i, chunk) in chunks.iter().enumerate() {
        let span = chunks.span(i, Some(start), end);
        let (bytes, rows) = chunk.file.span_share(span_start(&span), span_end(&span));
        total.0 += bytes;
        total.1 += rows;
    }
    total
}

/// The first key of `span`, `None` if it is unbounded below.
pub(super) fn span_start(span: &crate::bytes_range::BytesRange) -> Option<&[u8]> {
    match std::ops::RangeBounds::start_bound(span) {
        std::ops::Bound::Included(key) | std::ops::Bound::Excluded(key) => Some(key.as_ref()),
        std::ops::Bound::Unbounded => None,
    }
}

/// The key that ends `span`, `None` if it is unbounded above.
pub(super) fn span_end(span: &crate::bytes_range::BytesRange) -> Option<&[u8]> {
    match std::ops::RangeBounds::end_bound(span) {
        std::ops::Bound::Included(key) | std::ops::Bound::Excluded(key) => Some(key.as_ref()),
        std::ops::Bound::Unbounded => None,
    }
}

/// The number of pieces that a split at attach cuts a base of `bytes` into: about one per `target`
/// bytes, at least two.
pub(super) fn attach_pieces(bytes: u64, target: u64) -> usize {
    let pieces = bytes.saturating_add(target / 2) / target.max(1);
    usize::try_from(pieces).unwrap_or(usize::MAX).max(2)
}

/// Keys that cut `chunks`, the base of the range from `start` to `end`, into `pieces` parts of
/// about equal bytes, ascending and strictly inside the range: for each cut, the first key of the
/// page of a chunk's file nearest to its share of the bytes, or the start of a chunk if that is
/// nearer. Fewer, or none, if the base has too few pages. Searches of the fence indexes on the
/// heap, and a copy of each key.
pub(super) fn cut_keys(
    chunks: &Chunks,
    start: &Bytes,
    end: Option<&Bytes>,
    pieces: usize,
) -> Vec<Bytes> {
    let shares = chunk_shares(chunks, start, end);
    let total: u64 = shares.iter().sum();
    let mut cuts: Vec<Bytes> = Vec::with_capacity(pieces.saturating_sub(1));
    for piece in 1..pieces {
        let target = (u128::from(total) * piece as u128 / pieces as u128) as u64;
        let mut before = 0u64;
        for (i, share) in shares.iter().enumerate() {
            let last = i + 1 == shares.len();
            if target >= before + share && !last {
                before += share;
                continue;
            }
            // The nearest of: a page inside chunk `i`, chunk `i`'s start, the next chunk's start.
            let mut best: Option<(u64, Bytes)> = None;
            let mut consider = |at: u64, key: Bytes| {
                let distance = at.abs_diff(target);
                if best.as_ref().is_none_or(|(d, _)| distance < *d) {
                    best = Some((distance, key));
                }
            };
            if let Some(key) = chunks.start(i) {
                consider(before, key.clone());
            }
            if let Some(key) = chunks.start(i + 1) {
                consider(before + share, key.clone());
            }
            let span = chunks.span(i, Some(start), end);
            let inside = chunks[i].file.cut_near(
                span_start(&span),
                span_end(&span),
                target.saturating_sub(before),
            );
            if let Some((at, key)) = inside {
                consider(before + at, key);
            }
            if let Some((_, key)) = best {
                cuts.push(key);
            }
            break;
        }
    }
    cuts.sort();
    cuts.dedup();
    cuts.retain(|key| key > start && end.is_none_or(|end| key < end));
    cuts
}

/// The chunks of `chunks`, the base of a range, that hold keys from `start` to `end` (`None`: the
/// range's end), keys of the range: one chunk as [`Chunks::One`], several as a list whose first
/// chunk starts at `start`. A chunk that `start` or `end` cuts is kept whole: the child reads its
/// file within its own keys.
fn chunks_within(chunks: &Chunks, start: &[u8], end: Option<&[u8]>) -> Chunks {
    match chunks {
        Chunks::Loading => Chunks::Loading,
        Chunks::One(chunk) => Chunks::One(chunk.clone()),
        Chunks::Many(list) => {
            let first = list.index_of(start);
            let last = match end {
                None => list.chunks.len() - 1,
                // The chunk of the last key before `end`.
                Some(end) => list.starts.partition_point(|s| s.as_ref() < end),
            };
            let last = last.max(first);
            if first == last {
                Chunks::One(list.chunks[first].clone())
            } else {
                Chunks::Many(Arc::new(ChunkList::new(
                    list.chunks[first..=last].to_vec(),
                    list.starts[first..last].to_vec(),
                )))
            }
        }
    }
}

/// Plans a zero-copy split of `[start, end)` at ascending `cuts`, restricting the parent view.
pub(super) fn plan(
    layers: &Layers,
    chunks: &Chunks,
    start: &Bytes,
    end: Option<&Bytes>,
    cuts: &[Bytes],
) -> io::Result<Option<SplitPlan>> {
    if chunks.loading() || cuts.is_empty() {
        return Ok(None);
    }
    let mut bounds: Vec<(Bytes, Option<Bytes>)> = Vec::with_capacity(cuts.len() + 1);
    let mut from = start.clone();
    for cut in cuts {
        if *cut <= from || end.is_some_and(|end| cut >= end) {
            return Ok(None);
        }
        bounds.push((from, Some(cut.clone())));
        from = cut.clone();
    }
    bounds.push((from, end.cloned()));
    let mut children = Vec::with_capacity(bounds.len());
    for (child_start, child_end) in bounds {
        let child_chunks = chunks_within(chunks, &child_start, child_end.as_deref());
        let floor = child_chunks.iter().map(|c| c.base_seq).min().unwrap_or(0);
        // Runs are newest first and contiguous, so the ones that every chunk of the child absorbed
        // are the oldest.
        let kept = layers.runs.iter().take_while(|r| r.hi > floor).count();
        let (runs, dropped) = layers.runs.split_at(kept);
        if runs.last().is_some_and(|oldest| oldest.lo != floor) {
            return Ok(None);
        }
        let view = match &layers.view {
            Some(view) => view
                .restrict(&child_start, child_end.as_deref(), dropped)?
                .map(Arc::new),
            None => None,
        };
        if view.is_some() == runs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "range cache split: a view that does not match the runs",
            ));
        }
        children.push(SplitChild {
            start: child_start,
            end: child_end,
            chunks: child_chunks,
            runs: runs.to_vec(),
            view,
        });
    }
    Ok(Some(SplitPlan {
        chunks_was: chunks.clone(),
        runs_was: layers.runs.clone(),
        view_was: layers.view.clone(),
        children,
        rate: 0.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound::{self, Excluded, Included};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    use fail_parallel::FailPointRegistry;
    use slatedb_common::metrics::{
        lookup_metric, lookup_metric_with_labels, DefaultMetricsRecorder,
    };

    use crate::config::{ReadOptions, ScanOptions, TracingOptions};
    use crate::db::Db;
    use crate::merge_operator::{MergeOperator, MergeOperatorError};
    use crate::object_store::memory::InMemory;
    use crate::oracle::Oracle;
    use crate::range_cache::stats::{
        FLUSH_COUNT, PIN_RETRY_COUNT, REBUILD_COUNT, SEAL_COUNT, SPLIT_COUNT, SPLIT_RANGE_COUNT,
        SPLIT_REJECTED_COUNT, UNUSABLE_COUNT,
    };
    use crate::range_cache::{
        CachedRange, CachedRangeInfo, Footprint, RangeCache, RangeCacheOptions,
    };

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

        fn ranges(&self) -> Vec<Arc<CachedRange>> {
            self.cache().index.load().values().cloned().collect()
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
                lookup_metric(&self.recorder, name).unwrap_or(0)
            } else {
                lookup_metric_with_labels(&self.recorder, name, labels).unwrap_or(0)
            }
        }

        fn splits(&self, when: &str) -> i64 {
            self.metric(SPLIT_COUNT, &[("when", when)])
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

        /// Splits the range that starts at `start` at `cuts` as the maintenance task does.
        async fn split(&self, start: &Bytes, cuts: &[usize]) -> bool {
            let range = self.range(start);
            let cuts = cuts.iter().map(|i| key(*i)).collect();
            self.cache().split(&range, cuts).await.unwrap()
        }

        /// Asserts that cached reads of the keys in `[a, b)`, at the latest state and in
        /// `snapshot`, equal the normal read path at the same state, and that the ranges tile
        /// their keys and keep the layer sets' invariants.
        async fn assert_reads(&self, a: usize, b: usize, snapshot: Option<&crate::DbSnapshot>) {
            for range in &self.ranges() {
                let layers = range.layers.load_full();
                layers.check(&range.start, *crate::range_cache::MAX_RUNS_RANGE.end());
                if let Some(view) = &layers.view {
                    view.check_in(&range.start, range.end.as_deref()).unwrap();
                }
            }
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
                for (from, to) in [(a, b), (a + (b - a) / 3, b), (a, a + (b - a) / 2)] {
                    let mut cached = reader.scan(key(from)..key(to)).await.unwrap();
                    let mut normal = reader
                        .scan_with_options(
                            key(from)..key(to),
                            &ScanOptions::default()
                                .with_tracing_options(Some(TracingOptions::new("normal"))),
                        )
                        .await
                        .unwrap();
                    loop {
                        let (c, n) = (cached.next().await.unwrap(), normal.next().await.unwrap());
                        assert_eq!(c, n, "{label} scan from {from} to {to}");
                        if c.is_none() {
                            break;
                        }
                    }
                }
            }
        }
    }

    impl Drop for TestDb {
        /// A test that fails while a fail point holds tasks must end, not wait for them: the
        /// runtime's shutdown waits for its threads.
        fn drop(&mut self) {
            for name in [
                "range-cache-pin",
                "range-cache-flush",
                "range-cache-load",
                "range-cache-attach",
            ] {
                let _ = fail_parallel::cfg(self.fp_registry.clone(), name, "off");
            }
        }
    }

    async fn open(tune: impl FnOnce(&mut RangeCacheOptions)) -> TestDb {
        let dir = tempfile::tempdir().unwrap();
        let mut options = RangeCacheOptions::new(dir.path());
        options.tuning.stats_fold_events = 1;
        // No major merge, and no split, unless a test asks for one.
        options.tuning.min_rebuild_delta_bytes = usize::MAX;
        options.tuning.range_target_bytes = 0;
        tune(&mut options);
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let fp_registry = Arc::new(FailPointRegistry::new());
        let db = Db::builder("range_cache_split", Arc::new(InMemory::new()))
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

    /// A 200-byte value: a 4 KiB page holds about 15 rows.
    fn value(i: usize, round: usize) -> Bytes {
        let mut v = format!("value-{i}-{round}-").into_bytes();
        v.resize(200, b'.');
        Bytes::from(v)
    }

    fn span(a: usize, b: usize) -> (Bound<Bytes>, Bound<Bytes>) {
        (Included(key(a)), Excluded(key(b)))
    }

    fn normal() -> ReadOptions {
        ReadOptions::default().with_tracing_options(Some(TracingOptions::new("normal")))
    }

    async fn put_all(t: &TestDb, a: usize, b: usize) {
        for i in a..b {
            t.db.put(key(i), value(i, 0)).await.unwrap();
        }
    }

    /// The base file of a range of one chunk.
    fn base_file(range: &CachedRange) -> Arc<super::super::file::RangeFile> {
        range.layers.load().only_chunk().file.clone()
    }

    /// A split shares the base file as narrowed chunks, the runs and the view: each child's view
    /// lists its own keys, the bases count once against the disk budget, and each range's base is
    /// its pages of the file.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_split_without_copying_and_count_shared_objects_once() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        put_all(&t, 0, 300).await;
        t.db.cache_range(span(0, 300)).await.unwrap();
        // A run, so that the children share a run and the view.
        let seals = t.seals();
        t.write_until(0, 300, |t| t.seals() > seals).await;
        t.wait("the flush", |t| {
            let info = t.info(&key(0));
            info.sealed_buffers == 0 && info.runs == 1
        })
        .await;
        let parent = t.range(&key(0));
        let before = parent.layers.load_full();
        let base = base_file(&parent);
        let whole = Footprint::of(t.ranges().iter());
        let files = t.cache_files();
        assert!(t.split(&key(0), &[120]).await);
        assert_eq!(t.splits("maintenance"), 1);
        assert_eq!(t.metric(SPLIT_RANGE_COUNT, &[]), 2);
        let (left, right) = (t.range(&key(0)), t.range(&key(120)));
        assert_eq!(left.end, Some(key(120)));
        assert_eq!(right.end, Some(key(300)));
        for child in [&left, &right] {
            let layers = child.layers.load_full();
            assert!(Arc::ptr_eq(&layers.only_chunk().file, &base));
            assert_eq!(layers.only_chunk().base_seq, before.only_chunk().base_seq);
            assert_eq!(layers.runs.len(), 1);
            assert!(Arc::ptr_eq(&layers.runs[0], &before.runs[0]));
            assert_eq!(layers.covered_to, before.covered_to);
            assert!(!Arc::ptr_eq(
                layers.view.as_ref().unwrap(),
                before.view.as_ref().unwrap()
            ));
        }
        // The views split the parent's entries, and no file was written.
        let entries = |r: &CachedRange| r.layers.load().view.as_ref().unwrap().len();
        assert_eq!(
            entries(&left) + entries(&right),
            before.view.as_ref().unwrap().len()
        );
        assert_eq!(t.cache_files(), files);
        // The disk budget counts the shared base and run once; the bases of the ranges add up to
        // the base file's data, by pages (the page that holds the cut, which is not the first key
        // of a page here, counts in both).
        let split = Footprint::of(t.ranges().iter());
        assert_eq!(split.disk, whole.disk);
        let infos = t.db.cached_ranges();
        let bases: u64 = infos.iter().map(|r| r.file_bytes).sum();
        assert!(
            bases >= base.data_len() && bases <= base.data_len() + 4096,
            "{bases} for {}",
            base.data_len()
        );
        let rows: u64 = infos.iter().map(|r| r.file_rows).sum();
        assert!(
            rows.abs_diff(base.rows()) <= 20,
            "{rows} of {}",
            base.rows()
        );
        assert!(t.cache().within_budgets());
        drop(before);
        t.assert_reads(0, 300, None).await;
        t.db.close().await.unwrap();
    }

    /// A reader pinned to the parent before a split at `W` retries on `S = W + 1` and reads the
    /// child holding the new batch.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_retry_a_reader_of_the_parent_after_a_split() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 100).await;
        t.db.cache_range(span(0, 100)).await.unwrap();
        // A version in the parent's active buffer, which becomes a sealed buffer of the children;
        // their flushes wait until the test has looked at it.
        t.db.put(key(10), "before the split").await.unwrap();
        t.fail("range-cache-flush", "pause");
        let parent = t.range(&key(0));
        // The reader pins the parent's layers and waits before it loads its sequence number.
        t.fail("range-cache-pin", "pause");
        let reader = {
            let db = t.db.clone();
            tokio::spawn(async move { db.get(key(70)).await.unwrap() })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!reader.is_finished());
        let pinned = parent.layers.load_full();
        assert!(t.split(&key(0), &[50]).await);
        let w = pinned.active.sealed_at().load(Ordering::SeqCst);
        assert_eq!(w, t.committed());
        t.db.put(key(70), "after the split").await.unwrap();
        assert!(t.committed() > w);
        assert!(RangeCache::usable(&pinned, w).is_ok());
        assert_eq!(
            RangeCache::usable(&pinned, w + 1),
            Err(crate::range_cache::stats::Unfit::Sealed)
        );
        t.fail("range-cache-pin", "off");
        let got = tokio::time::timeout(PROMPT, reader).await.unwrap().unwrap();
        assert_eq!(got, Some(Bytes::from("after the split")));
        let unfit = |name, reason| t.metric(name, &[("op", "get"), ("reason", reason)]);
        assert_eq!(unfit(PIN_RETRY_COUNT, "sealed"), 1);
        assert_eq!(unfit(UNUSABLE_COUNT, "sealed"), 0);
        // The child that holds the batch has it in its active buffer, above `W`; both children
        // share the parent's active buffer, sealed at `W`.
        for (start, writes) in [(0, 0), (50, 1)] {
            let layers = t.range(&key(start)).layers.load_full();
            assert_eq!(layers.active.lo(), w);
            assert_eq!(layers.active.len(), writes);
            assert!(Arc::ptr_eq(&layers.sealed[0], &pinned.active));
        }
        drop(pinned);
        t.assert_reads(0, 100, None).await;
        t.fail("range-cache-flush", "off");
        t.wait("the flushes", |t| t.metric(FLUSH_COUNT, &[]) == 2)
            .await;
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    /// A reader pinned to the parent at `S <= W` reads the parent's layers and keeps files alive
    /// even after both children are removed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_serve_an_old_reader_of_the_parent_from_its_own_layers() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 100).await;
        t.db.cache_range(span(0, 100)).await.unwrap();
        // Versions in the parent's active buffer, below the snapshot.
        for i in (0..100).step_by(7) {
            t.db.put(key(i), value(i, 1)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        let pinned = t.range(&key(0)).layers.load_full();
        assert!(t.split(&key(0), &[40]).await);
        let w = pinned.active.sealed_at().load(Ordering::SeqCst);
        assert!(snapshot.seq() <= w);
        for i in (0..100).step_by(5) {
            t.db.put(key(i), value(i, 2)).await.unwrap();
        }
        // The pinned parent's layers read the snapshot's state, as the normal path does.
        let read =
            |layers: &Layers, i: usize| match super::super::lookup(layers, &key(i), snapshot.seq())
                .unwrap()
            {
                super::super::Found::Value(kv) => Some(kv.value),
                super::super::Found::Absent => None,
                super::super::Found::Merge => panic!("no merge operand here"),
            };
        assert!(RangeCache::usable(&pinned, snapshot.seq()).is_ok());
        for i in 0..100 {
            let normal = snapshot.get_with_options(key(i), &normal()).await.unwrap();
            assert_eq!(read(&pinned, i), normal, "key {i}");
        }
        // The children serve the snapshot too: their shared sources hold every version up to `W`.
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // Both children go; the pin keeps the parent's files and buffers.
        t.db.uncache_range(span(0, 100)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        assert!(t.db.cached_ranges().is_empty());
        assert_eq!(t.cache_files(), 1);
        for i in 0..100 {
            let normal = snapshot.get_with_options(key(i), &normal()).await.unwrap();
            assert_eq!(read(&pinned, i), normal, "key {i} after the children left");
        }
        drop(pinned);
        t.wait("the base file to go", |t| t.cache_files() == 0)
            .await;
        t.db.close().await.unwrap();
    }

    /// Each child flushes only its own keys from the shared sealed buffer; buffer memory drops
    /// once the last child finishes flushing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_flush_a_shared_buffer_once_per_child_within_its_keys() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 100).await;
        t.db.cache_range(span(0, 100)).await.unwrap();
        for i in 0..100 {
            t.db.put(key(i), value(i, 1)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        let shared = t.range(&key(0)).layers.load().active.clone();
        assert_eq!(shared.len(), 100);
        // Both flushes wait until both children hold the buffer.
        t.fail("range-cache-flush", "pause");
        assert!(t.split(&key(0), &[30]).await);
        let (left, right) = (t.range(&key(0)), t.range(&key(30)));
        for child in [&left, &right] {
            let layers = child.layers.load();
            assert_eq!(layers.sealed.len(), 1);
            assert!(Arc::ptr_eq(&layers.sealed[0], &shared));
            assert!(layers.active.is_empty());
        }
        t.assert_reads(0, 100, Some(&snapshot)).await;
        t.fail("range-cache-flush", "off");
        t.wait("both flushes", |t| t.metric(FLUSH_COUNT, &[]) == 2)
            .await;
        let runs: Vec<Arc<Run>> = [&left, &right]
            .iter()
            .map(|c| {
                let layers = c.layers.load();
                assert!(layers.sealed.is_empty());
                assert_eq!(layers.runs.len(), 1);
                layers.runs[0].clone()
            })
            .collect();
        assert_eq!((runs[0].file.rows(), runs[1].file.rows()), (30, 70));
        assert!(runs[0].max_key < key(30) && runs[1].min_key >= key(30));
        t.assert_reads(0, 100, Some(&snapshot)).await;
        // The shared buffer's memory goes with its last holder, once the flush jobs end.
        t.wait("the flush jobs", |_| Arc::strong_count(&shared) == 1)
            .await;
        let memory = t.cache().delta_memory.bytes();
        let held = shared.memory_bytes();
        assert!(memory >= held && held > 0);
        drop(shared);
        assert_eq!(t.cache().delta_memory.bytes(), memory - held);
        t.db.close().await.unwrap();
    }

    /// The writer rejects a split plan when a concurrent flush changed the range's runs and view.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_reject_a_split_whose_range_moved_on() {
        let t = open(|o| o.tuning.range_seal_bytes = 16 * 1024).await;
        put_all(&t, 0, 100).await;
        t.db.cache_range(span(0, 100)).await.unwrap();
        let range = t.range(&key(0));
        let layers = range.layers.load_full();
        let planned = plan(
            &layers,
            &layers.chunks,
            &range.start,
            range.end.as_ref(),
            &[key(50)],
        )
        .unwrap()
        .unwrap();
        let stale = plan(
            &layers,
            &layers.chunks,
            &range.start,
            range.end.as_ref(),
            &[key(50)],
        )
        .unwrap()
        .unwrap();
        drop(layers);
        let seals = t.seals();
        t.write_until(0, 100, |t| t.seals() > seals).await;
        t.wait("the flush", |t| t.info(&key(0)).runs == 1).await;
        assert!(!t.cache().install_split(&range, planned).await);
        assert_eq!(t.metric(SPLIT_REJECTED_COUNT, &[]), 1);
        assert_eq!(t.db.cached_ranges().len(), 1);
        assert!(t.cache().is_indexed(&range));
        // A removed range rejects it too.
        t.db.uncache_range(span(0, 100)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        assert!(!t.cache().install_split(&range, stale).await);
        assert_eq!(t.metric(SPLIT_REJECTED_COUNT, &[]), 2);
        assert_eq!(t.splits("maintenance"), 0);
        t.db.cache_range(span(0, 100)).await.unwrap();
        t.assert_reads(0, 100, None).await;
        t.db.close().await.unwrap();
    }

    /// A loading range or a range with an active flush attaches whole and waits until the flush
    /// finishes before maintenance splits it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_split_only_when_no_job_runs() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.range_target_bytes = 32 * 1024;
        })
        .await;
        put_all(&t, 0, 400).await;
        t.fail("range-cache-load", "pause");
        t.fail("range-cache-flush", "pause");
        let load = {
            let db = t.db.clone();
            tokio::spawn(async move { db.cache_range(span(0, 400)).await })
        };
        t.wait("the registration", |t| t.db.cached_ranges().len() == 1)
            .await;
        // A seal of the loading range, and its flush, which waits.
        let seals = t.seals();
        t.write_until(0, 400, |t| t.seals() > seals).await;
        let range = t.range(&key(0));
        t.wait("the flush", |_| range.jobs.flushing()).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(t.splits("maintenance"), 0);
        t.fail("range-cache-load", "off");
        tokio::time::timeout(PROMPT, load)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        // It attached whole: its flush runs.
        assert_eq!(t.splits("attach"), 0);
        assert_eq!(t.metric(SPLIT_REJECTED_COUNT, &[]), 1);
        assert!(!t.info(&key(0)).loading);
        tokio::time::sleep(Duration::from_millis(200)).await;
        // The task planned no split meanwhile: the writer would have rejected it.
        assert_eq!(t.splits("maintenance"), 0);
        assert_eq!(t.metric(SPLIT_REJECTED_COUNT, &[]), 1);
        assert_eq!(t.db.cached_ranges().len(), 1);
        assert!(t.info(&key(0)).rebuilding);
        t.assert_reads(0, 400, None).await;
        t.fail("range-cache-flush", "off");
        t.wait("the splits", |t| {
            let ranges = t.db.cached_ranges();
            ranges.len() > 1
                && ranges
                    .iter()
                    .all(|r| r.file_bytes <= 64 * 1024 && r.sealed_buffers == 0)
        })
        .await;
        assert!(t.splits("maintenance") >= 1);
        t.assert_reads(0, 400, None).await;
        t.db.close().await.unwrap();
    }

    /// A large load splits on attach into target-sized ranges sharing the base file and any run
    /// flushed during the load.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_split_a_large_load_as_it_attaches() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.range_target_bytes = 16 * 1024;
        })
        .await;
        put_all(&t, 0, 400).await;
        t.fail("range-cache-load", "pause");
        let load = {
            let db = t.db.clone();
            tokio::spawn(async move { db.cache_range(span(0, 400)).await })
        };
        t.wait("the registration", |t| t.db.cached_ranges().len() == 1)
            .await;
        let seals = t.seals();
        t.write_until(0, 400, |t| t.seals() > seals).await;
        t.wait("the flush", |t| {
            t.info(&key(0)).runs == 1 && !t.range(&key(0)).jobs.flushing()
        })
        .await;
        let run = t.range(&key(0)).layers.load().runs[0].clone();
        let snapshot = t.db.snapshot().unwrap();
        t.fail("range-cache-load", "off");
        tokio::time::timeout(PROMPT, load)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(t.splits("attach"), 1);
        assert_eq!(t.metric(SPLIT_REJECTED_COUNT, &[]), 0);
        let ranges = t.ranges();
        let base = base_file(&ranges[0]);
        assert!(ranges.len() >= 5, "{} ranges", ranges.len());
        assert_eq!(t.metric(SPLIT_RANGE_COUNT, &[]), ranges.len() as i64);
        assert_eq!(ranges[0].start, key(0));
        assert_eq!(ranges.last().unwrap().end, Some(key(400)));
        for (i, range) in ranges.iter().enumerate() {
            let layers = range.layers.load();
            assert!(Arc::ptr_eq(&layers.only_chunk().file, &base));
            assert!(Arc::ptr_eq(&layers.runs[0], &run), "range {i}");
            if let Some(next) = ranges.get(i + 1) {
                assert_eq!(range.end.as_ref(), Some(&next.start));
            }
        }
        for info in t.db.cached_ranges() {
            assert!(!info.loading);
            assert!(info.file_bytes <= 2 * 16 * 1024, "{info:?}");
        }
        t.assert_reads(0, 400, Some(&snapshot)).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(t.splits("maintenance"), 0);
        t.db.close().await.unwrap();
    }

    /// A split at attach that a flush of the loading range overtakes between its plan and its
    /// store attaches the range whole; the maintenance task splits it afterwards.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_attach_whole_when_a_flush_overtakes_the_split() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.range_target_bytes = 16 * 1024;
        })
        .await;
        put_all(&t, 0, 400).await;
        t.fail("range-cache-attach", "pause");
        let load = {
            let db = t.db.clone();
            tokio::spawn(async move { db.cache_range(span(0, 400)).await })
        };
        t.wait("the registration", |t| t.db.cached_ranges().len() == 1)
            .await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        let seals = t.seals();
        t.write_until(0, 400, |t| t.seals() > seals).await;
        t.wait("the flush", |t| {
            t.info(&key(0)).runs == 1 && !t.range(&key(0)).jobs.flushing()
        })
        .await;
        t.fail("range-cache-attach", "off");
        tokio::time::timeout(PROMPT, load)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(t.splits("attach"), 0);
        assert_eq!(t.metric(SPLIT_REJECTED_COUNT, &[]), 1);
        t.wait("the maintenance splits", |t| {
            t.db.cached_ranges()
                .iter()
                .all(|r| r.file_bytes <= 2 * 16 * 1024)
        })
        .await;
        assert!(t.splits("maintenance") >= 2);
        t.assert_reads(0, 400, None).await;
        t.db.close().await.unwrap();
    }

    /// A range whose base passes twice the target as major merges add its new keys is split by
    /// the maintenance task, until no base is above twice the target.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_split_a_range_that_grows_past_twice_the_target() {
        let target = 32 * 1024;
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.range_target_bytes = target;
            o.tuning.min_rebuild_delta_bytes = 1;
        })
        .await;
        put_all(&t, 0, 100).await;
        t.db.cache_range(span(0, 1000)).await.unwrap();
        assert_eq!(t.splits("attach"), 0);
        // New keys land in the range and reach its base through major merges.
        t.write_until(0, 1000, |t| t.splits("maintenance") >= 2)
            .await;
        t.wait("quiet", |t| {
            t.db.cached_ranges()
                .iter()
                .all(|r| !r.rebuilding && r.file_bytes <= 2 * target)
        })
        .await;
        assert!(t.metric(REBUILD_COUNT, &[]) >= 2);
        assert!(t.db.cached_ranges().len() >= 3);
        t.assert_reads(0, 1000, None).await;
        t.db.close().await.unwrap();
    }

    /// Removing one child leaves the other whole; a child's major merge reads the shared base
    /// within its keys and writes its own; the shared base goes when no range or reader holds it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_keep_a_child_whole_when_the_other_merges_or_leaves() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.min_rebuild_delta_bytes = 1;
            o.tuning.major_merge_factor = 0.05;
        })
        .await;
        put_all(&t, 0, 300).await;
        t.db.cache_range(span(0, 300)).await.unwrap();
        let shared = base_file(&t.range(&key(0)));
        assert!(t.split(&key(0), &[100, 200]).await);
        assert_eq!(t.db.cached_ranges().len(), 3);
        // The middle child merges: its base is its own keys now.
        let seals = t.seals();
        t.write_until(100, 200, |t| t.seals() > seals).await;
        t.wait("the middle child's major merge", |t| {
            t.metric(REBUILD_COUNT, &[]) >= 1 && !t.info(&key(100)).rebuilding
        })
        .await;
        let middle = base_file(&t.range(&key(100)));
        assert!(!Arc::ptr_eq(&middle, &shared));
        assert_eq!(middle.rows(), 100);
        assert!(Arc::ptr_eq(&base_file(&t.range(&key(0))), &shared));
        assert!(Arc::ptr_eq(&base_file(&t.range(&key(200))), &shared));
        t.assert_reads(0, 300, None).await;
        // The first child leaves; the last still reads the shared base.
        t.db.uncache_range(span(0, 100)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        assert_eq!(t.db.cached_ranges().len(), 2);
        t.assert_reads(100, 300, None).await;
        let weak = Arc::downgrade(&shared);
        drop(shared);
        assert!(weak.upgrade().is_some());
        t.db.uncache_range(span(200, 300)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        t.wait("the shared base to go", |_| weak.upgrade().is_none())
            .await;
        t.assert_reads(100, 200, None).await;
        t.db.close().await.unwrap();
    }

    /// An automatic range split on attach defers unpaid major merges until reads pay for them;
    /// read children merge and stay, while unread children evict at their run limit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_keep_the_read_children_of_an_automatic_load_split_at_attach() {
        use crate::range_cache::maintain;
        use crate::range_cache::stats::REMOVE_COUNT;
        let t = open(|o| {
            // Bases of about 64 KiB, and runs of 32 KiB.
            o.tuning.range_seal_bytes = 32 * 1024;
            o.tuning.range_target_bytes = 64 * 1024;
            o.tuning.min_rebuild_delta_bytes = 1;
            // Promotion on, so that automatic ranges take the payback test, but no sample.
            o.auto_promote = Some(crate::range_cache::AutoPromoteOptions {
                sample_one_in: u32::MAX,
                ..crate::range_cache::AutoPromoteOptions::default()
            });
        })
        .await;
        let not_worth_it = |t: &TestDb| t.metric(REMOVE_COUNT, &[("reason", "not_worth_it")]);
        put_all(&t, 0, 2000).await;
        t.fail("range-cache-load", "pause");
        let load = {
            let inner = t.db.inner.clone();
            tokio::spawn(async move {
                let range =
                    crate::bytes_range::BytesRange::new(Included(key(0)), Excluded(key(2000)));
                let origin = crate::range_cache::Origin::Auto {
                    read_cost_ns: 1000,
                    read_rate: 0.0,
                };
                inner.cache_range_with(range, origin).await
            })
        };
        t.wait("the registration", |t| t.db.cached_ranges().len() == 1)
            .await;
        // Writes across the range while it loads: six runs, under half of the base.
        let seals = t.seals();
        let mut i = 0usize;
        while t.seals() < seals + 6 {
            let k = (i * 7919) % 2000;
            t.db.put(key(k), value(k, i)).await.unwrap();
            i += 1;
        }
        t.wait("the flushes", |t| {
            t.info(&key(0)).sealed_buffers == 0 && !t.range(&key(0)).jobs.flushing()
        })
        .await;
        t.fail("range-cache-load", "off");
        tokio::time::timeout(PROMPT, load)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(t.splits("attach"), 1);
        let children = t.db.cached_ranges();
        assert!(children.len() >= 5, "{} ranges", children.len());
        let tuning = t.cache().options.tuning.clone();
        for range in t.ranges() {
            let layers = range.layers.load();
            assert!(range.auto);
            // Due, and waiting for reads: not under way.
            assert!(
                maintain::major_due(&range, &layers, &tuning).is_some(),
                "{:?}",
                range.info(t.cache())
            );
            assert!(!range.info(t.cache()).rebuilding);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(not_worth_it(&t), 0);
        assert_eq!(t.db.cached_ranges().len(), children.len());
        // The first child is read, so its merge pays (about 64 KiB at 5 ns a byte against 1 µs a
        // read): it merges, and stays.
        for i in 0..1000 {
            t.db.get(key(i % 100)).await.unwrap();
        }
        t.wait("the first child's merge", |t| {
            t.metric(REBUILD_COUNT, &[]) >= 1 && !t.info(&key(0)).rebuilding
        })
        .await;
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 1);
        assert_eq!(not_worth_it(&t), 0);
        assert_eq!(t.db.cached_ranges().len(), children.len());
        // The last child is written and never read: it leaves at its run limit.
        let last = children.last().unwrap().start.clone();
        assert!(last <= key(1950));
        t.write_until(1950, 2000, |t| not_worth_it(t) >= 1).await;
        t.wait("the removal", |t| {
            t.db.cached_ranges().iter().all(|r| r.start != last)
        })
        .await;
        assert_eq!(not_worth_it(&t), 1);
        assert_eq!(t.db.cached_ranges().len(), children.len() - 1);
        t.assert_reads(0, 300, None).await;
        t.db.close().await.unwrap();
    }

    /// A split divides the delta and the base in one proportion: with writes spread over the
    /// range, a range whose runs are below a quarter of its base splits into children whose parts
    /// of those runs are below a quarter of their bases, so the split makes no major merge due.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_not_make_a_major_merge_due_by_splitting() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 32 * 1024;
            o.tuning.min_rebuild_delta_bytes = 1;
        })
        .await;
        put_all(&t, 0, 4000).await;
        t.db.cache_range(span(0, 4000)).await.unwrap();
        // Four runs over the whole range: about an eighth of its base, and about half of what a
        // quarter of a child's base is, but more than that counted whole.
        let mut i = 0usize;
        let seals = t.seals();
        while t.seals() < seals + 4 {
            let k = (i * 7919) % 4000;
            t.db.put(key(k), value(k, i)).await.unwrap();
            i += 1;
        }
        t.wait("the flushes", |t| {
            t.info(&key(0)).sealed_buffers == 0 && !t.range(&key(0)).jobs.flushing()
        })
        .await;
        let whole = t.info(&key(0));
        assert!(whole.run_bytes * 4 < whole.file_bytes, "{whole:?}");
        assert!(!whole.rebuilding);
        let runs: u64 = t
            .range(&key(0))
            .layers
            .load()
            .runs
            .iter()
            .map(|r| r.file_bytes())
            .sum();
        assert!(t.split(&key(0), &[1000, 2000, 3000]).await);
        for child in t.db.cached_ranges() {
            assert!(child.run_bytes * 4 < child.file_bytes, "{child:?}");
            assert!(runs * 4 > child.file_bytes, "{runs} {child:?}");
            assert!(!child.rebuilding, "{child:?}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(t.metric(REBUILD_COUNT, &[]), 0);
        t.assert_reads(0, 4000, None).await;
        t.db.close().await.unwrap();
    }

    /// A split's children carry their share of the reads that the parent's base served since it
    /// was written, by the bytes of their bases, for the payback test of their major merges; a
    /// child's own major merge starts its count again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_carry_the_reads_of_the_parent_base_to_its_children() {
        let t = open(|o| {
            o.tuning.range_seal_bytes = 16 * 1024;
            o.tuning.min_rebuild_delta_bytes = 1;
        })
        .await;
        put_all(&t, 0, 300).await;
        t.db.cache_range(span(0, 300)).await.unwrap();
        for i in 0..600 {
            t.db.get(key(i % 300)).await.unwrap();
        }
        let parent = t.range(&key(0));
        assert_eq!(parent.reads_since_build(), 600);
        assert!(t.split(&key(0), &[150]).await);
        let (left, right) = (t.range(&key(0)), t.range(&key(150)));
        let (l, r) = (left.reads_since_build(), right.reads_since_build());
        assert!(
            l.abs_diff(r) < 100 && (l + r).abs_diff(600) <= 1,
            "{l} and {r}"
        );
        // Reads of the child after the split count on top.
        for i in 0..50 {
            t.db.get(key(i)).await.unwrap();
        }
        assert_eq!(left.reads_since_build(), l + 50);
        // Its major merge writes a new base: the count starts again.
        let seals = t.seals();
        t.write_until(0, 150, |t| t.seals() > seals).await;
        t.wait("the major merge", |t| {
            t.metric(REBUILD_COUNT, &[]) >= 1 && !t.info(&key(0)).rebuilding
        })
        .await;
        assert_eq!(left.reads_since_build(), 0);
        assert_eq!(right.reads_since_build(), r);
        t.db.close().await.unwrap();
    }

    /// Cut keys by bytes from the fence index: the middle page for two pieces, pieces of about
    /// the target for a split at attach, none for a base of one page.
    #[tokio::test]
    async fn should_cut_by_bytes_at_page_starts() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 300).await;
        t.db.cache_range(span(0, 300)).await.unwrap();
        t.db.cache_range(span(400, 405)).await.unwrap();
        let range = t.range(&key(0));
        let layers = range.layers.load_full();
        let file = base_file(&range);
        let cuts = cut_keys(&layers.chunks, &range.start, range.end.as_ref(), 2);
        assert_eq!(cuts.len(), 1);
        let (left, _) = file.span_share(Some(range.start.as_ref()), Some(cuts[0].as_ref()));
        let (right, _) = file.span_share(Some(cuts[0].as_ref()), range.end.as_deref());
        assert!(left.abs_diff(right) <= 4096, "{left} and {right}");
        let pieces = attach_pieces(file.data_len(), file.data_len() / 4);
        assert_eq!(pieces, 4);
        let cuts = cut_keys(&layers.chunks, &range.start, range.end.as_ref(), pieces);
        assert_eq!(cuts.len(), 3);
        assert!(cuts.windows(2).all(|w| w[0] < w[1]));
        let small = t.range(&key(400));
        let layers = small.layers.load_full();
        assert!(cut_keys(&layers.chunks, &small.start, small.end.as_ref(), 2).is_empty());
        assert_eq!(attach_pieces(10, 100), 2);
        t.db.close().await.unwrap();
    }
}
