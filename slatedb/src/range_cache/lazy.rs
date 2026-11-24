//! Lazy range pinning for snapshot and transaction scans.
//!
//! Latest-state scans pin every cached range up front before reading the sequence number. Scans at
//! a registered sequence number `S` (snapshots and transactions) pin only the starting range up
//! front and pin subsequent ranges on demand in [`RangeCache::pin_next`]:
//! - If a ready range holds the next key and satisfies the sequence rules at `S`, the scan reads
//!   it directly (handling concurrent splits or quiet merges transparently).
//! - Otherwise the scan opens a *gap* on the normal read path at `S` up to the next cached range
//!   or scan end.
//!
//! Each snapshot or transaction scan iterator holds a `RetentionPin` keeping `S` registered until
//! the iterator drops, so flushes and compactions retain all versions visible at `S`. As a safety
//! guard against crate-internal callers without retention, [`GapReader::holds`] verifies after
//! capturing the state view that either `last_l0_seq <= S` or an active snapshot or transaction is
//! registered at or below `S`.
//!
//! Each scan holds a per-thread [`LazyHandle`] to avoid shared refcount contention on the cache.
//! Each lazily pinned range increments `lazy_pin_count` and its range scan count; the scan records
//! `scan_count` or `mixed_scan_count` on drop.

use std::io;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::ops::RangeBounds;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;

use super::delta::past_end;
use super::hits::HitStats;
use super::scan::{min_end, CachedScan};
use super::stats::{Fallback, FallbackOp, ReadCount};
use super::{Layers, RangeCache};
use crate::bytes_range::BytesRange;
use crate::config::ScanOptions;
use crate::db_iter::DbIterator;
use crate::db_state::{DbState, DbStateView};
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::merge_operator::MergeOperatorType;
use crate::reader::{DbStateReader, Reader, ScanContext};
use crate::snapshot_manager::SnapshotManager;
use crate::transaction_manager::TransactionManager;
use crate::types::RowEntry;

/// Dependencies needed to open a normal-path gap and verify sequence retention.
pub(crate) struct GapReader {
    pub(crate) reader: Reader,
    pub(crate) state: Arc<RwLock<DbState>>,
    pub(crate) snapshots: Arc<SnapshotManager>,
    pub(crate) transactions: Arc<TransactionManager>,
}

impl GapReader {
    /// Returns true if `view` retains every version visible at `seq`: either no SST holds versions
    /// above `seq`, or an active snapshot or transaction is registered at or below `seq`.
    fn holds(&self, view: &DbStateView, seq: u64) -> bool {
        view.core().last_l0_seq <= seq
            || [
                self.snapshots.min_active_seq(),
                self.transactions.min_active_seq(),
            ]
            .into_iter()
            .flatten()
            .min()
            .is_some_and(|registered| registered <= seq)
    }
}

/// Number of cache-line-aligned handles sharded across reading threads.
const LAZY_HANDLES: usize = 32;

/// Per-thread cache handle held by [`LazyScan`] so concurrent scans avoid contending on one `Arc`.
#[repr(align(128))]
pub(super) struct LazyHandle {
    cache: Weak<RangeCache>,
    hits: HitStats,
}

impl LazyHandle {
    pub(super) fn all(cache: &Weak<RangeCache>, hits: &HitStats) -> Box<[Arc<LazyHandle>]> {
        (0..LAZY_HANDLES)
            .map(|_| {
                Arc::new(LazyHandle {
                    cache: cache.clone(),
                    hits: hits.clone(),
                })
            })
            .collect()
    }

    pub(super) fn of(handles: &[Arc<LazyHandle>]) -> Arc<LazyHandle> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        thread_local! {
            static SLOT: usize = NEXT.fetch_add(1, Ordering::Relaxed);
        }
        let slot = SLOT.with(|slot| *slot);
        handles[slot % handles.len()].clone()
    }

    fn cache(&self) -> Result<Arc<RangeCache>, SlateDBError> {
        self.cache.upgrade().ok_or(SlateDBError::Closed)
    }
}

/// Next segment resolved by [`RangeCache::pin_next`].
pub(super) enum NextPart {
    /// Ready cached layers usable at the scan's sequence number.
    Cached(Arc<Layers>),
    /// Normal-path gap up to the returned end bound, with its fallback reason.
    Gap(Fallback, Bound<Bytes>),
}

/// Active segment of a [`LazyScan`]. Stores [`CachedScan`] inline to avoid an extra allocation.
#[allow(clippy::large_enum_variant)]
enum Current {
    None,
    Cached(CachedScan),
    /// Gap discovered during synchronous iteration, deferred until the next async poll.
    Pending {
        from: Bytes,
        end: Bound<Bytes>,
        reason: Fallback,
    },
    Gap(Box<DbIterator>),
}

/// Snapshot or transaction scan that pins cached ranges as iteration reaches them.
pub(crate) struct LazyScan {
    current: Current,
    part_start: Bytes,
    part_end: Bound<Bytes>,
    next: Option<Bytes>,
    scan_end: Bound<Bytes>,
    seq: u64,
    merge_operator: Option<MergeOperatorType>,
    handle: Arc<LazyHandle>,
    options: ScanOptions,
    prefix: Option<Bytes>,
    opened_gap: bool,
}

impl Drop for LazyScan {
    fn drop(&mut self) {
        self.handle.hits.count(if self.opened_gap {
            ReadCount::MixedScan
        } else {
            ReadCount::Scan
        });
    }
}

/// Returns the starting key after `end` if the scan continues past it.
fn after(end: &Bound<Bytes>, scan_end: &Bound<Bytes>) -> Option<Bytes> {
    match end {
        Excluded(key) if !past_end(scan_end, key) => Some(key.clone()),
        _ => None,
    }
}

impl LazyScan {
    /// Creates a lazy scan starting with the already pinned and validated `first` range layers.
    pub(super) fn new(
        first: Arc<Layers>,
        seq: u64,
        range: &BytesRange,
        merge_operator: Option<MergeOperatorType>,
        handle: Arc<LazyHandle>,
        options: ScanOptions,
        prefix: Option<Bytes>,
    ) -> io::Result<Self> {
        let scan_end = RangeBounds::end_bound(range).cloned();
        let part_end = min_end(&scan_end, &first.end);
        let next = after(&part_end, &scan_end);
        let part_start = match RangeBounds::start_bound(range) {
            Included(key) | Excluded(key) => key.clone(),
            Unbounded => Bytes::new(),
        };
        let scan = CachedScan::new(vec![first], seq, range, merge_operator.clone())?;
        Ok(Self {
            current: Current::Cached(scan),
            part_start,
            part_end,
            next,
            scan_end,
            seq,
            merge_operator,
            handle,
            options,
            prefix,
            opened_gap: false,
        })
    }

    /// Pins the next segment, opening it immediately if cached or deferring if it is a gap.
    fn advance(&mut self) -> Result<bool, SlateDBError> {
        let Some(from) = self.next.take() else {
            return Ok(false);
        };
        let next = self
            .handle
            .cache()?
            .pin_next(&from, &self.scan_end, self.seq);
        match next {
            NextPart::Cached(layers) => {
                let part_end = min_end(&self.scan_end, &layers.end);
                let range = BytesRange::new(Included(from.clone()), self.scan_end.clone());
                let mut scan =
                    CachedScan::new(vec![layers], self.seq, &range, self.merge_operator.clone())?;
                scan.set_point(false);
                self.next = after(&part_end, &self.scan_end);
                self.part_start = from;
                self.part_end = part_end;
                self.current = Current::Cached(scan);
            }
            NextPart::Gap(reason, end) => {
                self.current = Current::Pending { from, end, reason };
            }
        }
        Ok(true)
    }

    /// Opens a normal-path iterator for `[from, end)` at `self.seq` after verifying retention.
    async fn open_gap(
        &mut self,
        from: Bytes,
        end: Bound<Bytes>,
        reason: Fallback,
    ) -> Result<(), SlateDBError> {
        let next = after(&end, &self.scan_end);
        let Some(range) = BytesRange::try_new(Included(from.clone()), end.clone()) else {
            self.next = next;
            return Ok(());
        };
        let cache = self.handle.cache()?;
        let Some(gaps) = cache.gaps.get() else {
            return Err(
                io::Error::other("range cache: a lazily pinned scan without a gap reader").into(),
            );
        };
        let sampled = if matches!(reason, Fallback::Uncovered | Fallback::Loading) {
            cache.sample()
        } else {
            None
        };
        cache.hits.count(ReadCount::ScanGap);
        cache.count_fallback(FallbackOp::Gap, reason);
        self.opened_gap = true;
        let view = gaps.state.read().view();
        if !gaps.holds(&view, self.seq) {
            return Err(io::Error::other(format!(
                "range cache: the snapshot or transaction of a scan at {} ended while the scan \
                 read on, and a flush or compaction since may have dropped versions that it sees",
                self.seq
            ))
            .into());
        }
        let iter = gaps
            .reader
            .scan_with_options(
                range,
                &self.options,
                ScanContext {
                    db_state: &view,
                    write_batch_iter: None,
                    max_seq: Some(self.seq),
                    prefix: self.prefix.clone(),
                },
            )
            .await?;
        if let Some(sampled) = sampled {
            cache.record(sampled, &from);
        }
        self.next = next;
        self.part_start = from;
        self.part_end = end;
        self.current = Current::Gap(Box::new(iter));
        Ok(())
    }
}

#[async_trait]
impl RowEntryIterator for LazyScan {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        loop {
            match &mut self.current {
                Current::Cached(scan) => {
                    if let Some(entry) = scan.next().await? {
                        return Ok(Some(entry));
                    }
                    self.current = Current::None;
                }
                Current::Gap(iter) => {
                    if let Some(entry) = iter.next_entry().await? {
                        return Ok(Some(entry));
                    }
                    self.current = Current::None;
                }
                Current::Pending { .. } => {
                    let Current::Pending { from, end, reason } =
                        std::mem::replace(&mut self.current, Current::None)
                    else {
                        unreachable!("the part was pending");
                    };
                    self.open_gap(from, end, reason).await?;
                }
                Current::None => {
                    if !self.advance()? {
                        return Ok(None);
                    }
                }
            }
        }
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        let within = !past_end(&self.part_end, next_key);
        match &mut self.current {
            Current::Cached(scan) if within => {
                if next_key > self.part_start.as_ref() {
                    scan.seek(next_key).await?;
                }
                Ok(())
            }
            Current::Gap(iter) if within => {
                if next_key > self.part_start.as_ref() {
                    iter.seek_entry(next_key).await?;
                }
                Ok(())
            }
            Current::Pending { from, end, .. } if !past_end(end, next_key) => {
                if next_key > from.as_ref() {
                    *from = Bytes::copy_from_slice(next_key);
                }
                Ok(())
            }
            Current::None => {
                if let Some(from) = &self.next {
                    if next_key > from.as_ref() {
                        self.next = (!past_end(&self.scan_end, next_key))
                            .then(|| Bytes::copy_from_slice(next_key));
                    }
                }
                Ok(())
            }
            _ => {
                self.current = Current::None;
                self.next =
                    (!past_end(&self.scan_end, next_key)).then(|| Bytes::copy_from_slice(next_key));
                Ok(())
            }
        }
    }

    fn next_sync(&mut self) -> Option<Result<Option<RowEntry>, SlateDBError>> {
        loop {
            match &mut self.current {
                Current::Cached(scan) => match scan.next_sync()? {
                    Ok(Some(entry)) => return Some(Ok(Some(entry))),
                    Ok(None) => self.current = Current::None,
                    Err(e) => return Some(Err(e)),
                },
                Current::Gap(_) | Current::Pending { .. } => return None,
                Current::None => match self.advance() {
                    Ok(true) => continue,
                    Ok(false) => return Some(Ok(None)),
                    Err(e) => return Some(Err(e)),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound::{self, Excluded, Included, Unbounded};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use bytes::Bytes;
    use fail_parallel::FailPointRegistry;
    use proptest::prelude::*;
    use slatedb_common::metrics::{
        lookup_metric, lookup_metric_with_labels, DefaultMetricsRecorder,
    };

    use super::GapReader;
    use crate::config::{FlushOptions, FlushType, ScanOptions, TracingOptions};
    use crate::db::Db;
    use crate::db_iter::DbIterator;
    use crate::merge_operator::{MergeOperator, MergeOperatorError};
    use crate::object_store::memory::InMemory;
    use crate::oracle::Oracle;
    use crate::range_cache::stats::{
        FALLBACK_COUNT, LAZY_PIN_COUNT, MIXED_SCAN_COUNT, SCAN_COUNT, UNUSABLE_COUNT,
    };
    use crate::range_cache::{CachedRange, RangeCache, RangeCacheOptions};
    use crate::reader::DbStateReader;
    use crate::transaction_manager::IsolationLevel;
    use crate::types::KeyValue;

    const PROMPT: Duration = Duration::from_secs(20);

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
        _dir: tempfile::TempDir,
    }

    impl TestDb {
        fn cache(&self) -> &Arc<RangeCache> {
            self.db.inner.range_cache.as_ref().unwrap()
        }

        fn range(&self, start: usize) -> Arc<CachedRange> {
            self.cache().index.load().get(&key(start)).unwrap().clone()
        }

        fn starts(&self) -> Vec<usize> {
            self.cache()
                .index
                .load()
                .keys()
                .map(|k| {
                    std::str::from_utf8(k)
                        .unwrap()
                        .trim_start_matches("key")
                        .parse()
                        .unwrap_or(0)
                })
                .collect()
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

        fn pins(&self) -> i64 {
            self.metric(LAZY_PIN_COUNT, &[])
        }

        fn gaps(&self, reason: &str) -> i64 {
            self.metric(FALLBACK_COUNT, &[("op", "gap"), ("reason", reason)])
        }

        async fn split(&self, start: usize, cuts: &[usize]) -> bool {
            let cuts = cuts.iter().map(|i| key(*i)).collect();
            self.cache().split(&self.range(start), cuts).await.unwrap()
        }

        async fn quiet_merge(&self, left: usize, right: usize) -> bool {
            self.cache()
                .quiet_merge(&self.range(left), &self.range(right))
                .await
        }

        async fn remove(&self, k: usize) {
            self.db.uncache_range(key(k)..key(k + 1)).unwrap();
            self.db.wait_for_range_cache().await.unwrap();
        }

        async fn merge_above(&self, start: usize, a: usize, b: usize) -> u64 {
            let range = self.range(start);
            let mut i = 0;
            while range.layers.load().runs.is_empty() {
                let layers = range.layers.load_full();
                if layers.sealed.is_empty() {
                    let k = a + i % (b - a);
                    self.db.put(key(k), value(k, 100 + i)).await.unwrap();
                    i += 1;
                    if !layers.active.is_empty() && i % 64 == 0 {
                        self.cache().request_seal(&range, &layers.active);
                    }
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
                assert!(i < 100_000, "the range never flushed");
            }
            let layers = range.layers.load_full();
            self.cache()
                .major_merge(&range, layers.chunks.clone(), layers.runs.clone())
                .await
                .unwrap();
            let base = self.range(start).layers.load().chunks[0].base_seq;
            assert_eq!(base, layers.runs[0].hi, "the major merge did not install");
            base
        }
    }

    impl Drop for TestDb {
        fn drop(&mut self) {
            for name in [
                "range-cache-pin",
                "range-cache-lazy-pin",
                "range-cache-load",
            ] {
                let _ = fail_parallel::cfg(self.fp_registry.clone(), name, "off");
            }
        }
    }

    async fn open(tune: impl FnOnce(&mut RangeCacheOptions)) -> TestDb {
        let dir = tempfile::tempdir().unwrap();
        let mut options = RangeCacheOptions::new(dir.path());
        options.tuning.stats_fold_events = 1;
        options.tuning.min_rebuild_delta_bytes = usize::MAX;
        options.tuning.range_target_bytes = 0;
        tune(&mut options);
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let fp_registry = Arc::new(FailPointRegistry::new());
        let db = Db::builder("range_cache_lazy", Arc::new(InMemory::new()))
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
            _dir: dir,
        }
    }

    fn key(i: usize) -> Bytes {
        Bytes::from(format!("key{i:05}"))
    }

    fn value(i: usize, round: usize) -> Bytes {
        let mut v = format!("value-{i}-{round}-").into_bytes();
        v.resize(200, b'.');
        Bytes::from(v)
    }

    fn span(a: usize, b: usize) -> (Bound<Bytes>, Bound<Bytes>) {
        (Included(key(a)), Excluded(key(b)))
    }

    fn normal() -> ScanOptions {
        ScanOptions::default().with_tracing_options(Some(TracingOptions::new("normal")))
    }

    async fn put_all(t: &TestDb, a: usize, b: usize) {
        for i in a..b {
            t.db.put(key(i), value(i, 0)).await.unwrap();
        }
    }

    async fn mixed_writes(t: &TestDb, a: usize, b: usize, round: usize) {
        for i in a..b {
            match (i + round) % 4 {
                0 => {
                    t.db.put(key(i), value(i, round)).await.unwrap();
                }
                1 => {
                    t.db.delete(key(i)).await.unwrap();
                }
                2 => {
                    t.db.merge(key(i), format!("+{round}")).await.unwrap();
                }
                _ => {}
            }
        }
    }

    async fn rows(iter: &mut DbIterator, n: usize) -> Vec<KeyValue> {
        let mut out = Vec::new();
        while out.len() < n {
            match iter.next().await.unwrap() {
                Some(kv) => out.push(kv),
                None => break,
            }
        }
        out
    }

    /// Runs a cached scan and a normal-path scan in lockstep and asserts equality.
    struct Pair {
        cached: DbIterator,
        normal: DbIterator,
        last: Option<Bytes>,
        done: bool,
    }

    impl Pair {
        async fn new(reader: &crate::DbSnapshot, range: (Bound<Bytes>, Bound<Bytes>)) -> Self {
            Self {
                cached: reader.scan(range.clone()).await.unwrap(),
                normal: reader.scan_with_options(range, &normal()).await.unwrap(),
                last: None,
                done: false,
            }
        }

        async fn step(&mut self, n: usize, what: &str) {
            for _ in 0..n {
                if self.done {
                    return;
                }
                let cached = self.cached.next().await.unwrap();
                let normal = self.normal.next().await.unwrap();
                assert_eq!(cached, normal, "{what}");
                match cached {
                    Some(kv) => self.last = Some(kv.key),
                    None => self.done = true,
                }
            }
        }

        async fn seek(&mut self, to: Bytes) {
            if self.done || self.last.as_ref().is_some_and(|last| to <= *last) {
                return;
            }
            let (a, b) = (self.cached.seek(&to).await, self.normal.seek(&to).await);
            assert_eq!(a.is_ok(), b.is_ok(), "seek to {to:?}");
        }

        async fn finish(&mut self, what: &str) {
            self.step(usize::MAX, what).await;
        }
    }

    /// Snapshot and transaction scans pin each range on demand and read past the end as a gap;
    /// latest-state scans still pin all ranges up front.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_pin_each_range_when_a_snapshot_scan_reaches_it() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 450).await;
        for a in (0..400).step_by(100) {
            t.db.cache_range(span(a, a + 100)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        mixed_writes(&t, 0, 450, 1).await;

        let (pins, scans) = (t.pins(), t.metric(SCAN_COUNT, &[]));
        let mut pair = Pair::new(&snapshot, (Included(key(10)), Unbounded)).await;
        pair.step(10, "a short scan").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 1);
        assert_eq!(t.metric(SCAN_COUNT, &[]) - scans, 1);

        let (pins, mixed) = (t.pins(), t.metric(MIXED_SCAN_COUNT, &[]));
        let gaps = t.gaps("uncovered");
        let mut pair = Pair::new(&snapshot, (Included(key(10)), Unbounded)).await;
        pair.finish("a scan to the end").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 4);
        assert_eq!(t.gaps("uncovered") - gaps, 1);
        assert_eq!(t.metric(MIXED_SCAN_COUNT, &[]) - mixed, 1);

        let pins = t.pins();
        let mut pair = Pair::new(&snapshot, (Included(key(10)), Excluded(key(400)))).await;
        pair.step(3, "before the seek").await;
        pair.seek(key(350)).await;
        pair.finish("after the seek").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 2);

        let txn = t.db.begin(IsolationLevel::Snapshot).unwrap();
        let (pins, scans) = (t.pins(), t.metric(SCAN_COUNT, &[]));
        let got = rows(&mut txn.scan(key(150)..key(350)).await.unwrap(), usize::MAX).await;
        let want = rows(
            &mut txn
                .scan_with_options(key(150)..key(350), &normal())
                .await
                .unwrap(),
            usize::MAX,
        )
        .await;
        assert_eq!(got, want);
        assert_eq!(t.pins() - pins, 3);
        assert_eq!(t.metric(SCAN_COUNT, &[]) - scans, 1);

        let (pins, scans) = (t.pins(), t.metric(SCAN_COUNT, &[]));
        let got = rows(&mut t.db.scan(key(10)..key(390)).await.unwrap(), usize::MAX).await;
        assert_eq!(
            got.len(),
            rows(
                &mut t
                    .db
                    .scan_with_options(key(10)..key(390), &normal())
                    .await
                    .unwrap(),
                usize::MAX
            )
            .await
            .len()
        );
        assert_eq!(t.pins() - pins, 0);
        assert_eq!(t.metric(SCAN_COUNT, &[]) - scans, 1);
        t.db.close().await.unwrap();
    }

    /// A range major-merged above the snapshot before a lazy scan reaches it falls back to a gap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_read_a_range_merged_above_the_snapshot_as_a_gap() {
        let t = open(|o| o.tuning.range_seal_bytes = 4 * 1024).await;
        put_all(&t, 0, 300).await;
        for a in (0..300).step_by(100) {
            t.db.cache_range(span(a, a + 100)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        let mut pair = Pair::new(&snapshot, (Included(key(50)), Excluded(key(300)))).await;
        pair.step(5, "before the merge").await;
        let base = t.merge_above(100, 100, 200).await;
        assert!(base > snapshot.seq());
        let (gaps, unusable, pins) = (
            t.gaps("too_old"),
            t.metric(
                UNUSABLE_COUNT,
                &[("op", "scan_part"), ("reason", "too_old")],
            ),
            t.pins(),
        );
        pair.finish("after the merge").await;
        drop(pair);
        assert_eq!(t.gaps("too_old") - gaps, 1);
        assert_eq!(
            t.metric(
                UNUSABLE_COUNT,
                &[("op", "scan_part"), ("reason", "too_old")]
            ) - unusable,
            1
        );
        assert_eq!(t.pins() - pins, 1);
        t.db.close().await.unwrap();
    }

    /// A range split ahead of a lazy snapshot scan is read through its children without gaps.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_pin_the_children_of_a_range_split_after_the_scan_began() {
        let t = open(|o| o.tuning.range_seal_bytes = 4 * 1024).await;
        put_all(&t, 0, 300).await;
        for a in (0..300).step_by(100) {
            t.db.cache_range(span(a, a + 100)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        mixed_writes(&t, 0, 300, 1).await;
        let mut pair = Pair::new(&snapshot, (Included(key(50)), Excluded(key(300)))).await;
        pair.step(5, "before the split").await;
        assert!(t.split(100, &[130, 170]).await);
        assert_eq!(t.starts(), vec![0, 100, 130, 170, 200]);
        let (pins, gaps) = (
            t.pins(),
            t.gaps("too_old") + t.gaps("uncovered") + t.gaps("sealed"),
        );
        pair.finish("after the split").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 4);
        assert_eq!(
            t.gaps("too_old") + t.gaps("uncovered") + t.gaps("sealed"),
            gaps
        );
        t.db.close().await.unwrap();
    }

    /// A range removed ahead of a lazy snapshot scan is read as a gap, followed by cached ranges.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_read_a_range_removed_after_the_scan_began_as_a_gap() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 300).await;
        for a in (0..300).step_by(100) {
            t.db.cache_range(span(a, a + 100)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        mixed_writes(&t, 0, 300, 1).await;
        let mut pair = Pair::new(&snapshot, (Included(key(50)), Excluded(key(300)))).await;
        pair.step(5, "before the removal").await;
        t.remove(150).await;
        let (pins, gaps) = (t.pins(), t.gaps("uncovered"));
        pair.finish("after the removal").await;
        drop(pair);
        assert_eq!(t.gaps("uncovered") - gaps, 1);
        assert_eq!(t.pins() - pins, 1);
        t.db.close().await.unwrap();
    }

    /// Ranges quietly merged ahead of a lazy scan are read as a gap if merged above the snapshot,
    /// or pinned directly if merged at the snapshot sequence number.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_read_ranges_merged_quietly_after_the_scan_began() {
        for above in [true, false] {
            let t = open(|_| {}).await;
            put_all(&t, 0, 300).await;
            t.db.cache_range(span(0, 100)).await.unwrap();
            t.db.cache_range(span(100, 300)).await.unwrap();
            assert!(t.split(100, &[200]).await);
            let snapshot = t.db.snapshot().unwrap();
            let mut pair = Pair::new(&snapshot, (Included(key(50)), Excluded(key(300)))).await;
            pair.step(5, "before the merge").await;
            if above {
                t.db.put(key(5), value(5, 1)).await.unwrap();
            }
            assert!(t.quiet_merge(100, 200).await);
            let merged = t.range(100);
            assert_eq!(
                merged.layers.load().chunks[0].base_seq > snapshot.seq(),
                above
            );
            let (pins, gaps) = (t.pins(), t.gaps("too_old"));
            pair.finish("after the merge").await;
            drop(pair);
            assert_eq!(t.gaps("too_old") - gaps, i64::from(above), "above: {above}");
            assert_eq!(t.pins() - pins, i64::from(!above), "above: {above}");
            t.db.close().await.unwrap();
        }
    }

    /// A snapshot or transaction scan iterator retains its sequence number until dropped, keeping
    /// both lazily pinned cached parts and normal-path gaps exact across flushes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_read_at_its_sequence_number_after_its_snapshot_or_transaction_ended() {
        for case in [
            "snapshot dropped",
            "snapshot dropped, no flush",
            "older snapshot held",
            "transaction committed",
            "transaction rolled back",
            "transaction dropped",
        ] {
            let t = open(|_| {}).await;
            put_all(&t, 0, 200).await;
            t.db.cache_range(span(0, 100)).await.unwrap();
            t.db.cache_range(span(100, 150)).await.unwrap();
            let older = (case == "older snapshot held").then(|| t.db.snapshot().unwrap());
            if older.is_some() {
                t.db.put(key(5), value(5, 1)).await.unwrap();
            }
            let transaction = case.starts_with("transaction");
            let (snapshot, txn) = if transaction {
                (None, Some(t.db.begin(IsolationLevel::Snapshot).unwrap()))
            } else {
                (Some(t.db.snapshot().unwrap()), None)
            };
            let seq = match (&snapshot, &txn) {
                (Some(snapshot), _) => snapshot.seq(),
                (_, Some(txn)) => txn.seqnum(),
                _ => unreachable!(),
            };
            let mut iter = match (&snapshot, &txn) {
                (Some(snapshot), _) => snapshot.scan(key(50)..).await.unwrap(),
                (_, Some(txn)) => txn.scan(key(50)..).await.unwrap(),
                _ => unreachable!(),
            };
            let mut got = rows(&mut iter, 5).await;
            let registered = || match transaction {
                true => t.db.inner.txn_manager.min_active_seq(),
                false => t.db.inner.snapshot_manager.min_active_seq(),
            };
            let after_end = older.as_ref().map(|older| older.seq());
            match case {
                "transaction committed" => {
                    let txn = txn.unwrap();
                    txn.put(key(300), value(300, 1)).unwrap();
                    txn.commit().await.unwrap();
                }
                "transaction rolled back" => txn.unwrap().rollback(),
                _ => drop((snapshot, txn)),
            }
            assert_eq!(registered(), Some(after_end.unwrap_or(seq)), "{case}");
            t.db.put(key(120), value(120, 2)).await.unwrap();
            t.db.put(key(170), value(170, 2)).await.unwrap();
            let last_l0_seq = || t.db.inner.state.read().state().core().last_l0_seq;
            if case.ends_with("no flush") {
                assert!(last_l0_seq() <= seq, "{case}");
            } else {
                t.db.flush_with_options(FlushOptions {
                    flush_type: FlushType::MemTable,
                })
                .await
                .unwrap();
                assert!(last_l0_seq() > seq, "{case}");
            }
            let (pins, gaps) = (t.pins(), t.gaps("uncovered"));
            got.extend(rows(&mut iter, usize::MAX).await);
            let got: Vec<(Bytes, Bytes)> = got.into_iter().map(|kv| (kv.key, kv.value)).collect();
            let want: Vec<(Bytes, Bytes)> = (50..200).map(|i| (key(i), value(i, 0))).collect();
            assert_eq!(got, want, "{case}");
            assert_eq!(
                (t.pins() - pins, t.gaps("uncovered") - gaps),
                (1, 1),
                "{case}"
            );
            assert_eq!(registered(), Some(after_end.unwrap_or(seq)), "{case}");
            drop(iter);
            assert_eq!(registered(), after_end, "{case}");
            drop(older);
            t.db.close().await.unwrap();
        }
    }

    /// Verifies [`GapReader::holds`] under both rules (`last_l0_seq <= seq` and active snapshot or
    /// transaction registration at or below `seq`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_check_that_a_gap_view_holds_what_its_sequence_number_sees() {
        let t = open(|_| {}).await;
        let inner = &t.db.inner;
        let gaps = GapReader {
            reader: inner.reader.clone(),
            state: inner.state.clone(),
            snapshots: inner.snapshot_manager.clone(),
            transactions: inner.txn_manager.clone(),
        };
        let flush = || async {
            t.db.flush_with_options(FlushOptions {
                flush_type: FlushType::MemTable,
            })
            .await
            .unwrap();
        };
        let view = || inner.state.read().view();
        put_all(&t, 0, 10).await;
        flush().await;
        let seq = t.committed();
        assert!(gaps.holds(&view(), seq));
        put_all(&t, 0, 10).await;
        flush().await;
        assert!(view().core().last_l0_seq > seq);
        assert!(!gaps.holds(&view(), seq));
        let (id, at) = inner.snapshot_manager.new_snapshot(Some(seq));
        assert_eq!(at, seq);
        assert!(gaps.holds(&view(), seq));
        inner.snapshot_manager.drop_snapshot(&id);
        assert!(!gaps.holds(&view(), seq));
        let txn = t.db.begin(IsolationLevel::Snapshot).unwrap();
        let txn_seq = txn.seqnum();
        put_all(&t, 0, 10).await;
        flush().await;
        assert!(view().core().last_l0_seq > txn_seq);
        assert!(gaps.holds(&view(), txn_seq));
        assert!(gaps.holds(&view(), txn_seq + 1));
        assert!(!gaps.holds(&view(), txn_seq - 1));
        drop(txn);
        assert!(!gaps.holds(&view(), txn_seq));
        t.db.close().await.unwrap();
    }

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Value,
        Tombstone,
        Merge,
    }

    fn kind() -> impl Strategy<Value = Kind> {
        prop_oneof![4 => Just(Kind::Value), 2 => Just(Kind::Tombstone), 2 => Just(Kind::Merge)]
    }

    #[derive(Clone, Debug)]
    enum Event {
        Split(u8, u8),
        Merge(u8),
        Remove(u8),
        Major(u8),
        Write(u8, Kind),
        Read(u8, u8),
        Seek(u8, u8),
    }

    fn event() -> impl Strategy<Value = Event> {
        prop_oneof![
            2 => (any::<u8>(), 0u8..80).prop_map(|(i, k)| Event::Split(i, k)),
            2 => any::<u8>().prop_map(Event::Merge),
            1 => (0u8..80).prop_map(Event::Remove),
            2 => any::<u8>().prop_map(Event::Major),
            3 => (0u8..90, kind()).prop_map(|(k, kind)| Event::Write(k, kind)),
            3 => (any::<u8>(), 0u8..20).prop_map(|(i, n)| Event::Read(i, n)),
            1 => (any::<u8>(), 0u8..95).prop_map(|(i, k)| Event::Seek(i, k)),
        ]
    }

    fn pkey(k: u8) -> Bytes {
        Bytes::from(format!("k{k:03}"))
    }

    async fn write(t: &TestDb, k: u8, kind: Kind, seq: usize) {
        match kind {
            Kind::Value => t.db.put(pkey(k), format!("{k}@{seq}")).await.unwrap(),
            Kind::Tombstone => t.db.delete(pkey(k)).await.unwrap(),
            Kind::Merge => t.db.merge(pkey(k), format!("+{seq}")).await.unwrap(),
        };
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// Lazily pinned snapshot scans match the normal path while ranges ahead split, merge
        /// quietly, are removed, or major-merge above the snapshot, interleaved with writes and
        /// seeks.
        #[test]
        fn should_read_lazily_pinned_parts_as_the_normal_path(
            before in proptest::collection::vec((0u8..90, kind()), 0..150),
            cuts in proptest::collection::btree_set(1u8..60, 0..4),
            scans in proptest::collection::vec(
                (0u8..90, proptest::option::of(0u8..95), 0u8..8),
                1..3,
            ),
            events in proptest::collection::vec(event(), 1..10),
        ) {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let t = open(|o| o.tuning.range_seal_bytes = 1024).await;
                for (i, (k, kind)) in before.iter().enumerate() {
                    write(&t, *k, *kind, i).await;
                }
                t.db.cache_range(pkey(0)..pkey(60)).await.unwrap();
                t.db.cache_range(pkey(60)..pkey(80)).await.unwrap();
                if !cuts.is_empty() {
                    let range = t.cache().index.load().get(&pkey(0)).unwrap().clone();
                    let cuts = cuts.iter().map(|k| pkey(*k)).collect();
                    t.cache().split(&range, cuts).await.unwrap();
                }
                let snapshot = t.db.snapshot().unwrap();
                let mut pairs = Vec::new();
                for (start, end, n) in &scans {
                    let end = match end {
                        Some(end) if end > start => Excluded(pkey(*end)),
                        _ => Unbounded,
                    };
                    let mut pair = Pair::new(&snapshot, (Included(pkey(*start)), end)).await;
                    pair.step(*n as usize, "before the events").await;
                    pairs.push(pair);
                }
                let ranges = || -> Vec<Arc<CachedRange>> {
                    t.cache().index.load().values().cloned().collect()
                };
                for (seq, event) in events.iter().enumerate() {
                    match event {
                        Event::Split(i, k) => {
                            let all = ranges();
                            let Some(range) = all.get(*i as usize % all.len().max(1)) else {
                                continue;
                            };
                            if range.contains(&pkey(*k))
                                && range.start != pkey(*k)
                                && !range.layers.load().chunks.loading()
                            {
                                t.cache().split(range, vec![pkey(*k)]).await.unwrap();
                            }
                        }
                        Event::Merge(i) => {
                            let all = ranges();
                            if all.len() >= 2 {
                                let at = *i as usize % (all.len() - 1);
                                t.cache().quiet_merge(&all[at], &all[at + 1]).await;
                            }
                        }
                        Event::Remove(k) => {
                            t.db.uncache_range(pkey(*k)..pkey(*k + 1)).unwrap();
                            t.db.wait_for_range_cache().await.unwrap();
                        }
                        Event::Major(i) => {
                            let all = ranges();
                            let Some(range) = all.get(*i as usize % all.len().max(1)) else {
                                continue;
                            };
                            if range.layers.load().chunks.loading() {
                                continue;
                            }
                            let keys: Vec<u8> =
                                (0u8..90).filter(|k| range.contains(&pkey(*k))).collect();
                            for (j, k) in keys.iter().take(6).enumerate() {
                                write(&t, *k, Kind::Value, 2000 + 10 * seq + j).await;
                            }
                            let layers = range.layers.load_full();
                            if !layers.active.is_empty() {
                                t.cache().request_seal(range, &layers.active);
                            }
                            let deadline = Instant::now() + PROMPT;
                            while {
                                let l = range.layers.load();
                                !l.sealed.is_empty() || (!keys.is_empty() && l.runs.is_empty())
                            } {
                                assert!(Instant::now() < deadline, "the flush never came");
                                tokio::time::sleep(Duration::from_millis(1)).await;
                            }
                            let layers = range.layers.load_full();
                            if !layers.runs.is_empty()
                                && !range.jobs.merging()
                                && layers.runs.len() < t.cache().options.tuning.max_runs
                            {
                                t.cache()
                                    .major_merge(range, layers.chunks.clone(), layers.runs.clone())
                                    .await
                                    .unwrap();
                            }
                        }
                        Event::Write(k, kind) => write(&t, *k, *kind, 1000 + seq).await,
                        Event::Read(i, n) => {
                            let at = *i as usize % pairs.len();
                            pairs[at].step(*n as usize, "between the events").await;
                        }
                        Event::Seek(i, k) => {
                            let at = *i as usize % pairs.len();
                            pairs[at].seek(pkey(*k)).await;
                        }
                    }
                }
                for pair in &mut pairs {
                    pair.finish("after the events").await;
                }
                drop(pairs);
                t.db.close().await.unwrap();
            });
        }
    }
}
