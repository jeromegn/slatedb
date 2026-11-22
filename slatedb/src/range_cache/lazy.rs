//! Scans in a snapshot or a transaction that pin each range when they reach it: *lazy pins*
//! (design 04 section 3.7, I16).
//!
//! A scan at the latest state pins every cached range that it can reach, up to its end bound,
//! before it loads its sequence number (see [`RangeCache::pin_scan`]): an open-ended scan pins
//! every range to the end of the key space, however few rows it reads. A scan at a *registered*
//! sequence number `S`, a snapshot's or a transaction's, pins only the range that holds its first
//! key up front. When it reaches the end of a part, it loads the index as it is then and pins the
//! part that starts there ([`RangeCache::pin_next`]):
//!
//! - the range that holds the part's first key, found by key in the current index: the range
//!   that the scan would have pinned, or the children of a split that replaced it, or a range
//!   that a quiet merge made, which then holds the key in its middle. The part ends where that
//!   range ends, or where the scan does;
//! - a *gap*, read through the normal path at `S` up to the next range's start (or the range's
//!   end, or the scan's), when no range holds the key (it was removed, or never cached), when the
//!   range loads, or when it fails the sequence number rules at `S`: a chunk with a `base_seq`
//!   above `S` (a major merge, or a quiet merge, after `S`; with no history, design 04 section 5
//!   was not built), or an active buffer sealed below `S` at every attempt.
//!
//! **Why a range pinned after `S` was loaded is complete for `S`.** The writer appends every
//! version of a batch to the active buffer of the range that holds its key before the batch
//! commits (design 01 I2; `RangeCache::apply_write` runs before the oracle's commit), so every
//! version at or below `S` of a cached key was appended before the scan loaded `S`, which was at
//! most the last committed number then. A layer set changes only in a store by the writer, and
//! each store keeps every version of the range's keys above the oldest chunk's `base_seq` or
//! raises that `base_seq`: a seal moves the active buffer to the sealed list, and stores
//! `sealed_at` on it first; a flush moves sealed buffers into a run, losslessly (see the `merge`
//! module); a major merge folds the oldest runs into a base at their newest `hi`; a split gives
//! each child the parent's sources at `W`, its active buffer sealed at `W` first (I13); a quiet
//! merge seals both active buffers at `W`, which hold no version, and labels the chunks `W`
//! (I14); an attach keeps the buffers that hold the versions above its `base_seq`, which is the
//! registration's `W`. So the layer set `L` that a lazy pin loads holds, for each chunk `c` that
//! the part meets, every version of `c`'s keys up to `c.base_seq` in `c`'s file, the versions in
//! `(c.base_seq, covered_to]` in runs (I4c), the ones in `(covered_to, active.lo]` in sealed
//! buffers, and every version above `active.lo` that the writer appended before it sealed `L`'s
//! active buffer. The pin checks `c.base_seq <= S` for each such chunk and then loads `sealed_at`
//! (SeqCst, as the writer stores it): if the buffer was not sealed, or was sealed at or above
//! `S`, it holds every version in `(active.lo, S]`, which were all appended before the load. A
//! seal below `S` cannot reach a pin that loaded the index after `S`: the writer seals at a
//! boundary `W` and stores what replaces `L` in the same step, before any batch above `W` commits,
//! so before `S > W` was committed. The pin still checks, and tries again with a new index as a
//! point read does (01 X1, X6); the checker's runs counted no such gap. Nothing in the range
//! cache drops a version that a later version at or below a registered `S` shadows (flushes are
//! lossless and a major merge's base is labelled with its sequence number), so the cached parts
//! do not depend on the registration; the gaps do.
//!
//! **Gaps.** A gap reads through the normal path from a state view `V` captured when the scan opens
//! it, with `max_seq = S`. That is exact while `S` stays registered: flush and compaction keep the
//! versions that a registered snapshot sees (the rule in `DbInner::range_cache_scan`). SlateDB's
//! API lets a caller drop the snapshot or the transaction while its scan reads on, and after that
//! a memtable flush or a compaction may drop a version that `S` sees: a flush takes its retention
//! boundary from the registrations live at its upload (`DbInner::compute_flush_retention_seq`),
//! and a compaction from the manifest's `recent_snapshot_min_seq`. The scan cannot keep the
//! registration itself without a lock per scan, so after it captures `V` it checks that one of
//! two things holds ([`GapReader::holds`]), and otherwise the gap fails with an error rather than
//! answer from a state that may lack a version:
//!
//! 1. `V`'s `last_l0_seq` is at or below `S`. The writer sets it in the same hold of the state's
//!    lock that adds the L0 (`apply_uploaded_state`), and flushes and compactions only write
//!    versions at or below it, so every version in an SST of `V` is at or below `S`: a version
//!    that one dropped was shadowed by a newer one at or below `S`, which `S` sees in its place.
//!    Memtables keep every version.
//! 2. A snapshot or a transaction at `X <= S` is registered when the check runs, after `V` was
//!    captured. A new registration takes the last committed number (`new_snapshot(None)`,
//!    `new_transaction`), so it was made when nothing above `X` was committed: every flush and
//!    compaction before it wrote only versions at or below `X`, and every one while it was live,
//!    up to `V`, kept every version above a boundary at or below `X` and the newest one at or
//!    below that boundary. Either way, a dropped version was shadowed by one at or below `S`. For
//!    `X < S` that registration was made before `S` was committed; for `X == S` it may have been
//!    made after the scan's own registration ended, but while the last committed number was
//!    still `S`.
//!
//! The manifest's `recent_snapshot_min_seq` is not enough: the writer updates it in a later hold
//! of the lock than the L0's, so a view captured between the two holds an L0 flushed with no
//! registration and the old value. Blobtree keeps its transaction for as long as its scan cursor.
//!
//! **Cost and counts.** Nothing on the hit path of a scan that ends in its first range, past the
//! reference count of its thread's handle of the cache, which it holds for its later parts (see
//! [`LazyHandle`]). Each range that the scan pins, the first one included, counts as a read of
//! the range on the scan's thread (`lazy_pin_count`, and the range's scans): so the per-range read
//! counts see every range that a scan reads (design 04 section 3.7's last paragraph). The scan
//! itself counts when it ends, as a scan that the cache served whole (`scan_count`) or, if it
//! opened a gap, in part (`mixed_scan_count`), as a mixed scan planned up front counts.

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

/// What a lazily pinned scan needs to read a gap through the normal path, and to check that the
/// gap's state view holds what the scan's sequence number sees (see the module documentation).
/// The database binds it once, at its first lazily pinned scan (see `DbInner::range_cache_scan`).
pub(crate) struct GapReader {
    pub(crate) reader: Reader,
    pub(crate) state: Arc<RwLock<DbState>>,
    pub(crate) snapshots: Arc<SnapshotManager>,
    pub(crate) transactions: Arc<TransactionManager>,
}

impl GapReader {
    /// True if `view`, captured before this call, holds every version that a read at `seq`, the
    /// sequence number of a lazily pinned scan, sees: if no SST of the view holds a version above
    /// `seq`, or if a snapshot or a transaction at or below `seq` is registered now. See the
    /// module documentation for why each one is enough. The registrations are read after the view
    /// was captured, with two read locks: on the gap path only, never on a hit.
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

/// The handles of a cache that lazily pinned scans hold (see [`LazyHandle`]).
const LAZY_HANDLES: usize = 32;

/// What a lazily pinned scan holds of its cache, from when it starts until it drops: the cache,
/// to pin its next parts, and the cache's read bookkeeping, to count itself when it ends. Each
/// cache has [`LAZY_HANDLES`] of them, each on its own cache line, and a scan takes the handle of
/// its thread, so that concurrent scans change no shared reference count: with one `Arc` of the
/// cache for every scan, 16 tasks' scans over 107 ranges ran 6% below one range's (one run
/// each). The cache is weak: a handle lives as long as the cache, and only a scan that reaches
/// its next part takes a strong reference, for the pin.
#[repr(align(128))]
pub(super) struct LazyHandle {
    cache: Weak<RangeCache>,
    hits: HitStats,
}

impl LazyHandle {
    /// The handles of the cache that `cache` will hold, with its bookkeeping `hits`.
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

    /// The handle of this thread among `handles`.
    pub(super) fn of(handles: &[Arc<LazyHandle>]) -> Arc<LazyHandle> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        thread_local! {
            static SLOT: usize = NEXT.fetch_add(1, Ordering::Relaxed);
        }
        let slot = SLOT.with(|slot| *slot);
        handles[slot % handles.len()].clone()
    }

    /// The cache, unless it was dropped.
    fn cache(&self) -> Result<Arc<RangeCache>, SlateDBError> {
        self.cache.upgrade().ok_or(SlateDBError::Closed)
    }
}

/// The part of a scan at a registered sequence number that starts at a given key. See
/// [`RangeCache::pin_next`].
pub(super) enum NextPart {
    /// The pinned layers of the range that holds the key, usable at the scan's sequence number.
    /// The part ends where the range ends, or where the scan ends.
    Cached(Arc<Layers>),
    /// The normal read path, because of the reason, up to the end.
    Gap(Fallback, Bound<Bytes>),
}

/// The part that the scan reads now. The cached part is inline: the scan lives in the one
/// allocation that its iterator makes, and a box would cost every scan another one.
#[allow(clippy::large_enum_variant)]
enum Current {
    /// None: the next part starts at `LazyScan::next`, or the scan has ended.
    None,
    Cached(CachedScan),
    /// A gap from `from` to `end` that `next_sync` found and left for `next` to open: opening
    /// it needs an await.
    Pending {
        from: Bytes,
        end: Bound<Bytes>,
        reason: Fallback,
    },
    Gap(Box<DbIterator>),
}

/// A scan at a registered sequence number that pins each range when it reaches it. See the
/// module documentation.
pub(crate) struct LazyScan {
    current: Current,
    /// The first key of the current part.
    part_start: Bytes,
    /// The end of the current part, within the scan.
    part_end: Bound<Bytes>,
    /// The first key of the next part, where the current one ends, if the scan goes on past it.
    next: Option<Bytes>,
    scan_end: Bound<Bytes>,
    seq: u64,
    merge_operator: Option<MergeOperatorType>,
    handle: Arc<LazyHandle>,
    /// The gaps' read options and prefix: the scan's.
    options: ScanOptions,
    prefix: Option<Bytes>,
    /// True once the scan opened a gap: it then counts as a mixed scan.
    opened_gap: bool,
}

impl Drop for LazyScan {
    /// Counts the scan, on the thread that drops it: as a scan that the cache served whole if it
    /// opened no gap, else as one that it served in part. The ranges that it pinned counted as
    /// they did.
    fn drop(&mut self) {
        self.handle.hits.count(if self.opened_gap {
            ReadCount::MixedScan
        } else {
            ReadCount::Scan
        });
    }
}

/// The first key of the part after one that ends at `end`, in a scan that ends at `scan_end`:
/// `None` if the scan ends there.
fn after(end: &Bound<Bytes>, scan_end: &Bound<Bytes>) -> Option<Bytes> {
    match end {
        Excluded(key) if !past_end(scan_end, key) => Some(key.clone()),
        _ => None,
    }
}

impl LazyScan {
    /// A scan of `range` at `seq`, a registered sequence number, whose first part reads `first`,
    /// the layers of the range that holds the start of `range`, which the caller pinned and
    /// checked.
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

    /// Pins the next part, if there is one, and opens it if it is cached. Returns false after the
    /// last part.
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
                // The scan covers more than one key: a scan of one key holds in one range.
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

    /// Opens the gap from `from` to `end` through the normal path, at the scan's sequence number.
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
        // I16: the view must hold every version that `S` sees. See the module documentation.
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
                // A part that starts at or after `next_key` is there already: `next_sync` opened
                // it and stopped before its first row.
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
                // The next part starts at the later of its start and `next_key`: the parts before
                // it hold no row at or after `next_key` that the scan has not returned.
                if let Some(from) = &self.next {
                    if next_key > from.as_ref() {
                        self.next = (!past_end(&self.scan_end, next_key))
                            .then(|| Bytes::copy_from_slice(next_key));
                    }
                }
                Ok(())
            }
            _ => {
                // The current part ends before `next_key`: the next part starts there.
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
                // A gap reads through the normal path, which needs an await.
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

    use crate::config::{FlushOptions, FlushType, ScanOptions, TracingOptions};
    use crate::db::Db;
    use crate::db_iter::DbIterator;
    use crate::merge_operator::{MergeOperator, MergeOperatorError};
    use crate::object_store::memory::InMemory;
    use crate::oracle::Oracle;
    use crate::range_cache::maintain::Trigger;
    use crate::range_cache::stats::{
        FALLBACK_COUNT, LAZY_PIN_COUNT, MIXED_SCAN_COUNT, PIN_RETRY_COUNT, QUIET_MERGE_COUNT,
        QUIET_MERGE_REJECTED_COUNT, SCAN_COUNT, SPLIT_COUNT, UNUSABLE_COUNT,
    };
    use crate::range_cache::{CachedRange, Chunks, RangeCache, RangeCacheOptions, MAX_RUNS_RANGE};
    use crate::transaction_manager::IsolationLevel;
    use crate::types::KeyValue;

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
        _dir: tempfile::TempDir,
    }

    impl TestDb {
        fn cache(&self) -> &Arc<RangeCache> {
            self.db.inner.range_cache.as_ref().unwrap()
        }

        fn range(&self, start: usize) -> Arc<CachedRange> {
            self.cache().index.load().get(&key(start)).unwrap().clone()
        }

        /// The ranges by start, keys as numbers (the first range may start at the empty key: 0).
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

        fn fail(&self, name: &str, action: &str) {
            fail_parallel::cfg(self.fp_registry.clone(), name, action).unwrap();
        }

        async fn wait(&self, what: &str, done: impl Fn(&Self) -> bool) {
            let deadline = Instant::now() + PROMPT;
            while !done(self) {
                assert!(Instant::now() < deadline, "timed out waiting for {what}");
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }

        /// Splits the range that starts at `start` at `cuts` as the maintenance task does.
        async fn split(&self, start: usize, cuts: &[usize]) -> bool {
            let cuts = cuts.iter().map(|i| key(*i)).collect();
            self.cache().split(&self.range(start), cuts).await.unwrap()
        }

        /// Merges the ranges that start at `left` and `right` as the maintenance task asks.
        async fn quiet_merge(&self, left: usize, right: usize) -> bool {
            self.cache()
                .quiet_merge(&self.range(left), &self.range(right))
                .await
        }

        /// Removes the range that holds `k`, and waits for the writer to apply it.
        async fn remove(&self, k: usize) {
            self.db.uncache_range(key(k)..key(k + 1)).unwrap();
            self.db.wait_for_range_cache().await.unwrap();
        }

        /// Writes rows of `[a, b)` (keys of the range that starts at `start`) until they seal
        /// and flush, then merges the range's runs into a new base, above every earlier
        /// sequence number. Returns the new base's `base_seq`.
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
                .major_merge(
                    &range,
                    layers.chunks.clone(),
                    layers.runs.clone(),
                    Trigger::Bytes,
                )
                .await
                .unwrap();
            let base = self.range(start).layers.load().chunks[0].base_seq;
            assert_eq!(base, layers.runs[0].hi, "the major merge did not install");
            base
        }
    }

    impl Drop for TestDb {
        /// A test that fails while a fail point holds tasks must end, not wait for them.
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
        // No major merge, split or quiet merge unless a test asks for one.
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

    /// A 200-byte value: a 4 KiB page holds about 15 rows.
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

    /// Overwrites, deletes and merge operands in `[a, b)`.
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

    /// A scan through the cache and the same scan through the normal path, read in lockstep.
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

    /// Design 04 section 3.7: a snapshot scan pins the range it starts in and each next range
    /// when it reaches it, and reads past the last cached range as a gap. A transaction scan
    /// does the same; a scan at the latest state pins every range up front.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_pin_each_range_when_a_snapshot_scan_reaches_it() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 450).await;
        for a in (0..400).step_by(100) {
            t.db.cache_range(span(a, a + 100)).await.unwrap();
        }
        let snapshot = t.db.snapshot().unwrap();
        mixed_writes(&t, 0, 450, 1).await;

        // A scan that ends in its first range pins no other, and counts as served.
        let (pins, scans) = (t.pins(), t.metric(SCAN_COUNT, &[]));
        let mut pair = Pair::new(&snapshot, (Included(key(10)), Unbounded)).await;
        pair.step(10, "a short scan").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 1);
        assert_eq!(t.metric(SCAN_COUNT, &[]) - scans, 1);

        // To the end: every range on its way, then the uncached keys as a gap.
        let (pins, mixed) = (t.pins(), t.metric(MIXED_SCAN_COUNT, &[]));
        let gaps = t.gaps("uncovered");
        let mut pair = Pair::new(&snapshot, (Included(key(10)), Unbounded)).await;
        pair.finish("a scan to the end").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 4);
        assert_eq!(t.gaps("uncovered") - gaps, 1);
        assert_eq!(t.metric(MIXED_SCAN_COUNT, &[]) - mixed, 1);

        // A seek across ranges skips them without pinning them.
        let pins = t.pins();
        let mut pair = Pair::new(&snapshot, (Included(key(10)), Excluded(key(400)))).await;
        pair.step(3, "before the seek").await;
        pair.seek(key(350)).await;
        pair.finish("after the seek").await;
        drop(pair);
        assert_eq!(t.pins() - pins, 2);

        // A transaction scan: the same path.
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

        // At the latest state, every range up front, as before.
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

    /// X34: a snapshot scan reaches a range whose base a major merge moved above the snapshot
    /// while the scan read the range before it. With no history, that part is a gap read through
    /// the normal path at the snapshot.
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
        // The range after it is cached again.
        assert_eq!(t.pins() - pins, 1);
        t.db.close().await.unwrap();
    }

    /// X37: the range after the one a snapshot scan reads is split while the scan reads: the
    /// scan pins its children.
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

    /// X37: the range after the one a snapshot scan reads is removed while the scan reads: the
    /// scan reads its keys as a gap, and pins the range after it.
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

    /// X37: the ranges after the one a snapshot scan reads are merged quietly while the scan
    /// reads. Merged above the snapshot, their chunks are too new: a gap. Merged at the
    /// snapshot's sequence number, the merged range serves the scan.
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
                // A write outside the merged ranges: the merge's boundary is above the snapshot.
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

    /// X36: a point read pins one side of a quiet merge before the merge and loads a sequence
    /// number above its boundary: the seal at the boundary makes it try again and read the merged
    /// range, which holds the write after the merge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_retry_a_read_that_pinned_a_side_of_a_quiet_merge() {
        let t = open(|_| {}).await;
        put_all(&t, 0, 300).await;
        t.db.cache_range(span(100, 300)).await.unwrap();
        assert!(t.split(100, &[200]).await);
        t.fail("range-cache-pin", "pause");
        let reader = {
            let db = t.db.clone();
            tokio::spawn(async move { db.get(key(120)).await.unwrap() })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        let w = t.committed();
        assert!(t.quiet_merge(100, 200).await);
        t.db.put(key(120), "after").await.unwrap();
        assert!(t.committed() > w);
        let retries = t.metric(PIN_RETRY_COUNT, &[("op", "get"), ("reason", "sealed")]);
        t.fail("range-cache-pin", "off");
        assert_eq!(reader.await.unwrap(), Some(Bytes::from("after")));
        assert!(t.metric(PIN_RETRY_COUNT, &[("op", "get"), ("reason", "sealed")]) > retries);
        t.db.close().await.unwrap();
    }

    /// Design 04 section 3.6 and I14: only two adjacent quiet ranges with a key between them that
    /// the cache cut merge; the merged range holds both bases labelled with the merge's boundary,
    /// two chunks of one file become one, and reads at and above the boundary are served while
    /// older snapshots use the normal path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_merge_only_quiet_neighbours_that_a_split_cut() {
        let t = open(|o| o.tuning.range_seal_bytes = 4 * 1024).await;
        put_all(&t, 0, 500).await;
        t.db.cache_range(span(0, 100)).await.unwrap();
        t.db.cache_range(span(100, 500)).await.unwrap();
        assert!(t.split(100, &[200, 300, 400]).await);
        let rejected = || t.metric(QUIET_MERGE_REJECTED_COUNT, &[]);
        // Ranges that the application named stay apart.
        assert!(!t.quiet_merge(0, 100).await);
        assert_eq!(rejected(), 1);
        // A version in the active buffer, then in a run: not quiet.
        t.db.put(key(250), value(250, 1)).await.unwrap();
        assert!(!t.quiet_merge(200, 300).await);
        t.merge_above(200, 200, 300).await;
        let range = t.range(200);
        if !range.layers.load().active.is_empty() || !range.layers.load().runs.is_empty() {
            assert!(!t.quiet_merge(200, 300).await);
        }
        // Not adjacent.
        assert!(!t.cache().quiet_merge(&t.range(100), &t.range(300)).await);

        // Children of one file, quiet: one chunk again, at the boundary.
        let before = t.db.snapshot().unwrap();
        let file = t.range(300).layers.load().chunks[0].file.clone();
        let merges = t.metric(QUIET_MERGE_COUNT, &[]);
        let w = t.committed();
        assert!(t.quiet_merge(300, 400).await);
        assert_eq!(t.metric(QUIET_MERGE_COUNT, &[]) - merges, 1);
        let merged = t.range(300).layers.load_full();
        match &merged.chunks {
            Chunks::One(chunk) => {
                assert!(Arc::ptr_eq(&chunk.file, &file));
                assert_eq!(chunk.base_seq, w);
            }
            _ => panic!("two chunks of one file were not joined"),
        }
        assert_eq!(merged.covered_to, w);
        assert!(merged.runs.is_empty() && merged.sealed.is_empty());
        assert_eq!(t.range(300).end, Some(key(500)));
        merged.check(&key(300), *MAX_RUNS_RANGE.end());

        // Ranges of two files, once the one written to is quiet again: a base of two chunks,
        // which every read uses.
        let range = t.range(200);
        let layers = range.layers.load_full();
        if !layers.active.is_empty() {
            t.cache().request_seal(&range, &layers.active);
            t.wait("the flush", |t| {
                let l = t.range(200).layers.load_full();
                l.sealed.is_empty() && l.active.is_empty()
            })
            .await;
        }
        let layers = range.layers.load_full();
        if !layers.runs.is_empty() {
            t.cache()
                .major_merge(
                    &range,
                    layers.chunks.clone(),
                    layers.runs.clone(),
                    Trigger::Bytes,
                )
                .await
                .unwrap();
        }
        let w = t.committed();
        assert!(t.quiet_merge(200, 300).await);
        let merged = t.range(200).layers.load_full();
        match &merged.chunks {
            Chunks::Many(list) => {
                assert_eq!(list.chunks.len(), 2);
                assert_eq!(list.starts.as_ref(), &[key(300)]);
                assert!(list.chunks.iter().all(|c| c.base_seq == w));
            }
            _ => panic!("ranges of two files did not make two chunks"),
        }
        merged.check(&key(200), *MAX_RUNS_RANGE.end());
        let now = t.db.snapshot().unwrap();
        mixed_writes(&t, 150, 450, 2).await;
        for (reader, label) in [(&now, "at the merge"), (&before, "before the merge")] {
            for (a, b) in [(150, 450), (250, 350), (290, 310)] {
                let mut pair = Pair::new(reader, span(a, b)).await;
                pair.finish(label).await;
            }
            for i in 190..410 {
                let cached = reader.get(key(i)).await.unwrap();
                let want = reader
                    .get_with_options(
                        key(i),
                        &crate::config::ReadOptions::default()
                            .with_tracing_options(Some(TracingOptions::new("normal"))),
                    )
                    .await
                    .unwrap();
                assert_eq!(cached, want, "{label}: key {i}");
            }
        }
        // A split of the range of two chunks, then a major merge of one child: one chunk each.
        assert!(t.split(200, &[250]).await);
        t.merge_above(250, 250, 400).await;
        assert!(matches!(t.range(250).layers.load().chunks, Chunks::One(_)));
        let latest = t.db.snapshot().unwrap();
        let mut pair = Pair::new(&latest, span(150, 450)).await;
        pair.finish("after the split").await;
        t.db.close().await.unwrap();
    }

    /// The maintenance task merges quiet neighbours whose bases are below an eighth of the range
    /// target, and the merged range does not split again.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_merge_small_quiet_neighbours_and_keep_them_merged() {
        let t = open(|o| {
            o.tuning.range_target_bytes = 64 * 1024;
            o.tuning.range_seal_bytes = 4 * 1024;
        })
        .await;
        put_all(&t, 0, 1600).await;
        t.db.cache_range(span(0, 1600)).await.unwrap();
        t.wait("the load", |t| {
            !t.db.cached_ranges().iter().any(|r| r.loading)
        })
        .await;
        let starts = t.starts();
        assert!(starts.len() >= 3, "the load was not split: {starts:?}");
        // Most keys of two neighbours are deleted, and their bases rewritten small. The first
        // range starts where the application asked, the second where the split cut.
        let (a, b, c) = (starts[0], starts[1], starts[2]);
        for i in a..c {
            if i % 50 != 0 {
                t.db.delete(key(i)).await.unwrap();
            }
        }
        for (start, end) in [(a, b), (b, c)] {
            let range = t.range(start);
            let layers = range.layers.load_full();
            if !layers.active.is_empty() {
                t.cache().request_seal(&range, &layers.active);
            }
            t.wait("the flush", |t| {
                let l = t.range(start).layers.load_full();
                l.sealed.is_empty() && l.active.is_empty()
            })
            .await;
            let layers = range.layers.load_full();
            t.cache()
                .major_merge(
                    &range,
                    layers.chunks.clone(),
                    layers.runs.clone(),
                    Trigger::Bytes,
                )
                .await
                .unwrap();
            let bytes = range.layers.load().base_share(&key(start)).0;
            assert!(bytes < 8 * 1024, "{start}..{end}: {bytes} bytes");
        }
        let splits = t.metric(SPLIT_COUNT, &[("when", "maintenance")]);
        t.wait("the quiet merge", |t| t.metric(QUIET_MERGE_COUNT, &[]) >= 1)
            .await;
        assert!(!t.starts().contains(&b));
        // A while later, the merged range is still there and nothing split.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(t.starts().contains(&a) && !t.starts().contains(&b));
        assert_eq!(t.metric(SPLIT_COUNT, &[("when", "maintenance")]), splits);
        let snapshot = t.db.snapshot().unwrap();
        let mut pair = Pair::new(&snapshot, (Unbounded, Unbounded)).await;
        pair.finish("after the merge").await;
        t.db.close().await.unwrap();
    }

    /// I16, the gap guard. A snapshot scan's snapshot ends while the scan reads its cached first
    /// range; then a key of the uncached stretch after it gets a version above the snapshot, and
    /// the memtable flushes. With no registration, the flush keeps only the newer version, so the
    /// gap must fail rather than return the stretch without the version that the snapshot sees.
    /// With another snapshot held, older than the scan's or at its sequence number, the flush keeps
    /// that version and the gap is served (rule 2); with no flush since the snapshot ended, no SST
    /// holds a version above it and the gap is served (rule 1).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_fail_a_gap_that_its_ended_snapshot_no_longer_holds() {
        for case in ["dropped", "older held", "equal held", "no flush"] {
            let t = open(|_| {}).await;
            put_all(&t, 0, 200).await;
            t.db.cache_range(span(0, 100)).await.unwrap();
            let older = (case == "older held").then(|| t.db.snapshot().unwrap());
            if older.is_some() {
                t.db.put(key(5), value(5, 1)).await.unwrap();
            }
            let snapshot = t.db.snapshot().unwrap();
            let seq = snapshot.seq();
            let equal = (case == "equal held").then(|| t.db.snapshot().unwrap());
            if let Some(older) = &older {
                assert!(older.seq() < seq);
            }
            let mut iter = snapshot.scan(key(50)..).await.unwrap();
            let mut got = rows(&mut iter, 5).await;
            drop(snapshot);
            t.db.put(key(150), value(150, 2)).await.unwrap();
            let last_l0_seq = || t.db.inner.state.read().state().core().last_l0_seq;
            if case == "no flush" {
                assert!(last_l0_seq() <= seq, "{case}");
            } else {
                t.db.flush_with_options(FlushOptions {
                    flush_type: FlushType::MemTable,
                })
                .await
                .unwrap();
                assert!(last_l0_seq() > seq, "{case}");
            }
            let result = loop {
                match iter.next().await {
                    Ok(Some(kv)) => got.push(kv),
                    Ok(None) => break Ok(()),
                    Err(e) => break Err(e),
                }
            };
            let got: Vec<(Bytes, Bytes)> = got.into_iter().map(|kv| (kv.key, kv.value)).collect();
            if case == "dropped" {
                assert!(
                    result.is_err(),
                    "{case}: the gap was served, {} rows",
                    got.len()
                );
                // The cached range's rows came before the gap.
                assert_eq!(got.len(), 50, "{case}");
            } else {
                result.unwrap();
                let want: Vec<(Bytes, Bytes)> = (50..200).map(|i| (key(i), value(i, 0))).collect();
                assert_eq!(got, want, "{case}");
            }
            drop((older, equal));
            t.db.close().await.unwrap();
        }
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

    /// Something that happens while the property test's scans are open.
    #[derive(Clone, Debug)]
    enum Event {
        /// Split the `i`-th range at the key.
        Split(u8, u8),
        /// Merge the `i`-th range with the next one, quietly, if it can.
        Merge(u8),
        /// Remove the range that holds the key.
        Remove(u8),
        /// Merge the runs of the `i`-th range into a new base, if it has runs.
        Major(u8),
        /// A write after the snapshot.
        Write(u8, Kind),
        /// Read rows from the `i`-th scan.
        Read(u8, u8),
        /// Seek the `i`-th scan to the key, if it is ahead.
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

        /// Snapshot scans with lazily pinned parts return what the normal path returns at the
        /// snapshot, while the ranges ahead of them split, merge quietly, are removed and are
        /// major-merged above the snapshot (X34, X36, X37), with writes, merge operands and seeks
        /// in between. The keys `k000..k060` start as one explicit range split at `cuts`,
        /// `k060..k080` as a second one, and `k080..` are not cached.
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
                            // Removals can leave no range.
                            let Some(range) = all.get(*i as usize % all.len().max(1)) else {
                                continue;
                            };
                            if range.contains(&pkey(*k)) && range.start != pkey(*k)
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
                            // Removals can leave no range.
                            let Some(range) = all.get(*i as usize % all.len().max(1)) else {
                                continue;
                            };
                            if range.layers.load().chunks.loading() {
                                continue;
                            }
                            // Writes to the range's keys, sealed and flushed into a run.
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
                                    .major_merge(
                                        range,
                                        layers.chunks.clone(),
                                        layers.runs.clone(),
                                        Trigger::Bytes,
                                    )
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
