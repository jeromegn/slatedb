//! The batch writer is the only task that changes the published state of the cache.
//!
//! The published state is the index, the layers of each range, the `sealed_at` word of each
//! delta, and the watermarks `applied` and `CachedRange::last_write_seq`. Other tasks prepare a
//! change (a loaded file, a rebuilt file, a removal) and send it as a [`Request`] through a
//! lock-free queue. The writer applies requests between batches, at a *writer boundary* `W`:
//! a moment when no batch is between its memtable insert and its commit. At a boundary, `W` is
//! the last committed sequence number, every batch at or below `W` is committed, and every
//! later batch starts after it.
//!
//! - Before the writer applies a batch, it drains the queue if the queue is not empty. That is
//!   one check per batch.
//! - A producer also sends [`BatchWriterMessage::RangeCache`], so that an idle writer drains.
//!   A flag keeps one such message in flight at a time.
//! - Each request checks the identity of what it changes (`Arc::ptr_eq` of the range and of
//!   the expected delta), and changes nothing if the state moved on.
//! - [`RangeCache::close`] refuses every queued request, and every later one.
//!
//! This replaces the cache mutex and the `inserted_seq` / `publish_epoch` pair. A range that
//! registers at `W` loads the state at `W`, and every batch above `W` finds the range in the
//! index. No batch is on both sides, so no version is both in the file and in a delta.

use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use log::warn;
use tokio::sync::oneshot;

use super::auto::Removal;
use super::file::RangeFile;
use super::stats::RemoveReason;
use super::{covering, uncovered, CachedRange, Delta, Layers, Origin, RangeCache, RangeIndex};
use crate::batch_write::BatchWriterMessage;
use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::oracle::Oracle;
use crate::types::RowEntry;
use crate::utils::{spawn_bg_task, SafeSender};

/// A value on its own cache line, so that the writer's stores to it do not slow down readers of
/// neighbouring fields. Apple silicon has 128-byte lines.
#[repr(align(128))]
#[derive(Debug, Default)]
pub(super) struct Padded<T>(pub(super) T);

/// A change that a task asks the writer to make. See the module documentation.
pub(super) enum Request {
    /// Register loading ranges for the parts of `[start, end)` that no range holds.
    Register {
        start: Bytes,
        end: Option<Bytes>,
        origin: Origin,
        reply: oneshot::Sender<Result<Registered, SlateDBError>>,
    },
    /// Publish the file of a loading range. The file holds the state at `file_seq`.
    Attach {
        range: Arc<CachedRange>,
        file: Arc<RangeFile>,
        file_seq: u64,
        reply: oneshot::Sender<bool>,
    },
    /// Replace the frozen delta `frozen` of `range`, and the range's file, with `file`, which
    /// holds the state at `file_seq`.
    Install {
        range: Arc<CachedRange>,
        frozen: Arc<Delta>,
        file: Arc<RangeFile>,
        file_seq: u64,
        reply: oneshot::Sender<bool>,
    },
    Remove {
        target: Target,
        reason: RemoveReason,
        reply: Option<oneshot::Sender<()>>,
    },
    /// Remove `victims`, in order, while the files of the cached ranges are larger than the disk
    /// budget.
    Evict {
        victims: Vec<Arc<CachedRange>>,
        reply: oneshot::Sender<()>,
    },
    /// Answer once every earlier request has been applied.
    Barrier { reply: oneshot::Sender<()> },
}

impl Request {
    /// Answers a request that the closed cache does not apply. A refused file is deleted when
    /// its last reference drops.
    fn refuse(self) {
        match self {
            Request::Register { reply, .. } => {
                let _ = reply.send(Err(SlateDBError::Closed));
            }
            Request::Attach { reply, .. } | Request::Install { reply, .. } => {
                let _ = reply.send(false);
            }
            Request::Remove { reply, .. } => {
                if let Some(reply) = reply {
                    let _ = reply.send(());
                }
            }
            Request::Evict { reply, .. } | Request::Barrier { reply } => {
                let _ = reply.send(());
            }
        }
    }
}

/// What a removal removes.
pub(super) enum Target {
    /// This range, if it is still indexed.
    Range(Arc<CachedRange>),
    /// Every range that overlaps this key range when the writer applies the request.
    Overlapping(BytesRange),
    /// Every range.
    All,
}

/// The answer to [`Request::Register`].
pub(super) struct Registered {
    /// The writer boundary `W` of the registration. The new ranges load the state at `W`.
    pub(super) seq: u64,
    pub(super) ranges: Vec<Arc<CachedRange>>,
}

/// The queue to the writer and the writer's own state. Only the writer task changes the state,
/// except `wake_pending`, which producers set.
pub(super) struct WriterQueue {
    tx: async_channel::Sender<Request>,
    rx: async_channel::Receiver<Request>,
    /// The batch writer's queue, for wake-up messages. The first caller that has it sets it.
    notifier: OnceLock<SafeSender<BatchWriterMessage>>,
    /// True while a wake-up message is on its way to the writer.
    wake_pending: AtomicBool,
    /// Set once, by [`RangeCache::close`].
    closed: AtomicBool,
    /// The applied watermark `A`: every batch at or below it has been applied to every range
    /// that it touches, or is in the state that a loading range loads. The writer stores it at
    /// the end of each batch and at each boundary. Nothing reads it yet: persistence will
    /// (design 01 section 6.1 step 7).
    applied: Padded<AtomicU64>,
    /// Slot numbers of removed ranges, for later ranges. Only the writer uses it.
    free_slots: (async_channel::Sender<u32>, async_channel::Receiver<u32>),
    /// The lowest slot number that no range has had. Only the writer uses it.
    next_slot: AtomicU32,
}

impl WriterQueue {
    pub(super) fn new() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self {
            tx,
            rx,
            notifier: OnceLock::new(),
            wake_pending: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            applied: Padded(AtomicU64::new(0)),
            free_slots: async_channel::unbounded(),
            next_slot: AtomicU32::new(0),
        }
    }

    pub(super) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// The applied watermark `A`.
    #[cfg(test)]
    pub(super) fn applied(&self) -> u64 {
        self.applied.0.load(Ordering::Acquire)
    }

    /// The number of queued requests.
    #[cfg(test)]
    pub(super) fn queued(&self) -> usize {
        self.rx.len()
    }

    /// Takes a slot number for a new range: a removed range's, or a new one.
    pub(super) fn take_slot(&self) -> u32 {
        match self.free_slots.1.try_recv() {
            Ok(slot) => slot,
            Err(_) => {
                let slot = self.next_slot.load(Ordering::Relaxed);
                self.next_slot.store(slot + 1, Ordering::Relaxed);
                slot
            }
        }
    }

    fn give_back_slot(&self, slot: u32) {
        let _ = self.free_slots.0.try_send(slot);
    }

    fn raise_applied(&self, seq: u64) {
        if seq > self.applied.0.load(Ordering::Relaxed) {
            self.applied.0.store(seq, Ordering::Release);
        }
    }
}

// Producers: any task.
impl RangeCache {
    /// Gives the cache the batch writer's queue, for the wake-up messages of its requests.
    /// Every path that can send the first request concerning a range calls it first.
    pub(crate) fn bind_writer(&self, notifier: &SafeSender<BatchWriterMessage>) {
        if self.writer.notifier.get().is_none() {
            let _ = self.writer.notifier.set(notifier.clone());
        }
    }

    /// Queues `request` for the writer and makes sure that an idle writer applies it.
    fn submit(&self, request: Request) {
        // The cache holds the receiver, so a send to the unbounded queue cannot fail.
        let _ = self.writer.tx.try_send(request);
        // Pairs with the fence in `close`. Either the writer's last drain sees this request, or
        // this side sees `closed` and refuses the queued requests itself. Refusing changes no
        // published state, so any task may do it once the cache is closed.
        fence(Ordering::SeqCst);
        if self.writer.is_closed() {
            self.refuse_queued();
            return;
        }
        self.wake();
    }

    fn wake(&self) {
        // Tests and checkers can delay the message or drop it.
        fail_parallel::fail_point!(Arc::clone(&self.fp_registry), "range-cache-wake", |_| {});
        // Without a bound queue, the next batch drains the request. The paths that send the
        // first request concerning a range bind the queue first, so this does not happen.
        let Some(notifier) = self.writer.notifier.get() else {
            return;
        };
        // One message in flight is enough: the writer clears the flag before it drains.
        if self.writer.wake_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        if notifier.send(BatchWriterMessage::RangeCache).is_err() {
            // The writer stopped. Its cleanup closes the cache, which refuses every queued
            // request.
            self.writer.wake_pending.store(false, Ordering::Release);
        }
    }

    fn refuse_queued(&self) {
        while let Ok(request) = self.writer.rx.try_recv() {
            request.refuse();
        }
    }

    /// Registers loading ranges for the parts of `[start, end)` that no range holds, at a
    /// writer boundary. See [`Request::Register`].
    pub(super) async fn register(
        &self,
        start: Bytes,
        end: Option<Bytes>,
        origin: Origin,
    ) -> Result<Registered, SlateDBError> {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Register {
            start,
            end,
            origin,
            reply,
        });
        answer.await.unwrap_or(Err(SlateDBError::Closed))
    }

    /// Publishes the file of a loading range. Returns false if the range is no longer indexed
    /// or no longer loading at `file_seq`: then the file is deleted when it drops.
    pub(super) async fn attach(
        &self,
        range: &Arc<CachedRange>,
        file: Arc<RangeFile>,
        file_seq: u64,
    ) -> bool {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Attach {
            range: range.clone(),
            file,
            file_seq,
            reply,
        });
        answer.await.unwrap_or(false)
    }

    /// Installs a rebuilt file. Returns false if the range is no longer indexed or `frozen` is
    /// no longer its frozen delta.
    pub(super) async fn install(
        &self,
        range: &Arc<CachedRange>,
        frozen: &Arc<Delta>,
        file: Arc<RangeFile>,
        file_seq: u64,
    ) -> bool {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Install {
            range: range.clone(),
            frozen: frozen.clone(),
            file,
            file_seq,
            reply,
        });
        answer.await.unwrap_or(false)
    }

    /// Asks the writer to remove `target`. The removal takes effect at the writer's next
    /// boundary. Reads stay exact meanwhile.
    pub(super) fn request_removal(&self, target: Target, reason: RemoveReason) {
        self.submit(Request::Remove {
            target,
            reason,
            reply: None,
        });
    }

    /// Removes `range` and waits until the writer did.
    pub(super) async fn remove_and_wait(&self, range: &Arc<CachedRange>, reason: RemoveReason) {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Remove {
            target: Target::Range(range.clone()),
            reason,
            reply: Some(reply),
        });
        let _ = answer.await;
    }

    /// Removes ranges until the range files fit the budget, never `skip`, and waits until the
    /// writer did. The order is computed here, off the writer: see
    /// [`RangeCache::eviction_order`]. The writer checks the budget again before each removal,
    /// so concurrent callers do not remove more than needed.
    pub(super) async fn evict(&self, skip: Option<&Arc<CachedRange>>) {
        let victims = self.eviction_order(skip);
        if victims.is_empty() {
            return;
        }
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Evict { victims, reply });
        let _ = answer.await;
    }

    /// Waits until the writer applied every request queued before the call, or the cache
    /// closed.
    pub(super) async fn barrier(&self) {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Barrier { reply });
        let _ = answer.await;
    }
}

// The writer: only the batch writer task calls these.
impl RangeCache {
    /// Applies one write batch. The batch writer calls this after it puts the entries in the
    /// memtable and before it advances the last committed sequence number. All entries have
    /// the same sequence number `s`.
    ///
    /// No lock, and no shared read-modify-write: one check of the queue, one index load, and
    /// for a batch that touches cached ranges, the delta inserts and one store per touched range
    /// and per batch.
    pub(crate) fn apply_write(self: &Arc<Self>, entries: &[RowEntry], oracle: &dyn Oracle) {
        let Some(seq) = entries.first().map(|e| e.seq) else {
            return;
        };
        if !self.writer.rx.is_empty() {
            // Batch `s` is in the memtable and not committed, and every earlier batch is
            // committed, so this is a writer boundary.
            self.drain(oracle);
        }
        {
            let index = self.index.load();
            if !index.is_empty() && entries.iter().any(|e| covering(&index, &e.key).is_some()) {
                self.apply_covered(&index, entries, seq);
            }
        }
        self.writer.raise_applied(seq);
    }

    fn apply_covered(self: &Arc<Self>, index: &RangeIndex, entries: &[RowEntry], seq: u64) {
        let mut touched: Vec<Arc<CachedRange>> = Vec::new();
        let mut expiring: Vec<Arc<CachedRange>> = Vec::new();
        for entry in entries {
            let Some(range) = covering(index, &entry.key) else {
                continue;
            };
            if entry.expire_ts.is_some() {
                // The cache does not hold rows that expire. See RFC 0040, "TTL".
                if !expiring.iter().any(|r| Arc::ptr_eq(r, range)) {
                    expiring.push(range.clone());
                }
                continue;
            }
            let layers = range.layers.load();
            // A range registered at `W`, or rebuilt at `W`, holds every batch at or below `W`
            // in its file, and receives only later batches.
            debug_assert!(
                entry.seq > layers.file_seq,
                "a batch at {} reached a range whose file holds {}",
                entry.seq,
                layers.file_seq
            );
            layers.active.table.put(entry.clone());
            if !touched.iter().any(|r| Arc::ptr_eq(r, range)) {
                range.last_write_seq.0.store(seq, Ordering::Release);
                touched.push(range.clone());
            }
        }
        for range in &expiring {
            // The delta does not hold this write, so seal it before the write.
            self.remove_at(range, seq - 1, RemoveReason::ExpiringWrite);
        }
        for range in &touched {
            if !expiring.iter().any(|r| Arc::ptr_eq(r, range)) {
                self.maybe_freeze(range, seq);
            }
        }
    }

    /// Starts a rebuild if the active delta of `range` is large enough.
    fn maybe_freeze(self: &Arc<Self>, range: &Arc<CachedRange>, seq: u64) {
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
            self.remove_at(range, seq, RemoveReason::NotWorthIt);
            return;
        }
        range
            .reads_at_build
            .store(range.heat.reads(), Ordering::Relaxed);
        let frozen = layers.active.clone();
        // The word is on the delta, so every set of layers that holds this delta as its active
        // delta, older ones included, sees the seal.
        frozen.sealed_at.store(seq, Ordering::SeqCst);
        range.layers.store(Arc::new(Layers {
            active: Arc::new(Delta::new()),
            frozen: Some(frozen.clone()),
            file: Some(file.clone()),
            file_seq: layers.file_seq,
            end: range.end.clone(),
        }));
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
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
                    failed_cache.request_removal(Target::Range(failed_range), RemoveReason::Error);
                }
            },
            async move { cache.rebuild(range, frozen, file, seq, file_id).await },
        );
    }

    /// Applies the queued requests at the writer boundary `oracle.last_committed_seq()`. The
    /// batch writer calls it for [`BatchWriterMessage::RangeCache`], when no batch is in flight.
    pub(crate) fn drain(self: &Arc<Self>, oracle: &dyn Oracle) {
        let w = oracle.last_committed_seq();
        fail_parallel::fail_point!(Arc::clone(&self.fp_registry), "range-cache-drain");
        // Clear the flag first. A request queued after this point sends its own message; one
        // queued before it is visible to the loop below (the swap reads the producer's flag).
        self.writer.wake_pending.swap(false, Ordering::AcqRel);
        self.writer.raise_applied(w);
        while let Ok(request) = self.writer.rx.try_recv() {
            self.apply_request(request, w);
        }
    }

    fn apply_request(self: &Arc<Self>, request: Request, w: u64) {
        if self.writer.is_closed() {
            request.refuse();
            return;
        }
        match request {
            Request::Register {
                start,
                end,
                origin,
                reply,
            } => {
                let _ = reply.send(Ok(self.register_at(start, end, origin, w)));
            }
            Request::Attach {
                range,
                file,
                file_seq,
                reply,
            } => {
                let _ = reply.send(self.attach_at(&range, file, file_seq));
            }
            Request::Install {
                range,
                frozen,
                file,
                file_seq,
                reply,
            } => {
                let _ = reply.send(self.install_at(&range, &frozen, file, file_seq));
            }
            Request::Remove {
                target,
                reason,
                reply,
            } => {
                self.remove_target(target, reason, w);
                if let Some(reply) = reply {
                    let _ = reply.send(());
                }
            }
            Request::Evict { victims, reply } => {
                for victim in victims {
                    if self.attached_bytes() <= self.options.max_disk_bytes {
                        break;
                    }
                    if victim.layers.load().file.is_some() {
                        self.remove_at(&victim, w, RemoveReason::Evicted);
                    }
                }
                let _ = reply.send(());
            }
            Request::Barrier { reply } => {
                let _ = reply.send(());
            }
        }
    }

    /// New loading ranges for the uncovered parts of `[start, end)`, which load the state at
    /// `w`: every batch at or below `w` is committed and in the load's view, and every later
    /// batch finds the new ranges and goes to their deltas.
    fn register_at(&self, start: Bytes, end: Option<Bytes>, origin: Origin, w: u64) -> Registered {
        let now = self.now_ms();
        let mut index = RangeIndex::clone(&self.index.load());
        let parts = uncovered(&index, &start, end.as_ref());
        let mut ranges = Vec::with_capacity(parts.len());
        for (part_start, part_end) in parts {
            let layers = Layers {
                active: Arc::new(Delta::new()),
                frozen: None,
                // A loading range serves no read.
                file: None,
                file_seq: w,
                end: part_end.clone(),
            };
            let range = Arc::new(CachedRange::new(
                part_start.clone(),
                part_end,
                layers,
                origin,
                now,
                self.writer.take_slot(),
            ));
            index.insert(part_start, range.clone());
            ranges.push(range);
        }
        if !ranges.is_empty() {
            self.index.store(Arc::new(index));
            self.update_gauges();
        }
        Registered { seq: w, ranges }
    }

    /// Publishes the file of a loading range. It keeps the active delta, which holds the
    /// batches after `file_seq`, so no reader of an earlier set of layers loses a signal.
    fn attach_at(&self, range: &Arc<CachedRange>, file: Arc<RangeFile>, file_seq: u64) -> bool {
        if !self.is_indexed(range) {
            // Removed, or removed and registered again: the file is deleted on drop.
            return false;
        }
        let layers = range.layers.load_full();
        if layers.file.is_some() || layers.file_seq != file_seq {
            return false;
        }
        debug_assert!(layers.frozen.is_none(), "a loading range never freezes");
        self.stats.loads.increment(1);
        self.stats.load_bytes.increment(file.file_len());
        let now = self.hits.access_clock().fetch_add(2, Ordering::Relaxed) + 2;
        range.heat.hits.set_last_access(now - 1);
        range.layers.store(Arc::new(Layers {
            active: layers.active.clone(),
            frozen: None,
            file: Some(file),
            file_seq,
            end: range.end.clone(),
        }));
        true
    }

    /// Replaces the frozen delta and the file of `range` with a rebuilt file. It keeps the
    /// active delta.
    fn install_at(
        &self,
        range: &Arc<CachedRange>,
        frozen: &Arc<Delta>,
        file: Arc<RangeFile>,
        file_seq: u64,
    ) -> bool {
        if !self.is_indexed(range) {
            return false;
        }
        let layers = range.layers.load_full();
        if !layers
            .frozen
            .as_ref()
            .is_some_and(|f| Arc::ptr_eq(f, frozen))
        {
            return false;
        }
        self.stats.rebuilds.increment(1);
        self.stats.rebuild_bytes.increment(file.file_len());
        range.layers.store(Arc::new(Layers {
            active: layers.active.clone(),
            frozen: None,
            file: Some(file),
            file_seq,
            end: range.end.clone(),
        }));
        self.hits.access_clock().fetch_add(2, Ordering::Relaxed);
        true
    }

    fn remove_target(&self, target: Target, reason: RemoveReason, w: u64) {
        match target {
            Target::Range(range) => {
                self.remove_at(&range, w, reason);
            }
            Target::Overlapping(bounds) => {
                let ranges: Vec<_> = self
                    .index
                    .load()
                    .values()
                    .filter(|r| r.bounds().intersect(&bounds).is_some())
                    .cloned()
                    .collect();
                for range in ranges {
                    self.remove_at(&range, w, reason);
                }
            }
            Target::All => {
                let ranges: Vec<_> = self.index.load().values().cloned().collect();
                for range in ranges {
                    self.remove_at(&range, w, reason);
                }
            }
        }
    }

    /// Removes `range` and seals its active delta at `sealed_at`, the last sequence number that
    /// the delta holds completely. Returns false if the range is not indexed.
    pub(super) fn remove_at(
        &self,
        range: &Arc<CachedRange>,
        sealed_at: u64,
        reason: RemoveReason,
    ) -> bool {
        if !self.is_indexed(range) {
            return false;
        }
        // Seal before the range leaves the index. A reader that pinned any set of layers of the
        // range sees the seal on the delta that they share, or an earlier seal of an earlier
        // active delta, tries again for a later sequence number, and then finds the range gone.
        range
            .layers
            .load()
            .active
            .sealed_at
            .store(sealed_at, Ordering::SeqCst);
        let mut index = RangeIndex::clone(&self.index.load());
        index.remove(&range.start);
        self.index.store(Arc::new(index));
        self.writer.give_back_slot(range.slot);
        self.stats.removed(reason);
        self.update_gauges();
        self.notify_removal(range, reason);
        true
    }

    /// Tells automatic promotion that an automatic range left. The promoter keeps the ghosts
    /// and the backoff of failed promotions.
    fn notify_removal(&self, range: &CachedRange, reason: RemoveReason) {
        let Some(auto) = &self.auto else {
            return;
        };
        if !range.auto || reason == RemoveReason::Closed {
            return;
        }
        let _ = auto.removals.0.try_send(Removal {
            start: range.start.clone(),
            end: range.end.clone(),
            failure: reason.is_failure(),
            at_ms: self.now_ms(),
        });
    }

    /// Removes every range and refuses every queued and later request. The batch writer calls
    /// it when it stops: on close, and on errors such as fencing. It seals each active delta at
    /// the applied watermark, which is at or above every committed batch.
    pub(crate) fn close(&self) {
        if !self.writer.is_closed() {
            let w = self.writer.applied.0.load(Ordering::Relaxed);
            let index = self.index.load_full();
            for range in index.values() {
                range
                    .layers
                    .load()
                    .active
                    .sealed_at
                    .store(w, Ordering::SeqCst);
                self.writer.give_back_slot(range.slot);
                self.stats.removed(RemoveReason::Closed);
            }
            self.index.store(Arc::new(RangeIndex::new()));
            self.update_gauges();
            self.writer.closed.store(true, Ordering::SeqCst);
        }
        // Pairs with the fence in `submit`.
        fence(Ordering::SeqCst);
        self.refuse_queued();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound::{Excluded, Included};
    use std::time::Duration;

    use fail_parallel::FailPointRegistry;
    use slatedb_common::metrics::{lookup_metric_with_labels, DefaultMetricsRecorder};

    use crate::config::ReadOptions;
    use crate::config::TracingOptions;
    use crate::db::Db;
    use crate::merge_operator::{MergeOperator, MergeOperatorError};
    use crate::object_store::memory::InMemory;
    use crate::range_cache::stats::{PIN_RETRY_COUNT, REMOVE_COUNT, UNUSABLE_COUNT};
    use crate::range_cache::RangeCacheOptions;

    /// How long a test waits for something that must happen promptly.
    const PROMPT: Duration = Duration::from_secs(10);

    #[test]
    fn should_reuse_free_slots() {
        let queue = WriterQueue::new();
        assert_eq!(
            (queue.take_slot(), queue.take_slot(), queue.take_slot()),
            (0, 1, 2)
        );
        queue.give_back_slot(2);
        queue.give_back_slot(0);
        assert_eq!(
            (queue.take_slot(), queue.take_slot(), queue.take_slot()),
            (2, 0, 3)
        );
    }

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

        fn range(&self, start: &Bytes) -> Arc<CachedRange> {
            self.cache().index.load().get(start).unwrap().clone()
        }

        fn committed(&self) -> u64 {
            self.db.inner.oracle.last_committed_seq()
        }

        fn metric(&self, name: &str, labels: &[(&str, &str)]) -> i64 {
            lookup_metric_with_labels(&self.recorder, name, labels).unwrap_or(0)
        }

        fn fail(&self, name: &str, action: &str) {
            fail_parallel::cfg(self.fp_registry.clone(), name, action).unwrap();
        }
    }

    async fn open(tune: impl FnOnce(&mut RangeCacheOptions)) -> TestDb {
        let dir = tempfile::tempdir().unwrap();
        let mut options = RangeCacheOptions::new(dir.path());
        // Exact counters at each read.
        options.tuning.stats_fold_events = 1;
        tune(&mut options);
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let fp_registry = Arc::new(FailPointRegistry::new());
        let db = Db::builder("range_cache_writer", Arc::new(InMemory::new()))
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

    fn span(a: usize, b: usize) -> (std::ops::Bound<Bytes>, std::ops::Bound<Bytes>) {
        (Included(key(a)), Excluded(key(b)))
    }

    async fn put_all(db: &Db, keys: std::ops::Range<usize>) {
        for i in keys {
            db.put(key(i), format!("value-{i}")).await.unwrap();
        }
    }

    /// Reads through the normal path: the cache never serves a traced read.
    fn normal() -> ReadOptions {
        ReadOptions::default().with_tracing_options(Some(TracingOptions::new("normal")))
    }

    /// Loads and removals race merges on the same keys. Each operand is unique, so an
    /// operand that the file and a delta both hold shows as a repeated token.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_apply_merges_that_race_loads_once() {
        let t = open(|o| {
            o.tuning.min_rebuild_delta_bytes = 64;
            o.tuning.page_size = 512;
        })
        .await;
        let db = &t.db;
        let writers: Vec<_> = (0..3)
            .map(|w| {
                let db = db.clone();
                tokio::spawn(async move {
                    for i in 0..300usize {
                        let k = key((i * 7 + w) % 16);
                        db.merge(k, format!("[{w}.{i}]")).await.unwrap();
                    }
                })
            })
            .collect();
        let loader = {
            let db = db.clone();
            tokio::spawn(async move {
                for i in 0..60usize {
                    let (a, b) = (i % 9, i % 9 + 3 + i % 5);
                    db.cache_range(span(a, b)).await.unwrap();
                    if i % 3 == 2 {
                        db.uncache_range(span(a + 1, b)).unwrap();
                    }
                }
            })
        };
        for writer in writers {
            writer.await.unwrap();
        }
        loader.await.unwrap();
        db.cache_range(span(0, 16)).await.unwrap();
        let snapshot = db.snapshot().unwrap();
        for i in 0..16 {
            let cached = snapshot.get(key(i)).await.unwrap().unwrap_or_default();
            let normal = snapshot
                .get_with_options(key(i), &normal())
                .await
                .unwrap()
                .unwrap_or_default();
            assert_eq!(cached, normal, "key {i}");
            let text = String::from_utf8(cached.to_vec()).unwrap();
            let tokens: Vec<_> = text.split_inclusive(']').collect();
            let unique: std::collections::BTreeSet<_> = tokens.iter().collect();
            assert_eq!(
                tokens.len(),
                unique.len(),
                "an operand applied twice: {text}"
            );
        }
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_load_and_remove_while_no_batch_arrives() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        // No write comes after this point: every request reaches the writer through the
        // wake-up message.
        tokio::time::timeout(PROMPT, t.db.cache_range(span(10, 90)))
            .await
            .expect("the load completes without a batch")
            .unwrap();
        assert_eq!(t.db.cached_ranges().len(), 1);
        t.db.uncache_range(span(10, 90)).unwrap();
        tokio::time::timeout(PROMPT, t.db.wait_for_range_cache())
            .await
            .expect("the removal completes without a batch")
            .unwrap();
        assert!(t.db.cached_ranges().is_empty());
        assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "explicit")]), 1);
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_need_the_wake_up_message_when_idle() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        // Drop every wake-up message: a request then waits for a batch.
        t.fail("range-cache-wake", "return");
        let load = {
            let db = t.db.clone();
            tokio::spawn(async move { db.cache_range(span(10, 90)).await })
        };
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!load.is_finished(), "the load completed without a writer");
        assert_eq!(t.cache().writer.queued(), 1);
        assert!(t.db.cached_ranges().is_empty());
        // A batch is a writer boundary: it applies the registration.
        t.db.put(key(500), "x").await.unwrap();
        tokio::time::timeout(PROMPT, async {
            while t.db.cached_ranges().is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        // The attach waits again, for the next batch or a wake-up message.
        t.fail("range-cache-wake", "off");
        t.db.put(key(501), "y").await.unwrap();
        tokio::time::timeout(PROMPT, load)
            .await
            .expect("the load completes after the batch")
            .unwrap()
            .unwrap();
        assert!(!t.db.cached_ranges()[0].loading);
        t.db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_register_at_a_writer_boundary() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        let before = t.committed();
        t.db.cache_range(span(10, 90)).await.unwrap();
        let range = t.range(&key(10));
        // The file holds the state at the boundary, the last committed sequence number then.
        assert_eq!(range.layers.load().file_seq, before);
        assert_eq!(t.cache().writer.applied(), before);
        // A later batch goes to the delta, above the file.
        t.db.put(key(20), "new").await.unwrap();
        let seq = t.committed();
        let layers = range.layers.load();
        assert_eq!(layers.active.table.metadata().entry_num, 1);
        assert!(seq > layers.file_seq);
        assert_eq!(t.db.get(key(20)).await.unwrap(), Some(Bytes::from("new")));
        t.db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_advance_the_applied_watermark_on_every_batch() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        t.db.cache_range(span(10, 20)).await.unwrap();
        let range = t.range(&key(10));
        assert_eq!(range.last_write_seq.0.load(Ordering::Acquire), 0);
        // A batch that touches no cached range advances `A`, not the range's last write.
        t.db.put(key(50), "outside").await.unwrap();
        assert_eq!(t.cache().writer.applied(), t.committed());
        assert_eq!(range.last_write_seq.0.load(Ordering::Acquire), 0);
        // A batch that touches the range advances both.
        t.db.put(key(15), "inside").await.unwrap();
        let seq = t.committed();
        assert_eq!(t.cache().writer.applied(), seq);
        assert_eq!(range.last_write_seq.0.load(Ordering::Acquire), seq);
        t.db.close().await.unwrap();
    }

    /// Stage 0 found that a removal sealed the active delta at the last batch that touched a
    /// cached range, which lags the last committed sequence number when later batches touch
    /// none, so snapshot reads that held the range fell back as `sealed`. The writer now
    /// seals at its boundary.
    #[tokio::test]
    async fn should_seal_a_removal_at_the_writer_boundary() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        t.db.cache_range(span(10, 20)).await.unwrap();
        let range = t.range(&key(10));
        // Batches that touch no cached range.
        put_all(&t.db, 50..60).await;
        let snapshot = t.db.snapshot().unwrap();
        let layers = range.layers.load_full();
        t.db.uncache_range(span(10, 20)).unwrap();
        t.db.wait_for_range_cache().await.unwrap();
        let sealed_at = layers.active.sealed_at.load(Ordering::SeqCst);
        assert_eq!(sealed_at, t.committed());
        // A snapshot read that pinned these layers before the removal can use them.
        assert!(snapshot.seq() <= sealed_at);
        assert!(RangeCache::usable(&layers, snapshot.seq()).is_ok());
        t.db.close().await.unwrap();
    }

    #[tokio::test]
    async fn should_not_retry_a_read_older_than_the_file() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        let snapshot = t.db.snapshot().unwrap();
        let txn = t.db.begin(crate::IsolationLevel::Snapshot).unwrap();
        t.db.put(key(500), "later").await.unwrap();
        t.db.cache_range(span(10, 90)).await.unwrap();
        // Both are older than the range's file: they fall back at once.
        let want = Some(Bytes::from("value-20"));
        assert_eq!(snapshot.get(key(20)).await.unwrap(), want);
        assert_eq!(txn.get(key(20)).await.unwrap(), want);
        drop(snapshot.scan(span(20, 30)).await.unwrap());
        let unfit = |name, op| t.metric(name, &[("op", op), ("reason", "too_old")]);
        assert_eq!(unfit(UNUSABLE_COUNT, "get"), 2);
        assert_eq!(unfit(UNUSABLE_COUNT, "scan"), 1);
        assert_eq!(unfit(PIN_RETRY_COUNT, "get"), 0);
        assert_eq!(unfit(PIN_RETRY_COUNT, "scan"), 0);
        drop(txn);
        t.db.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_refuse_requests_queued_at_close() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        t.db.cache_range(span(0, 10)).await.unwrap();
        // The requests stay queued: no wake-up message and no batch.
        t.fail("range-cache-wake", "return");
        let load = {
            let db = t.db.clone();
            tokio::spawn(async move { db.cache_range(span(20, 30)).await })
        };
        t.db.uncache_range(span(0, 10)).unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(t.cache().writer.queued(), 2);
        // The writer's cleanup closes the cache and answers both.
        tokio::time::timeout(PROMPT, t.db.close())
            .await
            .unwrap()
            .unwrap();
        let result = tokio::time::timeout(PROMPT, load).await.unwrap().unwrap();
        assert!(result.is_err());
        assert_eq!(t.cache().writer.queued(), 0);
        assert!(t.db.cached_ranges().is_empty());
        // Later requests are refused at once.
        t.db.uncache_range(span(0, 10)).unwrap();
        tokio::time::timeout(PROMPT, t.db.wait_for_range_cache())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(t.cache().writer.queued(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_refuse_requests_after_the_writer_failed() {
        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        t.db.cache_range(span(0, 10)).await.unwrap();
        // The next batch fails before it commits, and the writer stops.
        t.fail("write-batch-pre-commit", "return");
        assert!(t.db.put(key(5), "lost").await.is_err());
        t.fail("write-batch-pre-commit", "off");
        tokio::time::timeout(PROMPT, async {
            while !t.cache().writer.is_closed() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(t.db.cached_ranges().is_empty());
        // Loads and waits return at once.
        let load = tokio::time::timeout(PROMPT, t.db.cache_range(span(20, 30)))
            .await
            .unwrap();
        assert!(load.is_err());
        t.db.uncache_range(span(0, 10)).unwrap();
        tokio::time::timeout(PROMPT, t.db.wait_for_range_cache())
            .await
            .unwrap()
            .unwrap();
    }
}
