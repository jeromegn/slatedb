//! The batch writer is the only task that changes the published state of the cache.
//!
//! The published state is the index, the layers of each range, the `sealed_at` word of each
//! buffer, and the watermarks `applied` and `CachedRange::last_write_seq`. Other tasks prepare a
//! change (a loaded base, a flushed run, a merged base, a split, a removal) and send it as a
//! [`Request`] through a lock-free queue. The writer applies requests between batches, at a
//! *writer boundary* `W`: a moment when no batch is between its memtable insert and its commit.
//! At a boundary, `W` is the last committed sequence number, every batch at or below `W` is
//! committed, and every later batch starts after it.
//!
//! - Before the writer applies a batch, it drains the queue if the queue is not empty. That is
//!   one check per batch.
//! - A producer also sends [`BatchWriterMessage::RangeCache`], so that an idle writer drains.
//!   A flag keeps one such message in flight at a time.
//! - Each request checks the identity of what it changes (`Arc::ptr_eq` of the range and of
//!   the expected buffer, runs or base), and changes nothing if the state moved on (design 01
//!   section 6.2).
//! - [`RangeCache::close`] refuses every queued request, and every later one.
//!
//! This replaces the cache mutex and the `inserted_seq` / `publish_epoch` pair. A range that
//! registers at `W` loads the state at `W`, and every batch above `W` finds the range in the
//! index. No batch is on both sides, so no version is both in the base and in a buffer.
//!
//! The writer also seals a range's active buffer when it reaches `range_seal_bytes`, at the end
//! of the batch that took it there, and it enforces the memory cap of the buffers (design 01
//! section 9): a row whose append would take the memory of all buffers past the cap is not
//! appended, and its range is removed at `s - 1`, before the batch `s` commits. The buffer
//! reports the exact bytes that an append allocates before it allocates anything, and only the
//! writer adds to the memory count, so the check is exact and the count never passes the cap.

use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use tokio::sync::oneshot;

use super::auto::Removal;
use super::file::RangeFile;
use super::split::{SplitChild, SplitPlan};
use super::stats::RemoveReason;
use super::{
    covering, one_chunk, uncovered, CachedRange, Chunk, ChunkList, Chunks, Delta, DeltaView,
    Layers, Origin, RangeCache, RangeIndex, Run,
};
use crate::batch_write::BatchWriterMessage;
use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::oracle::Oracle;
use crate::types::RowEntry;
use crate::utils::SafeSender;

/// A value on its own cache line, so that the writer's stores to it do not slow down readers of
/// neighbouring fields. Apple silicon has 128-byte lines.
#[repr(align(128))]
#[derive(Debug, Default)]
pub(super) struct Padded<T>(pub(super) T);

/// A change that a task asks the writer to make. See the module documentation.
pub(super) enum Request {
    /// Register loading ranges for the parts of `[start, end)` that no range holds, split at
    /// `splits` (ascending keys) so that no range is much larger than the range target.
    Register {
        start: Bytes,
        end: Option<Bytes>,
        splits: Vec<Bytes>,
        origin: Origin,
        reply: oneshot::Sender<Result<Registered, SlateDBError>>,
    },
    /// Publish the base of a loading range, one chunk over the range that holds the state at
    /// `base_seq`, and with `split`, cut the range into the split's children in the same store
    /// (design 04 section 3.4). The reply holds the ranges that hold the base: the range, its
    /// children, or none if the range was removed meanwhile.
    AttachBase {
        range: Arc<CachedRange>,
        base: Arc<RangeFile>,
        base_seq: u64,
        split: Option<SplitPlan>,
        reply: oneshot::Sender<Vec<Arc<CachedRange>>>,
    },
    /// Install `run`, which a flush wrote from `buffers`, the oldest sealed buffers of `range`
    /// (newest first), and from `replaces`, its newest runs (newest first, empty for a plain
    /// flush).
    InstallRun {
        range: Arc<CachedRange>,
        run: Arc<Run>,
        buffers: Vec<Arc<Delta>>,
        replaces: Vec<Arc<Run>>,
        /// The view with the run and without `replaces`, and the view it was built on.
        view: ViewSwap,
        reply: oneshot::Sender<bool>,
    },
    /// Replace the base `chunks_was` of `range` and its oldest runs `absorbed` (newest first) with
    /// `base`, one chunk over the range that a major merge wrote with the state at `base_seq`.
    InstallMajor {
        range: Arc<CachedRange>,
        chunks_was: Chunks,
        absorbed: Vec<Arc<Run>>,
        base: Arc<RangeFile>,
        base_seq: u64,
        /// The view without the absorbed runs, and the view it was built on.
        view: ViewSwap,
        reply: oneshot::Sender<bool>,
    },
    /// Replace `range` by the children of `plan`, which share its files, runs and buffers (design
    /// 04 section 3.2), if it still holds the base, runs and view that the plan was built on.
    Split {
        range: Arc<CachedRange>,
        plan: SplitPlan,
        reply: oneshot::Sender<bool>,
    },
    /// Merge `left` and `right`, adjacent ranges, into one if both are still quiet (no run, no
    /// sealed buffer, an empty active buffer) and hold the bases `chunks_was` (design 04 section
    /// 3.6). `rate` is the sum of their read rates, read off the writer when the task asked.
    QuietMerge {
        left: Arc<CachedRange>,
        right: Arc<CachedRange>,
        chunks_was: [Chunks; 2],
        rate: f64,
        reply: Option<oneshot::Sender<bool>>,
    },
    /// Seal `buffer` if it is still the active buffer of `range` and holds a version.
    Seal {
        range: Arc<CachedRange>,
        buffer: Arc<Delta>,
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
            Request::AttachBase { reply, .. } => {
                let _ = reply.send(Vec::new());
            }
            Request::InstallRun { reply, .. }
            | Request::InstallMajor { reply, .. }
            | Request::Split { reply, .. } => {
                let _ = reply.send(false);
            }
            Request::QuietMerge { reply, .. } => {
                if let Some(reply) = reply {
                    let _ = reply.send(false);
                }
            }
            Request::Seal { .. } => {}
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

/// The view that an install publishes with its runs (`None` once no run is left), and the view
/// that it was built from, which must still be the range's: the writer publishes both or rejects
/// both. A job whose install the writer rejected only because the view moved on builds its view
/// again on the new one; its file stays good (see the `maintain` module).
pub(super) struct ViewSwap {
    pub(super) new: Option<Arc<DeltaView>>,
    pub(super) was: Option<Arc<DeltaView>>,
}

/// True if `a` and `b` are both `None` or the same view.
pub(super) fn same_view(a: &Option<Arc<DeltaView>>, b: &Option<Arc<DeltaView>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
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
        splits: Vec<Bytes>,
        origin: Origin,
    ) -> Result<Registered, SlateDBError> {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Register {
            start,
            end,
            splits,
            origin,
            reply,
        });
        answer.await.unwrap_or(Err(SlateDBError::Closed))
    }

    /// Publishes the base of a loading range, split as `split` says if the range has not moved
    /// on. Returns the ranges that hold it, or none if the range is no longer indexed or no longer
    /// loading at `base_seq`: then the file is deleted when it drops.
    pub(super) async fn attach_base(
        &self,
        range: &Arc<CachedRange>,
        base: Arc<RangeFile>,
        base_seq: u64,
        split: Option<SplitPlan>,
    ) -> Vec<Arc<CachedRange>> {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::AttachBase {
            range: range.clone(),
            base,
            base_seq,
            split,
            reply,
        });
        answer.await.unwrap_or_default()
    }

    /// Splits `range` as `plan` says. Returns false if the range moved on (see
    /// [`RangeCache::split_request_at`]).
    pub(super) async fn install_split(&self, range: &Arc<CachedRange>, plan: SplitPlan) -> bool {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::Split {
            range: range.clone(),
            plan,
            reply,
        });
        answer.await.unwrap_or(false)
    }

    /// Asks the writer to merge `left` and `right`, adjacent quiet ranges whose bases are
    /// `chunks_was` and whose read rates sum to `rate`, into one (see
    /// [`RangeCache::quiet_merge_at`]), without waiting for its answer.
    pub(super) fn ask_quiet_merge(
        &self,
        left: &Arc<CachedRange>,
        right: &Arc<CachedRange>,
        chunks_was: [Chunks; 2],
        rate: f64,
    ) {
        self.submit(Request::QuietMerge {
            left: left.clone(),
            right: right.clone(),
            chunks_was,
            rate,
            reply: None,
        });
    }

    /// Merges `left` and `right` as the maintenance task asks, with the bases that they hold now,
    /// and returns whether the writer did.
    #[cfg(test)]
    pub(super) async fn quiet_merge(
        &self,
        left: &Arc<CachedRange>,
        right: &Arc<CachedRange>,
    ) -> bool {
        let chunks_was = [
            left.layers.load().chunks.clone(),
            right.layers.load().chunks.clone(),
        ];
        let (reply, answer) = oneshot::channel();
        self.submit(Request::QuietMerge {
            left: left.clone(),
            right: right.clone(),
            chunks_was,
            rate: 0.0,
            reply: Some(reply),
        });
        answer.await.unwrap_or(false)
    }

    /// Installs a run that a flush wrote from `buffers` and `replaces`, with its view. Returns
    /// false if the state moved on (see [`RangeCache::install_run_at`]): then the run's file is
    /// deleted when it drops.
    pub(super) async fn install_run(
        &self,
        range: &Arc<CachedRange>,
        run: Arc<Run>,
        buffers: Vec<Arc<Delta>>,
        replaces: Vec<Arc<Run>>,
        view: ViewSwap,
    ) -> bool {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::InstallRun {
            range: range.clone(),
            run,
            buffers,
            replaces,
            view,
            reply,
        });
        answer.await.unwrap_or(false)
    }

    /// Installs a base that a major merge wrote from `chunks_was` and `absorbed`, with its view.
    /// Returns false if the state moved on (see [`RangeCache::install_major_at`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn install_major(
        &self,
        range: &Arc<CachedRange>,
        chunks_was: &Chunks,
        absorbed: Vec<Arc<Run>>,
        base: Arc<RangeFile>,
        base_seq: u64,
        view: ViewSwap,
    ) -> bool {
        let (reply, answer) = oneshot::channel();
        self.submit(Request::InstallMajor {
            range: range.clone(),
            chunks_was: chunks_was.clone(),
            absorbed,
            base,
            base_seq,
            view,
            reply,
        });
        answer.await.unwrap_or(false)
    }

    /// Asks the writer to seal `buffer`, the active buffer of `range`, at its next boundary.
    pub(super) fn request_seal(&self, range: &Arc<CachedRange>, buffer: &Arc<Delta>) {
        self.submit(Request::Seal {
            range: range.clone(),
            buffer: buffer.clone(),
        });
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

    /// Removes ranges until the range files fit the budget, never one of `skip`, and waits until
    /// the writer did. The order is computed here, off the writer: see
    /// [`RangeCache::eviction_order`]. The writer checks the budget again before each removal,
    /// so concurrent callers do not remove more than needed.
    pub(super) async fn evict(&self, skip: &[Arc<CachedRange>]) {
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
    /// No lock: one check of the queue, one index load, and for a batch that touches cached
    /// ranges, the copies into the buffers and one store per touched range and per batch. The
    /// only read-modify-writes are the claim of a buffer's writer role, once per run of entries
    /// on a line that readers do not load, and the count of buffer memory when a chunk is
    /// allocated. A commit never waits for the cache: a row that does not fit under the memory
    /// cap removes its range instead.
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
        let cap = self.options.tuning.max_recent_bytes;
        // The ranges are references into the index that the caller holds, so the batch changes
        // no reference count that readers share.
        let mut touched: Vec<&Arc<CachedRange>> = Vec::new();
        let mut expiring: Vec<&Arc<CachedRange>> = Vec::new();
        let mut overflowed: Vec<&Arc<CachedRange>> = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let Some(range) = covering(index, &entries[i].key) else {
                i += 1;
                continue;
            };
            // A batch's entries come in key order, so the entries of a range are adjacent. The
            // writer appends each run of them with one claim of the buffer.
            let layers = range.layers.load();
            let mut buffer = layers.active.writer();
            let mut appended = false;
            let mut full = false;
            while let Some(entry) = entries.get(i).filter(|e| range.contains(&e.key)) {
                i += 1;
                if entry.expire_ts.is_some() {
                    // The cache does not hold rows that expire. See RFC 0040, "TTL".
                    if !expiring.iter().any(|r| Arc::ptr_eq(r, range)) {
                        expiring.push(range);
                    }
                    continue;
                }
                if full {
                    continue;
                }
                // A range registered at `W`, or flushed or merged up to `W`, holds every batch
                // at or below `W` in its files, and receives only later batches.
                debug_assert!(
                    entry.seq > layers.covered_to,
                    "a batch at {} reached a range whose files hold {}",
                    entry.seq,
                    layers.covered_to
                );
                if buffer.append_within(entry, cap) {
                    appended = true;
                } else {
                    // The row does not fit under the cap. The range leaves the cache at
                    // `seq - 1`; its rows of this batch that the buffer already holds are above
                    // that seal, so no read uses them.
                    full = true;
                    overflowed.push(range);
                }
            }
            drop(buffer);
            if appended && !touched.iter().any(|r| Arc::ptr_eq(r, range)) {
                range.last_write_seq.0.store(seq, Ordering::Release);
                touched.push(range);
            }
        }
        for range in &expiring {
            // The buffer does not hold this write, so seal it before the write.
            self.remove_at(range, seq - 1, RemoveReason::ExpiringWrite);
        }
        if !overflowed.is_empty() {
            self.remove_at_cap(&overflowed, seq - 1);
        }
        // A removed range must keep its last layers: a reader that loaded the index before the
        // removal can still pin them, and a new active buffer would look complete to it.
        let removals = !expiring.is_empty() || !overflowed.is_empty();
        for range in &touched {
            if !removals || self.is_indexed(range) {
                self.maybe_seal(range, seq);
            }
        }
    }

    /// Removes `ranges`, whose rows of the current batch did not fit under the memory cap, at
    /// `sealed_at`, then removes the indexed ranges with the largest unflushed buffers until the
    /// memory would be under the low-water mark once readers and flushes drop them (design 01
    /// section 9).
    fn remove_at_cap(&self, ranges: &[&Arc<CachedRange>], sealed_at: u64) {
        let mut freed = 0u64;
        for range in ranges {
            let unflushed = range.unflushed_bytes();
            if self.remove_at(range, sealed_at, RemoveReason::Overflow) {
                freed += unflushed;
            }
        }
        let low = self.options.tuning.low_water();
        let projected = |freed: u64| self.delta_memory.bytes().saturating_sub(freed);
        if projected(freed) <= low {
            return;
        }
        let mut largest: Vec<(u64, Arc<CachedRange>)> = self
            .index
            .load()
            .values()
            .map(|r| (r.unflushed_bytes(), r.clone()))
            .filter(|(bytes, _)| *bytes > 0)
            .collect();
        largest.sort_by(|a, b| b.0.cmp(&a.0));
        for (unflushed, range) in largest {
            if projected(freed) <= low {
                break;
            }
            if self.remove_at(&range, sealed_at, RemoveReason::Overflow) {
                freed += unflushed;
            }
        }
    }

    /// Seals the active buffer of `range` at `seq`, the batch just applied, if its chunks reached
    /// `range_seal_bytes`.
    fn maybe_seal(&self, range: &Arc<CachedRange>, seq: u64) {
        // A guard and borrows: the check changes no reference count unless the buffer seals.
        let layers = range.layers.load();
        if layers.active.allocated_bytes() < self.options.tuning.range_seal_bytes as u64 {
            return;
        }
        self.seal_at(range, &layers, seq);
        self.stats.seals_full.increment(1);
    }

    /// Seals the active buffer of `layers`, the current layers of `range`, at `seq` (design 01
    /// section 7.1): the buffer holds every version of the range above its `lo` up to `seq`, and
    /// receives none after it. A new empty buffer for the versions above `seq` takes its place,
    /// and the sealed one goes first in the sealed list, for the maintenance task to flush.
    fn seal_at(&self, range: &Arc<CachedRange>, layers: &Layers, seq: u64) {
        let sealed = layers.active.clone();
        // The word is on the buffer, so every set of layers that holds this buffer as its active
        // buffer, older ones included, sees the seal.
        sealed.sealed_at().store(seq, Ordering::SeqCst);
        let mut buffers = Vec::with_capacity(layers.sealed.len() + 1);
        buffers.push(sealed);
        buffers.extend(layers.sealed.iter().cloned());
        self.store_layers(
            range,
            Layers {
                active: self.new_buffer(seq),
                sealed: buffers,
                runs: layers.runs.clone(),
                view: layers.view.clone(),
                chunks: layers.chunks.clone(),
                covered_to: layers.covered_to,
                end: range.end.clone(),
            },
        );
        self.maintenance.notify(range);
    }

    /// Publishes a new layer set of `range`.
    fn store_layers(&self, range: &CachedRange, layers: Layers) {
        layers.check(&range.start, self.options.tuning.max_runs);
        range.layers.store(Arc::new(layers));
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
                splits,
                origin,
                reply,
            } => {
                // Above the high-water mark, a load would add rows to buffers that are nearly
                // full (design 01 section 9).
                let answer = if self.delta_memory.bytes() > self.options.tuning.high_water() {
                    self.stats.loads_rejected_memory.increment(1);
                    Err(SlateDBError::RangeCacheLoadRejected(
                        "the range cache's write buffers are nearly full",
                    ))
                } else {
                    Ok(self.register_at(start, end, &splits, origin, w))
                };
                let _ = reply.send(answer);
            }
            Request::AttachBase {
                range,
                base,
                base_seq,
                split,
                reply,
            } => {
                let _ = reply.send(self.attach_at(&range, base, base_seq, split, w));
            }
            Request::Split { range, plan, reply } => {
                let _ = reply.send(self.split_request_at(&range, plan, w));
            }
            Request::InstallRun {
                range,
                run,
                buffers,
                replaces,
                view,
                reply,
            } => {
                let _ = reply.send(self.install_run_at(&range, run, &buffers, &replaces, view));
            }
            Request::InstallMajor {
                range,
                chunks_was,
                absorbed,
                base,
                base_seq,
                view,
                reply,
            } => {
                let installed =
                    self.install_major_at(&range, &chunks_was, &absorbed, base, base_seq, view);
                let _ = reply.send(installed);
            }
            Request::QuietMerge {
                left,
                right,
                chunks_was,
                rate,
                reply,
            } => {
                let merged = self.quiet_merge_at(&left, &right, &chunks_was, rate, w);
                if let Some(reply) = reply {
                    let _ = reply.send(merged);
                }
            }
            Request::Seal { range, buffer } => {
                if self.is_indexed(&range) {
                    let layers = range.layers.load();
                    if Arc::ptr_eq(&layers.active, &buffer) && !buffer.is_empty() {
                        self.seal_at(&range, &layers, w);
                        self.stats.seals_pressure.increment(1);
                    }
                }
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
                    if self.within_budgets() {
                        break;
                    }
                    if !victim.layers.load().chunks.loading() {
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
    /// batch finds the new ranges and goes to their buffers.
    fn register_at(
        &self,
        start: Bytes,
        end: Option<Bytes>,
        splits: &[Bytes],
        origin: Origin,
        w: u64,
    ) -> Registered {
        let now = self.now_ms();
        let mut index = RangeIndex::clone(&self.index.load());
        let parts = split_parts(uncovered(&index, &start, end.as_ref()), splits);
        let mut ranges = Vec::with_capacity(parts.len());
        for (part_start, part_end) in parts {
            let layers = Layers {
                active: self.new_buffer(w),
                sealed: Vec::new(),
                runs: Vec::new(),
                view: None,
                // A loading range serves no read. Its runs start at `w`, the base's `base_seq`.
                chunks: Chunks::Loading,
                covered_to: w,
                end: part_end.clone(),
            };
            let mut range = CachedRange::new(
                part_start.clone(),
                part_end,
                layers,
                origin,
                now,
                self.writer.take_slot(),
            );
            // `splits` are ascending: a part that starts at one starts where the cache cut.
            range.cut_start = splits.binary_search(&part_start).is_ok();
            let range = Arc::new(range);
            index.insert(part_start, range.clone());
            ranges.push(range);
        }
        if !ranges.is_empty() {
            self.index.store(Arc::new(index));
            self.update_gauges();
        }
        Registered { seq: w, ranges }
    }

    /// Publishes the base of a loading range (`AttachBase`), as one chunk over the range. It keeps
    /// the buffers, which hold the batches after `base_seq`, and the runs that flushes installed
    /// meanwhile, which start at `base_seq`, so no reader of an earlier set of layers loses a
    /// signal. Returns the ranges that hold the base: the range, or with `split`, the children
    /// that replace it at `w` (design 04 section 3.4) when the split was built on the range's runs
    /// and view and no flush or major merge runs for it; otherwise the range attaches whole, and
    /// the maintenance task splits it later. None if the range was removed or the base is not for
    /// its registration.
    fn attach_at(
        &self,
        range: &Arc<CachedRange>,
        base: Arc<RangeFile>,
        base_seq: u64,
        split: Option<SplitPlan>,
        w: u64,
    ) -> Vec<Arc<CachedRange>> {
        if !self.is_indexed(range) {
            // Removed, or removed and registered again: the file is deleted on drop.
            return Vec::new();
        }
        let layers = range.layers.load_full();
        if !layers.chunks.loading() || layers.floor() != base_seq {
            return Vec::new();
        }
        self.stats.loads.increment(1);
        self.stats.load_bytes.increment(base.file_len());
        let now = self.hits.access_clock().fetch_add(2, Ordering::Relaxed) + 2;
        range.heat.hits.set_last_access(now - 1);
        if let Some(plan) = split {
            let fits = plan.children.len() > 1
                && plan.children.iter().all(|c| {
                    c.chunks
                        .iter()
                        .all(|chunk| Arc::ptr_eq(&chunk.file, &base) && chunk.base_seq == base_seq)
                })
                && same_arcs(&layers.runs, &plan.runs_was)
                && same_view(&layers.view, &plan.view_was)
                && !range.jobs.flushing()
                && !range.jobs.merging();
            if fits {
                let children = self.split_at(range, &layers, plan.children, plan.rate, w);
                self.stats.splits_attach.increment(1);
                return children;
            }
            self.stats.splits_rejected.increment(1);
        }
        self.store_layers(
            range,
            Layers {
                active: layers.active.clone(),
                sealed: layers.sealed.clone(),
                runs: layers.runs.clone(),
                view: layers.view.clone(),
                chunks: one_chunk(base, base_seq),
                covered_to: layers.covered_to,
                end: range.end.clone(),
            },
        );
        self.maintenance.notify(range);
        vec![range.clone()]
    }

    /// Applies a split that the maintenance task planned (`Split`, design 04 section 3.2).
    ///
    /// Preconditions: the range is indexed and loaded; it still holds the base, the runs and the
    /// view that the plan was built on (seals since then are allowed); no flush or major merge
    /// runs for it (X28); the plan has at least two children. Otherwise the split is rejected,
    /// and the children's views go when the plan drops.
    fn split_request_at(&self, range: &Arc<CachedRange>, plan: SplitPlan, w: u64) -> bool {
        let layers = range.layers.load_full();
        let fits = self.is_indexed(range)
            && !layers.chunks.loading()
            && layers.chunks.same(&plan.chunks_was)
            && same_arcs(&layers.runs, &plan.runs_was)
            && same_view(&layers.view, &plan.view_was)
            && plan.children.len() > 1
            && !range.jobs.flushing()
            && !range.jobs.merging();
        if !fits {
            self.stats.splits_rejected.increment(1);
            return false;
        }
        self.split_at(range, &layers, plan.children, plan.rate, w);
        self.stats.splits_maintenance.increment(1);
        true
    }

    /// Replaces `range`, whose layers are `layers`, by `children` at the writer boundary `w`
    /// (design 04 section 3.2 and I13), and returns them in key order: the parent's active buffer
    /// ends at `w`, the children's layer sets share its sources (see
    /// [`super::split::children_layers`]), and one index store puts the children in place of the
    /// range. Each child takes a share of the range's read rate, and of the reads of its base, by
    /// the bytes of its base.
    fn split_at(
        &self,
        range: &Arc<CachedRange>,
        layers: &Layers,
        children: Vec<SplitChild>,
        rate: f64,
        w: u64,
    ) -> Vec<Arc<CachedRange>> {
        let shares: Vec<u64> = children
            .iter()
            .map(|c| super::split::base_share(&c.chunks, &c.start, c.end.as_ref()).0)
            .collect();
        let total: u64 = shares.iter().sum();
        let count = children.len();
        let now_ms = self.now_ms();
        let built = super::split::children_layers(
            layers,
            children,
            w,
            self.options.tuning.max_runs,
            |lo| self.new_buffer(lo),
        );
        let mut index = RangeIndex::clone(&self.index.load());
        index.remove(&range.start);
        let mut made = Vec::with_capacity(count);
        for ((start, end, child_layers), bytes) in built.into_iter().zip(shares) {
            let share = if total == 0 {
                1.0 / count as f64
            } else {
                bytes as f64 / total as f64
            };
            let inherited = super::Inherited {
                share,
                rate,
                now_ms,
            };
            let child = Arc::new(CachedRange::split_from(
                range,
                start.clone(),
                end,
                child_layers,
                inherited,
                self.writer.take_slot(),
            ));
            index.insert(start, child.clone());
            made.push(child);
        }
        self.index.store(Arc::new(index));
        self.writer.give_back_slot(range.slot);
        self.stats.split_ranges.increment(count as u64);
        self.update_gauges();
        for child in &made {
            self.maintenance.notify(child);
        }
        made
    }

    /// Merges `left` and `right` into one range at the writer boundary `w` (`QuietMerge`, design 04
    /// section 3.6 and I14), and returns whether it did.
    ///
    /// Preconditions: both are indexed and adjacent (`left` ends where `right` starts); both have
    /// a base, still `chunks_was`, and are quiet: no run, no sealed buffer, an empty active buffer,
    /// and no flush, major merge or split in flight; both are automatic, or both explicit with a
    /// key between them that the cache cut (see [`CachedRange::may_merge`]). Otherwise it rejects
    /// the merge, and the task may ask again. The sizes are the task's to check: a quiet range's
    /// base does not change.
    ///
    /// In a quiet range no version of a chunk's keys lies above the chunk's `base_seq`: with no run
    /// there is none up to `covered_to` (I4c), with no sealed buffer the active buffer starts
    /// there, and it holds none. So every chunk holds the state of its keys at `w`, and the merged
    /// range takes both chunk lists labelled `w`, no run, a new empty active buffer at `w` and
    /// `covered_to = w`. Two adjacent chunks of one file, as a split leaves them, become one chunk:
    /// the file holds the state of both spans at `w`. The writer stores `sealed_at = w` on both
    /// active buffers before the index store, as a split and a removal do: a reader that pinned
    /// either range and loads a sequence number above `w` tries again and finds the merged range
    /// (X36); one at or below `w` reads the old layers, complete for it. A reader below `w` finds
    /// the merged range's chunks too new and uses the normal path; quiet ranges are seldom read at
    /// old snapshots.
    fn quiet_merge_at(
        &self,
        left: &Arc<CachedRange>,
        right: &Arc<CachedRange>,
        chunks_was: &[Chunks; 2],
        rate: f64,
        w: u64,
    ) -> bool {
        let (a, b) = (left.layers.load_full(), right.layers.load_full());
        let quiet = |range: &CachedRange, layers: &Layers| {
            !layers.chunks.loading()
                && layers.runs.is_empty()
                && layers.sealed.is_empty()
                && layers.active.is_empty()
                && !range.jobs.flushing()
                && !range.jobs.merging()
                && !range.jobs.splitting()
        };
        let fits = self.is_indexed(left)
            && self.is_indexed(right)
            && left.end.as_ref() == Some(&right.start)
            && CachedRange::may_merge(left, right)
            && quiet(left, &a)
            && quiet(right, &b)
            && a.chunks.same(&chunks_was[0])
            && b.chunks.same(&chunks_was[1]);
        if !fits {
            left.jobs.quiet_merge_asked.store(false, Ordering::Relaxed);
            right.jobs.quiet_merge_asked.store(false, Ordering::Relaxed);
            self.stats.quiet_merges_rejected.increment(1);
            return false;
        }
        // I14: both active buffers end at `w` before the index changes.
        a.active.sealed_at().store(w, Ordering::SeqCst);
        b.active.sealed_at().store(w, Ordering::SeqCst);
        let layers = Layers {
            active: self.new_buffer(w),
            sealed: Vec::new(),
            runs: Vec::new(),
            view: None,
            chunks: merged_chunks(&a.chunks, &right.start, &b.chunks, w),
            covered_to: w,
            end: right.end.clone(),
        };
        layers.check(&left.start, self.options.tuning.max_runs);
        let merged = Arc::new(CachedRange::merged_from(
            left,
            right,
            layers,
            rate,
            self.now_ms(),
            self.writer.take_slot(),
        ));
        let mut index = RangeIndex::clone(&self.index.load());
        index.remove(&right.start);
        index.insert(left.start.clone(), merged.clone());
        self.index.store(Arc::new(index));
        self.writer.give_back_slot(left.slot);
        self.writer.give_back_slot(right.slot);
        self.stats.quiet_merges.increment(1);
        self.update_gauges();
        self.maintenance.notify(&merged);
        true
    }

    /// Installs a flushed run (`InstallRun`, design 01 sections 6.2 and 7.3): one store that puts
    /// the run first in place of the runs it replaces, raises `covered_to` to its `hi`, removes
    /// its buffers from the sealed list and publishes the range's view with the run.
    ///
    /// Preconditions: the range is indexed; `buffers` (newest first) are still its oldest sealed
    /// buffers; `replaces` (newest first) are still its newest runs; the run starts where the
    /// oldest replaced run starts, or at `covered_to` for a plain flush, and ends at the newest
    /// buffer's seal; the range holds at most `max_runs` runs afterwards; the view was built on
    /// the range's view (see [`ViewSwap`]). Otherwise the run is rejected, and its file is
    /// deleted when it drops. A major merge in flight that holds older runs does not matter: the
    /// run goes first and the other runs stay.
    pub(super) fn install_run_at(
        &self,
        range: &Arc<CachedRange>,
        run: Arc<Run>,
        buffers: &[Arc<Delta>],
        replaces: &[Arc<Run>],
        view: ViewSwap,
    ) -> bool {
        if !self.is_indexed(range) {
            self.stats.installs_rejected.increment(1);
            return false;
        }
        let layers = range.layers.load_full();
        let kept = layers.sealed.len().checked_sub(buffers.len());
        let oldest = !buffers.is_empty()
            && kept.is_some_and(|kept| same_arcs(&layers.sealed[kept..], buffers));
        let newest = replaces.len() <= layers.runs.len()
            && same_arcs(&layers.runs[..replaces.len()], replaces);
        let lo = replaces.last().map_or(layers.covered_to, |r| r.lo);
        if !oldest
            || !newest
            || !same_view(&layers.view, &view.was)
            || view.new.is_none()
            || run.lo != lo
            || run.hi != buffers[0].sealed_at().load(Ordering::Acquire)
            || layers.runs.len() - replaces.len() >= self.options.tuning.max_runs
        {
            self.stats.installs_rejected.increment(1);
            return false;
        }
        self.stats.flushes.increment(1);
        self.stats.flushed_buffers.increment(buffers.len() as u64);
        if buffers.len() > 1 {
            self.stats.multi_buffer_flushes.increment(1);
        }
        if replaces.is_empty() {
            self.stats.flush_bytes.increment(run.file_bytes());
        } else {
            self.stats.flush_merges.increment(1);
            self.stats.flush_merge_bytes.increment(run.file_bytes());
            self.stats.flush_merge_runs.increment(replaces.len() as u64);
        }
        let mut runs = Vec::with_capacity(layers.runs.len() + 1 - replaces.len());
        let covered_to = run.hi;
        let covered_bytes = run.file_bytes();
        runs.push(run);
        runs.extend(layers.runs[replaces.len()..].iter().cloned());
        let sealed = layers.sealed[..kept.unwrap_or(0)].to_vec();
        // Counted before the store: a look at the new layers sees the count.
        range.jobs.record_flush(covered_bytes);
        self.store_layers(
            range,
            Layers {
                active: layers.active.clone(),
                sealed,
                runs,
                view: view.new,
                chunks: layers.chunks.clone(),
                covered_to,
                end: range.end.clone(),
            },
        );
        true
    }

    /// Installs a base from a major merge (`InstallMajor`, design 01 section 6.2): the new base,
    /// one chunk over the range, holds the state at `base_seq`, the `hi` of the newest absorbed
    /// run, and replaces the old chunks and the absorbed runs, which no chunk needs any more. Runs
    /// that flushes installed during the merge stay, and so do `covered_to` and the buffers. No
    /// history is kept (stage 3b): a snapshot older than the new base uses the normal read path.
    ///
    /// Preconditions: the range is indexed; its base is still `chunks_was` (see [`Chunks::same`]);
    /// `absorbed` are still its oldest runs; `base_seq` is the newest absorbed run's `hi`; the
    /// view, the range's view without the absorbed runs, was built on the range's view (see
    /// [`ViewSwap`]).
    pub(super) fn install_major_at(
        &self,
        range: &Arc<CachedRange>,
        chunks_was: &Chunks,
        absorbed: &[Arc<Run>],
        base: Arc<RangeFile>,
        base_seq: u64,
        view: ViewSwap,
    ) -> bool {
        if !self.is_indexed(range) {
            self.stats.installs_rejected.increment(1);
            return false;
        }
        let layers = range.layers.load_full();
        let same_base = layers.chunks.same(chunks_was);
        let kept = layers.runs.len().checked_sub(absorbed.len());
        let oldest = kept.is_some_and(|kept| {
            layers.runs[kept..]
                .iter()
                .zip(absorbed)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        });
        if !same_base
            || !oldest
            || !same_view(&layers.view, &view.was)
            || absorbed.first().is_none_or(|r| r.hi != base_seq)
        {
            self.stats.installs_rejected.increment(1);
            return false;
        }
        let kept = kept.unwrap_or(0);
        self.stats.rebuilds.increment(1);
        self.stats.rebuild_bytes.increment(base.file_len());
        self.stats
            .rebuild_flushes
            .increment(range.jobs.record_major(base.file_len()));
        self.store_layers(
            range,
            Layers {
                active: layers.active.clone(),
                sealed: layers.sealed.clone(),
                runs: layers.runs[..kept].to_vec(),
                view: view.new,
                chunks: one_chunk(base, base_seq),
                covered_to: layers.covered_to,
                end: range.end.clone(),
            },
        );
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
            .sealed_at()
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
                    .sealed_at()
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

/// The base of the range that a quiet merge at `w` makes of a range with base `left` and the range
/// with base `right` that starts at `right_start`: their chunks in key order, each labelled `w`
/// (see [`RangeCache::quiet_merge_at`]). Adjacent chunks of one file become one chunk, so a merge
/// of the children of a split, with no major merge since, gives back one chunk.
fn merged_chunks(left: &Chunks, right_start: &Bytes, right: &Chunks, w: u64) -> Chunks {
    let mut chunks: Vec<Chunk> = Vec::with_capacity(left.len() + right.len());
    let mut starts: Vec<Bytes> = Vec::with_capacity(left.len() + right.len());
    let all = left
        .iter()
        .enumerate()
        .map(|(i, chunk)| (left.start(i), chunk))
        .chain(right.iter().enumerate().map(|(i, chunk)| {
            let start = if i == 0 {
                Some(right_start)
            } else {
                right.start(i)
            };
            (start, chunk)
        }));
    for (start, chunk) in all {
        if chunks
            .last()
            .is_some_and(|last| Arc::ptr_eq(&last.file, &chunk.file))
        {
            continue;
        }
        if let (Some(start), false) = (start, chunks.is_empty()) {
            starts.push(start.clone());
        }
        chunks.push(Chunk::new(chunk.file.clone(), w));
    }
    match chunks.len() {
        1 => Chunks::One(chunks.remove(0)),
        _ => Chunks::Many(Arc::new(ChunkList::new(chunks, starts))),
    }
}

/// Returns true if `a` and `b` hold the same objects, in the same order.
fn same_arcs<T>(a: &[Arc<T>], b: &[Arc<T>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y))
}

/// Splits each part `[start, end)` at the keys of `splits` (ascending) that fall strictly inside
/// it.
fn split_parts(
    parts: Vec<(Bytes, Option<Bytes>)>,
    splits: &[Bytes],
) -> Vec<(Bytes, Option<Bytes>)> {
    if splits.is_empty() {
        return parts;
    }
    let mut out = Vec::with_capacity(parts.len() + splits.len());
    for (start, end) in parts {
        let mut from = start;
        for split in splits {
            let inside = *split > from && end.as_ref().is_none_or(|end| split < end);
            if inside {
                out.push((from, Some(split.clone())));
                from = split.clone();
            }
        }
        out.push((from, end));
    }
    out
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
        assert_eq!(range.layers.load().only_chunk().base_seq, before);
        assert_eq!(t.cache().writer.applied(), before);
        // A later batch goes to the delta, above the file.
        t.db.put(key(20), "new").await.unwrap();
        let seq = t.committed();
        let layers = range.layers.load();
        assert_eq!(layers.active.len(), 1);
        assert!(seq > layers.only_chunk().base_seq);
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
        let sealed_at = layers.active.sealed_at().load(Ordering::SeqCst);
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

    /// Every read that the cache does not serve counts once, by its reason (design 04 section
    /// 8.1), and every read counts once in all: served or fallen back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn should_count_each_fallback_once_by_reason() {
        use crate::config::{DurabilityLevel, ScanOptions};
        use crate::iter::IterationOrder;
        use crate::range_cache::stats::{FALLBACK_COUNT, GET_COUNT, MIXED_SCAN_COUNT, SCAN_COUNT};

        let t = open(|_| {}).await;
        put_all(&t.db, 0..100).await;
        let fallback =
            |op: &str, reason: &str| t.metric(FALLBACK_COUNT, &[("op", op), ("reason", reason)]);
        let scan = |range: (std::ops::Bound<Bytes>, std::ops::Bound<Bytes>),
                    options: ScanOptions| {
            let db = t.db.clone();
            async move {
                let mut iter = db.scan_with_options(range, &options).await.unwrap();
                while iter.next().await.unwrap().is_some() {}
            }
        };
        let traced = || ScanOptions::default().with_tracing_options(Some(TracingOptions::new("n")));

        // No range at all.
        t.db.get(key(5)).await.unwrap();
        scan(span(5, 10), ScanOptions::default()).await;
        assert_eq!(
            (fallback("get", "uncovered"), fallback("scan", "uncovered")),
            (1, 1)
        );

        // Read options that the cache does not serve, the first in order.
        let remote = ReadOptions::default().with_durability_filter(DurabilityLevel::Remote);
        t.db.get_with_options(key(5), &remote).await.unwrap();
        let dirty = ReadOptions::default().with_dirty(true);
        t.db.get_with_options(key(5), &dirty).await.unwrap();
        t.db.get_with_options(key(5), &normal()).await.unwrap();
        let descending = ScanOptions::default().with_order(IterationOrder::Descending);
        scan(span(5, 10), descending.clone()).await;
        scan(
            span(5, 10),
            descending.with_tracing_options(Some(TracingOptions::new("n"))),
        )
        .await;
        scan(span(5, 10), traced()).await;
        for (op, reason, n) in [
            ("get", "remote", 1),
            ("get", "dirty", 1),
            ("get", "traced", 1),
            ("scan", "descending", 2),
            ("scan", "traced", 1),
        ] {
            assert_eq!(fallback(op, reason), n, "{op} {reason}");
        }

        // A range that loads.
        t.fail("range-cache-load", "pause");
        let load = {
            let db = t.db.clone();
            tokio::spawn(async move { db.cache_range(span(10, 50)).await })
        };
        tokio::time::timeout(PROMPT, async {
            while !t.db.cached_ranges().first().is_some_and(|r| r.loading) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        t.db.get(key(20)).await.unwrap();
        scan(span(20, 30), ScanOptions::default()).await;
        assert_eq!(
            (fallback("get", "loading"), fallback("scan", "loading")),
            (1, 1)
        );
        t.fail("range-cache-load", "off");
        load.await.unwrap().unwrap();

        // Served: a get, a scan, and a mixed scan whose gap the normal path reads.
        t.db.get(key(20)).await.unwrap();
        scan(span(20, 30), ScanOptions::default()).await;
        scan(span(40, 60), ScanOptions::default()).await;
        assert_eq!(t.metric(SCAN_COUNT, &[]), 1);
        assert_eq!(t.metric(MIXED_SCAN_COUNT, &[]), 1);
        assert_eq!(fallback("gap", "uncovered"), 1);

        // Every read counted once: 6 gets and 7 scans.
        let served_gets: i64 = ["hit", "absent", "merge"]
            .iter()
            .map(|r| t.metric(GET_COUNT, &[("result", r)]))
            .sum();
        let get_fallbacks: i64 = ["uncovered", "loading", "remote", "dirty", "traced"]
            .iter()
            .map(|r| fallback("get", r))
            .sum();
        assert_eq!(served_gets + get_fallbacks, 6);
        let scan_fallbacks: i64 = ["uncovered", "loading", "descending", "traced"]
            .iter()
            .map(|r| fallback("scan", r))
            .sum();
        assert_eq!(
            t.metric(SCAN_COUNT, &[]) + t.metric(MIXED_SCAN_COUNT, &[]) + scan_fallbacks,
            7
        );
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
