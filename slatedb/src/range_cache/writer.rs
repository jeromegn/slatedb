//! Batch-writer ownership of range cache state transitions.
//!
//! Only the batch writer mutates the range index, range layers, `sealed_at` words, and write
//! watermarks. Other tasks send [`Request`] values over a lock-free queue; the writer drains
//! them at writer boundaries (between batches, when no batch sits between memtable insert and
//! commit) and wakes via [`BatchWriterMessage::RangeCache`] when idle. Each install verifies
//! pointer identity (`Arc::ptr_eq`) and rejects stale changes. The writer also seals full buffers
//! and removes ranges to enforce the hard write-buffer memory cap without blocking commits.

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
    /// Register loading ranges for uncovered parts of `[start, end)`, split at `splits`.
    Register {
        start: Bytes,
        end: Option<Bytes>,
        splits: Vec<Bytes>,
        origin: Origin,
        reply: oneshot::Sender<Result<Registered, SlateDBError>>,
    },
    /// Publish the base of a loading range at `base_seq`, optionally splitting it into children
    /// in the same store. The reply lists the resulting range(s), or is empty if removed meanwhile.
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
    /// Replace `range` by the children of `plan`, sharing files, runs, and buffers if the range
    /// still holds the planned base, runs, and view.
    Split {
        range: Arc<CachedRange>,
        plan: SplitPlan,
        reply: oneshot::Sender<bool>,
    },
    /// Merge adjacent quiet ranges `left` and `right` into one if both remain quiet and hold
    /// `chunks_was`. `rate` is the sum of their read rates.
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

/// Compare-and-swap pair for delta views published with an install (`was` must match current).
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
    /// Highest sequence number applied to all affected ranges (or covered by loading ranges).
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

    /// Registers loading ranges for uncovered parts of `[start, end)` at a writer boundary.
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

    /// Asks the writer to merge adjacent quiet ranges `left` and `right` without waiting.
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

    /// Requests a quiet merge of `left` and `right` and returns whether it succeeded.
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

    /// Evicts ranges off the writer until disk and metadata budgets hold, skipping `skip`.
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
    /// Applies one write batch after memtable insert and before advancing the committed sequence
    /// number. Never blocks: writes exceeding the buffer memory cap remove their range instead.
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
                    // Range cache does not store expiring rows.
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

    /// Removes overflowed `ranges` at `sealed_at`, then removes ranges with the largest unflushed
    /// buffers until projected buffer memory falls below the low-water mark.
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

    /// Seals `layers.active` at `seq`, installs a fresh active buffer, and notifies maintenance.
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
                // Reject loads above the high-water mark to avoid filling nearly full buffers.
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

    /// Registers loading ranges for uncovered parts of `[start, end)` at writer boundary `w`.
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
            // `splits` are ascending: a part matching one starts at a cache-chosen boundary.
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

    /// Publishes the base of a loading range while preserving buffers and runs, splitting
    /// immediately at `w` when `split` matches the current layers.
    fn attach_at(
        &self,
        range: &Arc<CachedRange>,
        base: Arc<RangeFile>,
        base_seq: u64,
        split: Option<SplitPlan>,
        w: u64,
    ) -> Vec<Arc<CachedRange>> {
        if !self.is_indexed(range) {
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

    /// Applies a planned zero-copy split if the range is still indexed, loaded, holds the planned
    /// base, runs, and view, has no active flush or major merge, and has at least two children.
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

    /// Replaces `range` with `children` at writer boundary `w` without copying data.
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

    /// Merges adjacent quiet ranges `left` and `right` at writer boundary `w` without copying.
    ///
    /// Both ranges must be indexed, adjacent, quiet (no runs, no sealed buffers, empty active
    /// buffer, no in-flight jobs), eligible under [`CachedRange::may_merge`], and still holding
    /// `chunks_was`. Seals both active buffers at `w` before publishing the merged range with all
    /// chunks labeled `w` (coalescing adjacent chunks from the same file).
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

    /// Installs a flushed run, removes its sealed buffers, updates `covered_to`, and swaps the
    /// delta view atomically. Rejects the install if range identity, buffers, runs, or view moved.
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
        if replaces.is_empty() {
            self.stats.flush_bytes.increment(run.file_bytes());
        } else {
            self.stats.flush_merges.increment(1);
            self.stats.flush_merge_bytes.increment(run.file_bytes());
        }
        let mut runs = Vec::with_capacity(layers.runs.len() + 1 - replaces.len());
        let covered_to = run.hi;
        let covered_bytes = run.file_bytes();
        runs.push(run);
        runs.extend(layers.runs[replaces.len()..].iter().cloned());
        let sealed = layers.sealed[..kept.unwrap_or(0)].to_vec();
        // Count before storing layers so maintenance sees the updated flush count.
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

    /// Installs a major-merged single-chunk base at `base_seq`, dropping `absorbed` runs and
    /// swapping the delta view. Newer runs flushed during the merge remain intact.
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
        range.jobs.record_major(base.file_len());
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

    /// Notifies automatic promotion when an automatic range leaves so it updates ghosts and
    /// failure backoff.
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

    /// Removes every range and refuses all queued and future requests when the writer stops.
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

/// Combines chunks from `left` and `right` (starting at `right_start`) labeled at `w`.
/// Adjacent chunks backed by the same file coalesce into one chunk.
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

    /// Removals seal at the writer boundary so snapshots opened after non-overlapping batches
    /// remain valid.
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

    /// Every unserved read counts once by fallback reason so served plus fallbacks equals total.
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

#[cfg(test)]
mod quiet_merge_tests {
    use std::ops::Bound::{self, Excluded, Included, Unbounded};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use bytes::Bytes;
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
        PIN_RETRY_COUNT, QUIET_MERGE_COUNT, QUIET_MERGE_REJECTED_COUNT, SPLIT_COUNT,
    };
    use crate::range_cache::{CachedRange, Chunks, RangeCache, RangeCacheOptions, MAX_RUNS_RANGE};

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

        async fn split(&self, start: usize, cuts: &[usize]) -> bool {
            let cuts = cuts.iter().map(|i| key(*i)).collect();
            self.cache().split(&self.range(start), cuts).await.unwrap()
        }

        async fn quiet_merge(&self, left: usize, right: usize) -> bool {
            self.cache()
                .quiet_merge(&self.range(left), &self.range(right))
                .await
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
            let _ = fail_parallel::cfg(self.fp_registry.clone(), "range-cache-pin", "off");
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
        let db = Db::builder("range_cache_quiet_merge", Arc::new(InMemory::new()))
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

    fn normal_scan() -> ScanOptions {
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

    async fn assert_scan_eq(
        reader: &crate::DbSnapshot,
        range: (Bound<Bytes>, Bound<Bytes>),
        label: &str,
    ) {
        let mut cached = reader.scan(range.clone()).await.unwrap();
        let mut normal = reader
            .scan_with_options(range, &normal_scan())
            .await
            .unwrap();
        loop {
            let (c, n) = (cached.next().await.unwrap(), normal.next().await.unwrap());
            assert_eq!(c, n, "{label}");
            if c.is_none() {
                break;
            }
        }
    }

    /// A point read pinned to one side of a quiet merge retries for sequence numbers above the
    /// merge boundary and reads the merged range.
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

    /// Only adjacent quiet ranges separated by a cache-chosen split merge; same-file chunks
    /// coalesce into one chunk while distinct files produce a multi-chunk base.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn should_merge_only_quiet_neighbours_that_a_split_cut() {
        let t = open(|o| o.tuning.range_seal_bytes = 4 * 1024).await;
        put_all(&t, 0, 500).await;
        t.db.cache_range(span(0, 100)).await.unwrap();
        t.db.cache_range(span(100, 500)).await.unwrap();
        assert!(t.split(100, &[200, 300, 400]).await);
        let rejected = || t.metric(QUIET_MERGE_REJECTED_COUNT, &[]);
        // Ranges named separately by the application stay separate.
        assert!(!t.quiet_merge(0, 100).await);
        assert_eq!(rejected(), 1);
        // Active buffer or runs disqualify a range from quiet merge.
        t.db.put(key(250), value(250, 1)).await.unwrap();
        assert!(!t.quiet_merge(200, 300).await);
        t.merge_above(200, 200, 300).await;
        let range = t.range(200);
        if !range.layers.load().active.is_empty() || !range.layers.load().runs.is_empty() {
            assert!(!t.quiet_merge(200, 300).await);
        }
        // Non-adjacent ranges do not merge.
        assert!(!t.cache().quiet_merge(&t.range(100), &t.range(300)).await);

        // Quiet children sharing one file coalesce back into one chunk at `w`.
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

        // Quiet ranges with distinct files produce a two-chunk base.
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
                .major_merge(&range, layers.chunks.clone(), layers.runs.clone())
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
                assert_scan_eq(reader, span(a, b), label).await;
            }
            for i in 190..410 {
                let cached = reader.get(key(i)).await.unwrap();
                let want = reader
                    .get_with_options(
                        key(i),
                        &ReadOptions::default()
                            .with_tracing_options(Some(TracingOptions::new("normal"))),
                    )
                    .await
                    .unwrap();
                assert_eq!(cached, want, "{label}: key {i}");
            }
        }
        // Splitting the two-chunk range then major-merging one child restores a single chunk.
        assert!(t.split(200, &[250]).await);
        t.merge_above(250, 250, 400).await;
        assert!(matches!(t.range(250).layers.load().chunks, Chunks::One(_)));
        let latest = t.db.snapshot().unwrap();
        assert_scan_eq(&latest, span(150, 450), "after the split").await;
        t.db.close().await.unwrap();
    }

    /// Maintenance merges quiet neighbours below `range_target_bytes / 8` without triggering
    /// another split.
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
                .major_merge(&range, layers.chunks.clone(), layers.runs.clone())
                .await
                .unwrap();
            let bytes = range.layers.load().base_share(&key(start)).0;
            assert!(bytes < 8 * 1024, "{start}..{end}: {bytes} bytes");
        }
        let splits = t.metric(SPLIT_COUNT, &[("when", "maintenance")]);
        t.wait("the quiet merge", |t| t.metric(QUIET_MERGE_COUNT, &[]) >= 1)
            .await;
        assert!(!t.starts().contains(&b));
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(t.starts().contains(&a) && !t.starts().contains(&b));
        assert_eq!(t.metric(SPLIT_COUNT, &[("when", "maintenance")]), splits);
        let snapshot = t.db.snapshot().unwrap();
        assert_scan_eq(&snapshot, (Unbounded, Unbounded), "after the merge").await;
        t.db.close().await.unwrap();
    }
}
