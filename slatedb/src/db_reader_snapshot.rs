//! Snapshots of a [`DbReader`](crate::DbReader).
//!
//! A [`DbReaderSnapshot`] keeps the reader state that was current when it was taken: the manifest,
//! the tables that WAL replay built, and the sequence number of that state. Its reads see that
//! state only, however far the reader moves on.
//!
//! The replayed tables are in memory, so a snapshot needs no WAL object. It needs the SSTs of its
//! manifest. In [`DbReaderMode::ManagedCheckpoint`](crate::DbReaderMode::ManagedCheckpoint), the
//! reader's checkpoint on that manifest protects them from garbage collection. A snapshot leases
//! that checkpoint: while a lease is held, the poller does not remove the checkpoint when it
//! replaces it, and it refreshes the checkpoint as it refreshes its own. The poller deletes a kept
//! checkpoint at the first poll after its last lease drops.
//!
//! The [`SnapshotRegistry`] keeps the live snapshots' sequence numbers and the leases. Snapshot
//! creation, snapshot drop and the poller take its lock. Reads do not.

use std::collections::btree_map;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use tokio::sync::watch;
use uuid::Uuid;

use crate::bytes_range::{ByteRangeBounds, BytesRange};
use crate::config::{ReadOptions, ScanOptions};
use crate::db_iter::DbIterator;
use crate::db_reader::{DbReaderInner, ReaderState};
use crate::error::SlateDBError;
use crate::manifest::ManifestCore;
use crate::snapshot_manager::RetentionPin;
use crate::types::KeyValue;
use crate::{Checkpoint, DbReadOps};

/// A consistent, read-only view of a [`DbReader`](crate::DbReader) at one sequence number.
///
/// Get one with [`DbReader::snapshot`](crate::DbReader::snapshot). Its reads see the reader
/// state that was current when it was taken, at that state's sequence number, while the reader
/// goes on following the database. The API is the same as the one of
/// [`DbSnapshot`](crate::DbSnapshot) on the writer.
///
/// ## Protection
///
/// In [`DbReaderMode::ManagedCheckpoint`](crate::DbReaderMode::ManagedCheckpoint), the snapshot
/// leases the reader checkpoint of its state. The reader keeps and refreshes that checkpoint while
/// the snapshot lives, so garbage collection does not delete the snapshot's SSTs. If the reader
/// cannot keep the checkpoint (it finds the checkpoint gone, or the checkpoint expires because the
/// poller could not refresh it), every later read of the snapshot fails with an error of kind
/// [`ErrorKind::Unavailable`](crate::ErrorKind::Unavailable). Take a new snapshot then. A long
/// snapshot keeps its manifest's SSTs in the store and its replayed tables in memory.
///
/// In [`DbReaderMode::Checkpoint`](crate::DbReaderMode::Checkpoint), the checkpoint that the
/// reader is pinned to protects the snapshot, as it protects the reader. In
/// [`DbReaderMode::FollowLatest`](crate::DbReaderMode::FollowLatest), nothing protects it.
///
/// The iterators of its scans hold the snapshot's sequence number and lease too, so an iterator
/// that outlives the snapshot still reads the files the lease protects. The lease ends when the
/// snapshot and every such iterator have dropped.
pub struct DbReaderSnapshot {
    seq: u64,
    state: Arc<ReaderState>,
    lease: Option<Arc<CheckpointLease>>,
    inner: Arc<DbReaderInner>,
}

impl DbReaderSnapshot {
    pub(crate) fn new(
        inner: Arc<DbReaderInner>,
        state: Arc<ReaderState>,
        lease: Option<Arc<CheckpointLease>>,
    ) -> Self {
        Self {
            seq: state.seq(),
            state,
            lease,
            inner,
        }
    }

    /// Get the sequence number this snapshot reads at: the durable sequence number of the reader
    /// state it was taken from.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Get a value from the snapshot with default read options.
    ///
    /// ## Arguments
    /// - `key`: the key to get
    ///
    /// ## Returns
    /// - `Result<Option<Bytes>, Error>`: the value if it exists, None otherwise
    pub async fn get<K: AsRef<[u8]> + Send>(&self, key: K) -> Result<Option<Bytes>, crate::Error> {
        self.get_with_options(key, &ReadOptions::default()).await
    }

    /// Get a value from the snapshot with custom read options.
    ///
    /// ## Arguments
    /// - `key`: the key to get
    /// - `options`: the read options to use (Note that [`ReadOptions::read_level`] has no effect
    ///   for readers, which can only observe committed state).
    ///
    /// ## Returns
    /// - `Result<Option<Bytes>, Error>`: the value if it exists, None otherwise
    pub async fn get_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<Bytes>, crate::Error> {
        self.get_key_value_with_options(key, options)
            .await
            .map(|kv_opt| kv_opt.map(|kv| kv.value))
    }

    /// Get a key-value pair from the snapshot with default read options.
    pub async fn get_key_value<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
    ) -> Result<Option<KeyValue>, crate::Error> {
        self.get_key_value_with_options(key, &ReadOptions::default())
            .await
    }

    /// Get a key-value pair from the snapshot with custom read options.
    pub async fn get_key_value_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<KeyValue>, crate::Error> {
        self.check_readable()?;
        self.inner
            .get_key_value_at(&self.state, self.seq, key, options)
            .await
            .map_err(Into::into)
    }

    /// Scan a range of keys using the default scan options.
    ///
    /// ## Arguments
    /// - `range`: the range of keys to scan
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    pub async fn scan<T>(&self, range: T) -> Result<DbIterator, crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        self.scan_with_options(range, &ScanOptions::default()).await
    }

    /// Scan a range of keys with the provided options.
    ///
    /// ## Arguments
    /// - `range`: the range of keys to scan
    /// - `options`: the scan options to use
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    pub async fn scan_with_options<T>(
        &self,
        range: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        let start = range.start_bound().map(Bytes::copy_from_slice);
        let end = range.end_bound().map(Bytes::copy_from_slice);
        let range = BytesRange::from((start, end));
        self.scan_inner(range, options, None).await
    }

    /// Scan keys that share the provided prefix, restricted to `subrange`,
    /// using the default scan options.
    ///
    /// The subrange bounds are key *suffixes* interpreted relative to the
    /// prefix: a bound `s` selects the full key `prefix ++ s`. Pass `..` to
    /// scan the prefix's entire keyspace.
    ///
    /// ## Arguments
    /// - `prefix`: the key prefix to scan
    /// - `subrange`: the range of key suffixes (relative to `prefix`) to
    ///   scan; `..` scans all keys with the prefix
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    pub async fn scan_prefix<P, T>(
        &self,
        prefix: P,
        subrange: T,
    ) -> Result<DbIterator, crate::Error>
    where
        P: AsRef<[u8]> + Send,
        T: ByteRangeBounds + Send,
    {
        self.scan_prefix_with_options(prefix, subrange, &ScanOptions::default())
            .await
    }

    /// Scan keys that share the provided prefix, restricted to `subrange`,
    /// with custom options. See [`Self::scan_prefix`] for the subrange
    /// semantics.
    ///
    /// ## Arguments
    /// - `prefix`: the key prefix to scan
    /// - `subrange`: the range of key suffixes (relative to `prefix`) to
    ///   scan; `..` scans all keys with the prefix
    /// - `options`: the scan options to use
    ///
    /// ## Returns
    /// - `Result<DbIterator, Error>`: An iterator with the results of the scan
    pub async fn scan_prefix_with_options<P, T>(
        &self,
        prefix: P,
        subrange: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        P: AsRef<[u8]> + Send,
        T: ByteRangeBounds + Send,
    {
        let prefix = Bytes::copy_from_slice(prefix.as_ref());
        let range = BytesRange::from_prefix_and_subrange(prefix.as_ref(), subrange);
        self.scan_inner(range, options, Some(prefix)).await
    }

    async fn scan_inner(
        &self,
        range: BytesRange,
        options: &ScanOptions,
        prefix: Option<Bytes>,
    ) -> Result<DbIterator, crate::Error> {
        self.check_readable()?;
        let iter = self
            .inner
            .scan_at(&self.state, self.seq, range, options, prefix)
            .await?;
        Ok(iter.retaining(RetentionPin::reader_snapshot(
            Arc::clone(&self.inner),
            self.seq,
            self.lease.clone(),
        )))
    }

    #[cfg(test)]
    pub(crate) fn has_lease(&self) -> bool {
        self.lease.is_some()
    }

    /// Fails if the reader is closed, or if the checkpoint that protects this snapshot's files
    /// lapsed.
    fn check_readable(&self) -> Result<(), SlateDBError> {
        self.inner.check_closed()?;
        if let Some(lease) = &self.lease {
            lease.check(self.inner.now_millis(), self.seq)?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl DbReadOps for DbReaderSnapshot {
    async fn get_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<Bytes>, crate::Error> {
        DbReaderSnapshot::get_with_options(self, key, options).await
    }

    async fn get_key_value_with_options<K: AsRef<[u8]> + Send>(
        &self,
        key: K,
        options: &ReadOptions,
    ) -> Result<Option<KeyValue>, crate::Error> {
        DbReaderSnapshot::get_key_value_with_options(self, key, options).await
    }

    async fn scan_with_options<T>(
        &self,
        range: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        T: ByteRangeBounds + Send,
    {
        DbReaderSnapshot::scan_with_options(self, range, options).await
    }

    async fn scan_prefix_with_options<P, T>(
        &self,
        prefix: P,
        subrange: T,
        options: &ScanOptions,
    ) -> Result<DbIterator, crate::Error>
    where
        P: AsRef<[u8]> + Send,
        T: ByteRangeBounds + Send,
    {
        DbReaderSnapshot::scan_prefix_with_options(self, prefix, subrange, options).await
    }
}

impl Drop for DbReaderSnapshot {
    fn drop(&mut self) {
        self.inner.snapshots.release(self.seq, self.lease.as_ref());
    }
}

/// A reader checkpoint that live snapshots read through.
///
/// The checkpoint protects the SSTs of its manifest while it is in the manifest. The reader never
/// removes a leased checkpoint, and it refreshes it before it expires. The lease lapses when the
/// reader finds the checkpoint gone, or when the expiry time that the reader last wrote has passed.
/// After that, garbage collection can delete the checkpoint and the SSTs. A lapsed lease stays
/// lapsed.
pub(crate) struct CheckpointLease {
    id: Uuid,
    /// The expiry time that the reader last wrote for the checkpoint, in milliseconds since the
    /// Unix epoch. `i64::MAX` if the checkpoint does not expire.
    expire_millis: AtomicI64,
    lapsed: AtomicBool,
}

impl CheckpointLease {
    fn new(id: Uuid, expire_millis: i64) -> Self {
        Self {
            id,
            expire_millis: AtomicI64::new(expire_millis),
            lapsed: AtomicBool::new(false),
        }
    }

    fn is_lapsed(&self) -> bool {
        self.lapsed.load(Ordering::Acquire)
    }

    fn lapse(&self) {
        self.lapsed.store(true, Ordering::Release);
    }

    /// Fails if the lease lapsed, and marks it lapsed if its expiry time has passed at
    /// `now_millis`.
    fn check(&self, now_millis: i64, seq: u64) -> Result<(), SlateDBError> {
        if !self.is_lapsed() && now_millis < self.expire_millis.load(Ordering::Acquire) {
            return Ok(());
        }
        self.lapse();
        Err(SlateDBError::ReaderSnapshotExpired {
            checkpoint_id: self.id,
            seq,
        })
    }
}

/// The time in milliseconds since the Unix epoch, or `i64::MAX` for no time.
pub(crate) fn millis(time: Option<DateTime<Utc>>) -> i64 {
    time.map_or(i64::MAX, |time| time.timestamp_millis())
}

/// The live snapshots of a reader and the checkpoints they lease.
///
/// Snapshot creation, snapshot drop and the poller take its lock. Reads never do.
/// [`Self::min_seq`] reads an atomic value.
pub(crate) struct SnapshotRegistry {
    inner: Mutex<RegistryInner>,
    /// The lowest sequence number that a live snapshot reads at, or `u64::MAX` if no snapshot is
    /// live. Written under the lock.
    min_seq: AtomicU64,
    /// Counts the checkpoint replacements that ended. A snapshot that finds the checkpoint of the
    /// current state retiring waits for the next count.
    replaced: watch::Sender<u64>,
}

#[derive(Default)]
struct RegistryInner {
    /// The sequence numbers of the live snapshots, each with the number of snapshots at it.
    seqs: BTreeMap<u64, usize>,
    /// The leased checkpoints, by id.
    leases: HashMap<Uuid, Leased>,
    /// The checkpoint that the poller removes in the manifest write that adds its replacement.
    /// No snapshot leases it.
    retiring: Option<Uuid>,
    /// The checkpoints that the poller replaced but kept for their leases.
    retained: Vec<Uuid>,
    /// The leased checkpoints that the reader found gone. A new lease on one is lapsed at once.
    gone: HashSet<Uuid>,
}

struct Leased {
    lease: Arc<CheckpointLease>,
    holders: usize,
}

/// A registered snapshot's state and its lease, if any.
pub(crate) type Registered = (Arc<ReaderState>, Option<Arc<CheckpointLease>>);

/// What the poller does with the kept checkpoints at one poll.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct RetainedPlan {
    /// Kept checkpoints that no snapshot leases any more.
    pub(crate) delete: Vec<Uuid>,
    /// Leased kept checkpoints that are due for a refresh.
    pub(crate) refresh: Vec<Uuid>,
}

impl RetainedPlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.delete.is_empty() && self.refresh.is_empty()
    }
}

impl SnapshotRegistry {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(RegistryInner::default()),
            min_seq: AtomicU64::new(u64::MAX),
            replaced: watch::channel(0).0,
        }
    }

    /// The lowest sequence number that a live snapshot reads at, or `None` if no snapshot is
    /// live. Takes no lock.
    pub(crate) fn min_seq(&self) -> Option<u64> {
        let seq = self.min_seq.load(Ordering::Acquire);
        (seq != u64::MAX).then_some(seq)
    }

    /// Registers a snapshot of the state that `current` returns. `current` runs under the
    /// registry's lock, so the poller cannot release the state's checkpoint between the read of
    /// the state and the lease. With `lease`, the snapshot leases the state's checkpoint.
    ///
    /// Returns the state and the lease. If the poller is removing the state's checkpoint, returns
    /// a receiver instead: wait for its next value, then try again.
    pub(crate) fn register(
        &self,
        current: impl FnOnce() -> Arc<ReaderState>,
        lease: bool,
    ) -> Result<Registered, watch::Receiver<u64>> {
        let mut inner = self.inner.lock();
        let state = current();
        let lease = match state.checkpoint().filter(|_| lease) {
            None => None,
            Some(checkpoint) => {
                if inner.retiring == Some(checkpoint.id) {
                    return Err(self.replaced.subscribe());
                }
                Some(inner.lease(checkpoint))
            }
        };
        *inner.seqs.entry(state.seq()).or_default() += 1;
        self.publish_min_seq(&inner);
        Ok((state, lease))
    }

    /// One more holder of a live snapshot at `seq` and of its `lease`: an iterator of the
    /// snapshot's scans. [`Self::release`] ends the hold. A lapsed lease that a new one replaced
    /// gets no holder, as [`Self::release`] ignores it.
    pub(crate) fn hold(&self, seq: u64, lease: Option<&Arc<CheckpointLease>>) {
        let mut inner = self.inner.lock();
        *inner.seqs.entry(seq).or_default() += 1;
        if let Some(lease) = lease {
            if let Some(leased) = inner.leases.get_mut(&lease.id) {
                if Arc::ptr_eq(&leased.lease, lease) {
                    leased.holders += 1;
                }
            }
        }
        self.publish_min_seq(&inner);
    }

    /// Releases a snapshot at `seq` and its lease.
    pub(crate) fn release(&self, seq: u64, lease: Option<&Arc<CheckpointLease>>) {
        let mut inner = self.inner.lock();
        if let btree_map::Entry::Occupied(mut entry) = inner.seqs.entry(seq) {
            *entry.get_mut() -= 1;
            if *entry.get() == 0 {
                entry.remove();
            }
        }
        if let Some(lease) = lease {
            let id = lease.id;
            // A lapsed lease may have been replaced by a new one on the same checkpoint. Only
            // the holders of the lease in the map count.
            if let Some(leased) = inner.leases.get_mut(&id) {
                if Arc::ptr_eq(&leased.lease, lease) {
                    leased.holders -= 1;
                    if leased.holders == 0 {
                        inner.leases.remove(&id);
                    }
                }
            }
        }
        self.publish_min_seq(&inner);
    }

    /// Starts the replacement of checkpoint `old`. Returns `true` if live snapshots lease it: the
    /// poller then adds the new checkpoint and keeps `old`. Otherwise the poller removes `old` in
    /// the same manifest write, and snapshots of the state on `old` wait for
    /// [`Self::finish_replace`].
    pub(crate) fn begin_replace(&self, old: Uuid) -> bool {
        let mut inner = self.inner.lock();
        let leased = inner
            .leases
            .get(&old)
            .is_some_and(|leased| !leased.lease.is_lapsed());
        if !leased {
            inner.retiring = Some(old);
        }
        leased
    }

    /// Ends the replacement of checkpoint `old`. `kept` is what [`Self::begin_replace`] returned.
    /// `installed` tells whether the state on the new checkpoint is installed: a kept checkpoint
    /// is retained only then.
    pub(crate) fn finish_replace(&self, old: Uuid, kept: bool, installed: bool) {
        let mut inner = self.inner.lock();
        if inner.retiring == Some(old) {
            inner.retiring = None;
        }
        if kept && installed && !inner.retained.contains(&old) {
            inner.retained.push(old);
        }
        drop(inner);
        self.replaced.send_modify(|count| *count += 1);
    }

    /// Marks lapsed the leases on checkpoints that `manifest` does not hold.
    pub(crate) fn lapse_missing(&self, manifest: &ManifestCore) {
        let mut guard = self.inner.lock();
        let RegistryInner { leases, gone, .. } = &mut *guard;
        for leased in leases.values() {
            if manifest.find_checkpoint(leased.lease.id).is_none() {
                leased.lease.lapse();
                gone.insert(leased.lease.id);
            }
        }
    }

    /// Records that checkpoint `id` is gone, and marks its lease lapsed, if any.
    pub(crate) fn lapse(&self, id: Uuid) {
        let mut inner = self.inner.lock();
        inner.gone.insert(id);
        if let Some(leased) = inner.leases.get(&id) {
            leased.lease.lapse();
        }
    }

    /// Records that the reader wrote `expire_time` for checkpoint `id`.
    pub(crate) fn extend(&self, id: Uuid, expire_time: Option<DateTime<Utc>>) {
        if let Some(leased) = self.inner.lock().leases.get(&id) {
            leased
                .lease
                .expire_millis
                .fetch_max(millis(expire_time), Ordering::AcqRel);
        }
    }

    /// Plans what to do with the kept checkpoints, given the latest `manifest`: delete those that
    /// no live lease holds, and refresh those that expire before `refresh_before_millis`. Leases
    /// on kept checkpoints that `manifest` lacks lapse. The checkpoints to delete leave the
    /// retained set.
    pub(crate) fn plan_retained(
        &self,
        manifest: &ManifestCore,
        refresh_before_millis: i64,
    ) -> RetainedPlan {
        let mut guard = self.inner.lock();
        let RegistryInner {
            leases,
            retained,
            gone,
            ..
        } = &mut *guard;
        let mut plan = RetainedPlan::default();
        retained.retain(|id| {
            let present = manifest.find_checkpoint(*id).is_some();
            match leases.get(id).filter(|leased| !leased.lease.is_lapsed()) {
                Some(leased) if present => {
                    if leased.lease.expire_millis.load(Ordering::Acquire) < refresh_before_millis {
                        plan.refresh.push(*id);
                    }
                    true
                }
                Some(leased) => {
                    leased.lease.lapse();
                    gone.insert(*id);
                    false
                }
                None => {
                    if present {
                        plan.delete.push(*id);
                    }
                    false
                }
            }
        });
        plan
    }

    /// The checkpoints that the poller replaced but kept for their leases.
    pub(crate) fn retained(&self) -> Vec<Uuid> {
        self.inner.lock().retained.clone()
    }

    fn publish_min_seq(&self, inner: &RegistryInner) {
        let min = inner.seqs.keys().next().copied().unwrap_or(u64::MAX);
        self.min_seq.store(min, Ordering::Release);
    }
}

impl RegistryInner {
    /// One more holder of a lease on `checkpoint`. A lapsed lease is not reused: the new holder
    /// gets a new lease, which lapses again unless the reader refreshes the checkpoint. A lease on
    /// a checkpoint known to be gone is lapsed at once.
    fn lease(&mut self, checkpoint: &Checkpoint) -> Arc<CheckpointLease> {
        if self.gone.contains(&checkpoint.id) {
            let lease = CheckpointLease::new(checkpoint.id, millis(checkpoint.expire_time));
            lease.lapse();
            return Arc::new(lease);
        }
        let expire = millis(checkpoint.expire_time);
        let leased = self
            .leases
            .entry(checkpoint.id)
            .and_modify(|leased| {
                if leased.lease.is_lapsed() {
                    *leased = Leased {
                        lease: Arc::new(CheckpointLease::new(checkpoint.id, expire)),
                        holders: 0,
                    };
                }
            })
            .or_insert_with(|| Leased {
                lease: Arc::new(CheckpointLease::new(checkpoint.id, expire)),
                holders: 0,
            });
        leased.holders += 1;
        leased
            .lease
            .expire_millis
            .fetch_max(expire, Ordering::AcqRel);
        Arc::clone(&leased.lease)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_report_the_lowest_live_seq() {
        let registry = SnapshotRegistry::new();
        assert_eq!(registry.min_seq(), None);
        let mut inner = registry.inner.lock();
        *inner.seqs.entry(7).or_default() += 1;
        *inner.seqs.entry(5).or_default() += 1;
        *inner.seqs.entry(5).or_default() += 1;
        registry.publish_min_seq(&inner);
        drop(inner);
        assert_eq!(registry.min_seq(), Some(5));

        registry.release(5, None);
        assert_eq!(registry.min_seq(), Some(5));
        registry.release(5, None);
        assert_eq!(registry.min_seq(), Some(7));
        registry.release(7, None);
        assert_eq!(registry.min_seq(), None);
    }

    #[test]
    fn should_keep_a_leased_checkpoint_and_retire_an_unleased_one() {
        let registry = SnapshotRegistry::new();
        let leased = Uuid::new_v4();
        let unleased = Uuid::new_v4();
        let lease = registry.inner.lock().lease(&checkpoint(leased, None));

        assert!(registry.begin_replace(leased));
        assert_eq!(registry.inner.lock().retiring, None);
        registry.finish_replace(leased, true, true);
        assert_eq!(registry.retained(), vec![leased]);

        assert!(!registry.begin_replace(unleased));
        assert_eq!(registry.inner.lock().retiring, Some(unleased));
        let replaced = registry.replaced.subscribe();
        registry.finish_replace(unleased, false, true);
        assert_eq!(registry.inner.lock().retiring, None);
        assert!(replaced.has_changed().unwrap());
        assert_eq!(registry.retained(), vec![leased]);

        // The last holder drops: the kept checkpoint is deleted at the next plan.
        registry.release(0, Some(&lease));
        let manifest = manifest_with(&[leased]);
        let plan = registry.plan_retained(&manifest, i64::MIN);
        assert_eq!(plan.delete, vec![leased]);
        assert!(registry.retained().is_empty());
    }

    #[test]
    fn should_refresh_kept_checkpoints_and_lapse_missing_ones() {
        let registry = SnapshotRegistry::new();
        let due = Uuid::new_v4();
        let later = Uuid::new_v4();
        let gone = Uuid::new_v4();
        let due_lease = registry.inner.lock().lease(&checkpoint(due, Some(1_000)));
        let later_lease = registry.inner.lock().lease(&checkpoint(later, Some(9_000)));
        let gone_lease = registry.inner.lock().lease(&checkpoint(gone, Some(9_000)));
        for id in [due, later, gone] {
            assert!(registry.begin_replace(id));
            registry.finish_replace(id, true, true);
        }

        let plan = registry.plan_retained(&manifest_with(&[due, later]), 5_000);
        assert_eq!(plan.refresh, vec![due]);
        assert!(plan.delete.is_empty());
        assert_eq!(registry.retained(), vec![due, later]);
        assert!(gone_lease.check(0, 1).is_err());
        assert!(later_lease.check(0, 1).is_ok());

        registry.extend(due, DateTime::from_timestamp_millis(20_000));
        assert!(due_lease.check(10_000, 1).is_ok());
        assert!(due_lease.check(20_000, 1).is_err());
        // A lapsed lease stays lapsed, even if the reader refreshes the checkpoint.
        registry.extend(due, DateTime::from_timestamp_millis(40_000));
        assert!(due_lease.check(10_000, 1).is_err());
    }

    #[test]
    fn should_not_reuse_a_lapsed_lease() {
        let registry = SnapshotRegistry::new();
        let id = Uuid::new_v4();
        let first = registry.inner.lock().lease(&checkpoint(id, Some(1_000)));
        assert!(first.check(2_000, 1).is_err());
        let second = registry.inner.lock().lease(&checkpoint(id, Some(3_000)));
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(second.check(2_000, 1).is_ok());

        // Releasing the lapsed lease leaves the new one held.
        registry.release(0, Some(&first));
        assert_eq!(registry.inner.lock().leases.get(&id).unwrap().holders, 1);
        registry.release(0, Some(&second));
        assert!(registry.inner.lock().leases.is_empty());
    }

    #[test]
    fn should_lapse_new_leases_on_a_gone_checkpoint() {
        let registry = SnapshotRegistry::new();
        let id = Uuid::new_v4();
        let held = registry.inner.lock().lease(&checkpoint(id, None));
        registry.lapse_missing(&manifest_with(&[]));
        assert!(held.check(0, 1).is_err());
        let later = registry.inner.lock().lease(&checkpoint(id, None));
        assert!(later.check(0, 1).is_err());
        // The poller does not keep a checkpoint whose leases lapsed.
        assert!(!registry.begin_replace(id));
    }

    fn checkpoint(id: Uuid, expire_millis: Option<i64>) -> Checkpoint {
        Checkpoint {
            id,
            manifest_id: 1,
            expire_time: expire_millis.and_then(DateTime::from_timestamp_millis),
            create_time: DateTime::UNIX_EPOCH,
            name: None,
        }
    }

    fn manifest_with(ids: &[Uuid]) -> ManifestCore {
        let mut core = ManifestCore::new();
        core.checkpoints = ids.iter().map(|id| checkpoint(*id, None)).collect();
        core
    }
}
