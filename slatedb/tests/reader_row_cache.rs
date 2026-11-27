//! Integration tests for a [`RowCache`] attached to a `DbReader`.
//!
//! A writer and a reader share one in-memory object store. The reader's periodic poll runs once
//! at open and then once an hour, so the tests drive polls with `DbReader::refresh`. The test
//! cache covers the keys under `hot/`, which the writer writes only after the reader opened, so
//! every such key reaches the cache through the poller. It serves reads of those keys, and misses
//! on everything else.

#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::future::Future;
use std::num::NonZeroU64;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use slatedb::admin::Admin;
use slatedb::config::{
    CheckpointOptions, CheckpointScope, CompactionWorkerOptions, CompactorOptions, DbReaderOptions,
    FlushOptions, FlushType, ReadOptions, ScanOptions, Settings,
    SizeTieredCompactionSchedulerOptions,
};
use slatedb::db_stats::REQUEST_COUNT;
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use slatedb::row_cache::{
    GetOutcome, Host, MissToken, OpenContext, RowCache, ScanIterator, ScanOutcome,
};
use slatedb::{
    Db, DbIterator, DbReader, DbReaderMode, Error, ErrorKind, KeyValue, RowEntry, ValueDeletable,
    WriteBatch,
};
use slatedb_common::metrics::{lookup_metric_with_labels, DefaultMetricsRecorder};
use tokio::sync::Notify;

const PATH: &str = "reader_row_cache";
const HOT: &[u8] = b"hot/";

#[derive(Default)]
struct Events {
    lifecycle: Vec<&'static str>,
    /// `(committed_seq, batch)` of each `apply_write`.
    applied_batches: Vec<(u64, Vec<RowEntry>)>,
    drained_seqs: Vec<u64>,
    /// `(sealed_at, new_seq)` of each `invalidate_all`.
    invalidations: Vec<(u64, u64)>,
    recorded_misses: Vec<Bytes>,
    /// `at` of each read the cache served.
    served_at: Vec<Option<u64>>,
}

struct TestRowCache {
    host: OnceLock<Host>,
    events: Mutex<Events>,
    versions: Mutex<BTreeMap<Bytes, Vec<RowEntry>>>,
    /// The sequence number the cache holds every write up to.
    applied: AtomicU64,
    sample_misses: AtomicBool,
    next_token: AtomicU64,
    reject_open: AtomicBool,
    drain_notify: Notify,
    wake_on_close: AtomicBool,
}

impl TestRowCache {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            host: OnceLock::new(),
            events: Mutex::new(Events::default()),
            versions: Mutex::new(BTreeMap::new()),
            applied: AtomicU64::new(0),
            sample_misses: AtomicBool::new(true),
            next_token: AtomicU64::new(1),
            reject_open: AtomicBool::new(false),
            drain_notify: Notify::new(),
            wake_on_close: AtomicBool::new(false),
        })
    }

    fn host(&self) -> &Host {
        self.host.get().expect("row cache was opened")
    }

    fn token(&self) -> Option<MissToken> {
        if !self.sample_misses.load(Ordering::Relaxed) {
            return None;
        }
        NonZeroU64::new(self.next_token.fetch_add(1, Ordering::Relaxed)).map(MissToken)
    }

    fn read_seq(&self, at: Option<u64>) -> u64 {
        let committed = self.host().last_committed_seq();
        at.map_or(committed, |at| at.min(committed))
    }

    /// The row of `key` at `seq`, or `None` if the cache does not know it.
    fn lookup(
        versions: &BTreeMap<Bytes, Vec<RowEntry>>,
        key: &[u8],
        seq: u64,
    ) -> Option<Option<KeyValue>> {
        let entry = versions.get(key)?.iter().rev().find(|e| e.seq <= seq)?;
        match &entry.value {
            ValueDeletable::Value(value) => Some(Some(KeyValue::new(
                entry.key.clone(),
                value.clone(),
                entry.seq,
                entry.create_ts.unwrap_or(0),
                entry.expire_ts,
            ))),
            ValueDeletable::Tombstone => Some(None),
            ValueDeletable::Merge(_) => None,
        }
    }

    fn batches(&self) -> Vec<(u64, Vec<RowEntry>)> {
        self.events.lock().unwrap().applied_batches.clone()
    }

    fn invalidations(&self) -> Vec<(u64, u64)> {
        self.events.lock().unwrap().invalidations.clone()
    }

    fn served(&self) -> usize {
        self.events.lock().unwrap().served_at.len()
    }

    fn last_served_at(&self) -> Option<Option<u64>> {
        self.events.lock().unwrap().served_at.last().copied()
    }
}

struct VecScanIterator {
    rows: Vec<KeyValue>,
    pos: usize,
}

impl ScanIterator for VecScanIterator {
    fn next_sync(&mut self) -> Option<Result<Option<KeyValue>, Error>> {
        let row = self.rows.get(self.pos).cloned();
        self.pos += 1;
        Some(Ok(row))
    }

    fn next(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Option<KeyValue>, Error>> + Send + '_>> {
        Box::pin(async move { self.next_sync().unwrap() })
    }

    fn seek<'a>(
        &'a mut self,
        next_key: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            while self.pos < self.rows.len() && self.rows[self.pos].key.as_ref() < next_key {
                self.pos += 1;
            }
            Ok(())
        })
    }
}

impl RowCache for TestRowCache {
    fn open(&self, ctx: OpenContext) -> Result<(), Error> {
        self.events.lock().unwrap().lifecycle.push("open");
        assert!(!ctx.has_compaction_filter);
        self.applied
            .store(ctx.host.last_committed_seq(), Ordering::SeqCst);
        let _ = self.host.set(ctx.host);
        if self.reject_open.load(Ordering::Relaxed) {
            return Err(Error::invalid("row cache rejected open".to_string()));
        }
        Ok(())
    }

    fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            self.events.lock().unwrap().lifecycle.push("close");
            if self.wake_on_close.load(Ordering::Relaxed) {
                let notified = self.drain_notify.notified();
                assert!(
                    self.host().wake_writer(),
                    "the poller must still accept wakes during RowCache::close"
                );
                notified.await;
            }
        })
    }

    fn apply_write(&self, entries: &[RowEntry], committed_seq: u64) {
        let seq = entries.first().expect("a batch holds a write").seq;
        assert_eq!(
            committed_seq,
            self.applied.load(Ordering::SeqCst),
            "batches chain from the cache's watermark"
        );
        assert!(committed_seq < seq, "batches come in ascending seq order");
        assert!(
            self.host().last_committed_seq() < seq,
            "a batch reaches the cache before the reader's seq covers it"
        );
        assert!(entries.iter().all(|e| e.seq == seq), "one seq per batch");
        assert!(
            entries.windows(2).all(|w| w[0].key < w[1].key),
            "a batch comes in key order"
        );
        let mut versions = self.versions.lock().unwrap();
        for entry in entries {
            versions
                .entry(entry.key.clone())
                .or_default()
                .push(entry.clone());
        }
        drop(versions);
        self.applied.store(seq, Ordering::SeqCst);
        self.events
            .lock()
            .unwrap()
            .applied_batches
            .push((committed_seq, entries.to_vec()));
    }

    fn drain_writer(&self, committed_seq: u64) {
        assert!(committed_seq >= self.applied.load(Ordering::SeqCst));
        self.applied.store(committed_seq, Ordering::SeqCst);
        self.events.lock().unwrap().drained_seqs.push(committed_seq);
        self.drain_notify.notify_waiters();
    }

    fn invalidate_all(&self, sealed_at: u64, new_seq: u64) {
        assert_eq!(sealed_at, self.applied.load(Ordering::SeqCst));
        assert!(new_seq > sealed_at);
        assert_eq!(
            self.host().last_committed_seq(),
            sealed_at,
            "invalidation runs before the reader's seq advances"
        );
        // This cache keeps no history below its newest rows, so it drops everything.
        self.versions.lock().unwrap().clear();
        self.applied.store(new_seq, Ordering::SeqCst);
        self.events
            .lock()
            .unwrap()
            .invalidations
            .push((sealed_at, new_seq));
    }

    fn close_writer(&self) {
        self.events.lock().unwrap().lifecycle.push("close_writer");
    }

    fn get(&self, key: &[u8], _options: &ReadOptions, at: Option<u64>) -> GetOutcome {
        if !key.starts_with(HOT) {
            return GetOutcome::Miss(self.token());
        }
        let versions = self.versions.lock().unwrap();
        match Self::lookup(&versions, key, self.read_seq(at)) {
            Some(row) => {
                drop(versions);
                self.host().record_get_requests(1);
                self.events.lock().unwrap().served_at.push(at);
                GetOutcome::Hit(Ok(row))
            }
            None => GetOutcome::Miss(self.token()),
        }
    }

    fn scan(
        &self,
        range: (Bound<&[u8]>, Bound<&[u8]>),
        _options: &ScanOptions,
        _prefix: Option<&[u8]>,
        at: Option<u64>,
    ) -> ScanOutcome {
        let start = match range.0 {
            Bound::Included(start) | Bound::Excluded(start) => start,
            Bound::Unbounded => &[],
        };
        let hot_end: &[u8] = b"hot0";
        let within_hot = start.starts_with(HOT)
            && matches!(range.1, Bound::Included(end) | Bound::Excluded(end) if end <= hot_end);
        if !within_hot {
            return ScanOutcome::Miss(self.token());
        }
        let seq = self.read_seq(at);
        let versions = self.versions.lock().unwrap();
        let rows = versions
            .range::<[u8], _>(range)
            .filter_map(|(key, _)| Self::lookup(&versions, key, seq).flatten())
            .collect();
        drop(versions);
        self.host().record_scan_requests(1);
        self.events.lock().unwrap().served_at.push(at);
        ScanOutcome::Hit(Ok(Box::new(VecScanIterator { rows, pos: 0 })))
    }

    fn record_miss(&self, _token: MissToken, key: &[u8]) {
        self.events
            .lock()
            .unwrap()
            .recorded_misses
            .push(Bytes::copy_from_slice(key));
    }
}

/// A reader that polls at open, then only on `refresh`.
fn reader_options() -> DbReaderOptions {
    DbReaderOptions {
        manifest_poll_interval: Duration::from_secs(3600),
        checkpoint_lifetime: Duration::from_secs(3 * 3600),
        ..DbReaderOptions::default()
    }
}

/// A writer with no compactor and no garbage collector of its own.
fn quiet_settings() -> Settings {
    Settings {
        flush_interval: None,
        compactor_options: None,
        garbage_collector_options: None,
        ..Settings::default()
    }
}

/// A writer whose compactor merges every two L0 SSTs, quickly.
fn compacting_settings() -> Settings {
    Settings {
        flush_interval: None,
        garbage_collector_options: None,
        compactor_options: Some(CompactorOptions {
            poll_interval: Duration::from_millis(5),
            scheduler_options: SizeTieredCompactionSchedulerOptions {
                min_compaction_sources: 2,
                ..Default::default()
            }
            .into(),
            worker: Some(CompactionWorkerOptions {
                compactions_poll_interval: Duration::from_millis(5),
                ..Default::default()
            }),
            checkpoint_lifetime: Duration::from_millis(1),
            ..CompactorOptions::default()
        }),
        ..Settings::default()
    }
}

async fn open_writer(store: &Arc<dyn ObjectStore>, settings: Settings) -> Db {
    Db::builder(PATH, Arc::clone(store))
        .with_settings(settings)
        .build()
        .await
        .unwrap()
}

async fn open_reader(store: &Arc<dyn ObjectStore>, cache: &Arc<TestRowCache>) -> DbReader {
    DbReader::builder(PATH, Arc::clone(store))
        .with_options(reader_options())
        .with_row_cache(cache.clone())
        .build()
        .await
        .unwrap()
}

fn hot(name: &str) -> Bytes {
    Bytes::from(format!("hot/{name}"))
}

/// Writes each pair in its own batch, then makes them durable in the WAL.
async fn put_durable(db: &Db, rows: &[(Bytes, &'static str)]) {
    for (key, value) in rows {
        db.put(key, value.as_bytes()).await.unwrap();
    }
    db.flush().await.unwrap();
}

async fn flush_memtable(db: &Db) {
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::MemTable,
    })
    .await
    .unwrap();
}

async fn collect(mut iter: DbIterator) -> Vec<(Bytes, Bytes)> {
    let mut rows = Vec::new();
    while let Some(row) = iter.next().await.unwrap() {
        rows.push((row.key, row.value));
    }
    rows
}

fn row(key: Bytes, value: &'static str) -> (Bytes, Bytes) {
    (key, Bytes::from_static(value.as_bytes()))
}

/// Point reads, scans, prefix scans and snapshot reads consult the cache first, at the reader's
/// sequence number or at the snapshot's; misses fall through to the LSM and are reported.
#[tokio::test]
async fn reads_scans_and_snapshot_reads_consult_the_row_cache() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    put_durable(&db, &[(Bytes::from_static(b"cold/a"), "a1")]).await;

    let cache = TestRowCache::new();
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let reader = DbReader::builder(PATH, Arc::clone(&store))
        .with_options(reader_options())
        .with_metrics_recorder(recorder.clone())
        .with_row_cache(cache.clone())
        .build()
        .await
        .unwrap();
    put_durable(&db, &[(hot("x"), "x1"), (hot("y"), "y1")]).await;
    reader.refresh().await.unwrap();

    // A key the cache does not cover misses, falls through to the LSM, and is reported.
    let gets_before = lookup_metric_with_labels(&recorder, REQUEST_COUNT, &[("op", "get")]);
    assert_eq!(
        reader.get(b"cold/a").await.unwrap(),
        Some(Bytes::from_static(b"a1"))
    );
    assert_eq!(cache.served(), 0);
    assert_eq!(
        cache.events.lock().unwrap().recorded_misses,
        vec![Bytes::from_static(b"cold/a")]
    );

    // Covered keys are served at the reader's sequence number, and count as get requests.
    assert_eq!(
        reader.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x1"))
    );
    let kv = reader.get_key_value(hot("y")).await.unwrap().unwrap();
    assert_eq!(kv.value, Bytes::from_static(b"y1"));
    assert_eq!(cache.served(), 2);
    assert_eq!(cache.last_served_at(), Some(None));
    assert_eq!(
        lookup_metric_with_labels(&recorder, REQUEST_COUNT, &[("op", "get")]),
        gets_before.map(|count| count + 3)
    );

    // Scans and prefix scans of covered ranges are served; others miss with their start key.
    let rows = collect(reader.scan(hot("a")..hot("z")).await.unwrap()).await;
    assert_eq!(rows, vec![row(hot("x"), "x1"), row(hot("y"), "y1")]);
    let rows = collect(reader.scan_prefix(HOT, ..).await.unwrap()).await;
    assert_eq!(rows, vec![row(hot("x"), "x1"), row(hot("y"), "y1")]);
    assert_eq!(cache.served(), 4);
    let rows = collect(reader.scan_prefix(b"cold/", ..).await.unwrap()).await;
    assert_eq!(rows, vec![row(Bytes::from_static(b"cold/a"), "a1")]);
    assert_eq!(
        cache.events.lock().unwrap().recorded_misses.last(),
        Some(&Bytes::from_static(b"cold/"))
    );

    // A snapshot reads through the cache at its own sequence number while the reader moves on.
    let snapshot = reader.snapshot().await.unwrap();
    let seq = snapshot.seq();
    put_durable(&db, &[(hot("x"), "x2")]).await;
    reader.refresh().await.unwrap();
    assert_eq!(
        reader.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x2"))
    );
    assert_eq!(
        snapshot.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x1"))
    );
    assert_eq!(cache.last_served_at(), Some(Some(seq)));
    let rows = collect(snapshot.scan_prefix(HOT, ..).await.unwrap()).await;
    assert_eq!(rows, vec![row(hot("x"), "x1"), row(hot("y"), "y1")]);
    assert_eq!(cache.last_served_at(), Some(Some(seq)));
    assert_eq!(
        snapshot.get(b"cold/a").await.unwrap(),
        Some(Bytes::from_static(b"a1"))
    );

    // A scan iterator from the cache holds the snapshot's sequence number after it drops.
    let iter = snapshot.scan(hot("a")..hot("z")).await.unwrap();
    drop(snapshot);
    assert_eq!(reader.min_snapshot_seq(), Some(seq));
    assert_eq!(
        collect(iter).await,
        vec![row(hot("x"), "x1"), row(hot("y"), "y1")]
    );
    assert_eq!(reader.min_snapshot_seq(), None);

    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// WAL replay feeds the cache one batch per sequence number, in ascending order and in key order
/// within a batch, each chained from the previous one and fed before the reader's sequence
/// number covers it.
#[tokio::test]
async fn replay_feeds_ordered_batches_before_their_seq_is_visible() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let cache = TestRowCache::new();
    let reader = open_reader(&store, &cache).await;
    let opened_at = cache.host().last_committed_seq();

    let mut batch = WriteBatch::new();
    batch.put(hot("c"), b"c1");
    batch.put(hot("a"), b"a1");
    batch.delete(hot("b"));
    db.write(batch).await.unwrap();
    put_durable(&db, &[(hot("z"), "z1"), (hot("a"), "a2")]).await;
    assert!(reader.refresh().await.unwrap());

    let batches = cache.batches();
    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].0, opened_at);
    let keys: Vec<Bytes> = batches[0].1.iter().map(|e| e.key.clone()).collect();
    assert_eq!(keys, vec![hot("a"), hot("b"), hot("c")]);
    assert_eq!(batches[0].1[1].value, ValueDeletable::Tombstone);
    assert_eq!(batches[1].1[0].key, hot("z"));
    assert_eq!(batches[2].1[0].key, hot("a"));
    for pair in batches.windows(2) {
        assert_eq!(pair[1].0, pair[0].1[0].seq);
    }
    let last_seq = batches[2].1[0].seq;
    assert_eq!(cache.host().last_committed_seq(), last_seq);
    assert_eq!(
        reader.get_key_value(hot("a")).await.unwrap().unwrap().seq,
        last_seq
    );
    assert_eq!(reader.get(hot("b")).await.unwrap(), None);

    // A poll with nothing new feeds nothing.
    assert!(!reader.refresh().await.unwrap());
    assert_eq!(cache.batches().len(), 3);
    assert!(cache.invalidations().is_empty());

    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// Flushing and compacting writes the reader already replayed changes nothing in the cache: no
/// invalidation, and the cache keeps serving.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flush_and_compaction_of_replayed_writes_invalidate_nothing() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, compacting_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    let cache = TestRowCache::new();
    let reader = open_reader(&store, &cache).await;

    put_durable(&db, &[(hot("x"), "x1")]).await;
    reader.refresh().await.unwrap();
    flush_memtable(&db).await;
    reader.refresh().await.unwrap();
    put_durable(&db, &[(hot("x"), "x2"), (hot("y"), "y1")]).await;
    reader.refresh().await.unwrap();
    flush_memtable(&db).await;
    for _ in 0..2000 {
        let manifest = admin.read_manifest(None).await.unwrap().unwrap();
        if manifest.l0().is_empty() && !manifest.compacted().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    put_durable(&db, &[(hot("z"), "z1")]).await;
    reader.refresh().await.unwrap();

    let manifest = cache.host().manifest().unwrap();
    assert!(manifest.l0().is_empty() && !manifest.compacted().is_empty());
    assert!(cache.invalidations().is_empty());
    assert_eq!(cache.batches().len(), 4);
    let served = cache.served();
    assert_eq!(
        reader.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x2"))
    );
    assert_eq!(
        reader.get(hot("z")).await.unwrap(),
        Some(Bytes::from_static(b"z1"))
    );
    assert_eq!(cache.served(), served + 2);

    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// When the reader moves to a state whose L0 holds writes it never replayed, the cache drops its
/// rows above the reader's sequence number first, then receives the WAL writes above the L0.
#[tokio::test]
async fn l0_jump_invalidates_the_cache() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let cache = TestRowCache::new();
    let reader = open_reader(&store, &cache).await;
    put_durable(&db, &[(hot("x"), "x1")]).await;
    reader.refresh().await.unwrap();
    let snapshot = reader.snapshot().await.unwrap();
    let before = cache.host().last_committed_seq();

    // The writer flushes x2 to L0 before the reader replays its WAL, then writes z1 to the WAL.
    put_durable(&db, &[(hot("x"), "x2")]).await;
    flush_memtable(&db).await;
    put_durable(&db, &[(hot("z"), "z1")]).await;
    reader.refresh().await.unwrap();

    let last_l0_seq = cache.host().manifest().unwrap().last_l0_seq();
    assert!(last_l0_seq > before);
    assert_eq!(cache.invalidations(), vec![(before, last_l0_seq)]);
    let batches = cache.batches();
    let (committed, last) = batches.last().unwrap();
    assert_eq!(*committed, last_l0_seq);
    assert_eq!(last[0].key, hot("z"));

    // x2 comes from the L0, z1 from the cache, and the old snapshot still reads x1.
    assert_eq!(
        reader.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x2"))
    );
    let served = cache.served();
    assert_eq!(
        reader.get(hot("z")).await.unwrap(),
        Some(Bytes::from_static(b"z1"))
    );
    assert_eq!(cache.served(), served + 1);
    assert_eq!(
        snapshot.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x1"))
    );
    assert_eq!(snapshot.get(hot("z")).await.unwrap(), None);

    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// A reader that skips WAL replay feeds nothing, and invalidates at every L0 advance.
#[tokio::test]
async fn skip_wal_replay_invalidates_at_each_l0_advance() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let cache = TestRowCache::new();
    let reader = DbReader::builder(PATH, Arc::clone(&store))
        .with_options(DbReaderOptions {
            skip_wal_replay: true,
            ..reader_options()
        })
        .with_row_cache(cache.clone())
        .build()
        .await
        .unwrap();
    let before = cache.host().last_committed_seq();
    put_durable(&db, &[(hot("x"), "x1")]).await;
    reader.refresh().await.unwrap();
    assert!(cache.invalidations().is_empty());
    assert_eq!(reader.get(hot("x")).await.unwrap(), None);

    flush_memtable(&db).await;
    reader.refresh().await.unwrap();
    let last_l0_seq = cache.host().last_committed_seq();
    assert!(last_l0_seq > before);
    assert_eq!(cache.invalidations(), vec![(before, last_l0_seq)]);
    assert!(cache.batches().is_empty());
    assert_eq!(
        reader.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x1"))
    );

    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// The host reads the reader's state: views, retention, scans of a view, snapshots, the manifest
/// and SST indexes.
#[tokio::test]
async fn host_reads_the_reader_state() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let cache = TestRowCache::new();
    let reader = open_reader(&store, &cache).await;
    let host = cache.host().clone();
    assert!(host.check_closed().is_ok());

    put_durable(&db, &[(hot("x"), "x1")]).await;
    reader.refresh().await.unwrap();
    let x1_seq = host.last_committed_seq();
    let view = host.capture_state_view().unwrap();
    assert!(host.is_memtable_current(&view));
    assert!(host.retains_seq(&view, x1_seq));

    put_durable(&db, &[(hot("x"), "x2")]).await;
    reader.refresh().await.unwrap();
    assert!(!host.is_memtable_current(&view));
    let view = host.capture_state_view().unwrap();
    assert!(host.is_memtable_current(&view));

    // A view scan bypasses the cache and reads at the given seq, or the newest.
    let old = host
        .scan_view(
            &view,
            hot("a")..hot("z"),
            &ScanOptions::default(),
            None,
            Some(x1_seq),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(old.value, Bytes::from_static(b"x1"));
    let mut newest = host
        .scan_view(
            &view,
            hot("a")..hot("z"),
            &ScanOptions::default(),
            Some(Bytes::from_static(HOT)),
            None,
        )
        .await
        .unwrap();
    let entry = newest.next_entry().await.unwrap().unwrap();
    assert_eq!(
        entry.value,
        ValueDeletable::Value(Bytes::from_static(b"x2"))
    );
    assert!(host
        .scan_view(
            &view,
            hot("b")..hot("a"),
            &ScanOptions::default(),
            None,
            None
        )
        .await
        .is_err());

    // A host snapshot is a reader snapshot at the reader's seq.
    let snapshot = host.snapshot().await.unwrap();
    assert_eq!(snapshot.seq(), host.last_committed_seq());
    assert_eq!(reader.min_snapshot_seq(), Some(snapshot.seq()));
    drop(snapshot);
    assert_eq!(reader.min_snapshot_seq(), None);

    // After a flush to L0, a view keeps every version at or above the L0's seq only: the
    // writer's compactor does not know the reader's snapshots.
    let snapshot = reader.snapshot().await.unwrap();
    flush_memtable(&db).await;
    reader.refresh().await.unwrap();
    let view = host.capture_state_view().unwrap();
    let manifest = host.manifest().unwrap();
    let last_l0_seq = manifest.last_l0_seq();
    assert!(host.retains_seq(&view, last_l0_seq));
    assert!(!host.retains_seq(&view, x1_seq));
    let l0 = manifest.l0().front().expect("flushed L0 SST");
    assert!(!host.read_sst_index(&l0.sst).await.unwrap().is_empty());
    drop(snapshot);

    reader.close().await.unwrap();
    assert!(host.check_closed().is_err());
    drop(reader);
    assert!(host.capture_state_view().is_err());
    db.close().await.unwrap();
}

/// A wake runs `drain_writer` on the idle poller at the reader's seq. Close runs `close` while
/// the poller still answers wakes, then `close_writer` as the poller stops.
#[tokio::test]
async fn wake_drains_on_the_poller_and_close_hands_off_cleanly() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    put_durable(&db, &[(hot("x"), "x1")]).await;
    let cache = TestRowCache::new();
    cache.wake_on_close.store(true, Ordering::Relaxed);
    let reader = open_reader(&store, &cache).await;

    let notified = cache.drain_notify.notified();
    assert!(cache.host().wake_writer());
    notified.await;
    assert_eq!(
        cache.events.lock().unwrap().drained_seqs.last().copied(),
        Some(cache.host().last_committed_seq())
    );

    reader.close().await.unwrap();
    assert_eq!(
        cache.events.lock().unwrap().lifecycle,
        vec!["open", "close", "close_writer"]
    );
    assert!(!cache.host().wake_writer());
    db.close().await.unwrap();
}

/// A reader pinned to a checkpoint runs a poller for its cache, which answers wakes.
#[tokio::test]
async fn a_pinned_reader_runs_a_poller_for_its_cache() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    put_durable(&db, &[(hot("x"), "x1")]).await;
    flush_memtable(&db).await;
    let checkpoint = db
        .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
        .await
        .unwrap();
    let cache = TestRowCache::new();
    let reader = DbReader::builder(PATH, Arc::clone(&store))
        .with_reader_mode(DbReaderMode::Checkpoint(checkpoint.id))
        .with_row_cache(cache.clone())
        .build()
        .await
        .unwrap();
    assert!(!reader.refresh().await.unwrap());
    let notified = cache.drain_notify.notified();
    assert!(cache.host().wake_writer());
    notified.await;
    assert_eq!(
        reader.get(hot("x")).await.unwrap(),
        Some(Bytes::from_static(b"x1"))
    );
    reader.close().await.unwrap();
    assert_eq!(
        cache.events.lock().unwrap().lifecycle,
        vec!["open", "close", "close_writer"]
    );
    db.close().await.unwrap();
}

/// A cache that rejects open fails the reader's open, stops the poller (which calls
/// `close_writer`), and leaves no reader checkpoint behind.
#[tokio::test]
async fn open_rejection_stops_the_poller() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    let cache = TestRowCache::new();
    cache.reject_open.store(true, Ordering::Relaxed);
    let err = DbReader::builder(PATH, Arc::clone(&store))
        .with_options(reader_options())
        .with_row_cache(cache.clone())
        .build()
        .await
        .err()
        .expect("open fails");
    assert_eq!(err.kind(), ErrorKind::Invalid);
    assert_eq!(
        cache.events.lock().unwrap().lifecycle,
        vec!["open", "close_writer"]
    );
    assert!(!cache.host().wake_writer());
    assert!(admin.list_checkpoints(None).await.unwrap().is_empty());
    db.close().await.unwrap();
}
