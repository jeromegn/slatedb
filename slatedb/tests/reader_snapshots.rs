//! Integration tests for snapshots of a `DbReader` and for its on-demand refresh.
//!
//! A writer and a reader share one in-memory object store. The reader's periodic poll runs once
//! at open and then once an hour, so the tests drive polls with `DbReader::refresh`. The poll at
//! open may install a newer state before a test's first `refresh`, so these tests check states,
//! not what `refresh` returns after the store changed; the unit tests in `db_reader.rs` drive the
//! poller by hand for that. The garbage collector runs when a test calls it, with no minimum age,
//! so that only checkpoints protect SSTs.

#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use object_store::path::Path as OsPath;
use slatedb::admin::Admin;
use slatedb::config::{
    CheckpointOptions, CheckpointScope, CompactionWorkerOptions, CompactorOptions, DbReaderOptions,
    FlushOptions, FlushType, GarbageCollectorDirectoryOptions, GarbageCollectorOptions, Settings,
    SizeTieredCompactionSchedulerOptions,
};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use slatedb::{
    Db, DbIterator, DbReader, DbReaderMode, DbReaderSnapshot, ErrorKind, KeyValue, WriteBatch,
};

const PATH: &str = "reader_snapshots";

/// A reader that polls at open, then only on `refresh` in these tests.
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

/// A writer whose compactor merges every two L0 SSTs, quickly. Its compaction checkpoint expires
/// at once, so it protects nothing.
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

/// Garbage collection of every directory, with no minimum age.
fn gc_options() -> GarbageCollectorOptions {
    let directory = Some(GarbageCollectorDirectoryOptions {
        interval: None,
        min_age: Duration::ZERO,
        dry_run: false,
    });
    GarbageCollectorOptions {
        manifest_options: directory,
        wal_options: directory,
        compacted_options: directory,
        compactions_options: directory,
        ..GarbageCollectorOptions::default()
    }
}

async fn open_writer(store: &Arc<dyn ObjectStore>, settings: Settings) -> Db {
    Db::builder(PATH, Arc::clone(store))
        .with_settings(settings)
        .build()
        .await
        .unwrap()
}

async fn open_reader(store: &Arc<dyn ObjectStore>) -> DbReader {
    DbReader::builder(PATH, Arc::clone(store))
        .with_options(reader_options())
        .build()
        .await
        .unwrap()
}

fn key(i: usize) -> Bytes {
    Bytes::from(format!("key{i:04}"))
}

fn value(i: usize, round: usize) -> Bytes {
    Bytes::from(format!("value{i:04}-{round}"))
}

/// Writes keys `0..count` at `round` in one batch, and makes them durable.
async fn write_round(db: &Db, count: usize, round: usize) {
    let mut batch = WriteBatch::new();
    for i in 0..count {
        batch.put(key(i), value(i, round));
    }
    db.write(batch).await.unwrap();
    db.flush().await.unwrap();
}

async fn flush_memtable(db: &Db) {
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::MemTable,
    })
    .await
    .unwrap();
}

async fn collect(mut iter: DbIterator) -> Vec<KeyValue> {
    let mut rows = Vec::new();
    while let Some(row) = iter.next().await.unwrap() {
        rows.push(row);
    }
    rows
}

/// Checks that `snapshot` reads exactly keys `0..count` at `round`, by point reads and a scan.
async fn assert_round(snapshot: &DbReaderSnapshot, count: usize, round: usize) {
    for i in 0..count {
        assert_eq!(
            snapshot.get(key(i)).await.unwrap(),
            Some(value(i, round)),
            "key {i}"
        );
    }
    let rows = collect(snapshot.scan(..).await.unwrap()).await;
    let expected: Vec<(Bytes, Bytes)> = (0..count).map(|i| (key(i), value(i, round))).collect();
    let actual: Vec<(Bytes, Bytes)> = rows.into_iter().map(|row| (row.key, row.value)).collect();
    assert_eq!(actual, expected);
}

/// The SST objects in the store.
async fn sst_objects(store: &Arc<dyn ObjectStore>) -> BTreeSet<String> {
    let prefix = OsPath::from(format!("{PATH}/compacted"));
    store
        .list(Some(&prefix))
        .map(|meta| meta.unwrap().location.to_string())
        .collect()
        .await
}

/// Waits until the writer's compactor has merged L0 into sorted runs.
async fn wait_for_compaction(admin: &Admin) {
    for _ in 0..2000 {
        let manifest = admin.read_manifest(None).await.unwrap().unwrap();
        if manifest.l0().is_empty() && !manifest.compacted().is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the compactor did not merge L0");
}

async fn checkpoint_ids(admin: &Admin) -> Vec<uuid::Uuid> {
    admin
        .list_checkpoints(None)
        .await
        .unwrap()
        .into_iter()
        .map(|checkpoint| checkpoint.id)
        .collect()
}

/// A snapshot reads its state at its sequence number while the writer writes, flushes to L0 and
/// compacts, after the reader polls past it, after its checkpoint is replaced, and after the
/// garbage collector deletes every SST that the latest manifest no longer needs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn should_read_the_snapshot_state_through_flushes_compactions_and_gc() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, compacting_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    write_round(&db, 100, 1).await;
    flush_memtable(&db).await;

    let reader = open_reader(&store).await;
    let snapshot = reader.snapshot().await.unwrap();
    let first_ssts = sst_objects(&store).await;
    assert_eq!(first_ssts.len(), 1);
    assert_round(&snapshot, 100, 1).await;

    // Round 2 through the WAL, then a second L0 SST, which the compactor merges with the first
    // into a sorted run that holds round 2 only.
    write_round(&db, 100, 2).await;
    reader.refresh().await.unwrap();
    assert_round(&reader.snapshot().await.unwrap(), 100, 2).await;
    flush_memtable(&db).await;
    wait_for_compaction(&admin).await;
    write_round(&db, 100, 3).await;
    reader.refresh().await.unwrap();

    // The reader moved on to the compacted manifest and kept the snapshot's checkpoint.
    let latest = reader.snapshot().await.unwrap();
    assert_round(&latest, 100, 3).await;
    assert!(latest.seq() > snapshot.seq());
    assert_round(&snapshot, 100, 1).await;

    // The collector removes the compactor's expired checkpoints, and deletes nothing that the
    // kept checkpoint references.
    admin.run_gc_once(gc_options()).await.unwrap();
    assert_eq!(checkpoint_ids(&admin).await.len(), 2);
    assert!(first_ssts.is_subset(&sst_objects(&store).await));
    assert_round(&snapshot, 100, 1).await;
    assert_eq!(reader.min_snapshot_seq(), Some(snapshot.seq()));

    // Once the snapshot drops, the next poll deletes its checkpoint, and the collector its SST.
    drop(snapshot);
    assert_eq!(reader.min_snapshot_seq(), Some(latest.seq()));
    assert!(!reader.refresh().await.unwrap());
    assert_eq!(checkpoint_ids(&admin).await.len(), 1);
    admin.run_gc_once(gc_options()).await.unwrap();
    assert!(first_ssts.is_disjoint(&sst_objects(&store).await));
    assert_round(&latest, 100, 3).await;
    drop(latest);
    assert_eq!(reader.min_snapshot_seq(), None);

    reader.close().await.unwrap();
    assert!(checkpoint_ids(&admin).await.is_empty());
    db.close().await.unwrap();
}

/// Without a snapshot, a checkpoint replacement removes the old checkpoint: the reader keeps one
/// checkpoint, as before snapshots existed. That the replacement is one manifest write is checked
/// in `db_reader.rs`, where the test drives the poller by hand.
#[tokio::test]
async fn should_replace_the_checkpoint_without_snapshots() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    write_round(&db, 10, 1).await;
    let reader = open_reader(&store).await;
    let before = checkpoint_ids(&admin).await;
    assert_eq!(before.len(), 1);

    flush_memtable(&db).await;
    reader.refresh().await.unwrap();
    let after = checkpoint_ids(&admin).await;
    assert_eq!(after.len(), 1);
    assert_ne!(after, before);
    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// A snapshot's reads fail once its checkpoint is gone from the manifest, here deleted behind
/// the reader's back. The reader and its new snapshots go on reading.
#[tokio::test]
async fn should_fail_snapshot_reads_once_its_checkpoint_is_gone() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    write_round(&db, 10, 1).await;
    let reader = open_reader(&store).await;
    let snapshot = reader.snapshot().await.unwrap();

    // The writer flushes, and the reader keeps the snapshot's checkpoint.
    flush_memtable(&db).await;
    write_round(&db, 10, 2).await;
    reader.refresh().await.unwrap();
    let checkpoints = admin.list_checkpoints(None).await.unwrap();
    assert_eq!(checkpoints.len(), 2);
    // The kept checkpoint is the one on the older manifest.
    let kept = checkpoints
        .iter()
        .min_by_key(|checkpoint| checkpoint.manifest_id)
        .unwrap()
        .id;
    admin.delete_checkpoint(kept).await.unwrap();
    assert_round(&snapshot, 10, 1).await;

    // The next poll finds it gone.
    reader.refresh().await.unwrap();
    let err = snapshot.get(key(0)).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Unavailable);
    assert!(err.to_string().contains("reader snapshot expired"), "{err}");
    assert!(snapshot.scan(..).await.is_err());
    assert_eq!(reader.get(key(0)).await.unwrap(), Some(value(0, 2)));
    assert_round(&reader.snapshot().await.unwrap(), 10, 2).await;
    drop(snapshot);
    reader.refresh().await.unwrap();
    assert_eq!(checkpoint_ids(&admin).await.len(), 1);

    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// Closing a reader deletes its checkpoint and the ones it kept for live snapshots, in one
/// manifest write. The snapshots of a closed reader fail their reads.
#[tokio::test]
async fn should_delete_kept_checkpoints_at_close() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    write_round(&db, 4, 1).await;
    let reader = open_reader(&store).await;
    let snapshot = reader.snapshot().await.unwrap();
    flush_memtable(&db).await;
    reader.refresh().await.unwrap();
    assert_eq!(checkpoint_ids(&admin).await.len(), 2);

    let manifests = admin.list_manifests(..).await.unwrap().len();
    reader.close().await.unwrap();
    assert!(checkpoint_ids(&admin).await.is_empty());
    assert_eq!(admin.list_manifests(..).await.unwrap().len(), manifests + 1);
    let err = snapshot.get(key(0)).await.unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Closed(_)), "{err}");
    db.close().await.unwrap();
}

/// `refresh` installs what the writer made durable, reports `false` when nothing changed, and
/// fails once the reader is closed.
#[tokio::test]
async fn should_refresh_on_demand() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    write_round(&db, 4, 1).await;
    let reader = open_reader(&store).await;
    assert!(!reader.refresh().await.unwrap());

    write_round(&db, 4, 2).await;
    reader.refresh().await.unwrap();
    assert_eq!(reader.get(key(0)).await.unwrap(), Some(value(0, 2)));
    assert!(!reader.refresh().await.unwrap());

    flush_memtable(&db).await;
    write_round(&db, 4, 3).await;
    reader.refresh().await.unwrap();
    assert_eq!(reader.get(key(0)).await.unwrap(), Some(value(0, 3)));
    assert!(!reader.manifest().l0().is_empty());
    assert!(!reader.refresh().await.unwrap());

    reader.close().await.unwrap();
    let err = reader.refresh().await.unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Closed(_)));
    db.close().await.unwrap();
}

/// A reader pinned to a checkpoint never polls: `refresh` reports nothing, and its snapshots read
/// the checkpoint's state.
#[tokio::test]
async fn should_not_refresh_a_reader_pinned_to_a_checkpoint() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, quiet_settings()).await;
    write_round(&db, 4, 1).await;
    flush_memtable(&db).await;
    let checkpoint = db
        .create_checkpoint(CheckpointScope::All, &CheckpointOptions::default())
        .await
        .unwrap();
    write_round(&db, 4, 2).await;
    let reader = DbReader::builder(PATH, Arc::clone(&store))
        .with_reader_mode(DbReaderMode::Checkpoint(checkpoint.id))
        .build()
        .await
        .unwrap();
    assert!(!reader.refresh().await.unwrap());
    assert_round(&reader.snapshot().await.unwrap(), 4, 1).await;
    reader.close().await.unwrap();
    db.close().await.unwrap();
}

/// Snapshots taken while the writer writes and flushes, and while the reader polls, replaces
/// checkpoints and the collector runs, each read one consistent state: two keys written in one
/// batch always read equal, a later snapshot never reads an older state, and a snapshot held over
/// many polls reads the same state at the end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_read_consistent_snapshots_while_the_reader_polls() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = open_writer(&store, compacting_settings()).await;
    let admin = Admin::builder(PATH, Arc::clone(&store)).build();
    let put = |db: &Db, round: u64| {
        let db = db.clone();
        async move {
            let mut batch = WriteBatch::new();
            batch.put(b"a", round.to_be_bytes());
            batch.put(b"b", round.to_be_bytes());
            db.write(batch).await.unwrap();
            db.flush().await.unwrap();
        }
    };
    put(&db, 0).await;
    let reader = Arc::new(open_reader(&store).await);

    let writer = {
        let db = db.clone();
        tokio::spawn(async move {
            for round in 1..=60u64 {
                put(&db, round).await;
                if round % 10 == 0 {
                    flush_memtable(&db).await;
                }
            }
        })
    };
    let poller = {
        let reader = Arc::clone(&reader);
        tokio::spawn(async move {
            for _ in 0..80 {
                reader.refresh().await.unwrap();
                admin.run_gc_once(gc_options()).await.unwrap();
            }
        })
    };
    let mut readers = Vec::new();
    for _ in 0..3 {
        let reader = Arc::clone(&reader);
        readers.push(tokio::spawn(async move {
            let mut last = 0;
            let mut held: Vec<(Arc<DbReaderSnapshot>, u64)> = Vec::new();
            for step in 0..200 {
                let snapshot = reader.snapshot().await.unwrap();
                let a = snapshot.get(b"a").await.unwrap().unwrap();
                let b = snapshot.get(b"b").await.unwrap().unwrap();
                assert_eq!(a, b);
                let round = u64::from_be_bytes(a.as_ref().try_into().unwrap());
                assert!(round >= last, "round {round} after {last}");
                last = round;
                if step % 20 == 0 {
                    held.push((snapshot, round));
                }
                tokio::task::yield_now().await;
            }
            for (snapshot, round) in held {
                let a = snapshot.get(b"a").await.unwrap().unwrap();
                assert_eq!(u64::from_be_bytes(a.as_ref().try_into().unwrap()), round);
                let rows = collect(snapshot.scan(..).await.unwrap()).await;
                assert_eq!(rows.len(), 2);
            }
        }));
    }
    writer.await.unwrap();
    poller.await.unwrap();
    for task in readers {
        task.await.unwrap();
    }
    assert_eq!(reader.min_snapshot_seq(), None);
    reader.close().await.unwrap();
    db.close().await.unwrap();
}
