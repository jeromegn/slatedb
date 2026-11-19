//! Integration tests for the range cache (RFC 0040).
//!
//! Most tests compare reads that the range cache serves with reads through a snapshot that use
//! the normal read path. The cache serves snapshot reads too, so these reads carry a tracing
//! option: the cache never serves a read with one, and a tracing option changes no result. See
//! `normal_read` and `should_not_serve_traced_reads`.

#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use slatedb::config::SizeTieredCompactionSchedulerOptions;
use slatedb::config::{CompactionWorkerOptions, CompactorOptions, Settings};
use slatedb::config::{
    FlushOptions, FlushType, MergeOptions, PutOptions, ReadOptions, ScanOptions, TracingOptions,
    Ttl, WriteOptions,
};
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::ObjectStore;
use slatedb::range_cache_stats::{
    ADMISSION_COUNT, CANDIDATE_COUNT, GET_COUNT, LOAD_COUNT, PROMOTION_COUNT, REBUILD_COUNT,
    REJECTION_COUNT, REMOVE_COUNT, SAMPLE_COUNT, SCAN_COUNT,
};
use slatedb::{
    AutoPromoteOptions, CachedRangeInfo, Db, DbIterator, DbReadOps, ErrorKind, IsolationLevel,
    IterationOrder, KeyValue, MergeOperator, MergeOperatorError, RangeCacheOptions,
    RangeCachePageSource, WriteBatch,
};
use slatedb_common::clock::{MockSystemClock, SystemClock};
use slatedb_common::metrics::{lookup_metric, lookup_metric_with_labels, DefaultMetricsRecorder};
use tempfile::TempDir;
use tokio::runtime::Runtime;

struct ConcatMergeOperator;

impl MergeOperator for ConcatMergeOperator {
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
    dir: TempDir,
}

impl TestDb {
    fn metric(&self, name: &str, labels: &[(&str, &str)]) -> i64 {
        if labels.is_empty() {
            lookup_metric(&self.recorder, name).unwrap_or(0)
        } else {
            lookup_metric_with_labels(&self.recorder, name, labels).unwrap_or(0)
        }
    }

    fn cache_files(&self) -> usize {
        cache_files(self.dir.path())
    }
}

fn cache_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
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

fn options(dir: &Path) -> RangeCacheOptions {
    RangeCacheOptions::new(dir)
}

async fn open_db(
    object_store: Arc<dyn ObjectStore>,
    tune: impl FnOnce(&mut RangeCacheOptions),
) -> TestDb {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(dir.path());
    tune(&mut opts);
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let db = Db::builder("range_cache_test", object_store)
        .with_range_cache(opts)
        .with_merge_operator(Arc::new(ConcatMergeOperator))
        .with_metrics_recorder(recorder.clone())
        .build()
        .await
        .unwrap();
    TestDb { db, recorder, dir }
}

async fn open(tune: impl FnOnce(&mut RangeCacheOptions)) -> TestDb {
    open_db(Arc::new(InMemory::new()), tune).await
}

fn key(i: usize) -> Bytes {
    Bytes::from(format!("key{i:05}"))
}

fn value(i: usize, round: usize) -> Bytes {
    Bytes::from(format!("value-{i}-{round}"))
}

async fn flush_memtable(db: &Db) {
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::MemTable,
    })
    .await
    .unwrap();
}

async fn collect(mut iter: DbIterator) -> Vec<KeyValue> {
    let mut out = Vec::new();
    while let Some(kv) = iter.next().await.unwrap() {
        out.push(kv);
    }
    out
}

/// Read options that make a read take the normal read path: the range cache never serves a
/// read with a tracing option.
fn normal_read() -> ReadOptions {
    ReadOptions::default().with_tracing_options(Some(TracingOptions::new("normal-path")))
}

/// Scan options that make a scan take the normal read path. See [`normal_read`].
fn normal_scan() -> ScanOptions {
    ScanOptions::default().with_tracing_options(Some(TracingOptions::new("normal-path")))
}

/// Reads `key` in `reader` through the normal read path.
async fn normal_get<R: DbReadOps + Sync>(reader: &R, key: &Bytes) -> Option<KeyValue> {
    reader
        .get_key_value_with_options(key, &normal_read())
        .await
        .unwrap()
}

/// Scans `range` in `reader` through the normal read path.
async fn normal_scan_of<R: DbReadOps + Sync>(
    reader: &R,
    range: (Bound<Bytes>, Bound<Bytes>),
) -> Vec<KeyValue> {
    collect(
        reader
            .scan_with_options(range, &normal_scan())
            .await
            .unwrap(),
    )
    .await
}

/// Asserts that point reads of `keys` and a scan of `range` agree with the normal read path at
/// a snapshot.
async fn assert_matches_snapshot(
    db: &Db,
    keys: impl IntoIterator<Item = Bytes>,
    range: (Bound<Bytes>, Bound<Bytes>),
) {
    let snapshot = db.snapshot().unwrap();
    for k in keys {
        assert_eq!(
            db.get_key_value(&k).await.unwrap(),
            normal_get(snapshot.as_ref(), &k).await,
            "key {k:?}"
        );
    }
    assert_eq!(
        collect(db.scan(range.clone()).await.unwrap()).await,
        normal_scan_of(snapshot.as_ref(), range).await
    );
}

async fn wait_for(cond: impl Fn() -> bool) {
    for _ in 0..500 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("condition not met in time");
}

#[tokio::test]
async fn should_be_inert_when_not_configured() {
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let db = Db::builder("inert", Arc::new(InMemory::new()))
        .with_metrics_recorder(recorder.clone())
        .build()
        .await
        .unwrap();
    db.put(b"a", b"1").await.unwrap();

    let err = db.cache_range(..).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Invalid);
    assert_eq!(db.uncache_range(..).unwrap_err().kind(), ErrorKind::Invalid);
    assert_eq!(
        db.clear_range_cache().unwrap_err().kind(),
        ErrorKind::Invalid
    );
    assert!(db.cached_ranges().is_empty());
    assert_eq!(db.get(b"a").await.unwrap(), Some(Bytes::from_static(b"1")));
    // The cache registers no metrics.
    let snapshot = recorder.snapshot();
    assert!(snapshot
        .all()
        .iter()
        .all(|m| !m.name.starts_with("slatedb.range_cache.")));
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_serve_point_reads_from_the_cache() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in (0..200).step_by(2) {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    flush_memtable(db).await;
    for i in (100..200).step_by(4) {
        db.put(key(i), value(i, 1)).await.unwrap();
    }

    db.cache_range(key(50)..key(150)).await.unwrap();
    let ranges = db.cached_ranges();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].start, key(50));
    assert_eq!(ranges[0].end, Some(key(150)));
    assert!(!ranges[0].loading);
    assert_eq!(ranges[0].file_rows, 50);
    assert_eq!(t.cache_files(), 1);
    assert_eq!(t.metric(LOAD_COUNT, &[]), 1);

    assert_matches_snapshot(db, (40..160).map(key), (Unbounded, Unbounded)).await;
    // Keys 50..150 are cached: 50 values and 50 definite misses.
    assert_eq!(t.metric(GET_COUNT, &[("result", "hit")]), 50);
    assert_eq!(t.metric(GET_COUNT, &[("result", "absent")]), 50);

    // Reads with other options use the normal read path.
    let dirty = ReadOptions {
        dirty: true,
        ..ReadOptions::default()
    };
    assert_eq!(
        db.get_with_options(key(100), &dirty).await.unwrap(),
        Some(value(100, 1))
    );
    assert_eq!(t.metric(GET_COUNT, &[("result", "hit")]), 50);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_apply_puts_deletes_batches_and_transactions() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in 0..100 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(10)..key(90)).await.unwrap();
    let before = db.snapshot().unwrap();

    db.put(key(20), value(20, 1)).await.unwrap();
    db.delete(key(21)).await.unwrap();
    let mut batch = WriteBatch::new();
    batch.put(key(22), value(22, 1));
    batch.delete(key(23));
    batch.put(key(95), value(95, 1));
    batch.put(Bytes::from("key00030a"), Bytes::from("new"));
    db.write(batch).await.unwrap();
    let txn = db.begin(IsolationLevel::Snapshot).unwrap();
    txn.put(key(24), value(24, 1)).unwrap();
    txn.delete(key(25)).unwrap();
    txn.commit().await.unwrap();
    // A transaction that conflicts writes nothing.
    let loser = db.begin(IsolationLevel::Snapshot).unwrap();
    loser.put(key(26), value(26, 9)).unwrap();
    db.put(key(26), value(26, 1)).await.unwrap();
    assert!(loser.commit().await.is_err());

    assert_eq!(db.get(key(20)).await.unwrap(), Some(value(20, 1)));
    assert_eq!(db.get(key(21)).await.unwrap(), None);
    assert_eq!(db.get(key(22)).await.unwrap(), Some(value(22, 1)));
    assert_eq!(db.get(key(23)).await.unwrap(), None);
    assert_eq!(db.get(key(24)).await.unwrap(), Some(value(24, 1)));
    assert_eq!(db.get(key(25)).await.unwrap(), None);
    assert_eq!(db.get(key(26)).await.unwrap(), Some(value(26, 1)));
    assert_eq!(db.get("key00030a").await.unwrap(), Some(Bytes::from("new")));
    // A snapshot from before the writes still reads the old state.
    assert_eq!(before.get(key(20)).await.unwrap(), Some(value(20, 0)));
    assert_eq!(before.get(key(21)).await.unwrap(), Some(value(21, 0)));

    flush_memtable(db).await;
    assert_matches_snapshot(
        db,
        (0..100).map(key),
        (Included(key(10)), Excluded(key(90))),
    )
    .await;
    assert!(t.metric(GET_COUNT, &[("result", "hit")]) > 0);
    assert!(t.metric(SCAN_COUNT, &[]) > 0);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_serve_covered_scans_only() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in 0..100 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(10)..key(90)).await.unwrap();
    let snapshot = db.snapshot().unwrap();

    let covered = (Included(key(20)), Excluded(key(40)));
    assert_eq!(
        collect(db.scan(covered.clone()).await.unwrap()).await,
        normal_scan_of(snapshot.as_ref(), covered).await
    );
    assert_eq!(t.metric(SCAN_COUNT, &[]), 1);

    // Seek works on a scan that the cache serves.
    let mut iter = db.scan(key(10)..key(90)).await.unwrap();
    assert_eq!(iter.next().await.unwrap().unwrap().key, key(10));
    iter.seek(key(50)).await.unwrap();
    assert_eq!(iter.next().await.unwrap().unwrap().key, key(50));
    assert_eq!(t.metric(SCAN_COUNT, &[]), 2);

    // Scans that leave the range, and descending scans, use the normal read path.
    let crossing = (Included(key(80)), Excluded(key(95)));
    assert_eq!(collect(db.scan(crossing).await.unwrap()).await.len(), 15);
    assert_eq!(collect(db.scan(key(10)..).await.unwrap()).await.len(), 90);
    let descending = ScanOptions {
        order: IterationOrder::Descending,
        ..ScanOptions::default()
    };
    let rows = collect(
        db.scan_with_options(key(20)..key(30), &descending)
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(rows.first().unwrap().key, key(29));
    assert_eq!(t.metric(SCAN_COUNT, &[]), 2);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_read_batches_like_next() {
    let t = open(|o| o.tuning.min_rebuild_delta_bytes = 2048).await;
    let db = &t.db;
    for i in 0..300 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(0)..key(200)).await.unwrap();
    // Merge operands need an await in a cached scan, and tombstones are skipped.
    for i in (0..300).step_by(13) {
        db.merge(key(i), b"+m").await.unwrap();
    }
    for i in (0..300).step_by(17) {
        db.delete(key(i)).await.unwrap();
    }
    // A cached scan, and a scan that leaves the cached range and uses the normal path.
    for range in [key(0)..key(200), key(150)..key(300)] {
        let expected = collect(db.scan(range.clone()).await.unwrap()).await;
        for max in [1, 7, 64, 1000] {
            let mut iter = db.scan(range.clone()).await.unwrap();
            let mut got = Vec::new();
            loop {
                let n = iter.next_batch(&mut got, max).await.unwrap();
                assert!(n <= max);
                if n == 0 {
                    break;
                }
            }
            assert_eq!(got, expected, "{range:?} in batches of {max}");
            assert_eq!(iter.next().await.unwrap(), None);
        }
    }
    // A batch moves the iterator as `next` does: seek must go past its last row.
    let mut iter = db.scan(key(0)..key(200)).await.unwrap();
    let mut got = Vec::new();
    assert_eq!(iter.next_batch(&mut got, 10).await.unwrap(), 10);
    assert!(iter.seek(got[9].key.clone()).await.is_err());
    iter.seek(key(100)).await.unwrap();
    assert_eq!(iter.next().await.unwrap().unwrap().key, key(100));
    assert_eq!(iter.next_batch(&mut got, 0).await.unwrap(), 0);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_scan_across_adjacent_ranges() {
    let t = open(|o| o.tuning.min_rebuild_delta_bytes = 2048).await;
    let db = &t.db;
    for i in 0..300 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    // Three adjacent ranges, loaded one by one, and one more after a gap.
    db.cache_range(key(0)..key(100)).await.unwrap();
    db.cache_range(key(100)..key(200)).await.unwrap();
    db.cache_range(key(200)..key(250)).await.unwrap();
    db.cache_range(key(260)..key(300)).await.unwrap();
    assert_eq!(db.cached_ranges().len(), 4);

    // Writes land in every range, and some deltas rebuild.
    for round in 1..4 {
        for i in (0..300).step_by(7) {
            db.put(key(i), value(i, round)).await.unwrap();
        }
        db.delete(key(round * 50)).await.unwrap();
    }
    let all: Vec<Bytes> = (0..300).map(key).collect();
    let before = t.metric(SCAN_COUNT, &[]);
    assert_matches_snapshot(db, all.clone(), (Included(key(50)), Excluded(key(240)))).await;
    assert_matches_snapshot(db, all.clone(), (Included(key(0)), Excluded(key(250)))).await;
    assert_eq!(t.metric(SCAN_COUNT, &[]), before + 2);

    // A scan pins every part at one sequence number: writes after it opens stay invisible.
    let snapshot = db.snapshot().unwrap();
    let iter = db.scan(key(90)..key(210)).await.unwrap();
    db.put(key(95), b"later").await.unwrap();
    db.put(key(205), b"later").await.unwrap();
    db.delete(key(150)).await.unwrap();
    assert_eq!(
        collect(iter).await,
        normal_scan_of(snapshot.as_ref(), (Included(key(90)), Excluded(key(210)))).await
    );
    assert_eq!(t.metric(SCAN_COUNT, &[]), before + 3);

    // Seek across the parts.
    let mut iter = db.scan(key(0)..key(250)).await.unwrap();
    assert_eq!(iter.next().await.unwrap().unwrap().key, key(0));
    iter.seek(key(201)).await.unwrap();
    assert_eq!(iter.next().await.unwrap().unwrap().key, key(201));
    assert!(iter.seek(key(100)).await.is_err());
    assert_eq!(t.metric(SCAN_COUNT, &[]), before + 4);

    // A scan across the gap uses the normal read path.
    assert_eq!(
        collect(db.scan(key(240)..key(270)).await.unwrap())
            .await
            .len(),
        30
    );
    assert_eq!(t.metric(SCAN_COUNT, &[]), before + 4);

    // So does a scan across a removed part.
    db.uncache_range(key(150)..key(151)).unwrap();
    assert_matches_snapshot(db, [], (Included(key(50)), Excluded(key(240)))).await;
    assert_eq!(t.metric(SCAN_COUNT, &[]), before + 4);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_resolve_merge_operands() {
    let t = open(|o| o.tuning.min_rebuild_delta_bytes = 512).await;
    let db = &t.db;
    db.put(b"m1", b"base").await.unwrap();
    db.merge(b"m2", b"op0").await.unwrap();
    flush_memtable(db).await;
    db.cache_range(&b"m"[..]..&b"n"[..]).await.unwrap();

    for round in 0..50 {
        let operand = format!("+{round}");
        db.merge(b"m1", operand.as_bytes()).await.unwrap();
        db.merge_with_options(
            b"m2",
            operand.as_bytes(),
            &MergeOptions::default(),
            &WriteOptions::default(),
        )
        .await
        .unwrap();
        db.merge(b"m3", operand.as_bytes()).await.unwrap();
        assert_matches_snapshot(
            db,
            [b"m1", b"m2", b"m3"].map(|k| Bytes::from_static(k)),
            (Included(Bytes::from("m")), Excluded(Bytes::from("n"))),
        )
        .await;
    }
    assert!(t.metric(GET_COUNT, &[("result", "merge")]) > 0);
    wait_for(|| db.cached_ranges().iter().all(|r| !r.rebuilding)).await;
    assert!(
        t.metric(REBUILD_COUNT, &[]) > 0,
        "the merges caused no rebuild"
    );
    let expected: String = (0..50).map(|r| format!("+{r}")).collect();
    assert_eq!(
        db.get(b"m1").await.unwrap(),
        Some(Bytes::from(format!("base{expected}")))
    );
    assert_eq!(
        db.get(b"m2").await.unwrap(),
        Some(Bytes::from(format!("op0{expected}")))
    );
    assert_eq!(db.get(b"m3").await.unwrap(), Some(Bytes::from(expected)));
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_not_cache_rows_that_expire() {
    let t = open(|_| {}).await;
    let db = &t.db;
    let ttl = PutOptions {
        ttl: Ttl::ExpireAfterMillis(3_600_000),
    };
    db.put(key(1), value(1, 0)).await.unwrap();
    db.put_with_options(key(5), value(5, 0), &ttl, &WriteOptions::default())
        .await
        .unwrap();

    // A load of a range with an expiring row fails and caches nothing.
    let err = db.cache_range(key(0)..key(10)).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Invalid);
    assert!(db.cached_ranges().is_empty());
    assert_eq!(t.cache_files(), 0);

    // An expiring write to a cached range removes the range.
    db.cache_range(key(10)..key(20)).await.unwrap();
    db.put(key(11), value(11, 0)).await.unwrap();
    assert_eq!(db.cached_ranges().len(), 1);
    db.put_with_options(key(12), value(12, 0), &ttl, &WriteOptions::default())
        .await
        .unwrap();
    assert!(db.cached_ranges().is_empty());
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "expiring_write")]), 1);
    let kv = db.get_key_value(key(12)).await.unwrap().unwrap();
    assert!(kv.expire_ts.is_some());
    assert_eq!(db.get(key(11)).await.unwrap(), Some(value(11, 0)));
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_rebuild_grown_deltas() {
    let t = open(|o| {
        o.tuning.min_rebuild_delta_bytes = 2048;
        o.tuning.page_size = 512;
    })
    .await;
    let db = &t.db;
    for i in 0..200 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(..).await.unwrap();
    for round in 1..=10 {
        for i in (round..200).step_by(7) {
            db.put(key(i), value(i, round)).await.unwrap();
        }
        db.delete(key(round * 3)).await.unwrap();
    }
    wait_for(|| db.cached_ranges().iter().all(|r| !r.rebuilding)).await;
    assert!(t.metric(REBUILD_COUNT, &[]) > 1);
    assert_eq!(db.cached_ranges().len(), 1);
    assert_matches_snapshot(db, (0..200).map(key), (Unbounded, Unbounded)).await;
    // A rebuild deletes the old file.
    assert_eq!(t.cache_files(), 1);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_evict_the_least_recently_read_range() {
    // A file of 50 rows takes one page of 4096 bytes, a fence region and a trailer, so the
    // budget holds two ranges.
    let t = open(|o| {
        o.tuning.page_size = 4096;
        o.max_disk_bytes = 9000;
    })
    .await;
    let db = &t.db;
    for i in 0..300 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(0)..key(50)).await.unwrap();
    db.cache_range(key(100)..key(150)).await.unwrap();
    assert_eq!(db.cached_ranges().len(), 2);
    // Read the first range, so the second one is the least recently read range.
    db.get(key(10)).await.unwrap();
    db.cache_range(key(200)..key(250)).await.unwrap();
    let starts: Vec<_> = db.cached_ranges().into_iter().map(|r| r.start).collect();
    assert_eq!(starts, vec![key(0), key(200)]);
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]), 1);
    assert_eq!(t.cache_files(), 2);

    // A range larger than the budget is rejected.
    db.clear_range_cache().unwrap();
    let err = db.cache_range(..).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Invalid);
    assert_matches_snapshot(db, (0..300).map(key), (Unbounded, Unbounded)).await;
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_uncache_and_clear_ranges() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in 0..100 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(0)..key(10)).await.unwrap();
    db.cache_range(key(20)..key(30)).await.unwrap();
    db.cache_range(key(40)..key(50)).await.unwrap();
    // A range that overlaps cached ranges only loads the gaps.
    db.cache_range(key(5)..key(25)).await.unwrap();
    assert_eq!(db.cached_ranges().len(), 4);

    db.uncache_range(key(8)..key(9)).unwrap();
    let starts: Vec<_> = db.cached_ranges().into_iter().map(|r| r.start).collect();
    assert_eq!(starts, vec![key(10), key(20), key(40)]);
    db.clear_range_cache().unwrap();
    assert!(db.cached_ranges().is_empty());
    assert_eq!(t.cache_files(), 0);
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "explicit")]), 4);
    assert_matches_snapshot(db, (0..100).map(key), (Unbounded, Unbounded)).await;
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_drop_the_cache_on_close() {
    let t = open(|_| {}).await;
    let db = &t.db;
    db.put(key(1), value(1, 0)).await.unwrap();
    db.cache_range(..).await.unwrap();
    assert_eq!(t.cache_files(), 1);
    db.close().await.unwrap();
    assert!(db.cached_ranges().is_empty());
    assert_eq!(t.cache_files(), 0);
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "closed")]), 1);
    assert_eq!(
        db.cache_range(..).await.unwrap_err().kind(),
        ErrorKind::Closed(slatedb::CloseReason::Clean)
    );
}

#[tokio::test]
async fn should_remove_old_files_on_open() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("00000000000000000007.rcf"), b"old").unwrap();
    std::fs::write(dir.path().join("00000000000000000008.rcf.tmp"), b"old").unwrap();
    std::fs::write(dir.path().join("keep.txt"), b"not ours").unwrap();
    let db = Db::builder("old_files", Arc::new(InMemory::new()))
        .with_range_cache(options(dir.path()))
        .build()
        .await
        .unwrap();
    let mut names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["keep.txt".to_string()]);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_drop_the_cache_when_fenced() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let t = open_db(store.clone(), |_| {}).await;
    t.db.put(key(1), value(1, 0)).await.unwrap();
    t.db.cache_range(..).await.unwrap();
    assert_eq!(t.cache_files(), 1);

    let other = Db::open("range_cache_test", store).await.unwrap();
    // The old writer finds out that it is fenced when it flushes its WAL. A background flush
    // can find out first, and then the put fails.
    let err = match t.db.put(key(2), value(2, 0)).await {
        Ok(_) => t.db.flush().await.unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::Closed(slatedb::CloseReason::Fenced));
    wait_for(|| t.db.cached_ranges().is_empty()).await;
    assert_eq!(t.cache_files(), 0);
    assert!(t.db.get(key(1)).await.is_err());
    other.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_not_lose_writes_that_race_a_load() {
    for page_source in [
        RangeCachePageSource::Mmap,
        RangeCachePageSource::PositionalRead,
    ] {
        let t = open(|o| {
            o.page_source = page_source;
            o.tuning.min_rebuild_delta_bytes = 4096;
        })
        .await;
        let db = t.db.clone();
        for i in 0..2000 {
            db.put(key(i), value(i, 0)).await.unwrap();
        }
        let writer = {
            let db = db.clone();
            tokio::spawn(async move {
                for round in 1..=3 {
                    for i in (0..2000).step_by(3) {
                        if i % 2 == 0 {
                            db.put(key(i), value(i, round)).await.unwrap();
                        } else {
                            db.delete(key(i)).await.unwrap();
                        }
                    }
                }
            })
        };
        for start in (0..2000).step_by(250) {
            db.cache_range(key(start)..key(start + 250)).await.unwrap();
            db.uncache_range(key(start)..key(start + 1)).unwrap();
            db.cache_range(key(start)..key(start + 250)).await.unwrap();
        }
        writer.await.unwrap();
        wait_for(|| db.cached_ranges().iter().all(|r| !r.rebuilding)).await;
        assert_eq!(db.cached_ranges().len(), 8);
        assert_matches_snapshot(&db, (0..2000).map(key), (Unbounded, Unbounded)).await;
        db.close().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn should_not_serve_a_write_before_it_commits() {
    use slatedb::fail_parallel::{self, FailPointRegistry};

    let dir = tempfile::tempdir().unwrap();
    let fp_registry = Arc::new(FailPointRegistry::new());
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let db = Db::builder("pre_commit", Arc::new(InMemory::new()))
        .with_range_cache(options(dir.path()))
        .with_fp_registry(fp_registry.clone())
        .with_metrics_recorder(recorder.clone())
        .build()
        .await
        .unwrap();
    db.put(b"a1", b"old").await.unwrap();
    db.cache_range(&b"a"[..]..&b"b"[..]).await.unwrap();

    // Pause the writer after it applied the batch to the memtable and the cache, and before it
    // advances the last committed sequence number.
    fail_parallel::cfg(fp_registry.clone(), "write-batch-pre-commit", "pause").unwrap();
    let writer = {
        let db = db.clone();
        tokio::spawn(async move {
            let mut batch = WriteBatch::new();
            batch.put(b"a1", b"new");
            batch.put(b"a2", b"new");
            batch.put(b"x1", b"new");
            db.write(batch).await.unwrap();
        })
    };
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The cache serves the old state.
    assert_eq!(
        db.get(b"a1").await.unwrap(),
        Some(Bytes::from_static(b"old"))
    );
    assert_eq!(db.get(b"a2").await.unwrap(), None);
    let rows = collect(db.scan(&b"a"[..]..&b"b"[..]).await.unwrap()).await;
    assert_eq!(rows.len(), 1);
    let hits = lookup_metric_with_labels(&recorder, GET_COUNT, &[("result", "hit")]);
    assert_eq!(hits, Some(1));
    // A load while the write is in flight includes the write. Its range serves no read until
    // the write commits.
    db.cache_range(&b"x"[..]..&b"y"[..]).await.unwrap();
    assert_eq!(db.get(b"x1").await.unwrap(), None);

    fail_parallel::cfg(fp_registry.clone(), "write-batch-pre-commit", "off").unwrap();
    writer.await.unwrap();
    assert_eq!(
        db.get(b"a1").await.unwrap(),
        Some(Bytes::from_static(b"new"))
    );
    assert_eq!(
        db.get(b"a2").await.unwrap(),
        Some(Bytes::from_static(b"new"))
    );
    assert_eq!(
        db.get(b"x1").await.unwrap(),
        Some(Bytes::from_static(b"new"))
    );
    let hits = lookup_metric_with_labels(&recorder, GET_COUNT, &[("result", "hit")]);
    assert_eq!(hits, Some(4));
    db.close().await.unwrap();
}

/// A load registers its range in two steps: it publishes the range in the index, then it picks
/// its snapshot. The test pauses the load before and after the publish, and writes a key in the
/// range during the pause. A write before the publish does not find the range, so the snapshot
/// must include it. A write after the publish finds the range and waits to apply itself to the
/// delta. In both cases the cache must serve the write.
#[rstest::rstest]
#[case::before_publish("range-cache-register-publish", true)]
#[case::after_publish("range-cache-register-snapshot", false)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_not_lose_a_write_that_races_registration(
    #[case] fail_point: &str,
    #[case] write_commits_during_pause: bool,
) {
    use slatedb::fail_parallel::{self, FailPointRegistry};

    let dir = tempfile::tempdir().unwrap();
    let fp_registry = Arc::new(FailPointRegistry::new());
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let db = Db::builder("register_race", Arc::new(InMemory::new()))
        .with_range_cache(options(dir.path()))
        .with_fp_registry(fp_registry.clone())
        .with_metrics_recorder(recorder.clone())
        .build()
        .await
        .unwrap();
    db.put(b"a1", b"old").await.unwrap();

    // Pause a load inside the registration of its range, while it holds the cache mutex.
    fail_parallel::cfg(fp_registry.clone(), fail_point, "pause").unwrap();
    let loader = {
        let db = db.clone();
        tokio::spawn(async move { db.cache_range(&b"a"[..]..&b"b"[..]).await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Write a key in the range during the pause, and give the write time to commit.
    let mut writer = {
        let db = db.clone();
        tokio::spawn(async move { db.put(b"a1", b"new").await.map(|_| ()) })
    };
    let committed_during_pause = tokio::time::timeout(Duration::from_millis(500), &mut writer)
        .await
        .is_ok();
    assert_eq!(committed_during_pause, write_commits_during_pause);

    fail_parallel::cfg(fp_registry.clone(), fail_point, "off").unwrap();
    loader.await.unwrap().unwrap();
    if !committed_during_pause {
        writer.await.unwrap().unwrap();
    }

    let ranges = db.cached_ranges();
    assert_eq!(ranges.len(), 1);
    assert!(!ranges[0].loading);
    let hits = || lookup_metric_with_labels(&recorder, GET_COUNT, &[("result", "hit")]);
    let hits_before = hits().unwrap_or(0);
    let cached = db.get(b"a1").await.unwrap();
    assert_eq!(
        hits().unwrap_or(0),
        hits_before + 1,
        "the cache did not serve the read"
    );
    let snapshot = db.snapshot().unwrap();
    assert_eq!(
        normal_get(snapshot.as_ref(), &Bytes::from_static(b"a1"))
            .await
            .map(|kv| kv.value),
        Some(Bytes::from_static(b"new"))
    );
    assert_eq!(
        cached,
        Some(Bytes::from_static(b"new")),
        "the cache lost a write that raced the registration of its range"
    );
    db.close().await.unwrap();
}

#[cfg(feature = "compaction_filters")]
#[tokio::test]
async fn should_reject_compaction_filters() {
    use slatedb::{
        CompactionFilter, CompactionFilterError, CompactionFilterSupplier, CompactionJobContext,
        CompactorBuilder,
    };

    struct Supplier;

    #[async_trait::async_trait]
    impl CompactionFilterSupplier for Supplier {
        async fn create_compaction_filter(
            &self,
            _context: &CompactionJobContext,
        ) -> Result<Box<dyn CompactionFilter>, CompactionFilterError> {
            unimplemented!("never called")
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let compactor = CompactorBuilder::new("filters", store.clone())
        .with_compaction_filter_supplier(Arc::new(Supplier));
    let err = Db::builder("filters", store)
        .with_range_cache(options(dir.path()))
        .with_compactor_builder(compactor)
        .build()
        .await
        .err()
        .unwrap();
    assert_eq!(err.kind(), ErrorKind::Invalid);
}

/// Writes a mix of puts, deletes and merge operands in `0..n`, as round `round`.
async fn mixed_writes(db: &Db, n: usize, round: usize) {
    for i in (round % 5..n).step_by(5) {
        match (i + round) % 4 {
            0 => db.delete(key(i)).await.unwrap(),
            1 => db.merge(key(i), format!("+{round}")).await.unwrap(),
            _ => db.put(key(i), value(i, round)).await.unwrap(),
        };
    }
}

fn page_sources() -> [RangeCachePageSource; 2] {
    [
        RangeCachePageSource::Mmap,
        RangeCachePageSource::PositionalRead,
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_scan_one_state_while_writes_land() {
    for page_source in page_sources() {
        let t = open(|o| o.page_source = page_source).await;
        let db = &t.db;
        for i in 0..300 {
            db.put(key(i), value(i, 0)).await.unwrap();
        }
        db.cache_range(key(0)..key(300)).await.unwrap();
        // The file holds round 0. The delta holds rounds 1 and 2: overwrites, tombstones and
        // merge operands.
        mixed_writes(db, 300, 1).await;
        mixed_writes(db, 300, 2).await;

        let scans_before = t.metric(SCAN_COUNT, &[]);
        let snapshot = db.snapshot().unwrap();
        let mut iter = db.scan(key(10)..key(290)).await.unwrap();
        assert_eq!(t.metric(SCAN_COUNT, &[]), scans_before + 1);
        let mut got = Vec::new();
        for round in 3..8 {
            // Writes land in the scanned range while the scan runs, above its sequence number.
            mixed_writes(db, 300, round).await;
            db.put(Bytes::from(format!("key{:05}a", 20 * round)), "new")
                .await
                .unwrap();
            for _ in 0..40 {
                if let Some(kv) = iter.next().await.unwrap() {
                    got.push(kv);
                }
            }
        }
        while let Some(kv) = iter.next().await.unwrap() {
            got.push(kv);
        }
        let want = normal_scan_of(snapshot.as_ref(), (Included(key(10)), Excluded(key(290)))).await;
        assert_eq!(got, want, "page source {page_source:?}");
        // A new scan sees the new writes.
        assert_matches_snapshot(db, [], (Included(key(0)), Excluded(key(300)))).await;
        db.close().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_scan_across_a_rebuild() {
    use slatedb::fail_parallel::{self, FailPointRegistry};

    for page_source in page_sources() {
        let dir = tempfile::tempdir().unwrap();
        let fp_registry = Arc::new(FailPointRegistry::new());
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let mut opts = options(dir.path());
        opts.page_source = page_source;
        opts.tuning.page_size = 512;
        opts.tuning.min_rebuild_delta_bytes = 2048;
        let db = Db::builder("scan_rebuild", Arc::new(InMemory::new()))
            .with_range_cache(opts)
            .with_merge_operator(Arc::new(ConcatMergeOperator))
            .with_fp_registry(fp_registry.clone())
            .with_metrics_recorder(recorder.clone())
            .build()
            .await
            .unwrap();
        for i in 0..200 {
            db.put(key(i), value(i, 0)).await.unwrap();
        }
        db.cache_range(..).await.unwrap();

        // Pause the next rebuild after it froze the delta, and scan the three layers.
        fail_parallel::cfg(fp_registry.clone(), "range-cache-rebuild", "pause").unwrap();
        let mut round = 1;
        while !db.cached_ranges()[0].rebuilding {
            mixed_writes(&db, 200, round).await;
            round += 1;
        }
        let snapshot = db.snapshot().unwrap();
        let mut iter = db.scan(key(0)..key(200)).await.unwrap();
        let mut got = Vec::new();
        for _ in 0..50 {
            got.extend(iter.next().await.unwrap());
        }
        mixed_writes(&db, 200, round).await;
        fail_parallel::cfg(fp_registry.clone(), "range-cache-rebuild", "off").unwrap();
        // Let more rebuilds run and install new files under the scan.
        let rebuilds = || lookup_metric(&recorder, REBUILD_COUNT).unwrap_or(0);
        for round in round + 1..round + 20 {
            mixed_writes(&db, 200, round).await;
        }
        wait_for(|| rebuilds() >= 2 && db.cached_ranges().iter().all(|r| !r.rebuilding)).await;
        while let Some(kv) = iter.next().await.unwrap() {
            got.push(kv);
        }
        let want = normal_scan_of(snapshot.as_ref(), (Included(key(0)), Excluded(key(200)))).await;
        assert_eq!(got, want, "page source {page_source:?}");
        assert_matches_snapshot(&db, [], (Unbounded, Unbounded)).await;
        db.close().await.unwrap();
    }
}

#[tokio::test]
async fn should_seek_within_a_cached_scan() {
    for page_source in page_sources() {
        let t = open(|o| o.page_source = page_source).await;
        let db = &t.db;
        for i in (0..100).step_by(2) {
            db.put(key(i), value(i, 0)).await.unwrap();
        }
        db.cache_range(key(0)..key(100)).await.unwrap();
        db.delete(key(40)).await.unwrap();
        db.put(key(41), value(41, 1)).await.unwrap();
        db.merge(key(60), "+m").await.unwrap();

        let snapshot = db.snapshot().unwrap();
        let range = key(10)..key(90);
        let mut cached = db.scan(range.clone()).await.unwrap();
        let mut normal = snapshot
            .scan_with_options(range, &normal_scan())
            .await
            .unwrap();
        let scans = t.metric(SCAN_COUNT, &[]);
        assert!(scans > 0, "the cache did not serve the scan");
        let step = |a: Option<KeyValue>, b: Option<KeyValue>| assert_eq!(a, b);
        step(cached.next().await.unwrap(), normal.next().await.unwrap());
        // Forward seeks: to a present key, to a deleted key, to a key only in the delta, to a
        // merge operand, and between keys.
        for target in [key(20), key(40), key(59), Bytes::from("key00061x")] {
            cached.seek(&target).await.unwrap();
            normal.seek(&target).await.unwrap();
            step(cached.next().await.unwrap(), normal.next().await.unwrap());
        }
        // Errors: a key at or before the last returned key, and keys outside the range.
        for target in [key(62), key(5), key(90)] {
            let a = cached.seek(&target).await.unwrap_err();
            let b = normal.seek(&target).await.unwrap_err();
            assert_eq!(a.kind(), ErrorKind::Invalid);
            assert_eq!(a.kind(), b.kind());
        }
        // A seek past the last key ends both scans.
        cached.seek(key(89)).await.unwrap();
        normal.seek(key(89)).await.unwrap();
        step(cached.next().await.unwrap(), normal.next().await.unwrap());
        assert!(cached.next().await.unwrap().is_none());
        db.close().await.unwrap();
    }
}

// Reads in snapshots and transactions.

/// The reads that the range cache served: point reads of each result, and scans.
fn served(t: &TestDb) -> i64 {
    ["hit", "absent", "merge"]
        .iter()
        .map(|r| t.metric(GET_COUNT, &[("result", r)]))
        .sum::<i64>()
        + t.metric(SCAN_COUNT, &[])
}

/// Asserts that the default reads of `keys` and a scan of `range` in `reader` agree with the
/// normal read path in `reader`, and returns the reads that the cache served for them.
async fn assert_reader_matches_normal<R: DbReadOps + Sync>(
    t: &TestDb,
    reader: &R,
    keys: impl IntoIterator<Item = Bytes>,
    range: (Bound<Bytes>, Bound<Bytes>),
) -> i64 {
    let before = served(t);
    let mut normal_reads = Vec::new();
    for k in keys {
        let got = reader.get_key_value(&k).await.unwrap();
        let served_now = served(t);
        let want = normal_get(reader, &k).await;
        normal_reads.push(served(t) - served_now);
        assert_eq!(got, want, "key {k:?}");
    }
    let got = collect(reader.scan(range.clone()).await.unwrap()).await;
    let served_now = served(t);
    let want = normal_scan_of(reader, range).await;
    normal_reads.push(served(t) - served_now);
    assert_eq!(got, want);
    assert!(
        normal_reads.iter().all(|n| *n == 0),
        "the cache served a read with a tracing option"
    );
    served(t) - before
}

#[tokio::test]
async fn should_not_serve_traced_reads() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in 0..100 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(..).await.unwrap();
    let snapshot = db.snapshot().unwrap();
    let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    let range = (Included(key(10)), Excluded(key(90)));
    let before = served(&t);
    normal_get(db, &key(5)).await;
    normal_get(snapshot.as_ref(), &key(5)).await;
    normal_get(&txn, &key(5)).await;
    normal_scan_of(db, range.clone()).await;
    normal_scan_of(snapshot.as_ref(), range.clone()).await;
    normal_scan_of(&txn, range.clone()).await;
    assert_eq!(
        served(&t),
        before,
        "the oracle of these tests uses the cache"
    );
    // The same reads without the tracing option are served.
    db.get(key(5)).await.unwrap();
    snapshot.get(key(5)).await.unwrap();
    txn.get(key(5)).await.unwrap();
    collect(db.scan(range.clone()).await.unwrap()).await;
    collect(snapshot.scan(range.clone()).await.unwrap()).await;
    collect(txn.scan(range).await.unwrap()).await;
    assert_eq!(served(&t), before + 6);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_serve_snapshot_reads_from_the_cache() {
    for page_source in page_sources() {
        let t = open(|o| o.page_source = page_source).await;
        let db = &t.db;
        for i in 0..100 {
            db.put(key(i), value(i, 0)).await.unwrap();
        }
        for i in (0..100).step_by(9) {
            db.merge(key(i), "+m").await.unwrap();
        }
        flush_memtable(db).await;
        db.cache_range(key(10)..key(90)).await.unwrap();
        let mut snapshots = vec![db.snapshot().unwrap()];
        // Puts, deletes and merge operands after each snapshot, in and out of the range.
        for round in 1..5 {
            mixed_writes(db, 100, round).await;
            db.put(Bytes::from(format!("key{:05}a", 10 * round)), "new")
                .await
                .unwrap();
            snapshots.push(db.snapshot().unwrap());
        }
        let keys: Vec<Bytes> = (0..100).map(key).collect();
        for (i, snapshot) in snapshots.iter().enumerate() {
            let merges = t.metric(GET_COUNT, &[("result", "merge")]);
            let covered = (Included(key(10)), Excluded(key(90)));
            let n =
                assert_reader_matches_normal(&t, snapshot.as_ref(), keys.clone(), covered).await;
            // 80 point reads in the range, and the scan.
            assert_eq!(n, 81, "page source {page_source:?}");
            // The file holds resolved values. Later snapshots read merge operands in the delta.
            if i > 0 {
                assert!(t.metric(GET_COUNT, &[("result", "merge")]) > merges);
            }
            // A scan that leaves the range uses the normal path.
            let crossing = (Included(key(80)), Excluded(key(95)));
            assert_eq!(
                assert_reader_matches_normal(&t, snapshot.as_ref(), [], crossing).await,
                0
            );
        }
        // The oldest snapshot still reads the state before all writes.
        assert_eq!(snapshots[0].get(key(20)).await.unwrap(), Some(value(20, 0)));
        db.close().await.unwrap();
    }
}

#[tokio::test]
async fn should_fall_back_for_a_snapshot_older_than_the_load() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in 0..100 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    let old = db.snapshot().unwrap();
    mixed_writes(db, 100, 1).await;
    db.cache_range(key(0)..key(100)).await.unwrap();
    let keys: Vec<Bytes> = (0..100).map(key).collect();
    let all = (Included(key(0)), Excluded(key(100)));
    // The file holds a newer state than the snapshot, so the snapshot uses the normal path.
    assert_eq!(
        assert_reader_matches_normal(&t, old.as_ref(), keys.clone(), all.clone()).await,
        0
    );
    assert_eq!(old.get(key(1)).await.unwrap(), Some(value(1, 0)));
    // A snapshot after the load uses the cache.
    let new = db.snapshot().unwrap();
    assert_eq!(
        assert_reader_matches_normal(&t, new.as_ref(), keys, all).await,
        101
    );
    db.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_fall_back_for_a_snapshot_older_than_a_rebuild() {
    let t = open(|o| {
        o.tuning.min_rebuild_delta_bytes = 2048;
        o.tuning.page_size = 512;
    })
    .await;
    let db = &t.db;
    for i in 0..200 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(..).await.unwrap();
    mixed_writes(db, 200, 1).await;
    let old = db.snapshot().unwrap();
    let keys: Vec<Bytes> = (0..200).map(key).collect();
    let all = (Included(key(0)), Excluded(key(200)));
    // Before the rebuild, the cache serves the snapshot.
    assert_eq!(
        assert_reader_matches_normal(&t, old.as_ref(), keys.clone(), all.clone()).await,
        201
    );
    // An iterator opened before the rebuild keeps its layers.
    let mut open_iter = old.scan(all.clone()).await.unwrap();
    let first = open_iter.next().await.unwrap();
    let mut round = 2;
    while t.metric(REBUILD_COUNT, &[]) == 0 {
        mixed_writes(db, 200, round).await;
        round += 1;
        wait_for(|| db.cached_ranges().iter().all(|r| !r.rebuilding)).await;
    }
    // The new file holds a state after the snapshot, so the snapshot uses the normal path.
    assert_eq!(
        assert_reader_matches_normal(&t, old.as_ref(), keys.clone(), all.clone()).await,
        0
    );
    let mut rest: Vec<KeyValue> = first.into_iter().collect();
    rest.extend(collect(open_iter).await);
    assert_eq!(rest, normal_scan_of(old.as_ref(), all.clone()).await);
    // A snapshot after the rebuild uses the cache.
    let new = db.snapshot().unwrap();
    assert_eq!(
        assert_reader_matches_normal(&t, new.as_ref(), keys, all).await,
        201
    );
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_read_a_snapshot_while_its_range_is_removed() {
    // The budget holds two ranges of 50 rows. See `should_evict_the_least_recently_read_range`.
    let t = open(|o| {
        o.tuning.page_size = 4096;
        o.max_disk_bytes = 9000;
    })
    .await;
    let db = &t.db;
    for i in 0..300 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(0)..key(50)).await.unwrap();
    db.cache_range(key(100)..key(150)).await.unwrap();
    mixed_writes(db, 300, 1).await;
    let snapshot = db.snapshot().unwrap();
    mixed_writes(db, 300, 2).await;
    let first = (Included(key(0)), Excluded(key(50)));
    let second = (Included(key(100)), Excluded(key(150)));
    let mut first_iter = snapshot.scan(first.clone()).await.unwrap();
    let mut second_iter = snapshot.scan(second.clone()).await.unwrap();
    let mut first_rows = vec![first_iter.next().await.unwrap().unwrap()];
    let mut second_rows = vec![second_iter.next().await.unwrap().unwrap()];
    let scans = t.metric(SCAN_COUNT, &[]);

    // An explicit removal of the first range, and an eviction of the second one.
    db.uncache_range(key(0)..key(50)).unwrap();
    db.get(key(120)).await.unwrap();
    db.cache_range(key(200)..key(250)).await.unwrap();
    db.cache_range(key(250)..key(300)).await.unwrap();
    assert!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]) >= 1);
    assert!(db
        .cached_ranges()
        .iter()
        .all(|r| r.start != key(0) && r.start != key(100)));
    mixed_writes(db, 300, 3).await;

    // The open iterators finish on their pinned layers, and new reads use the normal path.
    first_rows.extend(collect(first_iter).await);
    second_rows.extend(collect(second_iter).await);
    assert_eq!(
        first_rows,
        normal_scan_of(snapshot.as_ref(), first.clone()).await
    );
    assert_eq!(
        second_rows,
        normal_scan_of(snapshot.as_ref(), second.clone()).await
    );
    assert_eq!(t.metric(SCAN_COUNT, &[]), scans);
    for range in [first, second] {
        let keys: Vec<Bytes> = (0..300).map(key).collect();
        assert_eq!(
            assert_reader_matches_normal(&t, snapshot.as_ref(), keys, range).await,
            0
        );
    }
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_read_own_writes_in_a_transaction() {
    for isolation in [
        IsolationLevel::Snapshot,
        IsolationLevel::SerializableSnapshot,
    ] {
        let t = open(|_| {}).await;
        let db = &t.db;
        for i in 0..100 {
            db.put(key(i), value(i, 0)).await.unwrap();
        }
        db.merge(key(30), "+m").await.unwrap();
        db.cache_range(key(10)..key(90)).await.unwrap();
        let txn = db.begin(isolation).unwrap();
        // Writes by others after the transaction started stay invisible to it. They touch no
        // key that the transaction writes, so that it can commit.
        for i in (51..100).step_by(3) {
            match i % 4 {
                0 => db.delete(key(i)).await.unwrap(),
                1 => db.merge(key(i), "+other").await.unwrap(),
                _ => db.put(key(i), value(i, 1)).await.unwrap(),
            };
        }
        // The transaction's own writes: a put, a delete, merge operands over a value, over a
        // put in the batch, over a merge operand in the database and over no value.
        txn.put(key(20), "own").unwrap();
        txn.delete(key(21)).unwrap();
        txn.merge(key(22), "+own").unwrap();
        txn.put(key(23), "own").unwrap();
        txn.merge(key(23), "+own").unwrap();
        txn.merge(key(30), "+own").unwrap();
        txn.merge(Bytes::from("key00040a"), "+own").unwrap();
        txn.delete(key(41)).unwrap();
        txn.put(key(41), "again").unwrap();
        // A write outside the cached range.
        txn.put(key(95), "own").unwrap();

        // The written keys: the batch answers puts and deletes, merges take the normal path.
        let written = [20, 21, 22, 23, 30, 41, 95]
            .map(key)
            .into_iter()
            .chain([Bytes::from("key00040a")]);
        let before = served(&t);
        for k in written {
            assert_eq!(
                txn.get_key_value(&k).await.unwrap(),
                normal_get(&txn, &k).await,
                "key {k:?}"
            );
        }
        assert_eq!(served(&t), before, "{isolation:?}");
        assert_eq!(txn.get(key(20)).await.unwrap(), Some(Bytes::from("own")));
        assert_eq!(txn.get(key(21)).await.unwrap(), None);
        assert_eq!(
            txn.get(key(22)).await.unwrap(),
            Some(Bytes::from("value-22-0+own"))
        );
        assert_eq!(
            txn.get(key(23)).await.unwrap(),
            Some(Bytes::from("own+own"))
        );
        assert_eq!(
            txn.get(key(30)).await.unwrap(),
            Some(Bytes::from("value-30-0+m+own"))
        );

        // Other keys come from the cache at the transaction's sequence number.
        let others: Vec<Bytes> = (50..90).map(key).collect();
        assert_eq!(
            assert_reader_matches_normal(&t, &txn, others, (Included(key(50)), Excluded(key(90))))
                .await,
            41,
            "{isolation:?}"
        );
        // A scan over the transaction's writes uses the normal path, and sees them.
        let with_writes = (Included(key(10)), Excluded(key(50)));
        assert_eq!(
            assert_reader_matches_normal(&t, &txn, [], with_writes.clone()).await,
            0
        );
        let rows = collect(txn.scan(with_writes).await.unwrap()).await;
        assert!(rows.iter().any(|kv| kv.key == key(20) && kv.value == "own"));
        assert!(rows.iter().all(|kv| kv.key != key(21)));
        // A write after a cached scan opened does not change that scan, as in the normal
        // path, where the scan copies the batch entries of its range when it opens.
        let range = (Included(key(60)), Excluded(key(70)));
        let iter = txn.scan(range.clone()).await.unwrap();
        let want = normal_scan_of(&txn, range).await;
        txn.put(key(65), "late").unwrap();
        assert_eq!(collect(iter).await, want);
        assert_eq!(txn.get(key(65)).await.unwrap(), Some(Bytes::from("late")));
        if isolation == IsolationLevel::SerializableSnapshot {
            // The transaction read keys that others wrote after it started, through the cache.
            let err = txn.commit().await.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::Transaction);
        } else {
            // The commit still applies every write.
            txn.commit().await.unwrap();
            assert_eq!(db.get(key(23)).await.unwrap(), Some(Bytes::from("own+own")));
        }
        assert_matches_snapshot(db, (0..100).map(key), (Unbounded, Unbounded)).await;
        db.close().await.unwrap();
    }
}

#[tokio::test]
async fn should_detect_conflicts_on_cached_reads() {
    let t = open(|_| {}).await;
    let db = &t.db;
    for i in 0..100 {
        db.put(key(i), value(i, 0)).await.unwrap();
    }
    db.cache_range(key(10)..key(90)).await.unwrap();

    // A point read that the cache serves, then a write to that key by another writer.
    let hits = t.metric(GET_COUNT, &[("result", "hit")]);
    let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    assert_eq!(txn.get(key(20)).await.unwrap(), Some(value(20, 0)));
    assert_eq!(t.metric(GET_COUNT, &[("result", "hit")]), hits + 1);
    txn.put(key(5), "x").unwrap();
    db.put(key(20), value(20, 1)).await.unwrap();
    let err = txn.commit().await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Transaction);

    // A scan that the cache serves, then a new key inside the scanned range. The scan never
    // returned that key, but the range counts as read.
    let scans = t.metric(SCAN_COUNT, &[]);
    let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    let rows = collect(txn.scan(key(30)..key(40)).await.unwrap()).await;
    assert_eq!(rows.len(), 10);
    assert_eq!(t.metric(SCAN_COUNT, &[]), scans + 1);
    txn.put(key(5), "x").unwrap();
    db.put(Bytes::from("key00035a"), "phantom").await.unwrap();
    let err = txn.commit().await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Transaction);

    // Without a conflicting write, both commit.
    let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    txn.get(key(20)).await.unwrap();
    collect(txn.scan(key(30)..key(40)).await.unwrap()).await;
    txn.put(key(5), "y").unwrap();
    db.put(key(50), value(50, 1)).await.unwrap();
    txn.commit().await.unwrap();
    // Snapshot isolation does not track reads, cached or not.
    let txn = db.begin(IsolationLevel::Snapshot).unwrap();
    txn.get(key(20)).await.unwrap();
    txn.put(key(5), "z").unwrap();
    db.put(key(20), value(20, 2)).await.unwrap();
    txn.commit().await.unwrap();
    db.close().await.unwrap();
}

/// Automatic promotion samples reads in transactions and snapshots, which the cache could
/// serve, as it samples other reads.
#[tokio::test]
async fn should_promote_ranges_that_transactions_read() {
    let (t, clock) = open_auto(|_| {}).await;
    let db = &t.db;
    load_padded(db, 2000).await;
    for _ in 0..40 {
        let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
        for i in 500..510 {
            txn.get(key(i)).await.unwrap();
        }
        collect(txn.scan(key(510)..key(520)).await.unwrap()).await;
        txn.rollback();
        let snapshot = db.snapshot().unwrap();
        for i in 500..510 {
            snapshot.get(key(i)).await.unwrap();
        }
        tick(&clock, 200).await;
        if holding(db, 500, 519).is_some() {
            break;
        }
    }
    assert!(holding(db, 500, 519).is_some(), "{:?}", db.cached_ranges());
    assert!(t.metric(SAMPLE_COUNT, &[("result", "recorded")]) >= 40);
    // Now the cache serves them.
    let hits = t.metric(GET_COUNT, &[("result", "hit")]);
    let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    for i in 500..520 {
        txn.get(key(i)).await.unwrap();
    }
    assert_eq!(t.metric(GET_COUNT, &[("result", "hit")]), hits + 20);
    db.close().await.unwrap();
}

/// A merge operand that a writer applies while a load registers its range must count once.
/// The load's snapshot holds the operand, and the writer then finds the new range. Before the
/// fix, the delta held the operand too, and reads applied it twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn should_apply_a_merge_that_races_registration_once() {
    use slatedb::fail_parallel::{self, FailPointRegistry};

    for fail_point in [
        "range-cache-register-publish",
        "range-cache-register-snapshot",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let fp_registry = Arc::new(FailPointRegistry::new());
        let db = Db::builder("merge_race", Arc::new(InMemory::new()))
            .with_range_cache(options(dir.path()))
            .with_merge_operator(Arc::new(ConcatMergeOperator))
            .with_fp_registry(fp_registry.clone())
            .build()
            .await
            .unwrap();
        db.put(b"a1", b"base").await.unwrap();
        fail_parallel::cfg(fp_registry.clone(), fail_point, "pause").unwrap();
        let loader = {
            let db = db.clone();
            tokio::spawn(async move { db.cache_range(&b"a"[..]..&b"b"[..]).await })
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        let writer = {
            let db = db.clone();
            tokio::spawn(async move { db.merge(b"a1", b"+op").await.map(|_| ()) })
        };
        tokio::time::sleep(Duration::from_millis(300)).await;
        fail_parallel::cfg(fp_registry.clone(), fail_point, "off").unwrap();
        loader.await.unwrap().unwrap();
        writer.await.unwrap().unwrap();
        let want = Some(Bytes::from_static(b"base+op"));
        let snapshot = db.snapshot().unwrap();
        let k = Bytes::from_static(b"a1");
        assert_eq!(
            normal_get(snapshot.as_ref(), &k).await.map(|kv| kv.value),
            want
        );
        assert_eq!(db.get(b"a1").await.unwrap(), want, "{fail_point}");
        assert_eq!(snapshot.get(b"a1").await.unwrap(), want, "{fail_point}");
        let rows = collect(db.scan(&b"a"[..]..&b"b"[..]).await.unwrap()).await;
        assert_eq!(
            rows.first().map(|kv| kv.value.clone()),
            want,
            "{fail_point}"
        );
        db.close().await.unwrap();
    }
}

/// One step of the model test.
#[derive(Debug, Clone)]
enum Op {
    Put(u8, u8),
    Delete(u8),
    Merge(u8, u8),
    Batch(Vec<(u8, Option<u8>)>),
    Flush,
    Cache(u8, u8),
    Uncache(u8, u8),
    Get(u8),
    Scan(u8, u8),
}

fn op() -> impl Strategy<Value = Op> {
    let k = 0u8..32;
    let range = (k.clone(), k.clone()).prop_map(|(a, b)| (a.min(b), a.max(b) + 1));
    prop_oneof![
        6 => (k.clone(), any::<u8>()).prop_map(|(k, v)| Op::Put(k, v)),
        2 => k.clone().prop_map(Op::Delete),
        2 => (k.clone(), any::<u8>()).prop_map(|(k, v)| Op::Merge(k, v)),
        1 => proptest::collection::vec((k.clone(), proptest::option::of(any::<u8>())), 1..4)
            .prop_map(Op::Batch),
        1 => Just(Op::Flush),
        2 => range.clone().prop_map(|(a, b)| Op::Cache(a, b)),
        1 => range.clone().prop_map(|(a, b)| Op::Uncache(a, b)),
        4 => k.prop_map(Op::Get),
        2 => range.prop_map(|(a, b)| Op::Scan(a, b)),
    ]
}

fn model_key(k: u8) -> Bytes {
    Bytes::from(vec![b'k', k])
}

async fn run_model(ops: Vec<Op>) {
    let t = open(|o| {
        o.tuning.min_rebuild_delta_bytes = 64;
        o.tuning.page_size = 512;
        o.max_disk_bytes = 4096;
    })
    .await;
    let db = &t.db;
    let mut model: BTreeMap<Bytes, Bytes> = BTreeMap::new();
    for op in ops {
        match op {
            Op::Put(k, v) => {
                db.put(model_key(k), [v]).await.unwrap();
                model.insert(model_key(k), Bytes::from(vec![v]));
            }
            Op::Delete(k) => {
                db.delete(model_key(k)).await.unwrap();
                model.remove(&model_key(k));
            }
            Op::Merge(k, v) => {
                db.merge(model_key(k), [v]).await.unwrap();
                let mut merged = model
                    .get(&model_key(k))
                    .map(|v| v.to_vec())
                    .unwrap_or_default();
                merged.push(v);
                model.insert(model_key(k), Bytes::from(merged));
            }
            Op::Batch(writes) => {
                let mut batch = WriteBatch::new();
                for (k, v) in writes {
                    match v {
                        Some(v) => {
                            batch.put(model_key(k), [v]);
                            model.insert(model_key(k), Bytes::from(vec![v]));
                        }
                        None => {
                            batch.delete(model_key(k));
                            model.remove(&model_key(k));
                        }
                    }
                }
                db.write(batch).await.unwrap();
            }
            Op::Flush => flush_memtable(db).await,
            Op::Cache(a, b) => {
                // A load is rejected when it does not fit the budget.
                let _ = db.cache_range(model_key(a)..model_key(b)).await;
            }
            Op::Uncache(a, b) => db.uncache_range(model_key(a)..model_key(b)).unwrap(),
            Op::Get(k) => {
                assert_eq!(
                    db.get(model_key(k)).await.unwrap(),
                    model.get(&model_key(k)).cloned(),
                    "get {k}"
                );
            }
            Op::Scan(a, b) => {
                let got: Vec<_> = collect(db.scan(model_key(a)..model_key(b)).await.unwrap())
                    .await
                    .into_iter()
                    .map(|kv| (kv.key, kv.value))
                    .collect();
                let want: Vec<_> = model
                    .range(model_key(a)..model_key(b))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                assert_eq!(got, want, "scan {a}..{b}");
            }
        }
    }
    db.close().await.unwrap();
}

#[test]
fn should_match_a_model() {
    let runtime = Runtime::new().unwrap();
    let mut runner = TestRunner::new(Config {
        cases: 48,
        ..Config::default()
    });
    runner
        .run(&proptest::collection::vec(op(), 1..120), |ops| {
            runtime.block_on(run_model(ops));
            Ok(())
        })
        .unwrap();
}

// Automatic promotion (stage 2).
//
// These tests run the database on a mock clock. Every read is sampled, and every sampled read
// costs `min_read_cost` (1 ms), since the mock clock does not move during a read. So the decisions
// depend on the keys read and on the clock, which the tests move. The promoter runs as a
// background task, so the tests poll for its effects.

/// Options that make promotion react in test time.
fn set_auto_options(opts: &mut RangeCacheOptions) {
    let mut auto = AutoPromoteOptions::default();
    auto.sample_one_in = 1;
    auto.target_range_bytes = 16 * 1024;
    auto.heat_half_life = Duration::from_secs(1);
    opts.auto_promote = Some(auto);
    let tuning = &mut opts.tuning;
    tuning.evaluate_interval = Duration::from_millis(100);
    tuning.max_candidates_per_evaluation = 4;
    tuning.min_evidence = 4;
    tuning.full_evidence = 16;
    tuning.min_reuse_span = Duration::from_millis(500);
    tuning.admission_margin = 2.0;
    tuning.ghost_ttl = Duration::from_secs(10);
    tuning.candidate_ttl = Duration::from_secs(10);
    tuning.candidate_recheck = Duration::from_millis(100);
    tuning.backoff_base = Duration::from_secs(5);
    tuning.reservoir_capacity = 32 * 64;
    tuning.queue_capacity = 100_000;
    tuning.min_read_cost = Duration::from_millis(1);
    tuning.rebuild_cost_ns_per_byte = 5.0;
}

async fn open_auto(tune: impl FnOnce(&mut RangeCacheOptions)) -> (TestDb, Arc<MockSystemClock>) {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(dir.path());
    set_auto_options(&mut opts);
    tune(&mut opts);
    let clock = Arc::new(MockSystemClock::new());
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let settings = Settings {
        compactor_options: None,
        ..Settings::default()
    };
    let db = Db::builder("range_cache_auto_test", Arc::new(InMemory::new()))
        .with_settings(settings)
        .with_range_cache(opts)
        .with_merge_operator(Arc::new(ConcatMergeOperator))
        .with_metrics_recorder(recorder.clone())
        .with_system_clock(clock.clone())
        .build()
        .await
        .unwrap();
    (TestDb { db, recorder, dir }, clock)
}

/// A 100-byte value, so that 16 KiB holds about 120 rows.
fn padded(i: usize, round: usize) -> Bytes {
    Bytes::from(format!("{round:04}-{i:0>95}"))
}

async fn load_padded(db: &Db, n: usize) {
    let mut batch = WriteBatch::new();
    for i in 0..n {
        batch.put(key(i), padded(i, 0));
        if i % 100 == 99 || i + 1 == n {
            db.write(std::mem::take(&mut batch)).await.unwrap();
        }
    }
}

async fn read_keys(db: &Db, keys: impl IntoIterator<Item = usize>) {
    for i in keys {
        db.get(key(i)).await.unwrap();
    }
}

/// Moves the mock clock and lets the promoter catch up.
async fn tick(clock: &MockSystemClock, ms: u64) {
    clock.advance(Duration::from_millis(ms)).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
}

/// The ready range that holds `first..=last`, if any.
fn holding(db: &Db, first: usize, last: usize) -> Option<CachedRangeInfo> {
    db.cached_ranges().into_iter().find(|r| {
        !r.loading && r.start <= key(first) && r.end.as_ref().is_none_or(|end| *end > key(last))
    })
}

#[tokio::test]
async fn should_be_inert_when_auto_promotion_is_off() {
    assert!(RangeCacheOptions::new("unused").auto_promote.is_none());
    let t = open(|_| {}).await;
    let db = &t.db;
    load_padded(db, 500).await;
    for _ in 0..20 {
        read_keys(db, 100..120).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(db.cached_ranges().is_empty());
    let snapshot = t.recorder.snapshot();
    assert!(
        !snapshot
            .all()
            .iter()
            .any(|m| m.name == SAMPLE_COUNT || m.name == PROMOTION_COUNT),
        "no automatic promotion metrics without automatic promotion"
    );
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_promote_a_repeatedly_read_range() {
    let (t, clock) = open_auto(|_| {}).await;
    let db = &t.db;
    load_padded(db, 2000).await;
    for _ in 0..40 {
        read_keys(db, 500..520).await;
        tick(&clock, 200).await;
        if holding(db, 500, 519).is_some() {
            break;
        }
    }
    let range = holding(db, 500, 519).expect("the hot keys are cached");
    assert!(range.automatic);
    // About the target size: 16 KiB of SST data, a little more as a file.
    assert!(range.file_rows > 50 && range.file_rows < 400, "{range:?}");
    assert_eq!(t.metric(PROMOTION_COUNT, &[]), 1);
    assert_eq!(t.metric(ADMISSION_COUNT, &[("result", "free")]), 1);
    assert!(t.metric(CANDIDATE_COUNT, &[("source", "scan")]) >= 1);
    assert!(t.metric(SAMPLE_COUNT, &[("result", "recorded")]) >= 40);

    // The cache now serves the hot keys, and agrees with the normal path.
    let hits = t.metric(GET_COUNT, &[("result", "hit")]);
    read_keys(db, 500..520).await;
    assert_eq!(t.metric(GET_COUNT, &[("result", "hit")]), hits + 20);
    db.put(key(505), padded(505, 1)).await.unwrap();
    db.delete(key(506)).await.unwrap();
    assert_matches_snapshot(
        db,
        (480..540).map(key),
        (Included(key(400)), Excluded(key(600))),
    )
    .await;
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_not_promote_a_single_pass() {
    // The budget holds about three automatic ranges.
    let (t, clock) = open_auto(|o| o.max_disk_bytes = 64 * 1024).await;
    let db = &t.db;
    load_padded(db, 2000).await;
    // One pass over all keys in order, over 2 s: a candidate's keys are read within about
    // 120 ms of each other, which is less than the reuse span.
    for i in 0..2000 {
        db.get(key(i)).await.unwrap();
        if i % 10 == 9 {
            tick(&clock, 10).await;
        }
    }
    for _ in 0..20 {
        tick(&clock, 200).await;
    }
    assert!(t.metric(REJECTION_COUNT, &[("reason", "not_reused")]) > 0);
    assert_eq!(t.metric(PROMOTION_COUNT, &[]), 0);
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]), 0);
    assert!(db.cached_ranges().is_empty());
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_follow_a_moving_hot_set() {
    // The budget holds one automatic range, not two.
    let (t, clock) = open_auto(|o| o.max_disk_bytes = 28 * 1024).await;
    let db = &t.db;
    load_padded(db, 2000).await;
    for _ in 0..40 {
        read_keys(db, 100..120).await;
        tick(&clock, 200).await;
        if holding(db, 100, 119).is_some() {
            break;
        }
    }
    assert!(holding(db, 100, 119).is_some(), "{:?}", db.cached_ranges());

    // The hot set moves. The first range cools down, and the second one displaces it.
    for _ in 0..60 {
        read_keys(db, 1500..1520).await;
        tick(&clock, 200).await;
        if holding(db, 1500, 1519).is_some() {
            break;
        }
    }
    assert!(
        holding(db, 1500, 1519).is_some(),
        "{:?}",
        db.cached_ranges()
    );
    assert!(holding(db, 100, 119).is_none());
    assert_eq!(t.metric(ADMISSION_COUNT, &[("result", "evicting")]), 1);
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]), 1);
    assert_eq!(t.metric(PROMOTION_COUNT, &[]), 2);
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_remove_ranges_that_writes_make_too_costly() {
    let (t, clock) = open_auto(|o| o.tuning.min_rebuild_delta_bytes = 1024).await;
    let db = &t.db;
    load_padded(db, 2000).await;
    for _ in 0..40 {
        read_keys(db, 100..120).await;
        read_keys(db, 1500..1520).await;
        tick(&clock, 200).await;
        if holding(db, 100, 119).is_some() && holding(db, 1500, 1519).is_some() {
            break;
        }
    }
    assert!(holding(db, 100, 119).is_some(), "{:?}", db.cached_ranges());
    assert!(
        holding(db, 1500, 1519).is_some(),
        "{:?}",
        db.cached_ranges()
    );

    // Both ranges take many writes. Only the second one is still read.
    for round in 1..=20 {
        for i in 100..130 {
            db.put(key(i), padded(i, round)).await.unwrap();
        }
        for i in 1500..1530 {
            db.put(key(i), padded(i, round)).await.unwrap();
        }
        read_keys(db, 1500..1540).await;
        tick(&clock, 200).await;
    }
    // The read-cold range left, for its own reason, and stays out while the writes go on.
    assert!(holding(db, 100, 119).is_none(), "{:?}", db.cached_ranges());
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "not_worth_it")]), 1);
    // The read-hot range stays and rebuilds, although its delta is as large.
    assert!(
        holding(db, 1500, 1519).is_some(),
        "{:?}",
        db.cached_ranges()
    );
    assert!(t.metric(REBUILD_COUNT, &[]) >= 2);
    assert_eq!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]), 0);
    assert_matches_snapshot(
        db,
        (90..140).chain(1490..1540).map(key),
        (Included(key(1400)), Excluded(key(1600))),
    )
    .await;
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_settle_when_the_hot_set_is_larger_than_the_budget() {
    // Four equally hot sets, and a budget for about two automatic ranges.
    let (t, clock) = open_auto(|o| o.max_disk_bytes = 56 * 1024).await;
    let db = &t.db;
    load_padded(db, 2000).await;
    let hot = [100, 600, 1100, 1600];
    let mut promotions_at_half = 0;
    let mut evictions_at_half = 0;
    for step in 0..80 {
        for start in hot {
            read_keys(db, start..start + 20).await;
        }
        tick(&clock, 200).await;
        if step == 39 {
            promotions_at_half = t.metric(PROMOTION_COUNT, &[]);
            evictions_at_half = t.metric(REMOVE_COUNT, &[("reason", "evicted")]);
        }
    }
    let cached = hot
        .iter()
        .filter(|s| holding(db, **s, **s + 19).is_some())
        .count();
    assert!(cached >= 1, "{:?}", db.cached_ranges());
    assert!(promotions_at_half >= 1);
    // No churn in the second half: candidates are no denser than what they would displace.
    assert!(t.metric(PROMOTION_COUNT, &[]) - promotions_at_half <= 1);
    assert!(t.metric(REMOVE_COUNT, &[("reason", "evicted")]) - evictions_at_half <= 1);
    assert!(t.metric(REJECTION_COUNT, &[("reason", "not_denser")]) > 0);
    db.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn should_take_candidate_bounds_from_the_sst_index() {
    // The real clock, and a compactor that moves every L0 SST into a sorted run.
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(dir.path());
    set_auto_options(&mut opts);
    opts.tuning.min_read_cost = Duration::from_micros(1);
    opts.tuning.evaluate_interval = Duration::from_millis(50);
    opts.tuning.min_reuse_span = Duration::from_millis(200);
    let settings = Settings {
        l0_sst_size_bytes: 64 * 1024,
        compactor_options: Some(CompactorOptions {
            poll_interval: Duration::from_millis(5),
            scheduler_options: SizeTieredCompactionSchedulerOptions {
                min_compaction_sources: 1,
                ..Default::default()
            }
            .into(),
            worker: Some(CompactionWorkerOptions {
                compactions_poll_interval: Duration::from_millis(5),
                ..Default::default()
            }),
            ..CompactorOptions::default()
        }),
        ..Settings::default()
    };
    let recorder = Arc::new(DefaultMetricsRecorder::new());
    let db = Db::builder("range_cache_auto_sst", Arc::new(InMemory::new()))
        .with_settings(settings)
        .with_range_cache(opts)
        .with_metrics_recorder(recorder.clone())
        .build()
        .await
        .unwrap();
    let t = TestDb { db, recorder, dir };
    let db = &t.db;
    load_padded(db, 4000).await;
    flush_memtable(db).await;
    wait_for(|| {
        let manifest = db.manifest();
        manifest.l0().is_empty() && !manifest.compacted().is_empty()
    })
    .await;
    for _ in 0..200 {
        read_keys(db, 2000..2020).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        if holding(db, 2000, 2019).is_some() {
            break;
        }
    }
    let range = holding(db, 2000, 2019).expect("the hot keys are cached");
    assert!(t.metric(CANDIDATE_COUNT, &[("source", "sst_index")]) >= 1);
    assert_eq!(t.metric(CANDIDATE_COUNT, &[("source", "scan")]), 0);
    assert!(range.file_rows > 50 && range.file_rows < 600, "{range:?}");
    assert_matches_snapshot(
        db,
        (1990..2030).map(key),
        (Included(key(1900)), Excluded(key(2100))),
    )
    .await;
    db.close().await.unwrap();
}

#[tokio::test]
async fn should_stop_promoting_when_fenced() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(dir.path());
    set_auto_options(&mut opts);
    opts.tuning.min_read_cost = Duration::from_micros(1);
    opts.tuning.evaluate_interval = Duration::from_millis(20);
    opts.tuning.min_reuse_span = Duration::from_millis(50);
    let db = Db::builder("range_cache_test", store.clone())
        .with_range_cache(opts)
        .build()
        .await
        .unwrap();
    load_padded(&db, 1000).await;
    for _ in 0..200 {
        read_keys(&db, 300..320).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        if holding(&db, 300, 319).is_some() {
            break;
        }
    }
    assert!(holding(&db, 300, 319).is_some());
    let other = Db::open("range_cache_test", store).await.unwrap();
    let err = match db.put(key(2), padded(2, 1)).await {
        Ok(_) => db.flush().await.unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::Closed(slatedb::CloseReason::Fenced));
    wait_for(|| db.cached_ranges().is_empty()).await;
    assert_eq!(cache_files(dir.path()), 0);
    assert!(db.get(key(300)).await.is_err());
    other.close().await.unwrap();
}
