//! Read-write conflicts of transactions whose read set stays in the transaction.
//!
//! A Serializable transaction records the keys that it reads, and `mark_read` records keys at
//! any isolation level. The commit checks these reads against later commits. These tests check
//! that the conflicts are the same as when the transaction manager held the read set.

#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytes::Bytes;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use slatedb::object_store::memory::InMemory;
use slatedb::{Db, ErrorKind, IsolationLevel};

async fn open() -> Db {
    Db::builder("db", Arc::new(InMemory::new()))
        .build()
        .await
        .unwrap()
}

fn key(i: usize) -> Vec<u8> {
    format!("k{i:03}").into_bytes()
}

fn is_conflict(result: &Result<Option<slatedb::WriteHandle>, slatedb::Error>) -> bool {
    matches!(result, Err(err) if err.kind() == ErrorKind::Transaction)
}

/// A read made right before the commit counts: the key changed after the transaction began.
#[tokio::test]
async fn should_conflict_on_read_made_just_before_commit() {
    for read_only in [false, true] {
        let db = open().await;
        db.put(b"k", b"0").await.unwrap();
        let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
        db.put(b"k", b"1").await.unwrap();
        // The snapshot still reads the old value.
        assert_eq!(txn.get(b"k").await.unwrap(), Some(Bytes::from_static(b"0")));
        if !read_only {
            txn.put(b"other", b"x").unwrap();
        }
        assert!(
            is_conflict(&txn.commit().await),
            "read_only={read_only}: the read of k must conflict"
        );
    }
}

/// `mark_read` makes a key conflict at Snapshot isolation; without it, there is no conflict.
#[tokio::test]
async fn should_conflict_on_marked_read_at_snapshot_isolation() {
    for marked in [false, true] {
        let db = open().await;
        let txn = db.begin(IsolationLevel::Snapshot).unwrap();
        let _ = txn.get(b"k").await.unwrap();
        if marked {
            txn.mark_read([b"k"]).unwrap();
        }
        db.put(b"k", b"1").await.unwrap();
        txn.put(b"other", b"x").unwrap();
        assert_eq!(is_conflict(&txn.commit().await), marked, "marked={marked}");
    }
}

/// A dropped transaction leaves nothing behind: its reads do not reach later transactions.
#[tokio::test]
async fn should_leave_nothing_behind_when_dropped() {
    let db = open().await;
    let dropped = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    for i in 0..100 {
        let _ = dropped.get(key(i)).await.unwrap();
    }
    let mut scan = dropped.scan(key(0)..key(100)).await.unwrap();
    while scan.next().await.unwrap().is_some() {}
    drop(scan);
    drop(dropped);
    for i in 0..100 {
        db.put(key(i), b"v").await.unwrap();
    }
    let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
    for i in 0..100 {
        assert!(txn.get(key(i)).await.unwrap().is_some());
    }
    txn.put(b"other", b"x").unwrap();
    assert!(txn.commit().await.unwrap().is_some());
}

/// One step of a schedule of [`should_conflict_as_model_on_random_schedules`].
#[derive(Clone, Debug)]
enum Step {
    Begin(usize),
    Get(usize, usize),
    Scan(usize, usize, usize),
    MarkRead(usize, usize),
    Put(usize, usize),
    Commit(usize),
}

/// What a transaction did, for the model.
#[derive(Default)]
struct ModelTxn {
    started: u64,
    serializable: bool,
    reads: BTreeSet<Vec<u8>>,
    ranges: Vec<(Vec<u8>, Vec<u8>)>,
    writes: BTreeSet<Vec<u8>>,
}

/// Random interleavings of transactions that read keys and ranges, mark reads and write. Each
/// commit result is compared with a model of the conflict rules: a transaction conflicts if a
/// commit after its start wrote a key that it wrote (write-write), or a key that it read or
/// marked read, or a key in a range that it scanned (read-write; reads count only at
/// Serializable isolation, marked reads at both).
#[tokio::test]
async fn should_conflict_as_model_on_random_schedules() {
    for seed in 0..200u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let db = open().await;
        let txns_total = 6;
        let mut steps = Vec::new();
        for t in 0..txns_total {
            steps.push(Step::Begin(t));
        }
        for _ in 0..40 {
            let t = rng.random_range(0..txns_total);
            let k = rng.random_range(0..12);
            steps.push(match rng.random_range(0..5) {
                0 => Step::Get(t, k),
                1 => Step::Scan(t, k, (k + rng.random_range(1..4)).min(12)),
                2 => Step::MarkRead(t, k),
                _ => Step::Put(t, k),
            });
        }
        for t in 0..txns_total {
            steps.push(Step::Commit(t));
        }
        // Shuffle the begins and commits into the middle, keeping each begin first and each
        // commit last for its transaction.
        let mut order: Vec<Step> = Vec::new();
        let mut pending: Vec<Step> = steps.clone();
        let mut begun = vec![false; txns_total];
        let mut committed = vec![false; txns_total];
        while !pending.is_empty() {
            let i = rng.random_range(0..pending.len());
            let step = pending[i].clone();
            let ok = match &step {
                Step::Begin(_) => true,
                Step::Commit(t) => {
                    begun[*t]
                        && !pending.iter().any(|s| match s {
                            Step::Get(u, _)
                            | Step::Scan(u, _, _)
                            | Step::MarkRead(u, _)
                            | Step::Put(u, _) => u == t,
                            _ => false,
                        })
                }
                Step::Get(t, _) | Step::Scan(t, _, _) | Step::MarkRead(t, _) | Step::Put(t, _) => {
                    begun[*t]
                }
            };
            if !ok {
                continue;
            }
            pending.remove(i);
            match &step {
                Step::Begin(t) => begun[*t] = true,
                Step::Commit(t) => committed[*t] = true,
                _ => {}
            }
            order.push(step);
        }

        let mut txns: Vec<Option<slatedb::DbTransaction>> = (0..txns_total).map(|_| None).collect();
        let mut model: Vec<ModelTxn> = (0..txns_total).map(|_| ModelTxn::default()).collect();
        // Committed writes: (commit position, keys).
        let mut history: Vec<(u64, BTreeSet<Vec<u8>>)> = Vec::new();
        let mut clock = 0u64;
        let mut values: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
        for step in &order {
            clock += 1;
            match step {
                Step::Begin(t) => {
                    let serializable = rng.random_bool(0.7);
                    let level = if serializable {
                        IsolationLevel::SerializableSnapshot
                    } else {
                        IsolationLevel::Snapshot
                    };
                    txns[*t] = Some(db.begin(level).unwrap());
                    model[*t] = ModelTxn {
                        started: clock,
                        serializable,
                        ..ModelTxn::default()
                    };
                }
                Step::Get(t, k) => {
                    let _ = txns[*t].as_ref().unwrap().get(key(*k)).await.unwrap();
                    if model[*t].serializable {
                        model[*t].reads.insert(key(*k));
                    }
                }
                Step::Scan(t, a, b) => {
                    let txn = txns[*t].as_ref().unwrap();
                    let mut it = txn.scan(key(*a)..key(*b)).await.unwrap();
                    while it.next().await.unwrap().is_some() {}
                    if model[*t].serializable {
                        model[*t].ranges.push((key(*a), key(*b)));
                    }
                }
                Step::MarkRead(t, k) => {
                    txns[*t].as_ref().unwrap().mark_read([key(*k)]).unwrap();
                    model[*t].reads.insert(key(*k));
                }
                Step::Put(t, k) => {
                    let v = values.entry(key(*k)).or_default();
                    *v += 1;
                    txns[*t]
                        .as_ref()
                        .unwrap()
                        .put(key(*k), v.to_string())
                        .unwrap();
                    model[*t].writes.insert(key(*k));
                }
                Step::Commit(t) => {
                    let txn = txns[*t].take().unwrap();
                    let m = &model[*t];
                    let later = history.iter().filter(|(at, _)| *at > m.started);
                    let mut expect_conflict = false;
                    for (_, written) in later {
                        let ww = !m.writes.is_disjoint(written);
                        let rw = !m.reads.is_disjoint(written)
                            || m.ranges
                                .iter()
                                .any(|(a, b)| written.iter().any(|w| w >= a && w < b));
                        // Read-write conflicts count for read-only commits too; a commit with
                        // no writes and no reads never conflicts.
                        if ww || rw {
                            expect_conflict = true;
                        }
                    }
                    let result = txn.commit().await;
                    assert_eq!(
                        is_conflict(&result),
                        expect_conflict,
                        "seed {seed}, step {step:?}: {result:?}"
                    );
                    if result.is_ok() && !m.writes.is_empty() {
                        history.push((clock, m.writes.clone()));
                    }
                }
            }
        }
    }
}

/// Many tasks run Serializable transactions over a few hot keys at once. Two invariants that
/// any serializable history keeps must hold at the end:
/// - transfers between accounts keep the total;
/// - no write skew: each transaction reads both "on call" flags and clears its own only if the
///   other is still set, so at least one flag stays set.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn should_keep_serializable_invariants_under_concurrency() {
    const ACCOUNTS: usize = 8;
    const INITIAL: i64 = 1000;
    let db = Arc::new(open().await);
    for a in 0..ACCOUNTS {
        db.put(key(a), INITIAL.to_string()).await.unwrap();
    }
    db.put(b"oncall-a", b"1").await.unwrap();
    db.put(b"oncall-b", b"1").await.unwrap();

    let mut tasks = Vec::new();
    for task in 0..16u64 {
        let db = Arc::clone(&db);
        tasks.push(tokio::spawn(async move {
            let mut rng = StdRng::seed_from_u64(task);
            let mut committed = 0;
            for _ in 0..300 {
                let txn = db.begin(IsolationLevel::SerializableSnapshot).unwrap();
                if rng.random_bool(0.2) {
                    // Write skew attempt.
                    let a = txn.get(b"oncall-a").await.unwrap().unwrap();
                    let b = txn.get(b"oncall-b").await.unwrap().unwrap();
                    if a.as_ref() == b"1" && b.as_ref() == b"1" {
                        let mine: &[u8] = if task % 2 == 0 {
                            b"oncall-a"
                        } else {
                            b"oncall-b"
                        };
                        txn.put(mine, b"0").unwrap();
                    }
                } else {
                    // Transfer, reading every account (a scan) to check the total too.
                    let from = rng.random_range(0..ACCOUNTS);
                    let to = (from + rng.random_range(1..ACCOUNTS)) % ACCOUNTS;
                    let mut it = txn.scan(key(0)..key(ACCOUNTS)).await.unwrap();
                    let mut balances = Vec::new();
                    while let Some(kv) = it.next().await.unwrap() {
                        balances.push(
                            std::str::from_utf8(&kv.value)
                                .unwrap()
                                .parse::<i64>()
                                .unwrap(),
                        );
                    }
                    drop(it);
                    assert_eq!(
                        balances.iter().sum::<i64>(),
                        INITIAL * ACCOUNTS as i64,
                        "a serializable snapshot sees a consistent total"
                    );
                    let amount = rng.random_range(1..50);
                    txn.put(key(from), (balances[from] - amount).to_string())
                        .unwrap();
                    txn.put(key(to), (balances[to] + amount).to_string())
                        .unwrap();
                }
                match txn.commit().await {
                    Ok(_) => committed += 1,
                    Err(err) if err.kind() == ErrorKind::Transaction => {}
                    Err(err) => panic!("{err}"),
                }
            }
            committed
        }));
    }
    let mut committed = 0;
    for task in tasks {
        committed += task.await.unwrap();
    }
    assert!(committed > 0);
    let mut total = 0;
    for a in 0..ACCOUNTS {
        let v = db.get(key(a)).await.unwrap().unwrap();
        total += std::str::from_utf8(&v).unwrap().parse::<i64>().unwrap();
    }
    assert_eq!(total, INITIAL * ACCOUNTS as i64);
    let a = db.get(b"oncall-a").await.unwrap().unwrap();
    let b = db.get(b"oncall-b").await.unwrap().unwrap();
    assert!(
        a.as_ref() == b"1" || b.as_ref() == b"1",
        "write skew: both flags were cleared"
    );
}
