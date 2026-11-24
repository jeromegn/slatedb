//! A benchmark of reads through the children of a split (an ignored test).
//!
//! The same rows are served two ways, as `RC_SPLIT_RANGES` adjacent ranges:
//!
//! - `own`: each range owns its base file, as ranges that `cache_range` registers or that a major
//!   merge rewrote;
//! - `split`: the children of one range split from one base file of all the rows, built by the
//!   planner and the layer construction that splits use (`split::plan` and
//!   `split::children_layers`): each child reads the shared file within its keys, so a point
//!   read searches the fence index of the whole file and a scan's cursor seeks in it.
//!
//! The rows are `RC_SPLIT_ROWS` 16-byte keys of the benchmark harness with 200-byte values. With
//! `runs=0` the ranges hold a base only; with `runs=4` they also hold four runs of
//! `RC_SPLIT_DELTA` versions of random keys, read through each range's view: for `own`, each
//! range's runs are files of its own keys; for `split`, the children share the parent's run files
//! and have the parent's view restricted to their keys. Point hits (`lookup` of a random key in
//! the range that holds it, found with a search of the range starts as the cache's index does)
//! and 10-row scans from a random key (`CachedScan` over the range that holds the start and the
//! next one) run on `RC_SPLIT_TASKS` threads, `own` and `split` interleaved, for `RC_SPLIT_ROUNDS`
//! rounds, and the benchmark prints each measurement and, per cell, the median and range of
//! each variant and the change of the medians. Results are checked against each other first.
//!
//! Run it alone on the machine, with its files on a local disk (`RC_SPLIT_DIR`, default a
//! temporary directory):
//!
//! ```sh
//! cargo test --profile bench -p slatedb --features range_cache --lib split_bench -- --ignored --nocapture
//! ```
//!
//! Output lines start with `SPLIT_BENCH` (one measurement), `SPLIT_SUMMARY` (a cell) and
//! `SPLIT_SHAPE` (the shared base file and the first own one: bytes on disk, and the heap bytes
//! of the fence index, which is all a base file holds in memory). The defaults make a shared file
//! of about 490 MiB and children of about 31 MiB, whose fence index is a sixteenth of the shared
//! file's.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::ops::Bound::{Included, Unbounded};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use bytes::Bytes;

use super::delta::{Delta, DeltaMemory};
use super::file::{FileBudget, RangeCachePageSource, RangeFile, RangeFileWriter};
use super::scan::CachedScan;
use super::view::{BuildCost, DeltaView, ViewMemory};
use super::{lookup, one_chunk, split, Found, Layers, Run, MAX_RUNS_RANGE};
use crate::bytes_range::BytesRange;
use crate::iter::RowEntryIterator;
use crate::types::{RowEntry, ValueDeletable};

const VALUE_LEN: usize = 200;
const BASE_SEQ: u64 = 1_000_000;
const RUNS: usize = 4;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `k` and 15 decimal digits of `2 i`, as `bench_harness::Dataset::key`.
fn key(i: u64) -> [u8; 16] {
    let mut k = *b"k000000000000000";
    let mut x = 2 * i;
    let mut pos = 15;
    while x > 0 {
        k[pos] = b'0' + (x % 10) as u8;
        x /= 10;
        pos -= 1;
    }
    k
}

fn value(i: u64, version: u64) -> Bytes {
    let mut v = Vec::with_capacity(VALUE_LEN);
    let mut rng = Rng(i ^ version.rotate_left(32) ^ 0xD6E8_FEB8_6659_FD93);
    while v.len() < VALUE_LEN {
        v.extend_from_slice(&rng.next().to_le_bytes());
    }
    v.truncate(VALUE_LEN);
    Bytes::from(v)
}

/// The keys of all rows, back to back.
struct KeyTable(Vec<u8>);

impl KeyTable {
    fn new(n: u64) -> Self {
        let mut bytes = Vec::with_capacity(n as usize * 16);
        for i in 0..n {
            bytes.extend_from_slice(&key(i));
        }
        Self(bytes)
    }

    fn get(&self, i: u64) -> &[u8] {
        let at = i as usize * 16;
        &self.0[at..at + 16]
    }
}

fn open(path: std::path::PathBuf) -> RangeFile {
    RangeFile::open(path, RangeCachePageSource::Mmap, FileBudget::detached()).unwrap()
}

/// A base file of rows `rows`.
fn write_base(
    dir: &std::path::Path,
    name: &str,
    table: &KeyTable,
    rows: std::ops::Range<u64>,
) -> Arc<RangeFile> {
    let mut writer = RangeFileWriter::create(dir, name, 4096).unwrap();
    for i in rows {
        writer
            .push(&RowEntry::new_value(table.get(i), &value(i, 0), 1))
            .unwrap();
    }
    Arc::new(open(writer.finish().unwrap()))
}

/// The runs of `versions` (key index, seq, run) whose keys are in `rows`, newest first, run `r`
/// covering `(BASE_SEQ + r * span, BASE_SEQ + (r + 1) * span]`.
fn write_runs(
    dir: &std::path::Path,
    tag: &str,
    table: &KeyTable,
    versions: &[(u64, u64, usize)],
    span: u64,
    rows: std::ops::Range<u64>,
) -> Vec<Arc<Run>> {
    let mut out = Vec::with_capacity(RUNS);
    for r in 0..RUNS {
        let mut picked: Vec<(u64, u64)> = versions
            .iter()
            .filter(|v| v.2 == r && rows.contains(&v.0))
            .map(|v| (v.0, v.1))
            .collect();
        picked.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut writer = RangeFileWriter::create_run(dir, &format!("{tag}-run{r}"), 4096).unwrap();
        for (i, seq) in picked {
            let row = RowEntry::new(
                Bytes::copy_from_slice(table.get(i)),
                ValueDeletable::Value(value(i, seq)),
                seq,
                None,
                None,
            );
            writer.push(&row).unwrap();
        }
        let file = open(writer.finish().unwrap());
        let lo = BASE_SEQ + r as u64 * span;
        out.push(Arc::new(Run::new(file, lo, lo + span).unwrap()));
    }
    out.reverse();
    out
}

/// Adjacent ranges that serve all rows, by start key.
struct Ranges {
    starts: Vec<Bytes>,
    layers: Vec<Arc<Layers>>,
    seq: u64,
}

impl Ranges {
    /// The index of the range that holds `key`.
    fn of(&self, key: &[u8]) -> usize {
        self.starts.partition_point(|s| s.as_ref() <= key) - 1
    }
}

/// The layer set of one range: `chunks` as its base, `runs` with their view, an empty active
/// buffer.
fn layers_of(
    start: &[u8],
    end: Option<Bytes>,
    file: &Arc<RangeFile>,
    runs: Vec<Arc<Run>>,
    memory: &Arc<ViewMemory>,
) -> Layers {
    let covered_to = runs.first().map_or(BASE_SEQ, |r| r.hi);
    let view = (!runs.is_empty())
        .then(|| Arc::new(DeltaView::from_runs(&runs, memory, &mut BuildCost::default()).unwrap()));
    let layers = Layers {
        active: Arc::new(Delta::for_test(&DeltaMemory::detached(), covered_to)),
        sealed: Vec::new(),
        runs,
        view,
        chunks: one_chunk(file.clone(), BASE_SEQ),
        covered_to,
        end,
    };
    layers.check(start, *MAX_RUNS_RANGE.end());
    layers
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    Get,
    Scan10,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Get => "get_hit",
            Op::Scan10 => "scan10",
        }
    }
}

/// One point hit or one 10-row scan from row index `i`.
fn one(ranges: &Ranges, table: &KeyTable, op: Op, i: u64) {
    let key = table.get(i);
    let at = ranges.of(key);
    match op {
        Op::Get => match lookup(&ranges.layers[at], key, ranges.seq).unwrap() {
            Found::Value(kv) => {
                black_box(kv);
            }
            _ => panic!("a miss"),
        },
        Op::Scan10 => {
            let parts = ranges.layers[at..(at + 2).min(ranges.layers.len())].to_vec();
            let bounds = BytesRange::new(Included(Bytes::copy_from_slice(key)), Unbounded);
            let mut scan = CachedScan::new(parts, ranges.seq, &bounds, None).unwrap();
            for _ in 0..10 {
                match scan.next_sync() {
                    Some(Ok(Some(row))) => {
                        black_box(row);
                    }
                    _ => panic!("a scan ended early"),
                }
            }
        }
    }
}

/// Runs `op` on `tasks` threads for `RC_SPLIT_MS` after a warm-up of `RC_SPLIT_WARM_MS`, and
/// returns operations per second.
fn measure(ranges: &Ranges, table: &KeyTable, n: u64, op: Op, tasks: usize, seed: u64) -> f64 {
    let warm = Duration::from_millis(env("RC_SPLIT_WARM_MS", 300));
    let secs = Duration::from_millis(env("RC_SPLIT_MS", 1500));
    let barrier = Barrier::new(tasks);
    let total = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        for t in 0..tasks {
            let (barrier, total, stop) = (&barrier, &total, &stop);
            s.spawn(move || {
                let mut rng = Rng(seed + t as u64 * 7919);
                barrier.wait();
                let started = Instant::now();
                while started.elapsed() < warm {
                    for _ in 0..64 {
                        one(ranges, table, op, rng.below(n - 10));
                    }
                }
                barrier.wait();
                let started = Instant::now();
                let mut ops = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..64 {
                        one(ranges, table, op, rng.below(n - 10));
                    }
                    ops += 64;
                    if t == 0 && started.elapsed() >= secs {
                        stop.store(true, Ordering::Relaxed);
                    }
                }
                total.fetch_add(ops, Ordering::Relaxed);
            });
        }
    });
    total.load(Ordering::Relaxed) as f64 / secs.as_secs_f64()
}

/// Checks that both ways serve the same rows: point reads and 10-row scans from random keys.
fn verify(own: &Ranges, split: &Ranges, table: &KeyTable, n: u64) {
    let mut rng = Rng(42);
    for _ in 0..20_000 {
        let i = rng.below(n - 10);
        let key = table.get(i);
        let (a, b) = (own.of(key), split.of(key));
        assert_eq!(a, b);
        let get = |r: &Ranges, at: usize| match lookup(&r.layers[at], key, r.seq).unwrap() {
            Found::Value(kv) => kv,
            _ => panic!("a miss"),
        };
        assert_eq!(get(own, a), get(split, b));
        let rows = |r: &Ranges, at: usize| {
            let parts = r.layers[at..(at + 2).min(r.layers.len())].to_vec();
            let bounds = BytesRange::new(Included(Bytes::copy_from_slice(key)), Unbounded);
            let mut scan = CachedScan::new(parts, r.seq, &bounds, None).unwrap();
            (0..10)
                .map(|_| scan.next_sync().unwrap().unwrap().unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(own, a), rows(split, b));
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "a benchmark: run it alone with --ignored --nocapture"]
fn split_bench() {
    let n = env("RC_SPLIT_ROWS", 2_000_000);
    let count = env("RC_SPLIT_RANGES", 16) as usize;
    let delta = env("RC_SPLIT_DELTA", 100_000);
    let rounds = env("RC_SPLIT_ROUNDS", 7) as usize;
    let tasks = env("RC_SPLIT_TASKS", 16) as usize;
    let dir = match std::env::var("RC_SPLIT_DIR") {
        Ok(root) => tempfile::tempdir_in(root).unwrap(),
        Err(_) => tempfile::tempdir().unwrap(),
    };
    let memory = Arc::new(ViewMemory::default());
    let table = KeyTable::new(n);
    // Range `j` holds rows `bounds[j]..bounds[j + 1]`.
    let bounds: Vec<u64> = (0..=count).map(|j| n * j as u64 / count as u64).collect();
    let start_of = |j: usize| match j {
        0 => Bytes::new(),
        j => Bytes::copy_from_slice(table.get(bounds[j])),
    };
    let end_of =
        |j: usize| (j + 1 < count).then(|| Bytes::copy_from_slice(table.get(bounds[j + 1])));
    let starts: Vec<Bytes> = (0..count).map(start_of).collect();
    let cuts: Vec<Bytes> = (1..count).map(start_of).collect();
    // The versions of the runs: random keys, the runs in consecutive intervals.
    let span = delta.div_ceil(RUNS as u64);
    let mut rng = Rng(0xDE17A);
    let versions: Vec<(u64, u64, usize)> = (0..delta)
        .map(|j| (rng.below(n), BASE_SEQ + 1 + j, (j / span) as usize))
        .collect();

    let whole = write_base(dir.path(), "whole", &table, 0..n);
    let own_files: Vec<Arc<RangeFile>> = (0..count)
        .map(|j| {
            write_base(
                dir.path(),
                &format!("own{j}"),
                &table,
                bounds[j]..bounds[j + 1],
            )
        })
        .collect();
    println!(
        "SPLIT_SHAPE rows={n} ranges={count} shared_file_bytes={} shared_fence_bytes={} own_file_bytes={} own_fence_bytes={}",
        whole.file_len(),
        whole.meta_bytes(),
        own_files[0].file_len(),
        own_files[0].meta_bytes(),
    );
    let shared_runs = write_runs(dir.path(), "shared", &table, &versions, span, 0..n);
    let own_runs: Vec<Vec<Arc<Run>>> = (0..count)
        .map(|j| {
            write_runs(
                dir.path(),
                &format!("own{j}"),
                &table,
                &versions,
                span,
                bounds[j]..bounds[j + 1],
            )
        })
        .collect();

    let mut setups: Vec<(usize, Ranges, Ranges)> = Vec::new();
    for runs in [0, RUNS] {
        let own = Ranges {
            starts: starts.clone(),
            layers: (0..count)
                .map(|j| {
                    let r = if runs == 0 {
                        Vec::new()
                    } else {
                        own_runs[j].clone()
                    };
                    Arc::new(layers_of(&starts[j], end_of(j), &own_files[j], r, &memory))
                })
                .collect(),
            seq: if runs == 0 {
                BASE_SEQ
            } else {
                BASE_SEQ + span * RUNS as u64
            },
        };
        let parent_runs = if runs == 0 {
            Vec::new()
        } else {
            shared_runs.clone()
        };
        let parent = layers_of(&[], None, &whole, parent_runs, &memory);
        let plan = split::plan(
            &parent,
            &parent.chunks,
            &Bytes::new(),
            None,
            &cuts,
            &mut BuildCost::default(),
        )
        .unwrap()
        .expect("a plan");
        let children = split::children_layers(
            &parent,
            plan.children,
            parent.covered_to,
            *MAX_RUNS_RANGE.end(),
            |lo| Arc::new(Delta::for_test(&DeltaMemory::detached(), lo)),
        );
        let split = Ranges {
            starts: children.iter().map(|(s, _, _)| s.clone()).collect(),
            layers: children.into_iter().map(|(_, _, l)| Arc::new(l)).collect(),
            seq: own.seq,
        };
        assert_eq!(split.starts, own.starts);
        verify(&own, &split, &table, n);
        setups.push((runs, own, split));
    }

    // (runs, op, variant) -> ops/s of each round.
    let mut results: BTreeMap<(usize, &'static str, &'static str), Vec<f64>> = BTreeMap::new();
    for round in 0..rounds {
        for (runs, own, split) in &setups {
            for op in [Op::Get, Op::Scan10] {
                let order: [(&'static str, &Ranges); 2] = if round % 2 == 0 {
                    [("own", own), ("split", split)]
                } else {
                    [("split", split), ("own", own)]
                };
                for (variant, ranges) in order {
                    let rate = measure(ranges, &table, n, op, tasks, 0x5EED + round as u64);
                    println!(
                        "SPLIT_BENCH round={round} ranges={count} runs={runs} op={} tasks={tasks} variant={variant} ops_per_sec={rate:.0}",
                        op.name()
                    );
                    results
                        .entry((*runs, op.name(), variant))
                        .or_default()
                        .push(rate);
                }
            }
        }
    }
    for runs in [0, RUNS] {
        for op in ["get_hit", "scan10"] {
            let mut own = results[&(runs, op, "own")].clone();
            let mut split = results[&(runs, op, "split")].clone();
            let (own_min, own_max) = (
                own.iter().cloned().fold(f64::MAX, f64::min),
                own.iter().cloned().fold(0.0, f64::max),
            );
            let (split_min, split_max) = (
                split.iter().cloned().fold(f64::MAX, f64::min),
                split.iter().cloned().fold(0.0, f64::max),
            );
            let (own_median, split_median) = (median(&mut own), median(&mut split));
            println!(
                "SPLIT_SUMMARY ranges={count} runs={runs} op={op} tasks={tasks} rounds={rounds} own_mops={:.2} ({:.2} to {:.2}) split_mops={:.2} ({:.2} to {:.2}) change={:+.1}%",
                own_median / 1e6,
                own_min / 1e6,
                own_max / 1e6,
                split_median / 1e6,
                split_min / 1e6,
                split_max / 1e6,
                (split_median / own_median - 1.0) * 100.0
            );
        }
    }
}
