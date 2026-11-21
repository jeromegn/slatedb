//! A benchmark of the delta view against per-run cursors (an ignored test).
//!
//! It builds one range as the cache holds it: a base of `RC_VIEW_ROWS` rows (16-byte keys of the
//! benchmark harness, or 40-byte keys shaped like TPC-C order lines), and a delta split into 1,
//! 4, 16 and 32 run files by sequence number, with an empty active buffer. Then, on 1 and 16
//! threads, it times 10-row scans from a random base key (`CachedScan`, the cache's own scan, at
//! the latest sequence number, with only a row limit as the harness's `db.scan(start..)`) and
//! point reads of random base keys (hits), with the layers' view off (per-run cursors and
//! `lookup`) and on (one view cursor and `lookup_view`).
//!
//! Two deltas:
//!
//! - `sparse`: `RC_VIEW_DELTA` versions of random keys (with repeats), the same bytes whatever
//!   the number of runs.
//! - `dense`: one version of every key, each in a random run: a scan's rows are all recent
//!   versions, spread over the runs (random updates of adjacent keys). With one run, the same rows
//!   sit in one merged run.
//!
//! It also prints the view's bytes per version and the cost of building it run by run, as
//! flushes do. Run it alone on the machine:
//!
//! ```sh
//! cargo test --profile bench -p slatedb --features range_cache --lib view_bench -- --ignored --nocapture
//! ```
//!
//! Output lines start with `VIEW_BENCH` (keys, delta, runs, op, tasks, variant, ops/s, ns per
//! op per thread) and `VIEW_BUILD`.

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
use super::{lookup, lookup_view, Found, Layers, Run};
use crate::bytes_range::BytesRange;
use crate::iter::RowEntryIterator;
use crate::types::{RowEntry, ValueDeletable};

const VALUE_LEN: usize = 200;
const BASE_SEQ: u64 = 1_000_000;

#[derive(Clone, Copy, Debug)]
enum Keys {
    /// `k` and 15 decimal digits of `2 i`, as `bench_harness::Dataset::key`.
    Harness,
    /// `blobtree/rec/order_line/` and four big-endian u32: warehouse (1 or 2), district (1 to
    /// 10), order and line (1 to 10). The range's keys share 27 bytes, and the next 8 bytes take
    /// few values; the bytes that tell rows apart come after them.
    Tpcc,
}

impl Keys {
    fn name(self) -> &'static str {
        match self {
            Keys::Harness => "harness",
            Keys::Tpcc => "tpcc",
        }
    }

    fn len(self) -> usize {
        match self {
            Keys::Harness => 16,
            Keys::Tpcc => 40,
        }
    }

    /// The key of row `i` of `n`, ascending in `i`.
    fn key(self, i: u64, n: u64) -> Vec<u8> {
        match self {
            Keys::Harness => {
                let mut k = *b"k000000000000000";
                let mut x = 2 * i;
                let mut pos = 15;
                while x > 0 {
                    k[pos] = b'0' + (x % 10) as u8;
                    x /= 10;
                    pos -= 1;
                }
                k.to_vec()
            }
            Keys::Tpcc => {
                let orders = n.div_ceil(2 * 10 * 10);
                let (line, rest) = (i % 10, i / 10);
                let (order, rest) = (rest % orders, rest / orders);
                let (district, warehouse) = (rest % 10, rest / 10);
                let mut k = b"blobtree/rec/order_line/".to_vec();
                for field in [warehouse + 1, district + 1, order + 1, line + 1] {
                    k.extend_from_slice(&(field as u32).to_be_bytes());
                }
                k
            }
        }
    }
}

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

fn value(i: u64, version: u64) -> Bytes {
    let mut v = Vec::with_capacity(VALUE_LEN);
    let mut rng = Rng(i ^ version.rotate_left(32) ^ 0xD6E8_FEB8_6659_FD93);
    while v.len() < VALUE_LEN {
        v.extend_from_slice(&rng.next().to_le_bytes());
    }
    v.truncate(VALUE_LEN);
    Bytes::from(v)
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The keys of all rows, back to back, so that an operation takes its key without building it.
struct KeyTable {
    bytes: Vec<u8>,
    len: usize,
}

impl KeyTable {
    fn new(keys: Keys, n: u64) -> Self {
        let mut bytes = Vec::with_capacity(n as usize * keys.len());
        for i in 0..n {
            bytes.extend_from_slice(&keys.key(i, n));
        }
        Self {
            bytes,
            len: keys.len(),
        }
    }

    fn get(&self, i: u64) -> &[u8] {
        let at = i as usize * self.len;
        &self.bytes[at..at + self.len]
    }
}

/// A range of `n` base rows and runs of `versions` (key index, seq), as the cache holds it.
struct Range {
    layers: Arc<Layers>,
    seq: u64,
}

fn write_base(dir: &std::path::Path, table: &KeyTable, n: u64) -> RangeFile {
    let mut writer = RangeFileWriter::create(dir, "base", 4096).unwrap();
    for i in 0..n {
        writer
            .push(&RowEntry::new_value(table.get(i), &value(i, 0), 1))
            .unwrap();
    }
    let path = writer.finish().unwrap();
    RangeFile::open(path, RangeCachePageSource::Mmap, FileBudget::detached()).unwrap()
}

/// Writes the runs of `versions`, (key index, seq, run), oldest run first, each run covering
/// `(BASE_SEQ + r * span, BASE_SEQ + (r + 1) * span]`. Returns them newest first.
fn write_runs(
    dir: &std::path::Path,
    table: &KeyTable,
    versions: &[(u64, u64, usize)],
    runs: usize,
    span: u64,
    tag: &str,
) -> Vec<Arc<Run>> {
    let mut out = Vec::with_capacity(runs);
    for r in 0..runs {
        let mut rows: Vec<(u64, u64)> = versions
            .iter()
            .filter(|v| v.2 == r)
            .map(|v| (v.0, v.1))
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut writer = RangeFileWriter::create_run(dir, &format!("{tag}-run{r}"), 4096).unwrap();
        for (i, seq) in rows {
            let row = RowEntry::new(
                Bytes::copy_from_slice(table.get(i)),
                ValueDeletable::Value(value(i, seq)),
                seq,
                None,
                None,
            );
            writer.push(&row).unwrap();
        }
        let path = writer.finish().unwrap();
        let file =
            RangeFile::open(path, RangeCachePageSource::Mmap, FileBudget::detached()).unwrap();
        let lo = BASE_SEQ + r as u64 * span;
        out.push(Arc::new(Run::new(file, lo, lo + span).unwrap()));
    }
    out.reverse();
    out
}

fn layers(base: &Arc<RangeFile>, runs: &[Arc<Run>], view: Option<Arc<DeltaView>>) -> Range {
    let covered_to = runs.first().map_or(BASE_SEQ, |r| r.hi);
    let layers = Layers {
        active: Arc::new(Delta::for_test(&DeltaMemory::detached(), covered_to)),
        sealed: Vec::new(),
        runs: runs.to_vec(),
        view,
        base: Some(base.clone()),
        base_seq: BASE_SEQ,
        covered_to,
        end: None,
    };
    layers.check(runs.len().max(4));
    Range {
        layers: Arc::new(layers),
        seq: covered_to,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Scan10,
    Get,
}

/// Runs `op` on `tasks` threads for `secs` after a warm-up, and returns (ops/s, ns per op per
/// thread).
fn measure(
    range: &Range,
    table: &KeyTable,
    n: u64,
    op: Op,
    view: bool,
    tasks: usize,
) -> (f64, f64) {
    let warm = Duration::from_millis(env("RC_VIEW_WARM_MS", 300));
    let secs = Duration::from_millis(env("RC_VIEW_MS", 1500));
    let barrier = Barrier::new(tasks);
    let total = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        for t in 0..tasks {
            let (barrier, total, stop) = (&barrier, &total, &stop);
            s.spawn(move || {
                let mut rng = Rng(0x5EED + t as u64 * 7919);
                let one = |rng: &mut Rng| match op {
                    Op::Scan10 => {
                        let start = Bytes::copy_from_slice(table.get(rng.below(n - 10)));
                        let bounds = BytesRange::new(Included(start), Unbounded);
                        let mut scan =
                            CachedScan::new(vec![range.layers.clone()], range.seq, &bounds, None)
                                .unwrap();
                        for _ in 0..10 {
                            match scan.next_sync() {
                                Some(Ok(Some(row))) => {
                                    black_box(row);
                                }
                                _ => panic!("a scan ended early"),
                            }
                        }
                    }
                    Op::Get => {
                        let key = table.get(rng.below(n));
                        let found = if view {
                            lookup_view(&range.layers, key, range.seq)
                        } else {
                            lookup(&range.layers, key, range.seq)
                        };
                        match found.unwrap() {
                            Found::Value(kv) => {
                                black_box(kv);
                            }
                            _ => panic!("a miss"),
                        }
                    }
                };
                barrier.wait();
                let started = Instant::now();
                while started.elapsed() < warm {
                    for _ in 0..64 {
                        one(&mut rng);
                    }
                }
                barrier.wait();
                let started = Instant::now();
                let mut ops = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..64 {
                        one(&mut rng);
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
    let ops = total.load(Ordering::Relaxed) as f64;
    let secs = secs.as_secs_f64();
    (ops / secs, secs * tasks as f64 * 1e9 / ops)
}

/// Checks that the two variants return the same rows, before they are timed.
fn verify(with: &Range, without: &Range, table: &KeyTable, n: u64) {
    let mut rng = Rng(42);
    for _ in 0..2000 {
        let start = Bytes::copy_from_slice(table.get(rng.below(n - 10)));
        let bounds = BytesRange::new(Included(start), Unbounded);
        let mut a = CachedScan::new(vec![with.layers.clone()], with.seq, &bounds, None).unwrap();
        let mut b =
            CachedScan::new(vec![without.layers.clone()], without.seq, &bounds, None).unwrap();
        for _ in 0..10 {
            let x = a.next_sync().unwrap().unwrap();
            let y = b.next_sync().unwrap().unwrap();
            assert_eq!(x, y);
        }
        let key = table.get(rng.below(n));
        let x = lookup_view(&with.layers, key, with.seq).unwrap();
        let y = lookup(&without.layers, key, without.seq).unwrap();
        match (x, y) {
            (Found::Value(x), Found::Value(y)) => assert_eq!(x, y),
            _ => panic!("a miss"),
        }
    }
}

#[test]
#[ignore = "a benchmark: run it alone with --ignored --nocapture"]
fn view_bench() {
    let n = env("RC_VIEW_ROWS", 262_144);
    let sparse_versions = env("RC_VIEW_DELTA", 65_536);
    let run_counts: Vec<usize> = std::env::var("RC_VIEW_RUNS")
        .unwrap_or_else(|_| "1,4,16,32".to_string())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let key_sets: Vec<Keys> = std::env::var("RC_VIEW_KEYS")
        .unwrap_or_else(|_| "harness,tpcc".to_string())
        .split(',')
        .map(|v| match v {
            "harness" => Keys::Harness,
            "tpcc" => Keys::Tpcc,
            v => panic!("RC_VIEW_KEYS: {v}"),
        })
        .collect();
    let deltas: Vec<String> = std::env::var("RC_VIEW_DELTAS")
        .unwrap_or_else(|_| "sparse,dense".to_string())
        .split(',')
        .map(str::to_string)
        .collect();
    let tasks: Vec<usize> = std::env::var("RC_VIEW_TASKS")
        .unwrap_or_else(|_| "1,16".to_string())
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let root = std::env::var("RC_VIEW_DIR").ok();
    let memory = Arc::new(ViewMemory::default());
    for &keys in &key_sets {
        let dir = match &root {
            Some(root) => tempfile::tempdir_in(root).unwrap(),
            None => tempfile::tempdir().unwrap(),
        };
        let table = KeyTable::new(keys, n);
        let base = Arc::new(write_base(dir.path(), &table, n));
        for delta in &deltas {
            for &runs in &run_counts {
                let mut rng = Rng(0xDE17A + runs as u64);
                // (key index, seq, run), with each run's versions in its own interval.
                let (versions, span): (Vec<(u64, u64, usize)>, u64) = match delta.as_str() {
                    "sparse" => {
                        let per = sparse_versions.div_ceil(runs as u64);
                        let v = (0..sparse_versions)
                            .map(|j| (rng.below(n), BASE_SEQ + 1 + j, (j / per) as usize))
                            .collect();
                        (v, per)
                    }
                    "dense" => {
                        let mut filled = vec![0u64; runs];
                        let v = (0..n)
                            .map(|i| {
                                let r = rng.below(runs as u64) as usize;
                                filled[r] += 1;
                                (i, BASE_SEQ + r as u64 * n + filled[r], r)
                            })
                            .collect();
                        (v, n)
                    }
                    d => panic!("RC_VIEW_DELTAS: {d}"),
                };
                let tag = format!("{}-{delta}-{runs}", keys.name());
                let run_files = write_runs(dir.path(), &table, &versions, runs, span, &tag);
                let run_bytes: u64 = run_files.iter().map(|r| r.file_bytes()).sum();
                // The view, built run by run as flushes build it; the last build is that of a
                // flush into a view of `runs - 1` runs.
                let mut cost = BuildCost::default();
                let oldest = run_files.last().unwrap().clone();
                let started = Instant::now();
                let mut view = DeltaView::from_runs(&[oldest], &memory, &mut cost).unwrap();
                cost.nanos = started.elapsed().as_nanos() as u64;
                let mut last = BuildCost::default();
                for r in run_files.iter().rev().skip(1) {
                    last = BuildCost::default();
                    let started = Instant::now();
                    view = view.with_run(r, &mut last).unwrap();
                    last.nanos = started.elapsed().as_nanos() as u64;
                    cost.nanos += last.nanos;
                    cost.keys_read += last.keys_read;
                    cost.rows_read += last.rows_read;
                    cost.bytes_read += last.bytes_read;
                }
                let mut scratch_cost = BuildCost::default();
                let started = Instant::now();
                let scratch = DeltaView::from_runs(&run_files, &memory, &mut scratch_cost).unwrap();
                scratch_cost.nanos = started.elapsed().as_nanos() as u64;
                assert_eq!(scratch.entries(), view.entries());
                drop(scratch);
                println!(
                    "VIEW_BUILD keys={} delta={delta} runs={runs} versions={} run_bytes={run_bytes} view_bytes={} bytes_per_version={:.2} incremental_ms={:.2} last_flush_ms={:.3} last_flush_rows={} last_flush_old_keys_read={} last_flush_bytes_read={} from_scratch_ms={:.2}",
                    keys.name(),
                    view.len(),
                    view.bytes(),
                    view.bytes() as f64 / view.len() as f64,
                    cost.nanos as f64 / 1e6,
                    last.nanos as f64 / 1e6,
                    last.rows_read,
                    last.keys_read,
                    last.bytes_read,
                    scratch_cost.nanos as f64 / 1e6,
                );
                let with = layers(&base, &run_files, Some(Arc::new(view)));
                let without = layers(&base, &run_files, None);
                verify(&with, &without, &table, n);
                let ops: &[Op] = if delta == "dense" {
                    &[Op::Scan10]
                } else {
                    &[Op::Scan10, Op::Get]
                };
                for &op in ops {
                    for &t in &tasks {
                        for (variant, range) in [("runs", &without), ("view", &with)] {
                            let (rate, ns) = measure(range, &table, n, op, variant == "view", t);
                            println!(
                                "VIEW_BENCH keys={} delta={delta} runs={runs} op={} tasks={t} variant={variant} ops_per_sec={rate:.0} ns_per_op={ns:.0}",
                                keys.name(),
                                if op == Op::Scan10 { "scan10" } else { "get_hit" },
                            );
                        }
                    }
                }
            }
        }
    }
}
