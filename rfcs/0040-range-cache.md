# SlateDB Range Cache

Table of Contents:

<!-- TOC start (generate with https://bitdowntoc.derlin.ch) -->

- [Summary](#summary)
- [Motivation](#motivation)
- [Goals](#goals)
- [Non-Goals](#non-goals)
- [Design](#design)
- [Impact Analysis](#impact-analysis)
- [Operations](#operations)
- [Testing](#testing)
- [Rollout](#rollout)
- [Alternatives](#alternatives)
- [Open Questions](#open-questions)
- [References](#references)
- [Appendix: Why reads see the right versions](#appendix-why-reads-see-the-right-versions)

<!-- TOC end -->

Status: Draft

Authors:

* [jeromegn](https://github.com/jeromegn)

## Summary

This RFC adds a range cache: local, immutable, sorted files for key ranges that the application or
the cache chooses. A cached range holds every live key of its range, so a point read in it is a
definite hit or miss, and a scan in it needs no other source. These reads skip the normal read
path, which visits the memtables, the L0 SSTs and the sorted runs. The batch writer keeps cached
ranges up to date. The cache is disposable: its files are local, never synced, and removed when
the database opens. The manifest, the WAL and the SSTs do not change. It is off by default. In the
benchmark below, cached point read hits and scans are 8 to 9 times faster than the normal read
path with every block in memory.

## Motivation

A point read at the latest state visits each source that can hold the key: the memtables, every L0
SST, and one SST per sorted run. A miss visits all of them, and a scan merges all of them. RFC 0037
and RFC 0038 (drafts) make each source cheaper, but do not reduce the number of sources. Many
workloads read a small part of the key space again and again, such as a tenant's rows. For these
ranges, a copy that holds the answer costs much less. A prototype outside SlateDB showed that the
idea works. Inside SlateDB, the cache can apply each write before the write becomes visible.

## Goals

- Serve point reads and ascending scans from cached ranges, at the latest committed state and in
  snapshots and transactions.
- Return the same values, sequence numbers and create timestamps as the normal read path.
- Keep cached ranges up to date with puts, deletes, merges, write batches and transaction commits.
- Cost nothing when the feature is off, and almost nothing for keys that are not cached.
- Bound the local disk and the local write amplification, and do not fill the block cache.

## Non-Goals

- Replace the block cache, the object store cache or the local object mirror.
- Keep the cache across restarts.
- Serve dirty reads, `DurabilityLevel::Remote` reads or descending scans.
- Serve a snapshot or a transaction that is older than the file of its range.
- Support `DbReader`, which has no write path for the cache to follow.
- Turn automatic promotion on by default.

## Design

### Terms

- **Cached range**: a half-open key range `[start, end)` that the cache holds completely.
- **Range cache file**: an immutable local file with a range's rows at sequence number `file_seq`.
- **Delta**: an in-memory `KVTable` with the writes to a cached range since its file was written.
- **Explicit range** or **automatic range**: one that `cache_range` or automatic promotion loaded.

### API

```rust
pub struct RangeCacheOptions {
    pub root_folder: PathBuf,                     // the cache owns this directory
    pub max_disk_bytes: u64,                      // default 1 GiB
    pub page_source: RangeCachePageSource,        // default Mmap
    pub auto_promote: Option<AutoPromoteOptions>, // default None: off
    #[doc(hidden)] pub tuning: RangeCacheTuning,  // not a stable API
}
pub struct AutoPromoteOptions {                   // implements Default
    pub target_range_bytes: u64,                  // default 64 MiB of SST data
    pub sample_one_in: u32,                       // default 64
    pub heat_half_life: Duration,                 // default 10 s
}
pub enum RangeCachePageSource { Mmap, PositionalRead }
pub struct CachedRangeInfo {
    pub start: Bytes, pub end: Option<Bytes>, pub loading: bool, pub rebuilding: bool,
    pub file_bytes: u64, pub file_rows: u64, pub delta_bytes: u64, pub automatic: bool,
}
impl RangeCacheOptions { pub fn new(root_folder: impl Into<PathBuf>) -> Self; }
impl DbBuilder { pub fn with_range_cache(self, options: RangeCacheOptions) -> Self; }
impl Db {
    pub async fn cache_range<T: ByteRangeBounds + Send>(&self, range: T) -> Result<(), Error>;
    pub fn uncache_range<T: ByteRangeBounds>(&self, range: T) -> Result<(), Error>;
    pub fn clear_range_cache(&self) -> Result<(), Error>;
    pub fn cached_ranges(&self) -> Vec<CachedRangeInfo>;
}
```

The API is behind the `range_cache` cargo feature and is experimental. These types and
`RangeCacheTuning` are `#[non_exhaustive]`. `RangeCacheTuning` is doc-hidden, is not a stable API,
and holds internal parameters such as the page size and the promotion thresholds. `cache_range`
loads the parts of the range that no cached range holds. It fails with `ErrorKind::Invalid` if
there is no cache, if the range has expiring rows, or if a part is larger than the disk budget.

### Structure of a cached range

The cache keeps an index of cached ranges by start key, which readers load without a lock. One
mutex guards every change. Each cached range has three layers, newest first, which the cache
replaces together: the active delta, which receives writes; the frozen delta, only during a
rebuild; and the file with its `file_seq`, absent while the range loads. A delta's `sealed_at` is
`u64::MAX` until the cache seals it at the last sequence number that it applied to the delta.

### Write path

`DbInner::write_batch` runs on the batch writer task. It puts a batch in the WAL buffer and the
memtable, then advances the last committed sequence number, which makes the batch visible. With a
cache, the writer runs these steps after the memtable insert and before that advance:

1. Advance the cache's `inserted_seq` counter to the batch sequence number. Then load its
   `publish_epoch` counter. Both operations are `SeqCst`.
2. Load the index without a lock. If no entry falls in a cached range, return.
3. Take the mutex. Put each covered entry, tombstones and merge operands included, in its delta,
   unless its sequence number is at or below the range's `file_seq`: the file holds it already.
4. If an entry has an expire time, remove its range. Seal its delta one below the batch sequence
   number, because the delta does not hold this write.
5. If an active delta is larger than its rebuild threshold, freeze it (see [Rebuild](#rebuild)).

The hook runs before the advance, so a write is in its delta before a read can use its sequence
number. Steps 1 and 2 pair with the order of a load (see [Loading a range](#loading-a-range)). A
write that fails after the memtable insert is fatal: the writer's cleanup closes the cache.

### Read path

The cache serves a read only if its durability filter is `Memory`, it is not dirty, and it has no
filter context or tracing option. A scan must also be ascending.

A point read takes no lock, and awaits only for a merge operand. It loads the index, the layers of
its range, and then the last committed sequence number, called `seq`. If the sequence number rules
fail, it retries up to three times, then uses the normal read path. It returns the newest version
at or below `seq` from the active delta, the frozen delta, then the file. A tombstone or no version
is a definite miss. For a merge operand, the read runs the normal iterator stack over the pinned
layers.

A scan walks the pinned layers with a cursor on the file and on each delta. Each cursor skips
versions above `seq`. Each step takes the smallest key, and the newest layer wins a tie. A
tombstone skips the key. A key with a merge operand is read alone through the normal iterator
stack. The file cursor copies each page that it enters into one buffer, and returns rows as slices
of it. So a kept row holds up to one page (4 KiB by default), as a value from an SST holds its
block. A scan over adjacent ranges pins all of them before it loads `seq`, because a rebuild could
otherwise install a file newer than `seq`. The scan sits behind the public `DbIterator`, so its
contract does not change. With `DbIterator::new_resolved` and `RowEntryIterator::next_sync`, it
returns most rows without an await, and checks Tokio's cooperative budget once every 32 rows.

A scan that cached ranges hold only in part is a mixed scan. From one load of the index, it splits
its range into parts in key order: a cached part for each ready range that intersects it, and a
gap for each stretch that no ready range holds. It pins the layers of every cached part, loads
`seq`, and turns each cached part that fails the rules into a gap. A gap reads through the normal
path from a state view `V` (memtables and manifest) with `max_seq = seq`, and opens only when the
scan reaches it, so a scan that ends early or seeks past a gap never opens it. A gap that no cached
range holds is sampled when it opens, with the time to open it as the cost. `V` must hold every
version visible at `seq`, but flush and compaction keep only the versions that the latest state and
registered snapshots need. At the latest state nothing is registered at `seq`, so the scan captures
`V` before it loads `seq`, and then checks that the active memtable is still the one in `V`. The
SSTs and immutable memtables of `V` never change, so it keeps every version written before the
capture; a later write at or below `seq` went into the active memtable, which `V` shares unless a
freeze happened. After a freeze the scan tries again, up to three times, then uses the normal path.
A snapshot or transaction is registered at `seq`, so its view can be captured after `seq`. A scan
that no cached range touches uses the normal path, and a fully cached scan does not change.

A read in a snapshot or a transaction works the same way. It still loads the last committed
sequence number after it pins the layers, and reads at the smaller of that number and the
snapshot's sequence number `S`, as the normal path does. `Db::snapshot` and `Db::begin` take `S`
from the last committed sequence number, so `S` is committed. Deltas keep every version, so the
cursors and lookups filter with `S`. A transaction reads its own writes first. For a point read,
a put in its batch is the answer, a delete means absent, and a merge operand uses the normal path.
A scan uses the normal path if the batch has a key in the scan range. Conflict tracking of
serializable transactions does not change: it records the read key, or the requested scan range,
before the read, and takes nothing from the iterator. Automatic promotion samples these reads
like the others.

### Sequence number rules

A read pins layers `L`, then loads `seq` (for a snapshot, the smaller of it and `S`). It uses `L`
only if both rules hold:

- `seq >= L.file_seq`. Right after a load or a rebuild, `seq` can be lower for a short time. The
  file holds the state at `file_seq`, so without this rule a read could see an uncommitted write.
  The file has no older versions, so a snapshot older than the file uses the normal path.
- `seq <= L.active.sealed_at`. Otherwise a write after the seal can be missing from the delta.

The second rule is stronger than linearizability needs: without it, a read could return an older
state that was current during the read. It makes a cached read equal the normal read path at `seq`.

### Loading a range

1. Under the mutex, in this order: publish a loading range with an empty active delta for each part
   that no range holds; advance `publish_epoch`, then read `inserted_seq`, both `SeqCst`; register
   a snapshot at the larger of `inserted_seq` and the last committed sequence number; record its
   sequence number as the `file_seq` of the new ranges.
2. Scan each part from the snapshot into a new file, with `dirty: true` and `cache_blocks: false`.
   The scan reads ahead like a compaction: requests of about 2 MiB, up to 4 in flight for each
   SST (`load_fetch_bytes` and `load_fetch_tasks` in the tuning). A load then costs about one
   request for each 2 MiB, and its time follows its bytes, not the latency of the store. Stop on a
   row with an expire time, or when the file grows larger than the disk budget.
3. Under the mutex, if the range is still in the index, attach the file, with the snapshot sequence
   number as `file_seq`. Then apply the disk budget.

The order of step 1 makes sure that no write is lost. The writer and the load each update one
counter and then read the other, all with `SeqCst`, so at least one side sees the other. If the
writer sees the new `publish_epoch`, it finds the new range and puts the write in its delta.
Otherwise the load sees the writer's `inserted_seq`, so the write is in the snapshot. Both can
happen to one write. Every write at or below the snapshot is in the memtable and visible to the
load, so the writer skips it (step 3 of the write path); a merge operand in both would apply twice.
The snapshot is dirty because a write at `inserted_seq` can still be in flight; that write commits,
or the database closes. The snapshot also keeps compaction from dropping the versions that the load
reads.

### Rebuild

When the active delta is larger than a quarter of its file and than a minimum size (1 MiB by
default), the writer seals it under the mutex, makes it the frozen delta, and installs a new empty
active delta. A background task merges the frozen delta and the file with the merge operator,
drops tombstones, writes a new file, and installs it under the mutex. A failed rebuild removes the
range. A rebuild absorbs at least a quarter of the file, so write amplification is at most about 4.

### Disk budget and eviction

After each load and rebuild, if the files are larger than `max_disk_bytes`, the cache removes
ranges until they fit, least recently read first. With automatic promotion, it removes automatic
ranges first, lowest read density first, and explicit ranges last. It never removes the range that
a load just installed. A read counts its range's reads and recency on its own thread. Each thread
adds them to the range every 4096 reads or about every 10 ms, so eviction and promotion use counts
that are up to that far behind, and miss the last counts of a thread that stopped reading.

### Range cache files

A file holds the live rows of one range, sorted by key, with sequence number and create timestamp.
It holds no tombstones and no rows with an expire time. Integers are little endian.

```text
[blob run][pad] page 0 [blob run][pad] page 1 ... page N-1
fence region:   for each page: u64 page offset | u32 key length | first key
trailer (64 B): magic u64 | version u32 | page size u32 | page count u64 | row count u64 |
                fence offset u64 | fence length u64 | data bytes u64 | max seq u64
page: u16 row count n | u16 hint skip s | n x u16 row offset | n x u32 key hint | rows | padding
row:  u8 flags | u64 seq | i64 create_ts | key ref | value ref
ref:  inline: u32 length | bytes; out of line: u64 file offset | u32 length
```

Pages have a fixed size and are aligned. Long keys and large values go into the blob run before
their page. Only the fence index (the first key of each page) is in memory. Key hints let a point
read usually touch only the page header and one row. The cache never syncs files, and deletes a
file after its last reader. At open, it deletes only `<digits>.rcf` and `<digits>.rcf.tmp` files
in its directory. Files never outlive the process, so the format can change freely.

### Close

The batch writer closes the cache in its cleanup, on `Db::close` or when a background task fails,
fencing included. Close removes all ranges and rejects later loads.

### Automatic promotion

With `auto_promote` set, a background task chooses ranges from sampled reads. The defaults suit a
hot set that changes over tens of seconds, and need tuning on real workloads.

- **Sampling.** The read path samples one in `sample_one_in` of the reads that the cache could
  serve but no cached range holds. It times the read with the database's `SystemClock`, and sends
  the key and time to a bounded queue, which drops samples when full. The time is what a cache hit
  would save. A task, the promoter, reads the queue. Its errors never fail the database. The
  promoter loads an admitted candidate in a separate task, so it keeps reading the queue during
  the load. It starts no other load and evaluates no candidate until the load ends.
- **Candidates.** The promoter forms candidates of about `target_range_bytes` around sampled keys.
  It uses block boundaries from the SST index of the largest sorted run, or bounded scans when the
  database has segments or no sorted run. A candidate stops at, and extends to, cached neighbours.
- **Admission.** A candidate needs a minimum number of sampled reads, which rises as the budget
  fills. Its samples must span a minimum time, so one pass over the data never qualifies. If it
  does not fit, its read density (normal-path cost per second per byte, decaying with
  `heat_half_life`) must beat by a fixed factor each automatic range that it would evict.
- **Guards against churn.** Promotion never evicts an explicit range. Over a recently removed
  automatic range, only newer samples count. A failed promotion starts a backoff that doubles each
  time. A large automatic range rebuilds only if its served reads paid for the rebuild. Otherwise
  the cache removes it with the reason `not_worth_it`.

## Impact Analysis

SlateDB features and components that this RFC interacts with. Check all that apply.

### Core API & Query Semantics

- [x] Basic KV API (`get`/`put`/`delete`)
- [x] Range queries, iterators, seek semantics
- [x] Range deletions
- [x] Error model, API errors

### Consistency, Isolation, and Multi-Versioning

- [x] Transactions
- [x] Snapshots
- [x] Sequence numbers

### Time, Retention, and Derived State

- [x] Time to live (TTL)
- [x] Compaction filters
- [x] Merge operator
- [ ] Change Data Capture (CDC)

### Metadata, Coordination, and Lifecycles

- [ ] Manifest format
- [x] Checkpoints
- [x] Clones
- [x] Garbage collection
- [ ] Database splitting and merging
- [ ] Multi-writer

### Compaction

- [ ] Compaction state persistence
- [x] Compaction filters
- [ ] Compaction strategies
- [ ] Distributed compaction
- [ ] Compactions format

### Storage Engine Internals

- [x] Write-ahead log (WAL)
- [x] Block cache
- [x] Object store cache
- [ ] Indexing (bloom filters, metadata)
- [ ] SST format or block format

### Ecosystem & Operations

- [ ] CLI tools
- [ ] Language bindings (Go/Python/etc)
- [x] Observability (metrics/logging/tracing)

### Notes on the main interactions

- **Merge operator.** Deltas keep merge operands, reads resolve them with the normal iterators,
  and rebuilds fold them. A file keeps a merge operand with no base value, as compaction does.
- **TTL.** Expiry happens only at flush and compaction, so an expiring row changes without a
  write. The cache holds no expiring rows, and does not help with `Settings::default_ttl_millis`.
- **Transactions and snapshots.** A commit is a write batch, so the cache applies it. Reads in
  transactions and snapshots use the cache when their snapshot is not older than the file.
- **Compaction.** It does not invalidate the cache: only expiry and compaction filters change
  visible data. The builder rejects a cache with an embedded compactor that has a filter. A
  compactor in another process must not use one.
- **`DbReader`.** Not supported. A later RFC could give it a feed of writes, such as a WAL tail.

## Operations

### Performance & Cost

All numbers come from a laptop running macOS; not yet measured on Linux or on object storage. The
main benchmark has 2M rows with uniform 16-byte keys and 200-byte values, on a local filesystem
object store, in one sorted run of 4 SSTs. The cache holds the whole key range. Each cell is the
p50 latency on one task, then operations per second on 1, 8 and 16 tasks.

| Engine | Get hit | Get miss | 10-row scan | 100-row scan |
| --- | --- | --- | --- | --- |
| Normal path, 2 GiB block cache holding every block | 3.04 µs; 0.32/0.76/0.80M | 1.38 µs; 0.69/1.46/1.47M | 9.21 µs; 0.10/0.45/0.64M | 56.3 µs; 0.02/0.11/0.18M |
| Cached range, mmap | 0.33 µs; 2.57/17.8/32.7M | 0.29 µs; 2.94/22.4/42.7M | 1.17 µs; 0.79/3.97/5.70M | 6.00 µs; 0.15/1.00/1.68M |
| Cached range, mmap, `next_batch` | - | - | 1.04 µs; 0.88/4.10/5.83M | 4.46 µs; 0.20/1.24/1.98M |
| Cached range, positional reads | 0.79 µs; 1.21/1.04/0.90M | 0.75 µs; 1.28/1.04/0.90M | 2.25 µs; 0.44/0.65/0.56M | 10.5 µs; 0.09/0.32/0.28M |

- The normal path holds about 1.6 GB of anonymous memory for its block cache. The cache file lives
  in the page cache, which the kernel can reclaim.
- Positional reads stop scaling near 1M reads per second on this machine, at any number of tasks.
  A profile shows the time inside the `pread` system call. `Mmap` does not have this limit, and is
  the default.
- Reads count on their own thread and fold their counts into the metrics, so a metrics recorder
  costs little: 31.5M point reads/s on 16 tasks with one, 31.9M/s without. Shared per-read
  counters cost about a quarter (22.6M/s).
- Files take about the size of the cached data. A load reads its range once from the object store.
- Automatic promotion, from an empty cache: with room for all the data, 99% of reads were served
  from the cache within about 12 seconds. With a budget smaller than the data, 16-task point reads
  were about 1.2x to 1.3x faster than with a block cache with enough metadata space. A 64 MiB
  block cache looked much worse: its 16 MiB metadata share could not hold index and filter blocks.

When the feature is off, its code does not exist. Without a configured cache, a read or a write
checks one `Option`. With a cache, a write clones each entry and makes 3 atomic operations,
and a read of a key that is not cached makes one atomic load and one index lookup. The table gives
the median of 8 runs on one task, with 100,000 keys in the memtables and an in-memory object store.
The machine was busy during these runs, so differences of a few percent are noise. The last row
shows the cost of writes to cached keys; the table above gives the read numbers.

| Configuration | Put | Batch, per entry | Get | 100-row scan |
| --- | --- | --- | --- | --- |
| `main`, no range cache code | 9.09 µs | 436 ns | 1.38 µs | 55.5 µs |
| Feature on, cache not configured | 9.13 µs | 428 ns | 1.36 µs | 56.0 µs |
| Cache configured, keys not cached | 9.22 µs | 428 ns | 1.39 µs | 55.7 µs |
| All keys cached, writes go to deltas | 9.51 µs | 794 ns | - | - |

### Observability

- Configuration: `DbBuilder::with_range_cache`. New components: rebuild and promoter tasks.
- Metrics under `slatedb.range_cache.`: reads served, mixed scans and the gaps they open, retries
  and fallbacks to the normal path by reason, loads, rebuilds, bytes, removals by reason, and
  gauges for ranges, file bytes and delta bytes. Automatic promotion adds its own counters. The
  read-path counters, and `slatedb.db.request_count` for reads that the cache serves, are counted
  on each thread and added to the shared counters at the same folds as the ranges' reads, so a
  read writes no shared counter. They are exact only after a fold.
- Logging: a warning when a rebuild or a cache read fails.

### Compatibility

- Object storage: no change. Checkpoints and clones do not include the cache.
- Public API: additive, experimental and behind a cargo feature. Bindings do not change.

## Testing

- Unit tests cover the file format with both page sources, the searches, and the index. A
  property test compares cached scans with the normal iterator stack, field by field. Another
  compares cached reads at any sequence number from `file_seq` up with the whole history.
- Integration tests mostly compare cached reads with snapshot reads, under writes, rebuilds,
  eviction, close, fencing, races with loads, and automatic promotion. The cache serves snapshot
  reads too, so these reads carry a tracing option, which the cache never serves.
- A separate checker, written from this specification and kept outside this branch, checks
  concurrent gets and scans for linearizability with 2, 6 and 16 tasks. It also checks reads in
  snapshots and transactions against the state at their sequence number. It found no violation,
  and it caught deliberate bugs.
- Fail points pause a batch before its commit, and a load around its publish step. Not yet
  covered: deterministic simulation, and runs on Linux.

## Rollout

The branch has five commits. Commits 1 and 2 do not depend on the cache and can merge first.

1. Fix: keep the buffered entry in `MergeOperatorIterator::seek`. It affects normal scans on `main`.
2. Check an atomic flag before reading the close result in `check_closed`.
3. This RFC.
4. The range cache, behind the `range_cache` feature. For review, it can be split into:
   - the file format;
   - the cache core, with explicit `cache_range` and point reads;
   - rebuilds and the disk budget;
   - the scan path;
   - automatic promotion, which stays off by default.
5. `DbIterator::next_batch`. This commit is optional.

## Alternatives

- **A larger block cache, with RFCs 0037 and 0038.** A read still visits each source.
- **A row cache.** It cannot answer a scan, and prior art found it disappointing (see References).
- **The memtable and newer L0 SSTs as the delta.** This needs no write hook, but each read visits
  those sources again. This design could suit `DbReader`.
- **An external wrapper with a write hook.** The hook would make internal order a public contract,
  and the wrapper must wrap every read and write method.
- **Values in memory only.** The kernel cannot reclaim that memory.

## Open Questions

- Should the cache survive restarts?
- Should `cache_range` return before the load ends?
- Should `DbIterator` get a batched read, such as `next_batch` in commit 5?

## References

- [RFC 0021: Metrics](./0021-metrics.md)
- [RFC 0023: Cache manager](./0023-cache-manager.md)
- [RFC 0024: Segment-oriented compaction](./0024-segment-oriented-compaction.md)
- [RFC 0034: Local object mirroring](./0034-local-object-mirroring.md)
- RFC 0036 (draft): reader manifest sync, [PR #2186](https://github.com/slatedb/slatedb/pull/2186)
- RFC 0037 (draft): pinned SST metadata, [PR #2206](https://github.com/slatedb/slatedb/pull/2206)
- RFC 0038 (draft): SST index search, [PR #2254](https://github.com/slatedb/slatedb/pull/2254)
- RFC 0039 (draft): range deletes, [PR #2260](https://github.com/slatedb/slatedb/pull/2260)
- [2-Tree, CIDR 2023](https://www.cidrdb.org/cidr2023/papers/p57-zhou.pdf)
- [DataStax: configuring caches](https://docs.datastax.com/en/cassandra/3.0/cassandra/operations/opsConfiguringCaches.html)
- [MySQL 8.4 adaptive hash index](https://dev.mysql.com/doc/refman/8.4/en/innodb-adaptive-hash.html)

## Updates

## Appendix: Why reads see the right versions

Claim: when a read at `seq` uses layers `L` under the sequence number rules, `L` holds every
version of each key in the range that is visible at `seq`, and the read uses no other version.
For a snapshot, `seq` is at most the last committed sequence number that the read loaded after
`L`, so every step below holds for it as it holds for that number.

1. The writer puts a write `w` in its delta before a `SeqCst` store makes `w` committed. A read
   loads `seq` (`SeqCst`) after it loads `L`, so if `w <= seq`, it sees the write in the delta.
2. Seals and writes to deltas happen under one mutex, in sequence number order. So a delta sealed
   at `s` holds each write to its range up to `s` that the older layers lack, and none above `s`.
   The seal comes before any write above `s` commits, so a read with `seq > s` retries.
3. A file holds the state of its range at `file_seq`: for a load, through the counter pair in
   [Loading a range](#loading-a-range); for a rebuild, at the seal of the frozen delta. Deltas
   hold only versions above `file_seq`. Reads filter delta versions with `max_seq = seq`, and use
   the file only if `file_seq <= seq`, since it lacks the versions that an older `seq` needs.
4. A removal seals the active delta, under the mutex, at the last sequence number that the writer
   applied. A read that pinned the range earlier then needs `seq` at or below that number. A writer
   that saw the range before the removal finds it gone under the mutex, and applies nothing.
5. Only the writer task freezes a delta, so a freeze never races its unlocked check. An install
   replaces the file and the frozen delta, and keeps the active delta. Automatic promotion adds
   and removes ranges only through the same load and removal code.
6. A scan pins every range before it loads `seq`, and checks the rules for each. The ranges hold
   disjoint keys that cover the scan range, so their answers together are the answer at `seq`. A
   mixed scan's gaps hold the other keys. Their view holds every version at or below `seq`: a
   snapshot's versions stay because it is registered; at the latest state, the view was captured
   before `seq` was loaded, and no freeze happened in between (see the read path). The normal path
   reads it with `max_seq = seq`, so every part answers at `seq`, and so does the scan.
7. A write that fails after the memtable insert closes the database. So `seq` never reaches a
   version that never commits, and reads fail with `Closed`.
