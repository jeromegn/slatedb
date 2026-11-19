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
- [Appendix A: Prototype measurements](#appendix-a-prototype-measurements)
- [Appendix B: Why reads see the right versions](#appendix-b-why-reads-see-the-right-versions)

<!-- TOC end -->

Status: Draft

Authors:

* [jeromegn](https://github.com/jeromegn)

## Summary

This RFC adds a range cache. The range cache keeps a local, disposable copy of selected key
ranges. For each cached range it holds every live key, so a point read in the range is a definite
hit or a definite miss, and a scan in the range needs no other source.

The block cache caches where bytes are. The range cache caches what the answer is for a key range,
including the answer "no key". The two caches complement each other. The block cache does not
change.

The batch writer updates cached ranges in the same step that makes a write visible. The range
cache is off by default, behind the `range_cache` cargo feature and a builder option. When it is
off, it costs nothing.

The application chooses the cached ranges (stage 1), or automatic promotion chooses them from
sampled reads (stage 2, also off by default). Both stages are implemented in the branch
that accompanies this RFC.

## Motivation

A point read at the latest state visits each source that can hold the key: the memtables, every
L0 SST, and one SST per sorted run. A miss visits all of them. A scan merges all of them. Each
source costs a filter probe, an index search, a block lookup in the block cache, and an iterator.
RFC 0037 (draft) and RFC 0038 (draft) make each of these steps cheaper. They do not reduce the
number of sources.

Many workloads read a small, hot part of the key space again and again: a tenant's rows, a
queue's head, the index of a table. For these ranges, a copy that already holds the answer is much
cheaper than the LSM read path. An out-of-tree prototype measured this on 2M rows (16-byte keys,
200-byte values). It compared the prototype with SlateDB 0.18.1, which had its own local mirror
and a 64 MiB block cache:

| Operation | Range cache | SlateDB 0.18.1 |
| --- | --- | --- |
| Point read, 1 task | 0.5 µs | 16 to 17 µs (3.1 µs with all data in SlateDB's memory cache) |
| 100-row scan, 1 task | 3.5 µs | 135 to 148 µs |
| Point reads per second, 16 tasks | 21 M | 0.20 to 0.25 M |

The prototype ran on macOS, against a MinIO server on the LAN and a local filesystem, with one run
for each configuration. Treat the numbers as an indication, not a benchmark. Appendix A has the
details and the costs that the prototype measured.

A point-only row cache is not enough. Prior art found row caches disappointing (see
[Alternatives](#alternatives)). A row cache cannot answer a scan, and it cannot prove that a key is
absent without a lookup in the LSM. The range cache can do both, because it is complete for each
range that it holds.

The prototype wrapped SlateDB from the outside. About half of its code existed only for that
reason. It had no write hook, so it needed a gate around each write, marks on keys with a write in
flight, bounded waits on those marks, and its own batch and iterator types. Inside SlateDB, none of
this is necessary. One task, the batch writer, assigns sequence numbers, writes to the memtable,
and then advances the last committed sequence number. The range cache can apply a write between
the memtable insert and that advance.

## Goals

- Serve point reads and ascending scans at the latest committed state from cached ranges.
- Give the same results as the normal read path, for every value, sequence number and create
  timestamp.
- Keep cached ranges up to date with puts, deletes, merges, write batches and transaction commits,
  in the batch writer.
- Cost nothing when the feature is off, and next to nothing for writes and reads of keys that are
  not cached.
- Bound the local disk that the cache uses, and bound its local write amplification.
- Keep the block cache as it is. Loads must not fill the block cache.
- Let users read cache files with memory maps or with positional reads.

## Non-Goals

- Replace the block cache, the object store cache or the object store mirror.
- Persist the cache across restarts. The cache starts empty each time the database opens.
- Serve reads in snapshots or transactions, dirty reads, reads with `DurabilityLevel::Remote`, or
  descending scans. These reads use the normal path.
- Support `DbReader` in the first version. A reader has no write path to hook.
- Turn automatic promotion on by default. It is a second stage, with its own option, and it
  stays off until the defaults have been tuned on real workloads.

## Design

### Names

This RFC uses these names. They avoid "segment", which RFC 0024 uses for key-space partitions, and
"run", which sorted runs use.

| Name | Meaning |
| --- | --- |
| Range cache | The feature and the per-database component (`RangeCache`). |
| Cached range | A half-open key range `[start, end)` that the cache holds completely. |
| Range cache file | An immutable local file with the rows of one cached range (`.rcf`). |
| Delta | The versions that the writer applied to a cached range after its file was written. |
| Load | Copy a key range from a snapshot into a new range cache file. |
| Rebuild | Merge a delta into the range cache file, which gives a new file. |
| Remove | Drop a cached range: on request, for space (evict), or after an event it cannot follow. |

### Configuration and API

The feature is behind the `range_cache` cargo feature. It is off by default. The feature is
experimental: its API may change in any release. With the feature, the builder takes the options:

```rust
#[non_exhaustive]
pub struct RangeCacheOptions {
    pub root_folder: PathBuf,               // the cache owns this directory
    pub max_disk_bytes: u64,                // default 1 GiB
    pub page_source: RangeCachePageSource,  // Mmap (default) or PositionalRead
    pub auto_promote: Option<AutoPromoteOptions>, // default None: off (stage 2)
    #[doc(hidden)]
    pub tuning: RangeCacheTuning,           // internal, see below
}

#[non_exhaustive]
pub struct AutoPromoteOptions {             // Default
    pub target_range_bytes: u64,            // default 64 MiB
    pub sample_one_in: u32,                 // default 64
    pub heat_half_life: Duration,           // default 10 s
}

#[non_exhaustive]
pub enum RangeCachePageSource { Mmap, PositionalRead }

#[doc(hidden)]
#[non_exhaustive]
pub struct RangeCacheTuning {               // Default
    pub page_size: usize,                   // default 4096
    pub min_rebuild_delta_bytes: usize,     // default 1 MiB
    // and the promotion parameters in "Parameters and defaults" below
}

impl DbBuilder {
    pub fn with_range_cache(self, options: RangeCacheOptions) -> Self;
}

impl Db {
    /// Loads the parts of `range` that no cached range holds.
    pub async fn cache_range<T: ByteRangeBounds + Send>(&self, range: T) -> Result<(), Error>;
    /// Removes each cached range that overlaps `range`.
    pub fn uncache_range<T: ByteRangeBounds>(&self, range: T) -> Result<(), Error>;
    pub fn clear_range_cache(&self) -> Result<(), Error>;
    pub fn cached_ranges(&self) -> Vec<CachedRangeInfo>;
}
```

`RangeCacheOptions`, `AutoPromoteOptions`, `RangeCachePageSource`, `CachedRangeInfo` and
`RangeCacheTuning` are `#[non_exhaustive]`, so later versions can add fields and variants without
breaking callers. Code outside SlateDB starts from `RangeCacheOptions::new(dir)` and
`AutoPromoteOptions::default()` and assigns fields, and a `match` on `RangeCachePageSource` needs a
wildcard arm. `RangeCacheTuning` holds internal parameters for tests and benchmarks: the page size,
the rebuild threshold and the promotion parameters other than the three in `AutoPromoteOptions`.
It is hidden from the documentation and is not a stable API: its fields can change or disappear in
any release.

The first version uses a builder method, not a `Settings` field (see
[Open Questions](#open-questions)). `cache_range` returns `ErrorKind::Invalid` if the database has
no range cache, if the range holds rows with an expire time, or if a part that it loads is larger
than the disk budget. Parts that loaded before the error stay cached.

### Structure

```text
RangeCache
 ├─ index: ArcSwap<BTreeMap<start, Arc<CachedRange>>>   readers load it without a lock
 ├─ state: Mutex<..>                                    changes, and writes to a non-empty cache
 └─ CachedRange [start, end)
     └─ layers: ArcSwap<Layers>                         replaced as a whole
         ├─ active delta   KVTable + sealed_at          receives writes
         ├─ frozen delta   KVTable + sealed_at          only during a rebuild
         ├─ file           RangeFile                    None while the range loads
         └─ file_seq       the file holds the range at this sequence number
```

A delta is a `KVTable`, the type of the memtable. It keeps every version, keyed by key and
sequence number. Reads at a sequence number `seq` use the newest version at or below `seq`.
`sealed_at` is `u64::MAX` while the delta receives writes. When the cache stops applying writes to
a delta, it seals the delta at the last sequence number that it applied.

### Write path

`DbInner::write_batch` runs on the batch writer task. It extracts the entries of a batch, appends
them to the WAL buffer, and puts them in the memtable. Then `track_recent_committed_*` advances the
last committed sequence number, which makes the batch visible.

With a configured cache, the writer gives the entries to the cache after the memtable insert and
before that advance:

1. Advance `inserted_seq` to the batch sequence number, then load `publish_epoch`. Both are
   `SeqCst`.
2. Load the index. If no entry falls in a cached range, return. This takes no lock.
3. Take the cache mutex. Find the cached range of each entry in the index again.
4. Put each covered entry in the active delta of its range. This includes tombstones and merge
   operands.
5. If an entry has an expire time, remove its range. Seal the delta at `seq - 1`, because the
   delta does not hold this write.
6. If an active delta is larger than its rebuild threshold, freeze it (see [Rebuild](#rebuild)).

The writer puts the entries in the memtable before step 1. Steps 1 and 2 pair with the order in
which a load publishes a range and picks its snapshot. Together they make step 2 safe without the
mutex (see [Loading a range](#loading-a-range)). The cost is one clone of each entry and two
`SeqCst` atomic operations for each write, when the cache is configured.

A write is fatal for the database if it fails after the memtable insert. The writer task then
stops, and its cleanup closes the cache. So a version in a delta is always committed, or the
database is closed.

### Read path

`DbInner::get_key_value_with_options` and `DbInner::scan_with_options` ask the cache first. The
cache serves a read only if all of these are true:

- the durability filter is `Memory`, `dirty` is false, and there is no filter context and no
  tracing option;
- for a scan, the order is ascending, and cached ranges hold the whole scan range: one range, or
  adjacent ranges where each starts at the end of the one before;
- the range is not loading, and the [sequence number rules](#sequence-number-rules) hold.

Otherwise the read uses the normal path. Snapshot and transaction reads never reach this code.

A point read takes no lock and does no await on the common path:

1. Load the index, find the range, load its layers.
2. Load the last committed sequence number `seq`.
3. Check the sequence number rules. If they fail, retry from step 1, at most three times.
4. Find the newest version at or below `seq` in the active delta, then in the frozen delta, then
   in the file.
5. A value is a hit. A tombstone or no version is a definite miss. A merge operand goes to the
   slow path.

The slow path builds a `DbIterator` over the pinned layers, with `max_seq = seq` and the
database's merge operator. This is the iterator stack of the normal read path. So merge operands
resolve in the same way, and `KeyValue` fields are the same.

### Scan path

A scan that the cache serves does not use the general iterator stack for each row. A
`CachedScan` walks the pinned layers directly:

- A file cursor walks the rows of the range file in order. When it enters a page, it copies the
  page into one reference-counted buffer. It compares keys in that buffer and returns each row as
  two slices of it, so a scan makes one allocation per page, not per row. See
  [Range cache files](#range-cache-files) for what a kept row keeps in memory.
- A cursor for each delta peeks the next key and its newest version at or below `seq`. It skips
  newer versions and older versions of the same key.
- Each step takes the smallest key of the three cursors. On a tie, the newest layer wins and the
  older layers skip the key. A tombstone skips the key. A value is the row.
- A merge operand needs the merge operator and the older versions of its key. For such a key, the
  scan reads that key alone through the normal iterator stack at `seq`. It uses the stack that
  the normal path uses for the same scan: the scan stack, or the point-lookup stack for a scan of
  one key. The two stacks give the same bytes but different entry kinds for a merge over a
  tombstone. Nothing else needs the normal stack: cached ranges hold no expiring rows, and files
  hold only values and base-less merge operands.

A scan can span adjacent cached ranges. Automatic promotion makes such ranges, since its
candidates stop at, and extend to, their cached neighbours. The read pins the layers of every
range that the scan range touches, then loads `seq` once and checks the sequence number rules for
each of them. If one fails, it retries, as a point read does. Each range's layers then hold every
version visible at `seq` for its keys (Appendix B applies to each range), and the ranges hold
disjoint keys, so the scan is the concatenation of their parts at one sequence number. It keeps
one reference per range, and opens the cursors of a range only when the scan reaches it. A scan
with an unbounded end pins every range after its start: pinning a range costs one reference
count. Pinning lazily, when the scan reaches a range, would not be safe: a rebuild that installs
a file newer than `seq` in between would leave no layers for `seq`.

`CachedScan` is a `RowEntryIterator` behind the public `DbIterator`, so the scan keeps the same
type and contract: bounds, ascending order, `seek` and its errors, and the fields of each row. Two
small changes to shared code connect it:

- `DbIterator::new_resolved` builds a `DbIterator` over a source that already returns one
  resolved entry per key. It skips the merge, filter and merge-operator layers.
- `RowEntryIterator::next_sync` is a provided method that returns `None` ("call `next`"). The
  `Box<dyn RowEntryIterator>` impl forwards it. `DbIterator` calls it only for a source from
  `new_resolved`, so other iterators do not pay for it. `CachedScan` returns most rows from it
  without an await, which saves two boxed futures for each row.
- For a source from `new_resolved`, `DbIterator` consumes Tokio's cooperative budget once every 32
  rows instead of once per row. A cached row costs well under 100 ns, so the scan still yields to
  the scheduler at least every few microseconds. Other iterators keep the check on each row.

These changes are behind the `range_cache` feature.

With positional reads, a scan reads whole pages into a buffer. The first read is one page, and
each refill doubles the read up to 64 KiB. So a 10-row scan makes one system call, and a 100-row
scan of 4 KiB pages makes about three. A point read reads one page.

### Sequence number rules

A read that pinned layers `L` at sequence number `seq` uses them only if:

- `seq >= L.file_seq`. The file holds the state at `file_seq`. A smaller `seq` can occur for a
  short time after a load or a rebuild, before the writer commits the newest write.
- `seq <= L.active.sealed_at`. Otherwise a write after the seal can be missing from the delta.

When both rules hold, the layers hold every version that is visible at `seq`. So a cached read
gives exactly the answer of the normal path at `seq`. Appendix B gives the argument.

The two rules have different roles. The first rule is necessary for correctness: without it, a
read could return a write that is not committed yet. The second rule is stronger than
linearizability needs. A reader only pins layers that were current when it loaded them. Without
the second rule, a read can return the state at an earlier sequence number, which was still the
current state at some moment during the read. The second rule makes the answer the state at
`seq` exactly. This keeps the cache equal to the normal path, also if a later version exposes the
read sequence number. The cost is one atomic load for each read.

### Loading a range

`cache_range` runs these steps:

1. Under the cache mutex, in this order:
   1. Compute the parts of the range that no cached range holds. Publish a loading range with an
      empty active delta for each part in the index.
   2. Advance `publish_epoch`, then read `inserted_seq`. Both are `SeqCst`.
   3. Register a snapshot at `last_seq`, the last sequence number that the writer assigned.
2. For each part, scan the snapshot with `dirty: true` and `cache_blocks: false`. Write the
   resolved rows to a new range cache file. Stop with an error on a row with an expire time, or
   when the file grows larger than the disk budget.
3. Under the cache mutex: if the range is still in the index, attach the file with `file_seq` set
   to the snapshot sequence number. The range now serves reads. Then apply the disk budget.

Each write is in the snapshot or in the delta, and no write is lost. The writer's unlocked check
(write path steps 1 and 2) and the load's publish (steps 1.1 to 1.3 above) form a store-buffering
pair: each side updates one counter and then loads the other, all with `SeqCst`. These operations
are in one total order, so at least one side sees the other:

- If the writer's load of `publish_epoch` sees the load's update, it synchronizes with it. So the
  writer's index load finds the new range. The writer then waits for the mutex and applies the
  write to the delta, after the load released the mutex.
- Otherwise the load's read of `inserted_seq` sees the writer's update, which the writer made
  after its memtable insert. So the load synchronizes with that insert. `last_seq` is also at or
  above the write, so the scan includes it.

The argument does not depend on the memory orderings of `ArcSwap`.

A write that gets its sequence number after the load read `last_seq` falls in the first case. The
snapshot uses `dirty: true`, because `last_seq` can be one above the last committed sequence number
while a write is in flight. Such a write is committed, or the database closes.

The order of step 1 matters. An earlier version picked the snapshot before it published the range.
A write could then get its sequence number after the snapshot, check the index before the publish,
skip the delta, and commit. Neither the file nor the delta held it. A fail point between the two
steps reproduced this. The regression test pauses the load at each step of the registration.

If a version is in both the snapshot and the delta, both copies have the same sequence number, and
the delta wins. The snapshot registration keeps compaction from dropping versions at `last_seq`
while the load runs.

### Rebuild

A delta grows with each write to its range. When the active delta is larger than
`max(min_rebuild_delta_bytes, file size / 4)`, the writer freezes it, still under the mutex:

1. Seal the delta at the sequence number of the current write.
2. Replace the layers: a new empty active delta, the old delta as the frozen delta, the same file.
3. Start a background task.

The task merges the frozen delta and the file through a `DbIterator` with the merge operator. It
drops tombstones and writes a new file. Then, under the mutex, it installs the new file and drops
the frozen delta, if the range still has the same frozen delta. A failed rebuild removes the range.

Each rebuild rewrites the whole file to absorb a quarter of its size or more. So the local write
amplification of rebuilds is at most about 4, plus 1 for the load. The `rebuild_bytes` metric
measures it.

A file can hold a merge operand with no base value below it. It keeps the operand kind, as
compaction does. A later read then gives the same result as the LSM.

### Disk budget and eviction

After each load and rebuild, if the files of cached ranges are larger than `max_disk_bytes`, the
cache removes ranges until the files fit. A load larger than the budget fails.

Without automatic promotion, it removes the least recently read ranges first. Recency uses an
access clock. Each load and rebuild advances the clock, and each read copies it to its range only
if the value changed. So a read does not write a shared cache line on each call.

With automatic promotion, it removes automatic ranges first, by increasing read density (see
[Automatic promotion](#automatic-promotion-stage-2)), and explicitly cached ranges only after all
automatic ones, least recently read first. It never removes the range that a load just
installed.

### Range cache files

A file holds the live rows of one range, sorted by key, with sequence number and create timestamp.
It never holds tombstones or rows with an expire time.

```text
[blob run][pad] page 0 [blob run][pad] page 1 ... page N-1
fence region:   for each page: u64 page offset | u32 key length | first key
trailer (64 B): magic u64 | version u32 | page size u32 | page count u64 | row count u64 |
                fence offset u64 | fence length u64 | data bytes u64 | max seq u64

page: u16 row count n | u16 hint skip s | n x u16 row offset | n x u32 key hint | rows | padding
row:  u8 flags | u64 seq | i64 create_ts | key ref | value ref
ref:  inline u32 length | bytes, or out of line u64 file offset | u32 length
```

Pages are fixed size and aligned. Long keys and large values go out of line, into a blob run before
the page, so each row fits in a page. Only the fence index (the first key of each page) is in
memory. The flags keep the merge-operand kind and whether the row has a create timestamp.

Lookups are bound by memory latency: a random page of a large file is not in the CPU caches, so
each row that a binary search reads costs a cache miss. Two indexes keep the misses down:

- In memory, next to the fence keys, eight bytes of each fence after the prefix that all fences
  share, as one array of `u64`. The search over pages compares these first, and compares whole
  fence keys only when they tie.
- In each page, a hint for each row: four bytes of its key after the `s` bytes that all keys of
  the page share (at most 60), big-endian. The search within a page compares hints, which sit in
  the first cache lines of the page, and reads a row only when the hints tie. A point read then
  touches the page header and one row.

On the benchmark below, the two indexes cut a cached point read on one task from 0.54 µs to
0.42 µs and then to 0.38 µs, and a cached miss from 0.54 µs to 0.29 µs.

A `PageSource` gives access to the bytes of a file: the whole file if it is mapped, or positional
reads into a caller's buffer. The first version has two sources: a memory map (the default), and
positional reads (`pread`). Neither lends out the file's bytes. A returned `Bytes` never keeps a
file, a mapping or a read-ahead buffer alive, so a deleted file leaves the disk when its last
reader is done, whatever the application keeps.

- A point read copies the key and the value of its row into one allocation.
- A scan copies each page it enters, from the first row that it reads to the end of the page,
  into one reference-counted buffer, and returns rows as slices of it. A row that the
  application keeps therefore keeps that buffer (at most one `page_size`, 4 KiB by default) in
  memory, as a value from an SST keeps its decoded block. Keeping
  one row of each page of a range keeps about the size of the range file in memory. An application
  that keeps rows for a long time and wants them small can copy them.
- Out-of-line keys and values (the ones that do not fit in a page) are copied on their own, so they
  never keep a page.

An earlier version copied each row on its own. One buffer per page saves an allocation and a free
per row, and lets the cursor compare keys in the buffer without decoding them again.
[Performance & Cost](#performance--cost) gives the effect of each step.

The cache does not sync files. It writes `<id>.rcf.tmp`, renames it to `<id>.rcf`, and deletes the
file when the last reader drops it. At open, it deletes old `<digits>.rcf` and `<digits>.rcf.tmp`
files in its directory, and only those.

### Close and fencing

The batch writer closes the cache in its cleanup. The cleanup runs on `Db::close`, and when a
background task fails, which includes fencing. Close removes all ranges and rejects later loads.
Readers that still hold layers finish, and the last of them deletes the files.

### Automatic promotion (stage 2)

With `auto_promote: Some(..)`, the cache chooses ranges by itself. The design ports the ideas of
the prototype's profiler and admission control, not its code.

**Signal.** A point read or an ascending scan that the cache could serve (latest committed state,
no filter context) but that no cached range holds takes the normal path. One such read in
`sample_one_in` is sampled: the read path times it with the database's `SystemClock`, and sends
the key (the start key for a scan) and the time to a bounded queue with `try_send`. A full queue
drops the sample. That is all the read path does. An unsampled read pays one draw from `DbRand`.

The weight of a sample is the time that the normal path took, at least `min_read_cost`. Time
measures what the cache saves: it grows with the number of sources a read visits, with block
cache misses and with object store latency, and it needs no new plumbing in the reader. Counting
object store requests instead would miss the cost of reads that hit the block cache but visit
many sources, which the cache also removes. The floor keeps the fastest reads from counting as
free, and makes a mock clock (which does not move during a read) give equal weights, so tests are
deterministic. The `SystemClock` keeps the timing compatible with deterministic simulation.

**Promoter.** A background task, `range_cache_promoter`, runs as a `MessageHandler` on the
database's `MessageHandlerExecutor`, with the queue as its channel and a ticker of
`evaluate_interval`. It also evaluates when a sample arrives and an evaluation is due, because the
dispatcher serves queued messages before ticks. `Db::close` shuts the task down, and a failed
background task (fencing included) cancels it with the others. A load that the promoter runs
checks every 1024 rows that the database is open, so shutdown does not wait for a large load.
Promotion failures are counted and logged. They never fail the database.

**Reservoir.** The promoter keeps the samples of a window of four heat half-lives, in 32 slots.
Each slot keeps a uniform random subset of its samples (reservoir sampling, at most
`reservoir_capacity / 32`), and their count. So the window covers the same time at any read rate,
and estimates scale each kept sample by its slot's count. The evidence for a key range is: the
estimated sampled reads in it, the time between its first and last sample, and its normal-path
reads and cost per second, decayed with `heat_half_life`.

**Candidates.** Each evaluation draws up to `max_candidates_per_evaluation` seed keys from the
newest samples, weighted by cost, and skips seeds that a cached range holds. For a new seed it
forms a candidate of about `target_range_bytes`:

- From the SST index, if the database is not segmented and has a sorted run. In the largest
  sorted run, it reads the index of the SST that holds the key (through the metadata cache), and
  grows a window of blocks around the key, alternating sides, until it holds the target. If the
  SST is used up on a side, it adds whole neighbouring SSTs by their size. The bounds are block
  boundaries. No data is read.
- Otherwise from bounded scans: a forward scan from the key and a reverse scan before it, each up
  to half the target, through the reader with `cache_blocks: false`. At most one scan-formed
  candidate per evaluation.

Either way the candidate stops at the neighbouring cached ranges, and reaches them when it runs
into them, so cached ranges stay adjacent and scans can span them.

**Admission.** A candidate:

1. needs at least the evidence threshold of sampled reads, and two kept samples. The threshold
   rises linearly from `min_evidence` to `full_evidence` as the budget fills from half to full.
   If an automatic range that overlaps the candidate was removed within `ghost_ttl` (a ghost),
   only samples after the removal count: re-promotion needs fresh evidence;
2. needs samples that span at least `min_reuse_span`. A range that one pass reads once never
   qualifies, since caching it would not pay. This is what keeps a sequential pass from churning;
3. is admitted if it fits in the budget;
4. otherwise is admitted only if its density beats, by `admission_margin`, the density of every
   automatic range that eviction would remove (lowest density first) to make room. Density is the
   normal-path cost per second per byte: for a candidate, its decayed cost rate over its
   estimated size; for a cached range, its decayed rate of served reads times the cost per read
   measured at its admission, over its file size. Promotion never displaces explicitly cached
   ranges.

A rejected candidate is remembered for `candidate_ttl`. A later seed inside it re-runs admission
from the reservoir, at most every `candidate_recheck`, without a new index read or scan, so it gets
in as soon as what it competes with cools down. An admitted candidate loads through the same
`register` and `load` path as `cache_range`, so every correctness argument above holds. A new
automatic range starts with the read rate of its evidence, so it is not the first one evicted.

**Heat.** Each cached range counts the reads that it serves, in steps of 16 per thread, so a hot
range's counter is not written by every read. The promoter and the budget turn the count into a
rate with exponential decay (`heat_half_life`). Explicit ranges have heat too. They use the
average cost of sampled reads as their cost per read.

**Write amplification.** When the active delta of an automatic range reaches its rebuild
threshold, the writer checks, under the mutex, whether the rebuild pays: the reads that the range
served since its file was written, times its cost per read, must be at least the size of the file
times `rebuild_cost_ns_per_byte`. If not, it removes the range with the reason `not_worth_it`.
Files smaller than four times `min_rebuild_delta_bytes` always rebuild. A range that is read hot
keeps rebuilding however hard it is written. No range is ever removed because its delta is the
largest: deltas are bounded by rebuilds, and there is no separate delta memory cap. Explicit ranges
always rebuild.

**Ghosts and backoff.** The cache remembers each automatic range that it removes (except on
close) for `ghost_ttl`. A removal that signals a failed promotion (`not_worth_it`,
`expiring_write`, `error`) also starts a backoff over the range: no candidate forms there for
`backoff_base`, doubling with each failure up to 2^9 times, and the level drops by one every eight
base periods.

**Parameters and defaults.** The first three are fields of `AutoPromoteOptions`. The others are
fields of `RangeCacheTuning`, which is not a stable API.

| Option | Struct | Default | Role |
| --- | --- | --- | --- |
| `sample_one_in` | `AutoPromoteOptions` | 64 | read path sampling |
| `target_range_bytes` | `AutoPromoteOptions` | 64 MiB | candidate size, in SST bytes |
| `heat_half_life` | `AutoPromoteOptions` | 10 s | decay of rates; the reservoir holds four half-lives |
| `evaluate_interval` | `RangeCacheTuning` | 250 ms | promoter cadence |
| `max_candidates_per_evaluation` | `RangeCacheTuning` | 2 | index reads or scans per evaluation |
| `min_evidence`, `full_evidence` | `RangeCacheTuning` | 8, 32 | sampled reads needed, at half and at full budget |
| `min_reuse_span` | `RangeCacheTuning` | 2 s | samples must span this long |
| `admission_margin` | `RangeCacheTuning` | 2.0 | density ratio needed to evict |
| `ghost_ttl` | `RangeCacheTuning` | 60 s | fresh evidence after a removal |
| `candidate_ttl`, `candidate_recheck` | `RangeCacheTuning` | 30 s, 1 s | memory of rejected candidates |
| `backoff_base` | `RangeCacheTuning` | 10 s | wait after a failed promotion |
| `reservoir_capacity` | `RangeCacheTuning` | 8192 | samples kept |
| `queue_capacity` | `RangeCacheTuning` | 4096 | samples in flight from the read path |
| `min_read_cost` | `RangeCacheTuning` | 1 µs | floor of a sample's weight |
| `rebuild_cost_ns_per_byte` | `RangeCacheTuning` | 5 | estimated rebuild cost, for the payback check |

The defaults suit a database whose hot set changes over tens of seconds. The prototype used a 2 s
half-life for its benchmarks. These values need tuning on real workloads before stage 2 can be on
by default (see [Open Questions](#open-questions)).

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

### Notes on each interaction

- **Merge operator.** Deltas keep merge operands as versions. Reads resolve them through the normal
  iterators. A rebuild folds them, and keeps the operand kind when no base value exists.
- **TTL.** `main` applies expiry only when it flushes a memtable and when it compacts. Reads return
  an expired value until then. The visible state of an expiring key thus changes without a write,
  and the cache cannot follow it. The first version does not cache rows with an expire time. A
  load of a range with such a row fails, and an expiring write removes its range. With
  `Settings::default_ttl_millis`, every write expires, so the cache is not useful. See
  [Open Questions](#open-questions).
- **Tombstones.** Deltas keep tombstones, so a delete is a definite miss at once. Files never hold
  tombstones.
- **Iterators.** Cached scans return the public `DbIterator`. Feature-gated additions to shared
  code connect the scan path: `DbIterator::new_resolved`, which also checks the cooperative
  budget every 32 rows for its source, and the provided method `RowEntryIterator::next_sync` (see
  [Scan path](#scan-path)). The work also fixes a seek bug in
  `MergeOperatorIterator` that affects normal scans on `main` (see [Testing](#testing)).
  `DbIterator::next_batch`, a batched form of `next`, is an optional, separate addition behind
  the feature (see [Performance & Cost](#performance--cost)).
- **Closed check.** Every read calls `DbInner::check_closed`. It cloned a `tokio` watch receiver
  and took its read lock, so concurrent reads wrote the same cache lines. A plain upstream fix,
  not gated: an atomic flag that the close result sets before it becomes visible, which reads
  check first. It is a separate PR (PR 0b), useful without the cache.
- **Transactions and snapshots.** A commit is a write batch, so the cache applies it. A conflict
  writes nothing. Reads in a transaction or a snapshot use the normal path. A load holds a snapshot
  registration, which holds back compaction retention like any snapshot.
- **Sequence numbers.** The cache relies on one rule: the writer advances the last committed
  sequence number after the memtable insert. User-supplied sequence numbers keep this rule.
- **Checkpoints and clones.** The cache is local and holds no state that the database lacks. A
  checkpoint or a clone does not include it. A clone opens with an empty cache.
- **`DbReader`.** Not supported. A reader has no write path. A reader could use the cache if a
  later RFC gives it an ordered feed of writes, for example a WAL tail with RFC 0036 (draft).
- **Compaction.** Compaction does not change the cache, and it does not invalidate it. This is a
  difference from the block cache, whose blocks die with their SST. Compaction can change visible
  data in two cases only: expiry (see TTL) and compaction filters.
- **Compaction filters.** A filter can drop or change rows without a write. The builder rejects a
  range cache together with an embedded compactor that has a filter. A compactor in another
  process with a filter is not supported, and the documentation says so.
- **Range deletes (RFC 0039, draft).** A range tombstone is a new `ValueDeletable` variant. The
  cache matches that enum exhaustively, so the new variant does not compile until the cache
  handles it. The simple, sound first answer is to remove each cached range that a range
  tombstone overlaps. A later version can keep range tombstones in deltas.
- **Block cache.** Loads scan with `cache_blocks: false`, so they do not evict hot blocks. Reads
  that the cache serves do not touch the block cache. The block cache keeps serving the rest of the
  key space.
- **Object store cache and mirror.** Loads read SSTs through the table store, so they use the
  mirror or the object store cache when one is configured. Nothing else changes.
- **Garbage collection.** None. Cache files are local, and the cache deletes them itself. A load
  holds its iterators only for the duration of the load.
- **WAL.** None. WAL replay runs before the database can hold a cached range.
- **Fencing and close.** See [Close and fencing](#close-and-fencing). Before a fenced writer
  learns that it is fenced, its own reads see its own writes, from the memtable and from the cache
  alike.
- **Error model.** Two new `SlateDBError` variants, `RangeCacheNotConfigured` and
  `RangeCacheLoadRejected`, map to `ErrorKind::Invalid`. Local I/O errors map to the existing
  `IoError`. A point read that fails on a cache file removes the range and uses the normal path.
  A scan that fails on a cache file returns the error, as a scan returns an object store error.
- **Segments (RFC 0024).** None for cached ranges, which are independent of segments. Automatic
  promotion forms candidates from SST indexes only in a database without segments; a segmented
  database uses bounded scans.
- **Background tasks.** Automatic promotion adds one task to the database's
  `MessageHandlerExecutor`. `Db::close` shuts it down, and a failed task (fencing included) cancels
  it with the others. The read path changes only to sample reads that the cache could have served,
  when automatic promotion is on.

## Operations

### Performance & Cost

When the feature is off, the code does not exist. When the feature is on and the builder does not
configure a cache, a write or a read checks one `Option`.

When the cache is configured, a write clones each entry for the memtable insert, makes two
`SeqCst` atomic operations and one atomic load, and looks up each entry in the index if the cache
holds a range. Only a write to a cached range takes the mutex. A read of a key that no range
holds makes one atomic load and one index lookup, with no allocation.

A micro-benchmark measured this overhead (see [Testing](#testing)). It ran on a shared machine
with other heavy work, so treat the numbers as an indication:

Each cell is the median of 8 interleaved runs, on one task, on an in-memory object store with
100,000 keys of 17 bytes and values of 200 bytes. A put includes the hop to the batch writer task.
The scan is a bounded 100-row scan. The first five rows keep the data in the memtables. The last
two rows flush it to an L0 SST first, so the normal path reads it through the block cache.

| Configuration | Put | Batch, per entry | Get | 100-row scan |
| --- | --- | --- | --- | --- |
| `main`, no range cache code | 9.09 µs | 436 ns | 1.38 µs | 55.5 µs |
| Feature on, cache not configured | 9.13 µs | 428 ns | 1.36 µs | 56.0 µs |
| Cache configured, no cached range | 9.20 µs | 426 ns | 1.38 µs | 55.7 µs |
| Cache configured, keys not cached | 9.22 µs | 428 ns | 1.39 µs | 55.7 µs |
| All keys cached, writes go to deltas | 9.51 µs | 794 ns | 0.59 µs | see below |

With the feature on and no cache configured, every column is within the spread between runs.
With a cache configured, a single put costs about 0.1 µs (1.3%) more: the entry clone and the two
`SeqCst` operations of the write path. Batches amortize this cost, and reads and scans of keys that
are not cached are within the spread. Writes to cached keys cost more, because each write goes to a
delta and rebuilds run in the background.

A second micro-benchmark measured reads of the same 100,000 rows in three places: the memtable,
one L0 SST, and one sorted run. Cached rows were loaded into a range cache file first, so the cached
columns are the same for each place. "Delta" overwrites 10% of the keys after the load, so the
active delta holds 10,000 versions. "Before" is the cached scan through the general iterator stack,
before the scan path. Each cell is the median of 5 interleaved runs on one task. The machine was
mostly quiet (load average about 2).

| Data on `main` | Read | `main` | Cached, mmap | Cached, pread | Cached + delta | Before |
| --- | --- | --- | --- | --- | --- | --- |
| Memtable | Get | 1.25 µs | 0.37 µs | 0.75 µs | 0.57 µs | 0.43 µs |
| Memtable | 10-row scan | 6.6 µs | 1.8 µs | 2.7 µs | 2.1 µs | 7.3 µs |
| Memtable | 100-row scan | 42.1 µs | 10.8 µs | 13.6 µs | 11.7 µs | 47.2 µs |
| L0 SST | Get | 4.46 µs | 0.37 µs | 0.74 µs | 0.56 µs | 0.42 µs |
| L0 SST | 10-row scan | 10.7 µs | 1.8 µs | 2.6 µs | 2.1 µs | 7.2 µs |
| L0 SST | 100-row scan | 61.3 µs | 10.8 µs | 13.4 µs | 11.5 µs | 47.1 µs |
| Sorted run | Get | 4.34 µs | 0.37 µs | 0.74 µs | 0.56 µs | 0.42 µs |
| Sorted run | 10-row scan | 11.2 µs | 1.8 µs | 2.6 µs | 2.1 µs | 7.2 µs |
| Sorted run | 100-row scan | 64.3 µs | 10.8 µs | 13.4 µs | 11.5 µs | 46.6 µs |

The table above predates the scan path changes below. A third benchmark uses the prototype's
benchmark harness, so its numbers compare directly with the prototype and with SlateDB 0.18.1:
2M rows, 16-byte keys, 200-byte values (491.6 MiB in one range cache file), on a local-filesystem
object store, after a flush and after compaction went quiet (one sorted run of 4 SSTs, 435 MiB).
"Normal path" has a 2 GiB block cache that holds every block, warmed by a full scan. The cached
rows cache the whole key range. Uniform keys; p50 latency on one task, and operations per second
on 1, 8 and 16 tasks; `anon` and `heap` are the process's anonymous and allocator memory in MiB.
"End of round one" and the final rows are the best of two interleaved runs, each after an idle
pause; the end-of-round-one build attaches a `DefaultMetricsRecorder`, the final rows do not
unless they say so (SlateDB's default is a no-op recorder).

| Engine | Get hit | Get miss | 10-row scan | 100-row scan | anon / heap |
| --- | --- | --- | --- | --- | --- |
| Normal path, all blocks cached | 3.00 µs; 0.33/0.73/0.81M | 1.38 µs; 0.68/1.34/1.41M | 9.63 µs; 0.10/0.42/0.59M | 57.0 µs; 0.02/0.10/0.16M | 1605 / 479 |
| Cached, mmap, before the scan changes | 0.58 µs; 1.64/4.43/5.72M | 0.54 µs; 1.71/4.53/5.78M | 2.00 µs; 0.47/2.14/2.77M | 12.1 µs; 0.08/0.47/0.77M | 70 / 13 |
| Cached, pread, before the scan changes | 1.00 µs; 0.95/1.10/0.94M | 1.00 µs; 0.97/1.09/0.94M | 2.92 µs; 0.32/0.68/0.55M | 16.3 µs; 0.06/0.28/0.27M | 77 / 14 |
| Cached, mmap, end of round one | 0.54 µs; 1.68/5.46/6.12M | 0.50 µs; 1.80/5.60/6.14M | 1.71 µs; 0.56/2.42/3.83M | 8.3 µs; 0.12/0.76/1.30M | 94 / 16 |
| Cached, mmap, final | 0.38 µs; 2.56/17.8/31.9M | 0.29 µs; 2.93/22.5/42.9M | 1.33 µs; 0.69/3.49/5.12M | 8.0 µs; 0.12/0.80/1.38M | 94 / 16 |
| Cached, mmap, final, with the synchronous path in `next` | - | - | 1.17 µs; 0.79/3.97/5.70M | 6.0 µs; 0.15/1.00/1.68M | |
| Cached, mmap, final, `next_batch` | - | - | 1.00 µs; 0.92/4.12/5.85M | 4.4 µs; 0.21/1.28/2.04M | |
| Cached, mmap, final, with a metrics recorder | 0.38 µs; 2.55/13.4/23.0M | 0.33 µs; 2.90/16.2/28.6M | 1.38 µs; 0.68/3.33/4.81M | 8.0 µs; 0.12/0.78/1.36M | |
| Cached, pread, final | 0.79 µs; 1.21/1.04/0.90M | 0.75 µs; 1.28/1.04/0.90M | 2.25 µs; 0.44/0.65/0.56M | 10.5 µs; 0.09/0.32/0.28M | 97 / 16 |
| Normal path, final | 3.04 µs; 0.32/0.76/0.80M | 1.38 µs; 0.69/1.46/1.47M | 9.21 µs; 0.10/0.45/0.64M | 56.3 µs; 0.02/0.11/0.18M | |
| SlateDB 0.18.1, all blocks cached (prototype's run) | 3.2 µs | - | 9.6 µs | 56 µs | - |
| Prototype, rerun interleaved with the final build | 0.54 µs; 1.75/12.1/21.6M | 0.54 µs; 1.78/12.1/21.7M | 0.92 µs; 0.96/4.98/7.86M | 3.54 µs; 0.22/1.37/2.42M | - |

Each step of the scan path work was measured on its own, with the same binary flags, on one
range cache file (mmap, p50 on one task, operations per second on 16 tasks):

| Step | 10-row scan | 100-row scan | 100-row scan, pread | Kept |
| --- | --- | --- | --- | --- |
| Before | 2.00 µs; 2.82M | 12.4 µs; 0.79M | 16.3 µs | |
| One reference-counted buffer per page, rows as slices | 2.21 µs; 3.07M | 11.8 µs; 0.92M | 15.3 µs | yes |
| Cooperative budget every 32 rows for resolved sources | 1.75 µs; 3.24M | 9.2 µs; 1.00M | 11.8 µs | yes |
| Skip the read span, move rows into `KeyValue` | 1.79 µs | 9.6 µs | - | no gain, dropped |
| Copy the page when the cursor enters it, compare keys in the buffer | 1.67 µs; 3.50M | 8.2 µs; 1.19M | 10.9 µs | yes |
| Point reads through `arc_swap` guards, no reference counts | - | - | - | yes: 16-task get hits 5.7M to 6.2M, 8 tasks 4.5M to 5.9M |

A cached 100-row scan then cost about 0.95 µs to start and about 74 ns for each row, down from
about 0.8 µs and 100 ns.

A second round of work put point reads ahead of the prototype and narrowed the scan gap: at 16
tasks, scans with `next` reach 69% to 73% of the prototype's throughput, and scans with
`next_batch` 74% to 82%. Numbers come from the same
harness on the same machine, a laptop that slows down under sustained load; each comparison
interleaves the builds, runs each after an idle pause, and keeps the best of three runs.

**Contention.** At 16 tasks, cached point reads stopped scaling at about 6.2M per second. A
profile showed `DbInner::check_closed` on every read: it cloned a `tokio` watch receiver and took
its read lock. An atomic flag that the close result sets before it becomes visible, checked
first, raised 16-task point reads to 20.1M (hits) and 21.1M (misses) per second with SlateDB's
default no-op metrics recorder, and to 16.4M and 17.3M with a `DefaultMetricsRecorder`. After it,
a 16-task profile shows no shared cache line on the read path: the time goes to the lookup
itself, which is bound by memory latency, and the single-task changes below carry over to 16
tasks (31.9M hits and 42.9M misses per second in the final table). With a recorder attached,
every read also increments shared atomic counters (`slatedb.request_count`, and the cache's
`get_count`), which costs about a quarter of the 16-task throughput of point reads.
Registering the cache's read counters at `MetricLevel::Debug` did not measurably help
(16.25M to 16.41M point hits per second at 16 tasks, best of three), so the branch keeps them at
the default level. A scan cursor that borrows the range file instead of holding a reference to
it, which saves one shared reference count per scan, did not measurably help either (16-task
10-row scans 5.12M to 5.20M per second, 100-row scans within the noise), and was dropped.

**Single-task cost.** Each step on one task (mmap, p50, operations per second):

| Step | Get hit | Get miss | 10-row scan | 100-row scan | Kept |
| --- | --- | --- | --- | --- | --- |
| Before | 0.54 µs; 1.70M | - | 1.67 µs; 0.58M | 7.83 µs; 0.12M | |
| Fence search on an in-memory array of 8-byte prefixes | 0.42 µs; 2.17M | - | 1.54 µs; 0.62M | 7.67 µs | yes |
| Skip deltas that are empty when the scan opens | 0.42 µs | - | 1.46 µs; 0.65M | 7.58 µs | yes |
| Copy a page from the first row that the scan reads | 0.42 µs; 2.24M | - | 1.42 µs; 0.66M | 7.58 µs | yes |
| Scan bounds by reference: no copy of the start key | 0.42 µs; 2.26M | 0.38 µs; 2.46M | 1.38 µs; 0.67M | 7.58 µs | yes |
| Key hints in page headers (file format version 2) | 0.38 µs; 2.57M | 0.29 µs; 2.96M | 1.33 µs; 0.70M | 7.54 µs | yes |
| Synchronous path in `DbIterator::next` for resolved sources (best of three, against 1.33 µs and 7.92 µs in the same runs) | - | - | 1.17 µs; 0.79M | 6.00 µs; 0.15M | yes |
| `DbIterator::next_batch` instead of `next` (optional API) | - | - | 1.00 µs; 0.92M | 4.21 µs; 0.21M | optional |

Skipping empty deltas is safe at the pinned sequence number: the writer puts each version in its
delta before it commits it, and the scan loaded `seq` before it checks the delta, so a delta that
is empty then holds no version at or below `seq` (Appendix B, step 1).

With `next`, a cached scan now costs about 0.63 µs to start and 54 ns per row (69 ns before
`next` returned rows of a resolved source without an await, skipping the general path's nested
futures and span; every 32nd row still takes the general path, which checks the cooperative
budget); with `next_batch`, the same start and 36 ns per row. The prototype measured 0.9 µs for
10 rows and 3.5 µs for 100, about 29 ns per row. The remaining per-row cost of `next` over
`next_batch` is its work for each call: the future, the copy of the last key for `seek`, and the
return of each `KeyValue` by value. `next_batch` pays these once per batch. The remaining start cost is the fence search,
the page header and first row of a random page (memory latency), the copy of the page, and the
iterator setup.

Positional reads do not scale past about 1M reads per second, at any number of tasks. A profile
of 16 tasks shows 96% of the time inside the `pread` system call (p50 17 µs, against 0.9 µs on one
task). Eight or sixteen descriptors per file, picked by thread, did not change it, and neither
did spreading the data over 17 range files. So the limit is in the kernel's read path on this
machine (macOS), not in a descriptor or a file. The memory map does not have it, and stays the
default. The normal path holds about 1.6 GB of anonymous memory for its block cache; the range
cache file lives in the page cache, which the kernel can reclaim.

Automatic promotion, measured with the same harness (`--range-cache auto`): 16 tasks read uniform
or Zipfian keys, the cache starts empty, the block cache has 64 MiB, and the promoter uses its
defaults (one read in 64 sampled, 64 MiB candidates, 10 s half-life). The harness's Zipfian
generator scatters hot keys over the key space, so at the scale of 64 MiB ranges the skewed load
is almost as even as the uniform one.

| Load, budget | Served from the cache: 50% / 90% / 99% of reads | Promotions, evictions | Then |
| --- | --- | --- | --- |
| Uniform, room for all data | 6.1 s / 9.1 s / 11.1 s | 9, 0 | 9 adjacent ranges, 491.6 MiB |
| Zipfian, room for all data | 6.1 s / 11.1 s / 12.1 s | 8, 0 | the whole key range |

Each load takes 0.5 to 1 s while the normal path saturates the machine, and the promoter loads
one candidate at a time, so filling the cache takes about one load time per 64 MiB. After
convergence, the cache that promotion built serves 16-task point reads at 6.1M per second,
10-row scans at 3.3M and 100-row scans at 1.2M, the same as the explicitly cached whole range
(single-task latencies are within 0.3 µs: scans with an unbounded end pin every following range).
These automatic promotion runs predate the closed-check fix and the lookup changes below, which
apply to automatic ranges as to explicit ones.

Churn at a budget of 128 MiB, about a quarter of the 491.6 MiB of data, over 120 s:

| Load, candidate size | Promotions, evictions | After the first 12 s | Cached | Reads served | 16-task point reads |
| --- | --- | --- | --- | --- | --- |
| Uniform, 64 MiB | 3, 2 | none | 1 range, 73.5 MiB | 15.0% | 0.14M/s |
| Zipfian, 64 MiB | 2, 0 | none | 2 ranges, 123.2 MiB | 29% | 0.36M/s |
| Uniform, 16 MiB | 15, 7 | none | 8 ranges, 120.2 MiB | 24.4% | 0.17M/s |
| Zipfian, 16 MiB | 13, 4 | one promotion into free space, no eviction | 9 ranges, 121.4 MiB | 28.4% | 0.36M/s |

Every run settles within about 12 seconds and then stays put: equally hot candidates do not beat
cached ranges by the admission margin. The share of reads served matches the share of bytes
cached for uniform keys, and is higher for Zipfian keys. With 64 MiB candidates, a 128 MiB budget
holds only one or two ranges, because a range file is about 15% larger than its SST data and the
estimate uses SST bytes. Smaller candidates use the budget better.

For comparison, the same 16-task point reads without the range cache: 0.04M/s for both key
distributions with the 64 MiB block cache of these runs, whose 16 MiB metadata share is too small
for the index and filter blocks of the four SSTs; and 0.12M/s (uniform) and 0.27M/s (Zipfian)
with 64 MiB for data blocks plus 64 MiB for metadata. So against a block cache with enough
metadata space, the gain at this budget is about 1.2x to 1.3x, in line with the share of reads
served. The larger gain against the 64 MiB block cache comes mostly from cached ranges needing no
index or filter blocks, which ends the metadata cache misses of that configuration. The benchmark
keys are scattered over the key space; it has no workload with a contiguous hot range, which is
the case that a range cache with a small budget serves best.

`DbIterator::next_batch(&mut Vec<KeyValue>, n)` appends up to `n` rows and returns how many. It
works for every iterator: for a normal scan it calls `next` in a loop; for a cached scan it reads
rows without a future per row, checks the cooperative budget every 32 rows, and records the last
key once per batch. It is behind the feature, in its own last PR, and nothing depends on it. It is
a public API change, so it needs its own discussion (see [Open Questions](#open-questions)).

- **Local disk.** About the size of the cached data, plus page padding and the fence region. The
  prototype measured 1.19 times the cached data.
- **Local write amplification.** Loads write each cached byte once. Rebuilds write at most about 4
  times the bytes written to cached ranges. `rebuild_bytes` measures it.
- **Memory.** The fence index (one key per page) and the deltas, which rebuilds bound. File pages
  live in the page cache, which the kernel can reclaim. The prototype's fence index took 2.2 MB for
  2M rows.
- **Object store.** Loads read the range once from the LSM. Cached reads make no object store
  request.
- **CPU.** Loads and rebuilds run in the background. Rebuilds run the merge operator.

### Observability

- Configuration: `DbBuilder::with_range_cache` and `RangeCacheOptions`.
- New component: none. Rebuilds run as background tasks on the database runtime. Automatic
  promotion runs as a task on the database's task executor.
- Metrics (all under `slatedb.range_cache.`):
  - `get_count{result=hit|absent|merge}` and `scan_count`: reads that the cache served.
  - `load_count`, `load_bytes`, `rebuild_count`, `rebuild_bytes`: background work and local write
    amplification.
  - `remove_count{reason=evicted|explicit|expiring_write|error|closed|not_worth_it}`.
  - `range_count`, `file_bytes` and `delta_bytes` (gauges). `delta_bytes` is the logical size of
    the deltas of cached ranges.
  - With automatic promotion only: `sample_count{result=recorded|dropped}`,
    `candidate_count{source=sst_index|scan|remembered}`, `admission_count{result=free|evicting}`,
    `rejection_count{reason=cold|not_reused|too_large|no_room|not_denser|backoff}`,
    `promotion_count` and `repromotion_count`. With `load_bytes`, `rebuild_bytes` and the removal
    reasons, every decision of the promoter has a counter.
- Logging: a warning when a rebuild or a cache read fails.

### Compatibility

- Data on object storage: no change. The manifest, WAL and SST formats do not change.
- Local files: a new, versioned format that only this process reads. The cache deletes old files
  when it opens, so a format change needs no migration.
- Public API: additive, behind a cargo feature, and experimental. The option types are
  `#[non_exhaustive]`, and `RangeCacheTuning` is not a stable API. Language bindings do not change
  in the first version.

## Testing

- Unit tests: the file format with both page sources (round trip, page boundaries, absent keys,
  out-of-line keys and values, merge kind, empty file, corrupt file, file lifetime); the fence
  search against a brute-force count, with keys that share long prefixes and tie on the
  in-memory prefixes; the in-page search with key hints, with keys that tie on their hints and a
  shared prefix longer than the hints skip; the index (uncovered parts, the covering range, scan
  containment, range conversion); the close flag of `DbInner::check_closed` (set before a value is
  visible, from another worker).
- Integration tests (`slatedb/tests/range_cache.rs`): inert when not configured; point reads;
  puts, deletes, batches, transaction commits and conflicts; covered and uncovered scans and seek;
  merge operands across rebuilds; expiring rows; rebuilds; eviction order and the budget; uncache
  and clear; close; old files at open; fencing; writes that race loads, for both page sources; and
  a proptest model with puts, deletes, merges, batches, flushes, loads, removals, reads and scans.
  Most tests compare each read with the same read through a snapshot, which uses the normal path.
- Independent checker: `jit-check`, a checker that the prototype's author wrote from the
  specification, drives the cache through `cache_range`, `uncache_range` and
  `clear_range_cache` while it writes, reads and scans. Its sequential tester compares each result
  with a model. Its concurrent tester checks linearizability rules for gets and scans with 2, 6
  and 16 tasks. It ran with configurations that force rebuilds, evictions, rejected loads, L0
  flushes and compaction, and positional reads. It found no violation. To show that the checker
  sees cache bugs, it also ran against mutants. It found a mutant that drops deletes at once. It
  found a mutant that loads from the last committed sequence number instead of `last_seq`, when a
  delay in the writer widened the race window. A mutant without the second sequence number rule
  passed, as the analysis in [Sequence number rules](#sequence-number-rules) predicts. The checker
  did not find the registration-order race in [Loading a range](#loading-a-range): its window is
  a few instructions long. A review found it, and a fail-point test reproduced it. Races of this
  kind need fail points or simulation, not only randomized testing.
- Fault-injection tests: a test pauses the writer at the `write-batch-pre-commit` fail point,
  after the cache applied a batch and before the batch commits. Reads must return the old state,
  and a load during the pause must not serve the batch early. Another test pauses a load before
  and after it publishes its range (fail points `range-cache-register-publish` and
  `range-cache-register-snapshot`), and writes a key of the range during the pause. This test
  reproduced a lost write in an earlier version of the registration order. A later PR adds fail
  points in the file writes of loads and rebuilds.
- Deterministic simulation tests: a later PR adds the cache to the `slatedb-dst` workloads. This
  needs the cache to do its file I/O through `Vfs`, so `DeterministicLocalFilesystem` can drive it.
- Scan path tests: a property test compares `CachedScan` with the normal iterator stack over the
  same layers, field by field, including the entry kind. It generates files, frozen and active
  deltas with values, tombstones, merge operands and several versions for each key, a pinned
  sequence number, bounds, and a forward seek. It passed 50,000 cases. It caught two mutants: a
  delta cursor without the sequence number filter, and a wrong tie between deltas. Integration
  tests scan while writes land in the range, scan across a paused and a finished rebuild, and seek
  forward and with errors, with both page sources.
- Upstream bug found: the property test found that `MergeOperatorIterator::seek` drops its
  buffered entry. A merge reads ahead one entry of the next key and buffers it. A `seek` to a key
  at or before that entry then cleared the buffer, but the delegate was already past it, so the
  next row was lost. On `main` this loses a row of a normal scan when the application seeks after
  a merge operand row. The fix keeps the buffered entry when its key is at or after the seek key.
- Formal methods verification: none planned. Appendix B gives the visibility argument.
- Batched reads: `next_batch` returns the same rows as `next`, in batches of 1, 7, 64 and 1000,
  for a cached scan with merge operands (which take the await path) and tombstones, and for a
  scan on the normal path; a seek after a batch must go past its last row.
- Scans across adjacent ranges: the property test above also splits the layers into two adjacent
  ranges at a generated key, and compares the scan over both parts with the normal stack over the
  unsplit layers, also after a seek. It caught mutants that do not skip a part on seek, or that
  open the next part one key early. An integration test scans across three adjacent ranges with
  writes and rebuilds, checks that every part is pinned at one sequence number, seeks across
  parts, and checks that a gap or a removed part sends the scan to the normal path.
- Automatic promotion: integration tests run on a mock clock, with every read sampled, so the
  decisions depend only on the keys read and the clock. A repeatedly read range is cached, and
  then serves reads that agree with the normal path. A single pass over all keys promotes nothing
  and evicts nothing (`not_reused`). A moving hot set is followed: the new set displaces the old
  one once it cools (`admission_count{result=evicting}`). A range that takes writes but no reads
  leaves with `not_worth_it` and stays out, while a range that is read and written as hard stays
  and rebuilds. Four hot sets under a budget for three settle: no promotion or eviction in the
  second half, the fourth set is rejected as `not_denser`. With the option off, nothing is
  promoted and no promotion metric exists. A test with a real compactor checks that candidates
  come from SST index boundaries, and a fencing test checks that the promoter stops with the
  database. Unit tests cover the reservoir.
- Checker with automatic promotion: two more configurations, with aggressive settings (every read
  sampled, 256-byte candidates, evaluations every millisecond, a 100 ms half-life, a tight budget
  and a high rebuild cost). `auto_only` disables the checker's explicit hooks, so only the
  promoter caches ranges. `auto_combined` adds explicit caching, rebuilds, eviction, L0 flushes
  and compaction. Both exercise every decision: free and evicting admissions, all rejection
  reasons that the settings allow, evictions, `not_worth_it` removals, re-promotions, and both
  candidate sources. They found no violation: 18,000 sequential and 1.23M concurrent operations
  in the quick run, and 90,000 sequential and 13.1M concurrent operations in a soak with 10
  seconds per task count (2, 6 and 16 tasks). The six stage 1 configurations still pass.
- Performance tests: a later PR adds range cache workloads to `slatedb-bencher`: hot point reads,
  hot scans, writes into cached ranges, and an equal-memory comparison with the block cache.

## Rollout

- Milestones / phases:
  1. This RFC.
  2. PR 0, independent of this RFC: the fix to `MergeOperatorIterator::seek` (see
     [Testing](#testing)), with its unit test. It can merge first. PR 0b, also independent: the
     fast path of `DbInner::check_closed`, with its unit tests.
  3. PR 1: the range cache file format, page sources, the file cursor and unit tests. No wiring.
  4. PR 2: the cache core: index, deltas, the write hook, the read path, loads, removal, close,
     the public API, metrics and integration tests. Scans use `DbIterator::new` over the
     pinned layers. Rebuild and eviction are not in this PR, so a delta grows until the range is
     removed. The PR documents this.
  5. PR 3: rebuilds and the disk budget.
  6. PR 4: the scan path (see [Scan path](#scan-path)): `CachedScan`,
     `DbIterator::new_resolved`, `RowEntryIterator::next_sync`, page buffers, the budget check
     every 32 rows, and the differential test.
  7. PR 5: fail points in loads and rebuilds, DST support through `Vfs`, and bencher workloads.
  8. Stage 2, in three PRs:
     1. PR 6: scans across adjacent cached ranges, with the split case of the differential test.
     2. PR 7: heat, density eviction, ghosts, backoff and the rebuild payback check in the cache
        core, with the `Origin` of a range. Inert without the option.
     3. PR 8: the promoter task, sampling on the read path, the options, metrics, the automatic
        promotion tests and the checker configurations.
  9. Stage 3, each item separate: scans past the end of the cached key space; expiring rows with a
     clock rule; range deletes; `DbReader` support; a batched read API.
- Feature flags / opt-in: the `range_cache` cargo feature, and `DbBuilder::with_range_cache`.
- Docs updates: a page in the design docs, and the options in the configuration docs.

## Alternatives

**Status quo, with a larger block cache and RFCs 0037 and 0038.** These make each source cheaper.
A read still visits each source, so misses and scans still pay once per L0 SST and sorted run. The
range cache removes the sources for hot ranges. The two approaches combine.

**A point-only row cache.** This is simpler: a hash map from key to value. Prior art found it
disappointing. In the 2-Tree paper, RocksDB with 90% of its memory as row cache ran a
read-modify-write workload up to about 9 times slower than plain RocksDB, and a scan-heavy mix lost
83% of its throughput. DataStax advises the Cassandra row cache only when about 95% of the load is
reads. MySQL 8.4 turned the InnoDB adaptive hash index off by default. A row cache also cannot
answer a scan, and it can prove absence only with a negative entry for each key.

**The memtable and the newer L0 SSTs as the delta.** In this design, a cache file holds a range at
the sequence number of an L0 flush or a compaction. A read merges the file with the memtables and
the L0 SSTs that are newer than the file. A flush or a compaction refreshes the file. This needs no
write hook and no deltas. But each read again visits the memtables and the newer L0 SSTs, which is
the per-source cost that the cache must remove. The refresh rate follows the flush rate, not the
write rate into cached ranges, so local write amplification depends on unrelated writes. This
design is a candidate for `DbReader`, which has no write hook.

**An external wrapper with a minimal write hook.** SlateDB would expose a callback at the point
where the cache applies writes. The cache would live in another crate. This keeps the change in
SlateDB small, but the hook becomes a public contract on internal ordering. The wrapper also needs
`RowEntry`, the merge iterators, and a way to build a `DbIterator`, which are internal today. The
prototype showed that a wrapper must also wrap every read and write method. Inside SlateDB, each
of these is a direct call.

**Values in memory only.** A cache in memory needs no file format and no page source. Its memory
grows with the cached data, and the kernel cannot reclaim it. Files with a small in-memory index
keep the non-reclaimable memory small: the prototype added no measurable non-reclaimable memory
beyond its 2.2 MB fence index.

## Open Questions

- Should the options live in `Settings`, which users can set from files and the environment, or
  stay a builder component?
- Should stage 1 make `cache_range` public, or keep it for tests and ship it with stage 2?
- Should the default page source be the memory map or positional reads? A memory map needs
  `unsafe` and turns I/O errors into signals. Positional reads cost a system call for each page.
- How should the cache handle expiring rows? One option: serve a row while its expire time is in
  the future by the database clock, and use the normal path after that. A compactor in another
  process uses its own clock, which makes this rule fragile.
- Should a scan continue in the normal path when it leaves the cached key space? It then needs a
  normal-path iterator at the same sequence number, which needs a snapshot registration for the
  life of the scan.
- Should `cache_range` return before the load ends?
- Should read counters use `MetricLevel::Debug`, to keep shared counters off the hot path?
- Should `DbIterator` get a batched form? The branch has one behind the feature,
  `next_batch(&mut Vec<KeyValue>, n)`, as an optional last PR that nothing depends on. It cuts
  the per-row cost of cached scans by about half (see [Performance & Cost](#performance--cost)),
  and works for every iterator. It is a public API, so it needs its own discussion: the name,
  whether it returns the rows or fills a caller's buffer, and whether normal scans should get a
  fast path too.
- With range deletes (RFC 0039), should the first version remove overlapping ranges, as this RFC
  proposes, or apply range tombstones to deltas at once?
- What should the defaults of automatic promotion be? They need real workloads. In particular the
  half-life (10 s), the reuse span (2 s) and the target size (64 MiB) trade reaction time against
  churn and load cost.
- Should the promoter load several candidates at once? It loads one at a time today, so filling a
  large budget takes one load time per candidate (see [Performance & Cost](#performance--cost)).
- Should the cost of a sample be the normal-path time, as now, or the object store requests and
  bytes that it caused? Time is simple and captures every source of cost, but CPU contention
  inflates it for all reads alike.

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
- [ScyllaDB row cache](https://github.com/scylladb/scylladb/blob/master/docs/dev/row_cache.md)
- [TiKV in-memory engine](https://docs.pingcap.com/tidb/stable/tikv-in-memory-engine/)
- [AC-Key, USENIX ATC 2020](https://www.usenix.org/system/files/atc20-wu-fenggang.pdf)
- [DataStax: configuring caches](https://docs.datastax.com/en/cassandra/3.0/cassandra/operations/opsConfiguringCaches.html)
- [MySQL 8.4: adaptive hash index](https://dev.mysql.com/doc/refman/8.4/en/innodb-adaptive-hash.html)

## Updates

## Appendix A: Prototype measurements

The prototype is an out-of-tree wrapper around the SlateDB 0.18.1 writer. It keeps the same
structure as this RFC: complete key intervals, immutable files of sorted fixed-size pages with an
in-memory fence index, an in-memory delta for each interval, background rebuilds, eviction under a
disk budget, and a profiler for automatic promotion.

Setup: 2M rows, 16-byte keys, 200-byte values. SlateDB 0.18.1 had its own local mirror and a 64
MiB block cache. macOS, MinIO on the LAN, local filesystem. One run per configuration.

| Measurement | Prototype | SlateDB 0.18.1 |
| --- | --- | --- |
| Point read latency, 1 task | 0.5 µs | 16 to 17 µs |
| Point read latency, 1 task, all data in SlateDB's memory cache | | 3.1 µs |
| 100-row scan latency, 1 task | 3.5 µs | 135 to 148 µs |
| Point reads per second, 16 tasks | 21 M | 0.20 to 0.25 M |

Costs that the prototype measured:

- Memory: no measurable non-reclaimable memory beyond a 2.2 MB fence index. The rest is page
  cache over the files.
- Local disk: 1.19 times the promoted data.
- Background I/O: loads, and rebuilds with write amplification of about file size divided by
  the delta threshold.
- Policy risk: promotion churn under a tight budget, before the prototype added admission
  control.

The prototype's write path needed a gate, marks for writes in flight, and bounded waits. This RFC
does not need them. The prototype also could not see row entries, so it sent merge operands and
expiring keys to SlateDB on each read. This RFC resolves merge operands in the cache.

## Appendix B: Why reads see the right versions

Claim: when a read at `seq` uses layers `L` under the sequence number rules, `L` holds every version
of each key in the range that is visible at `seq`, and the read uses no other version. So the read
gives the answer of the normal path at `seq`.

1. The writer puts a write with sequence number `w` in a delta before it advances the last
   committed sequence number to `w`. Both happen on the writer task. The advance is a `SeqCst`
   store, and the read loads `seq` with a `SeqCst` load after it loads `L`. So if `w <= seq`, the
   read sees the delta insert of `w`.
2. A delta receives the writes to its range from the moment it becomes active until it is sealed
   at `s`. The cache seals a delta under the mutex. The writer applies writes under the same mutex
   and in sequence number order. So the delta holds each write to the range with a sequence
   number at or below `s` that came after its predecessor layer, and no write above `s`. The seal
   is a `SeqCst` store before any write above `s` commits, so a read with `seq > s` sees the seal
   and retries.
3. A file holds the state of the range at `file_seq`. For a load, `file_seq` is the snapshot
   sequence number. The load publishes the range before it picks the snapshot, and the writer
   checks the index after its memtable insert. A pair of `SeqCst` counters orders the two sides
   (see [Loading a range](#loading-a-range)). So each write is in the snapshot, or the writer
   applies it to the new delta. A rebuild writes the state at the seal of the frozen delta, which
   is its `file_seq`. A read with `seq < file_seq` does not use the file.
4. Readers filter delta versions with `max_seq = seq`. Files hold no version above `file_seq`, and
   `file_seq <= seq`.
5. A removal unlinks the range under the mutex, and seals its active delta at the last sequence
   number that the writer applied under the mutex. A read that loaded the range before the
   removal uses it only if its `seq` is at or below that number. Each write to the range at or
   below that number is in the delta, or, if it came before the range was published, in the
   file. So the layers are still complete for that read. A writer that saw the range before the
   removal takes the mutex, finds the range gone, and applies nothing to it.
6. Only the writer task freezes deltas, inside step 6 of the write path under the mutex. Its
   unlocked check runs on the same task, so a freeze never races it. An install replaces the file
   and the frozen delta but keeps the active delta, so the writer's check does not depend on it.
   The payback check of automatic ranges runs at the same point: instead of a freeze, it removes
   the range and seals its active delta at the current write, which the delta holds.
7. A scan across adjacent ranges loads the layers of every range before it loads `seq`, and uses
   them only if the rules hold for each. Steps 1 to 6 then hold for each range on its own. The
   ranges hold disjoint keys and, together, every key of the scan range. So the concatenation of
   their answers is the answer of the normal path at `seq`.
8. Automatic promotion adds and removes ranges only through `register`, `load` and the removal
   under the mutex, as the explicit API does. It changes no layers of its own.

A failed write after the memtable insert closes the database, so a version in a delta that never
commits is never read: `seq` never reaches it, and reads fail with `Closed`.
