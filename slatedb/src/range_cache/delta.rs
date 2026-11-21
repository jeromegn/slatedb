//! The deltas of cached ranges.
//!
//! A [`Delta`] holds every version that the writer applied to a range since the range's file was
//! written, in a [`Buffer`] of the `buffer` module: a sorted map that the batch writer appends to
//! and that readers search without a reference count or any other write to shared memory. Its
//! `sealed_at` word is the readers' validity word (see the sequence number rules of the cache).
//!
//! The cache counts the memory of all deltas exactly, in [`DeltaMemory`]: each chunk from its
//! allocation, by the writer, until the last holder of its delta (the layers, a scan, a merge
//! resolution or a rebuild) drops it.

use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use slatedb_common::metrics::GaugeFn;

use super::buffer::{Buffer, Cursor, Kind, Version, Writer};
use super::file::copy_key_value;
use super::writer::Padded;
use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::types::{RowEntry, ValueDeletable};

/// The memory that the deltas of one cache hold: the sum of their chunks (design 01 calls it
/// `recent_bytes`). The writer adds a chunk when it allocates it; a delta subtracts its chunks
/// when it drops. The gauge follows each change.
pub(super) struct DeltaMemory {
    bytes: Padded<AtomicU64>,
    gauge: Arc<dyn GaugeFn>,
}

impl DeltaMemory {
    pub(super) fn new(gauge: Arc<dyn GaugeFn>) -> Arc<Self> {
        Arc::new(Self {
            bytes: Padded(AtomicU64::new(0)),
            gauge,
        })
    }

    /// A count that reports to no gauge, for tests.
    #[cfg(test)]
    pub(super) fn detached() -> Arc<Self> {
        Self::new(
            slatedb_common::metrics::MetricsRecorderHelper::noop()
                .gauge("range_cache_test_delta_memory")
                .register(),
        )
    }

    /// The bytes of the chunks of all live deltas.
    pub(super) fn bytes(&self) -> u64 {
        self.bytes.0.load(Ordering::Relaxed)
    }

    /// Sets the gauge to the current count.
    pub(super) fn publish(&self) {
        self.gauge.set(self.bytes() as i64);
    }

    fn add(&self, bytes: u64) {
        let total = self.bytes.0.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.gauge.set(total as i64);
    }

    fn release(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let total = self.bytes.0.fetch_sub(bytes, Ordering::Relaxed) - bytes;
        self.gauge.set(total as i64);
    }
}

/// The versions that the writer applied to a range since its file was written.
pub(super) struct Delta {
    buffer: Buffer,
    memory: Arc<DeltaMemory>,
}

impl Delta {
    /// An empty delta, whose chunks count in `memory`. It allocates nothing until the first
    /// version arrives.
    pub(super) fn new(memory: &Arc<DeltaMemory>) -> Self {
        Self {
            buffer: Buffer::new(),
            memory: memory.clone(),
        }
    }

    /// The cache applies no write with a greater sequence number to this delta. It is
    /// `u64::MAX` while the delta is active.
    pub(super) fn sealed_at(&self) -> &AtomicU64 {
        self.buffer.sealed_at()
    }

    /// The estimated size of the versions, as the memtable counts its rows. The rebuild threshold
    /// and the `delta_bytes` gauge use it.
    pub(super) fn size(&self) -> usize {
        self.buffer.estimated_bytes() as usize
    }

    /// The memory that the delta holds: its chunks.
    pub(super) fn allocated_bytes(&self) -> u64 {
        self.buffer.allocated_bytes()
    }

    /// The number of versions.
    #[cfg(test)]
    pub(super) fn len(&self) -> u64 {
        self.buffer.len()
    }

    /// Returns true if the delta holds no version. A reader that loaded its sequence number
    /// before this check, and finds the delta empty, can skip it: the writer puts each version
    /// in its delta before it commits it.
    pub(super) fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub(super) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The newest version of `key` at or below `seq`, read in place.
    pub(super) fn newest(&self, key: &[u8], seq: u64) -> Option<Version<'_>> {
        self.buffer.get(key, seq)
    }

    /// The writer of the delta. Only the batch writer task appends.
    pub(super) fn writer(&self) -> DeltaWriter<'_> {
        DeltaWriter {
            writer: self.buffer.writer(),
            memory: &self.memory,
        }
    }
}

impl Drop for Delta {
    fn drop(&mut self) {
        self.memory.release(self.allocated_bytes());
    }
}

/// Appends the rows of a batch to a delta. See [`Delta::writer`].
pub(super) struct DeltaWriter<'a> {
    writer: Writer<'a>,
    memory: &'a DeltaMemory,
}

impl DeltaWriter<'_> {
    /// Copies `entry` into the delta. The cache holds no row with an expire time.
    pub(super) fn append(&mut self, entry: &RowEntry) {
        debug_assert!(entry.expire_ts.is_none(), "a delta row with an expire time");
        let (kind, value): (Kind, &[u8]) = match &entry.value {
            ValueDeletable::Value(v) => (Kind::Value, v),
            ValueDeletable::Merge(v) => (Kind::Merge, v),
            ValueDeletable::Tombstone => (Kind::Tombstone, &[]),
        };
        let grown = self
            .writer
            .append(&entry.key, entry.seq, kind, entry.create_ts, value);
        if grown > 0 {
            self.memory.add(grown);
        }
    }
}

/// Copies a version out of a delta into a row, with one allocation for its key and value.
pub(super) fn to_row(version: &Version<'_>) -> RowEntry {
    let (key, value) = copy_key_value(version.key, version.value);
    RowEntry {
        key,
        value: match version.kind {
            Kind::Value => ValueDeletable::Value(value),
            Kind::Merge => ValueDeletable::Merge(value),
            Kind::Tombstone => ValueDeletable::Tombstone,
        },
        seq: version.seq,
        create_ts: version.create_ts,
        expire_ts: None,
    }
}

/// Returns true if `key` is past `end`.
pub(super) fn past_end(end: &Bound<Bytes>, key: &[u8]) -> bool {
    match end {
        Bound::Unbounded => false,
        Bound::Included(end) => key > end.as_ref(),
        Bound::Excluded(end) => key >= end.as_ref(),
    }
}

/// Every version of a delta in a key range, key ascending and newest first, for the normal
/// iterator stack, which filters them by sequence number. Rebuilds and merge resolution use it.
/// It holds the delta, so it outlives the layers that it came from.
pub(super) struct DeltaIterator {
    delta: Arc<Delta>,
    cursor: Cursor,
    end: Bound<Bytes>,
}

impl DeltaIterator {
    pub(super) fn new(delta: Arc<Delta>, range: &BytesRange) -> Self {
        use std::ops::RangeBounds;
        let start = range.start_bound().map(|key| key.as_ref());
        let cursor = delta.buffer.seek(start);
        Self {
            end: range.end_bound().cloned(),
            delta,
            cursor,
        }
    }

    fn current(&self) -> Option<Version<'_>> {
        self.delta
            .buffer
            .version(&self.cursor)
            .filter(|version| !past_end(&self.end, version.key))
    }
}

#[async_trait]
impl RowEntryIterator for DeltaIterator {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        let Some(row) = self.current().map(|version| to_row(&version)) else {
            return Ok(None);
        };
        self.delta.buffer.advance(&mut self.cursor);
        Ok(Some(row))
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        // Never move backwards, and never past the end again.
        if self.current().is_some_and(|version| version.key < next_key) {
            self.cursor = self.delta.buffer.seek(Bound::Included(next_key));
        }
        // Keep in-memory seeking cooperative, as the memtable's iterator does.
        tokio::task::coop::consume_budget().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes_range::BytesRange;
    use crate::mem_table::KVTable;
    use crate::types::RowEntry;
    use std::ops::Bound::{Excluded, Included, Unbounded};

    fn row(key: &str, seq: u64, value: Option<&str>) -> RowEntry {
        RowEntry::new(
            Bytes::copy_from_slice(key.as_bytes()),
            match value {
                Some(v) => ValueDeletable::Value(Bytes::copy_from_slice(v.as_bytes())),
                None => ValueDeletable::Tombstone,
            },
            seq,
            Some(seq as i64),
            None,
        )
    }

    async fn drain(iter: &mut dyn RowEntryIterator) -> Vec<RowEntry> {
        let mut out = Vec::new();
        while let Some(row) = iter.next().await.unwrap() {
            out.push(row);
        }
        out
    }

    /// The iterator returns what the memtable's iterator returned over the same rows, also
    /// after a seek.
    #[tokio::test]
    async fn should_iterate_as_the_memtable_does() {
        let memory = DeltaMemory::detached();
        let delta = Arc::new(Delta::new(&memory));
        let table = KVTable::new();
        let rows = [
            row("a", 3, Some("a3")),
            row("b", 1, Some("b1")),
            row("b", 4, None),
            row("c", 2, Some("c2")),
            row("c", 5, Some("c5")),
            row("d", 6, Some("d6")),
        ];
        let mut writer = delta.writer();
        for r in &rows {
            writer.append(r);
            table.put(r.clone());
        }
        drop(writer);
        assert_eq!(delta.size(), table.metadata().entries_size_in_bytes);
        assert_eq!(delta.len() as usize, table.metadata().entry_num);
        let b = |s: &str| Bytes::copy_from_slice(s.as_bytes());
        let ranges = [
            BytesRange::unbounded(),
            BytesRange::new(Included(b("b")), Excluded(b("d"))),
            BytesRange::new(Excluded(b("b")), Included(b("c"))),
            BytesRange::new(Included(b("bb")), Unbounded),
        ];
        for range in ranges {
            let mut ours = DeltaIterator::new(delta.clone(), &range);
            let mut theirs = table.range(
                range.clone(),
                crate::iter::IterationOrder::Ascending,
                crate::read_trace::ReadTrace::none(),
            );
            assert_eq!(
                drain(&mut ours).await,
                drain(&mut theirs).await,
                "{range:?}"
            );
            for target in ["", "b", "bz", "c", "z"] {
                let mut ours = DeltaIterator::new(delta.clone(), &range);
                let mut theirs = table.range(
                    range.clone(),
                    crate::iter::IterationOrder::Ascending,
                    crate::read_trace::ReadTrace::none(),
                );
                // One row, then a seek, then the rest.
                assert_eq!(ours.next().await.unwrap(), theirs.next().await.unwrap());
                ours.seek(target.as_bytes()).await.unwrap();
                theirs.seek(target.as_bytes()).await.unwrap();
                assert_eq!(
                    drain(&mut ours).await,
                    drain(&mut theirs).await,
                    "{range:?} after a seek to {target:?}"
                );
            }
        }
    }

    /// The cache's count of delta memory holds each delta's chunks until its last holder drops it.
    #[test]
    fn should_count_delta_memory_until_the_last_holder_drops_it() {
        let memory = DeltaMemory::detached();
        let first = Arc::new(Delta::new(&memory));
        let second = Arc::new(Delta::new(&memory));
        assert_eq!(memory.bytes(), 0);
        for (i, delta) in [&first, &second].into_iter().enumerate() {
            let mut writer = delta.writer();
            for j in 0..2_000u64 {
                writer.append(&row(
                    &format!("k{j:05}"),
                    i as u64 * 10_000 + j,
                    Some("value"),
                ));
            }
        }
        assert!(first.allocated_bytes() > 0);
        assert_eq!(
            memory.bytes(),
            first.allocated_bytes() + second.allocated_bytes()
        );
        // A scan holds the first delta after the layers dropped it.
        let held = DeltaIterator::new(first.clone(), &BytesRange::unbounded());
        let first_bytes = first.allocated_bytes();
        drop(first);
        drop(second);
        assert_eq!(memory.bytes(), first_bytes);
        drop(held);
        assert_eq!(memory.bytes(), 0);
    }
}
