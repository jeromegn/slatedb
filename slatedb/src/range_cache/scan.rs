//! The scan path of the range cache.
//!
//! [`CachedScan`] walks the pinned layers of cached ranges directly. It reads the file with a
//! cursor and overlays the frozen and the active delta, at the pinned sequence number. For each
//! key it returns the newest version at or below that number, and it skips tombstones.
//!
//! A scan can span adjacent cached ranges. It then walks the parts in key order, all pinned at
//! the same sequence number. The parts hold disjoint keys, so the scan is their concatenation.
//!
//! A merge operand needs the merge operator and the older versions of its key. For such a key,
//! the scan reads that key alone through the normal iterator stack, at the same sequence number.
//! No other row needs the normal stack: cached ranges hold no expiring rows, and a file holds
//! only values and base-less merge operands.

use std::collections::VecDeque;
use std::io;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::ops::RangeBounds;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use super::{Delta, Layers};
use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::iter::{IterationOrder, RowEntryIterator};
use crate::mem_table::MemTableIterator;
use crate::merge_operator::MergeOperatorType;
use crate::range_cache::file::RangeFileCursor;
use crate::read_trace::ReadTrace;
use crate::types::{RowEntry, ValueDeletable};

/// A cursor over the versions of a delta that are visible at one sequence number.
struct DeltaCursor {
    iter: MemTableIterator,
    seq: u64,
    /// The newest visible version of the next key.
    peeked: Option<RowEntry>,
    /// The key of the last version that `fill` peeked, so that it skips the older versions.
    last_key: Option<Bytes>,
    exhausted: bool,
}

impl DeltaCursor {
    fn new(delta: &Delta, range: &BytesRange, seq: u64) -> Self {
        Self {
            iter: delta
                .table
                .range(range.clone(), IterationOrder::Ascending, ReadTrace::none()),
            seq,
            peeked: None,
            last_key: None,
            exhausted: false,
        }
    }

    /// Peeks the newest visible version of the next key. A delta holds the versions of a key
    /// newest first, so the first version at or below `seq` is the newest visible one.
    fn fill(&mut self) {
        while self.peeked.is_none() && !self.exhausted {
            match self.iter.next_sync() {
                None => self.exhausted = true,
                Some(entry) => {
                    if entry.seq > self.seq || self.last_key.as_ref() == Some(&entry.key) {
                        continue;
                    }
                    self.last_key = Some(entry.key.clone());
                    self.peeked = Some(entry);
                }
            }
        }
    }

    fn key(&self) -> Option<&Bytes> {
        self.peeked.as_ref().map(|entry| &entry.key)
    }

    async fn seek(&mut self, key: &[u8]) -> Result<(), SlateDBError> {
        if let Some(entry) = &self.peeked {
            if entry.key.as_ref() >= key {
                return Ok(());
            }
            self.peeked = None;
        }
        if !self.exhausted {
            self.iter.seek(key).await?;
        }
        Ok(())
    }
}

/// Where the next row of a scan comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    File,
    Delta,
    /// The file and a delta hold the same key. The delta is newer.
    Both,
}

/// The smaller of a scan's end bound and a range's exclusive end.
fn min_end(scan_end: &Bound<Bytes>, range_end: &Option<Bytes>) -> Bound<Bytes> {
    let Some(range_end) = range_end else {
        return scan_end.clone();
    };
    match scan_end {
        Included(end) if end < range_end => Included(end.clone()),
        Excluded(end) if end <= range_end => Excluded(end.clone()),
        _ => Excluded(range_end.clone()),
    }
}

/// What one step of a scan found.
enum Step {
    Row(RowEntry),
    /// The newest version of this key is a merge operand.
    Resolve(Bytes),
    End,
}

/// A scan of the pinned layers of one or more adjacent cached ranges. See the module
/// documentation.
pub(crate) struct CachedScan {
    /// The layers of the current part.
    layers: Arc<Layers>,
    /// The end of the current part, within the scan range.
    part_end: Bound<Bytes>,
    /// The layers of the later parts, in key order. Each part starts where the one before ends.
    rest: VecDeque<Arc<Layers>>,
    /// The end of the scan range.
    scan_end: Bound<Bytes>,
    seq: u64,
    file: Option<RangeFileCursor>,
    /// Newest first: the active delta, then the frozen delta.
    deltas: Vec<DeltaCursor>,
    merge_operator: Option<MergeOperatorType>,
    /// A merge-operand key that `next_sync` found and left for `next` to resolve.
    unresolved: Option<Bytes>,
    /// The scan covers one key. The normal path then uses the point-lookup stack.
    point: bool,
}

impl CachedScan {
    /// Returns a scan of `range` over the layers of adjacent cached ranges, in key order. The
    /// first range holds the start of `range`, and together they hold all of `range`.
    pub(crate) fn new(
        parts: Vec<Arc<Layers>>,
        seq: u64,
        range: &BytesRange,
        merge_operator: Option<MergeOperatorType>,
    ) -> io::Result<Self> {
        let mut rest: VecDeque<_> = parts.into();
        let Some(layers) = rest.pop_front() else {
            return Err(io::Error::other("range cache scan without parts"));
        };
        let scan_end = RangeBounds::end_bound(range).cloned();
        let first_end = min_end(&scan_end, &layers.end);
        let mut scan = Self {
            layers,
            part_end: Unbounded,
            rest,
            scan_end,
            seq,
            file: None,
            deltas: Vec::new(),
            merge_operator,
            unresolved: None,
            point: range.as_point().is_some(),
        };
        let start = RangeBounds::start_bound(range).map(|key| key.as_ref());
        scan.open(start, first_end)?;
        Ok(scan)
    }

    /// Starts reading the current layers from `start` to `end`.
    fn open(&mut self, start: Bound<&[u8]>, end: Bound<Bytes>) -> io::Result<()> {
        let layers = &self.layers;
        self.file = match &layers.file {
            Some(file) => Some(RangeFileCursor::with_bounds(
                file.clone(),
                start,
                end.clone(),
            )?),
            None => None,
        };
        let mut range = None;
        // A delta that is empty now holds no version at or below `seq`: the writer puts each
        // version in its delta before it commits it, and `seq` was loaded before this call
        // (see Appendix B of RFC 0040). So the scan skips it.
        self.deltas = std::iter::once(&layers.active)
            .chain(layers.frozen.as_ref())
            .filter(|delta| !delta.table.is_empty())
            .map(|delta| {
                let range = range.get_or_insert_with(|| {
                    BytesRange::new(start.map(Bytes::copy_from_slice), end.clone())
                });
                DeltaCursor::new(delta, range, self.seq)
            })
            .collect();
        self.part_end = end;
        Ok(())
    }

    /// Moves to the next part, which starts where the current one ends. Returns false after
    /// the last part.
    fn next_part(&mut self) -> io::Result<bool> {
        let Some(start) = self.layers.end.clone() else {
            return Ok(false);
        };
        let reaches_next = match &self.scan_end {
            Unbounded => true,
            Included(end) => *end >= start,
            Excluded(end) => *end > start,
        };
        if !reaches_next {
            return Ok(false);
        }
        let Some(layers) = self.rest.pop_front() else {
            return Ok(false);
        };
        let end = min_end(&self.scan_end, &layers.end);
        self.layers = layers;
        self.open(Included(start.as_ref()), end)?;
        Ok(true)
    }

    /// Finds the next key and its newest visible version, without the normal iterator stack.
    fn step(&mut self) -> io::Result<Step> {
        loop {
            for delta in &mut self.deltas {
                delta.fill();
            }
            // The newest delta with the smallest key. On a tie, the newer delta wins.
            let mut best: Option<usize> = None;
            for i in 0..self.deltas.len() {
                if let Some(key) = self.deltas[i].key() {
                    let smaller = match best {
                        None => true,
                        Some(b) => self.deltas[b].key().is_some_and(|best_key| key < best_key),
                    };
                    if smaller {
                        best = Some(i);
                    }
                }
            }
            let file_key = self.file.as_ref().and_then(|file| file.key());
            let source = match (file_key, best) {
                (None, None) => {
                    if self.next_part()? {
                        continue;
                    }
                    return Ok(Step::End);
                }
                (Some(_), None) => Source::File,
                (None, Some(_)) => Source::Delta,
                (Some(file_key), Some(b)) => {
                    let delta_key = self.deltas[b].key().map_or(&[][..], |k| k.as_ref());
                    match file_key.cmp(delta_key) {
                        std::cmp::Ordering::Less => Source::File,
                        std::cmp::Ordering::Equal => Source::Both,
                        std::cmp::Ordering::Greater => Source::Delta,
                    }
                }
            };
            if source == Source::File {
                let row = match self.file.as_mut() {
                    Some(file) => file.take()?,
                    None => return Ok(Step::End),
                };
                return Ok(match row.value {
                    ValueDeletable::Merge(_) => Step::Resolve(row.key),
                    _ => Step::Row(row),
                });
            }
            if source == Source::Both {
                if let Some(file) = self.file.as_mut() {
                    file.skip()?;
                }
            }
            let Some(b) = best else {
                return Ok(Step::End);
            };
            let Some(entry) = self.deltas[b].peeked.take() else {
                return Ok(Step::End);
            };
            // Older deltas that hold the same key lose.
            for delta in &mut self.deltas {
                if delta.key() == Some(&entry.key) {
                    delta.peeked = None;
                }
            }
            match entry.value {
                ValueDeletable::Value(_) => return Ok(Step::Row(entry)),
                ValueDeletable::Tombstone => continue,
                ValueDeletable::Merge(_) => return Ok(Step::Resolve(entry.key)),
            }
        }
    }

    /// Reads `key` through the normal iterator stack, which resolves merge operands.
    ///
    /// It uses the stack that the normal path uses for this scan. The point-lookup stack
    /// returns a merge result over a tombstone as a merge operand, where the scan stack returns
    /// it as a value. The bytes are the same, but the entry kinds must match too. So for a scan
    /// of more than one key, the range is `[key, key + 0x00)`, which is not a point.
    async fn resolve(&self, key: Bytes) -> Result<Option<RowEntry>, SlateDBError> {
        let range = if self.point {
            BytesRange::new(Included(key.clone()), Included(key))
        } else {
            let mut after = key.to_vec();
            after.push(0);
            BytesRange::new(Included(key), Excluded(Bytes::from(after)))
        };
        let mut iter =
            super::stack_iterator(&self.layers, self.seq, range, self.merge_operator.clone())
                .await?;
        iter.next_entry().await
    }
}

#[async_trait]
impl RowEntryIterator for CachedScan {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        if let Some(key) = self.unresolved.take() {
            if let Some(entry) = self.resolve(key).await? {
                return Ok(Some(entry));
            }
        }
        loop {
            match self.step()? {
                Step::Row(entry) => return Ok(Some(entry)),
                Step::Resolve(key) => {
                    if let Some(entry) = self.resolve(key).await? {
                        return Ok(Some(entry));
                    }
                }
                Step::End => return Ok(None),
            }
        }
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        if self
            .unresolved
            .as_ref()
            .is_some_and(|key| key.as_ref() < next_key)
        {
            self.unresolved = None;
        }
        // Skip the parts that end at or before `next_key`.
        while !self.rest.is_empty()
            && match &self.part_end {
                Unbounded => false,
                Included(end) => next_key > end.as_ref(),
                Excluded(end) => next_key >= end.as_ref(),
            }
        {
            if !self.next_part()? {
                break;
            }
        }
        if let Some(file) = self.file.as_mut() {
            file.seek(next_key)?;
        }
        for delta in &mut self.deltas {
            delta.seek(next_key).await?;
        }
        Ok(())
    }

    fn next_sync(&mut self) -> Option<Result<Option<RowEntry>, SlateDBError>> {
        if self.unresolved.is_some() {
            return None;
        }
        match self.step() {
            Ok(Step::Row(entry)) => Some(Ok(Some(entry))),
            Ok(Step::End) => Some(Ok(None)),
            Ok(Step::Resolve(key)) => {
                self.unresolved = Some(key);
                None
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db_iter::DbIterator;
    use crate::range_cache::file::{RangeCachePageSource, RangeFile, RangeFileWriter};
    use crate::test_utils::StringConcatMergeOperator;
    use proptest::prelude::*;
    use std::collections::BTreeMap;
    use std::ops::Bound::{self, Unbounded};
    use std::sync::atomic::AtomicU64;

    /// The newest seq of a file row. Frozen deltas hold seqs above it, active deltas above
    /// `FROZEN_MAX_SEQ`.
    const FILE_SEQ: u64 = 100;
    const FROZEN_MAX_SEQ: u64 = 200;
    const ACTIVE_MAX_SEQ: u64 = 300;

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Value,
        Tombstone,
        Merge,
    }

    fn key(k: u8) -> Bytes {
        Bytes::from(vec![b'k', k])
    }

    fn entry(k: u8, kind: Kind, seq: u64) -> RowEntry {
        let value = Bytes::from(format!("{k}@{seq}"));
        let value = match kind {
            Kind::Value => ValueDeletable::Value(value),
            Kind::Tombstone => ValueDeletable::Tombstone,
            Kind::Merge => ValueDeletable::Merge(value),
        };
        RowEntry::new(
            key(k),
            value,
            seq,
            (!seq.is_multiple_of(3)).then_some(seq as i64),
            None,
        )
    }

    /// A version as the writer makes it: every write has a create time, from a clock that
    /// does not go back. A merge result takes the newest create time of its versions, so with
    /// such times a tombstone under a merge operand never changes the result. The file drops
    /// tombstones, as compaction does, so the history test depends on this.
    fn history_entry(k: u8, kind: Kind, seq: u64) -> RowEntry {
        let mut entry = entry(k, kind, seq);
        entry.create_ts = Some(seq as i64);
        entry
    }

    fn kind() -> impl Strategy<Value = Kind> {
        prop_oneof![4 => Just(Kind::Value), 2 => Just(Kind::Tombstone), 2 => Just(Kind::Merge)]
    }

    fn bound() -> impl Strategy<Value = Bound<u8>> {
        prop_oneof![
            Just(Unbounded),
            (0u8..40).prop_map(Included),
            (0u8..40).prop_map(Excluded),
        ]
    }

    /// Builds layers from generated versions: file rows (no tombstones), then frozen and active
    /// delta versions with seqs in their own bands. Only keys in `keys` go in.
    fn layers_of(
        dir: &std::path::Path,
        name: &str,
        file_rows: &BTreeMap<u8, (bool, u64)>,
        frozen: &[(u8, Kind, u64)],
        active: &[(u8, Kind, u64)],
        page_source: RangeCachePageSource,
        keys: impl Fn(u8) -> bool,
        end: Option<Bytes>,
    ) -> Arc<Layers> {
        let mut writer = RangeFileWriter::create(dir, name, 512).unwrap();
        for (k, (merge, seq)) in file_rows.iter().filter(|(k, _)| keys(**k)) {
            let kind = if *merge { Kind::Merge } else { Kind::Value };
            writer.push(&entry(*k, kind, *seq)).unwrap();
        }
        let path = writer.finish().unwrap();
        let file = RangeFile::open(path, page_source, Arc::new(AtomicU64::new(0))).unwrap();
        let delta = |versions: &[(u8, Kind, u64)]| {
            let delta = Delta::new();
            for (k, kind, seq) in versions.iter().filter(|(k, _, _)| keys(*k)) {
                delta.table.put(entry(*k, *kind, *seq));
            }
            Arc::new(delta)
        };
        Arc::new(Layers {
            active: delta(active),
            frozen: (!frozen.is_empty()).then(|| delta(frozen)),
            file: Some(Arc::new(file)),
            file_seq: FILE_SEQ,
            end,
        })
    }

    fn layers(
        dir: &std::path::Path,
        file_rows: &BTreeMap<u8, (bool, u64)>,
        frozen: &[(u8, Kind, u64)],
        active: &[(u8, Kind, u64)],
        page_source: RangeCachePageSource,
    ) -> Arc<Layers> {
        layers_of(
            dir,
            "f",
            file_rows,
            frozen,
            active,
            page_source,
            |_| true,
            None,
        )
    }

    async fn drain(iter: &mut DbIterator, limit: usize) -> Vec<RowEntry> {
        let mut out = Vec::new();
        while out.len() < limit {
            match iter.next_entry().await.unwrap() {
                Some(entry) => out.push(entry),
                None => break,
            }
        }
        out
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 300,
            // About one generated range in four is empty and rejected.
            max_global_rejects: 1_000_000,
            ..ProptestConfig::default()
        })]

        /// The direct scan returns exactly what the normal iterator stack returns over the same
        /// layers at the same sequence number, also after a seek. With `split`, the scan walks
        /// two adjacent ranges, split at that key, and the normal stack reads the unsplit layers.
        #[test]
        fn should_match_the_normal_iterator_stack(
            file_rows in proptest::collection::btree_map(0u8..40, (proptest::bool::weighted(0.2), 1..=FILE_SEQ), 0..60),
            frozen in proptest::collection::vec((0u8..40, kind(), FILE_SEQ + 1..=FROZEN_MAX_SEQ), 0..30),
            active in proptest::collection::vec((0u8..40, kind(), FROZEN_MAX_SEQ + 1..=ACTIVE_MAX_SEQ), 0..30),
            seq in FILE_SEQ..=ACTIVE_MAX_SEQ,
            start in bound(),
            end in bound(),
            prefix_rows in 0usize..20,
            seek_to in proptest::option::of(0u8..40),
            mmap in any::<bool>(),
            split in proptest::option::of(1u8..40),
        ) {
            let range = BytesRange::try_new(start.map(key), end.map(key));
            prop_assume!(range.is_some());
            let range = range.unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
            runtime.block_on(async {
                let dir = tempfile::tempdir().unwrap();
                let page_source = if mmap {
                    RangeCachePageSource::Mmap
                } else {
                    RangeCachePageSource::PositionalRead
                };
                let layers = layers(dir.path(), &file_rows, &frozen, &active, page_source);
                let merge: MergeOperatorType = Arc::new(StringConcatMergeOperator);
                let mut expected =
                    crate::range_cache::stack_iterator(&layers, seq, range.clone(), Some(merge.clone()))
                        .await
                        .unwrap();
                // A split scan starts in the part that holds its start.
                let parts = match split {
                    None => vec![layers.clone()],
                    Some(at) => {
                        let part = |name, low: bool, end| {
                            layers_of(dir.path(), name, &file_rows, &frozen, &active, page_source, |k| (k < at) == low, end)
                        };
                        let high = part("high", false, None);
                        let starts_low = match RangeBounds::start_bound(&range) {
                            Unbounded => true,
                            Included(k) | Excluded(k) => *k < key(at),
                        };
                        if starts_low {
                            vec![part("low", true, Some(key(at))), high]
                        } else {
                            vec![high]
                        }
                    }
                };
                let scan = CachedScan::new(parts, seq, &range, Some(merge)).unwrap();
                let mut actual = DbIterator::new_resolved(range.clone(), Box::new(scan), ReadTrace::none())
                    .await
                    .unwrap();

                let want = drain(&mut expected, prefix_rows).await;
                let got = drain(&mut actual, prefix_rows).await;
                assert_eq!(got, want);
                if let Some(target) = seek_to {
                    let target = key(target);
                    let allowed = RangeBounds::contains(&range, &target)
                        && want.last().is_none_or(|last| last.key < target);
                    let want_seek = expected.seek(&target).await;
                    let got_seek = actual.seek(&target).await;
                    assert_eq!(got_seek.is_ok(), want_seek.is_ok());
                    assert_eq!(got_seek.is_ok(), allowed);
                }
                assert_eq!(drain(&mut actual, usize::MAX).await, drain(&mut expected, usize::MAX).await);
            });
        }

        /// Cached reads at any sequence number from the file's to the newest return the state
        /// of the whole history at that number, as the normal path over all versions would.
        ///
        /// The history is one table of all versions. The file holds its state at `FILE_SEQ`,
        /// written the way a load or a rebuild writes it. The deltas hold the versions after
        /// the file, split at `FROZEN_MAX_SEQ` when there is a frozen delta. Unlike the test
        /// above, which reads the same layers both ways, the expected answer here does not
        /// come from the layers.
        #[test]
        fn should_match_the_history_at_any_read_seq(
            history in proptest::collection::vec((0u8..24, kind(), 1..=ACTIVE_MAX_SEQ), 0..120),
            with_frozen in any::<bool>(),
            seq in FILE_SEQ..=ACTIVE_MAX_SEQ,
            start in bound(),
            end in bound(),
            mmap in any::<bool>(),
        ) {
            let range = BytesRange::try_new(start.map(key), end.map(key));
            prop_assume!(range.is_some());
            let range = range.unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
            runtime.block_on(async {
                let dir = tempfile::tempdir().unwrap();
                let page_source = if mmap {
                    RangeCachePageSource::Mmap
                } else {
                    RangeCachePageSource::PositionalRead
                };
                let merge: MergeOperatorType = Arc::new(StringConcatMergeOperator);
                // One version per key and sequence number, as the writer applies them.
                let history: BTreeMap<(u8, u64), Kind> =
                    history.into_iter().map(|(k, kind, seq)| ((k, seq), kind)).collect();
                let all = Delta::new();
                for ((k, seq), kind) in &history {
                    all.table.put(history_entry(*k, *kind, *seq));
                }
                let all = Arc::new(Layers {
                    active: Arc::new(all),
                    frozen: None,
                    file: None,
                    file_seq: 0,
                    end: None,
                });

                // The file: the state at `FILE_SEQ`, through the normal iterator stack.
                let mut writer = RangeFileWriter::create(dir.path(), "f", 512).unwrap();
                let mut rows = crate::range_cache::stack_iterator(
                    &all,
                    FILE_SEQ,
                    BytesRange::unbounded(),
                    Some(merge.clone()),
                )
                .await
                .unwrap();
                while let Some(row) = rows.next_entry().await.unwrap() {
                    writer.push(&row).unwrap();
                }
                let path = writer.finish().unwrap();
                let file = RangeFile::open(path, page_source, Arc::new(AtomicU64::new(0))).unwrap();
                let delta = |low: u64, high: u64| {
                    let delta = Delta::new();
                    for ((k, seq), kind) in &history {
                        if *seq > low && *seq <= high {
                            delta.table.put(history_entry(*k, *kind, *seq));
                        }
                    }
                    Arc::new(delta)
                };
                let (active, frozen) = if with_frozen {
                    (
                        delta(FROZEN_MAX_SEQ, ACTIVE_MAX_SEQ),
                        Some(delta(FILE_SEQ, FROZEN_MAX_SEQ)),
                    )
                } else {
                    (delta(FILE_SEQ, ACTIVE_MAX_SEQ), None)
                };
                let layers = Arc::new(Layers {
                    active,
                    frozen,
                    file: Some(Arc::new(file)),
                    file_seq: FILE_SEQ,
                    end: None,
                });

                // Scans. Compare what a caller sees. The entry kinds can differ: the file drops
                // tombstones, so a merge over a tombstone in the history is a merge over nothing
                // in the cache, which the stack returns as a merge operand with the same bytes.
                let visible = |rows: Vec<RowEntry>| {
                    rows.into_iter().map(crate::types::KeyValue::from).collect::<Vec<_>>()
                };
                let mut expected =
                    crate::range_cache::stack_iterator(&all, seq, range.clone(), Some(merge.clone()))
                        .await
                        .unwrap();
                let scan = CachedScan::new(vec![layers.clone()], seq, &range, Some(merge.clone())).unwrap();
                let mut actual = DbIterator::new_resolved(range.clone(), Box::new(scan), ReadTrace::none())
                    .await
                    .unwrap();
                assert_eq!(
                    visible(drain(&mut actual, usize::MAX).await),
                    visible(drain(&mut expected, usize::MAX).await)
                );

                // Point reads, as `RangeCache::get` and `Pinned::get` answer them.
                for k in 0u8..24 {
                    let point = BytesRange::new(Included(key(k)), Included(key(k)));
                    let want = crate::range_cache::stack_iterator(&all, seq, point, Some(merge.clone()))
                        .await
                        .unwrap()
                        .next_entry()
                        .await
                        .unwrap()
                        .map(crate::types::KeyValue::from);
                    let got = match crate::range_cache::lookup(&layers, &key(k), seq).unwrap() {
                        Some(entry) => match entry.value {
                            ValueDeletable::Value(_) => Some(crate::types::KeyValue::from(entry)),
                            ValueDeletable::Tombstone => None,
                            ValueDeletable::Merge(_) => crate::range_cache::Pinned {
                                layers: layers.clone(),
                                seq,
                            }
                            .get(&key(k), Some(merge.clone()))
                            .await
                            .unwrap(),
                        },
                        None => None,
                    };
                    assert_eq!(got, want, "key {k} at {seq}");
                }
            });
        }
    }
}
