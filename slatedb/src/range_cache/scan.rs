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

use super::buffer::{Cursor, Kind, Version};
use super::delta::{past_end, to_row, Delta};
use super::Layers;
use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::merge_operator::MergeOperatorType;
use crate::range_cache::file::RangeFileCursor;
use crate::types::{RowEntry, ValueDeletable};

/// The deltas of a set of layers, newest first: the active delta, then the frozen delta.
const DELTAS: usize = 2;

/// The delta `i` of `layers`, in the order of [`DELTAS`].
fn delta_of(layers: &Layers, i: usize) -> Option<&Delta> {
    match i {
        0 => Some(&layers.active),
        _ => layers.frozen.as_deref(),
    }
}

/// A cursor over the versions of a delta that are visible at one sequence number. It is a
/// position in the delta's buffer: the scan reads it through the layers that it holds, so the
/// cursor takes no reference of its own and reads the buffer in place.
struct DeltaCursor {
    pos: Cursor,
    seq: u64,
    /// `pos` is at the newest visible version of its key, which the scan has not used yet.
    peeked: bool,
    /// The cursor passed the end of the part.
    done: bool,
}

impl DeltaCursor {
    fn new(delta: &Delta, start: Bound<&[u8]>, seq: u64) -> Self {
        Self {
            pos: delta.buffer().seek(start),
            seq,
            peeked: false,
            done: false,
        }
    }

    /// Moves to the newest visible version of the next key, unless the cursor is at one. A delta
    /// holds the versions of a key newest first, so the first version at or below `seq` is the
    /// newest visible one.
    fn fill(&mut self, delta: &Delta, end: &Bound<Bytes>) {
        let buffer = delta.buffer();
        while !self.peeked && !self.done {
            match buffer.version(&self.pos) {
                None => self.done = true,
                Some(version) if past_end(end, version.key) => self.done = true,
                Some(version) if version.seq <= self.seq => self.peeked = true,
                Some(_) => buffer.advance(&mut self.pos),
            }
        }
    }

    /// The version that `fill` found.
    fn version<'a>(&self, delta: &'a Delta) -> Option<Version<'a>> {
        if self.peeked {
            delta.buffer().version(&self.pos)
        } else {
            None
        }
    }

    fn key<'a>(&self, delta: &'a Delta) -> Option<&'a [u8]> {
        self.version(delta).map(|version| version.key)
    }

    /// Leaves the key of the version that `fill` found: moves past its older versions.
    fn next_key(&mut self, delta: &Delta) {
        let buffer = delta.buffer();
        if let Some(key) = self.key(delta) {
            buffer.advance(&mut self.pos);
            while buffer.version(&self.pos).is_some_and(|v| v.key == key) {
                buffer.advance(&mut self.pos);
            }
        }
        self.peeked = false;
    }

    /// Moves forward to the first key at or after `key`. It never moves backwards.
    fn seek(&mut self, delta: &Delta, key: &[u8]) {
        if self.done {
            return;
        }
        let buffer = delta.buffer();
        if buffer.version(&self.pos).is_some_and(|v| v.key < key) {
            self.pos = buffer.seek(Included(key));
            self.peeked = false;
        }
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
pub(super) fn min_end(scan_end: &Bound<Bytes>, range_end: &Option<Bytes>) -> Bound<Bytes> {
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
    /// A cursor over the file of the current layers, which it reads through `layers`.
    file: Option<RangeFileCursor>,
    /// A cursor for each non-empty delta of the current layers, in the order of [`DELTAS`].
    deltas: [Option<DeltaCursor>; DELTAS],
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
            deltas: [None, None],
            merge_operator,
            unresolved: None,
            point: range.as_point().is_some(),
        };
        let start = RangeBounds::start_bound(range).map(|key| key.as_ref());
        scan.open(start, first_end)?;
        Ok(scan)
    }

    /// Sets whether the scan that this one is part of covers one key. A part of a mixed scan
    /// resolves merge operands with the iterator stack of the whole scan. See `resolve`.
    pub(super) fn set_point(&mut self, point: bool) {
        self.point = point;
    }

    /// Starts reading the current layers from `start` to `end`.
    fn open(&mut self, start: Bound<&[u8]>, end: Bound<Bytes>) -> io::Result<()> {
        let layers = &self.layers;
        // The cursor reads the file of the pinned layers, so it takes no reference to it.
        self.file = match &layers.file {
            Some(file) => Some(RangeFileCursor::with_bounds(file, start, end.clone())?),
            None => None,
        };
        // A delta that is empty now holds no version at or below `seq`: the writer puts each
        // version in its delta before it commits it, and `seq` was loaded before this call
        // (see Appendix B of RFC 0040). So the scan skips it.
        let seq = self.seq;
        self.deltas = std::array::from_fn(|i| {
            delta_of(layers, i)
                .filter(|delta| !delta.is_empty())
                .map(|delta| DeltaCursor::new(delta, start, seq))
        });
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
            let layers = &*self.layers;
            for (i, cursor) in self.deltas.iter_mut().enumerate() {
                if let (Some(cursor), Some(delta)) = (cursor, delta_of(layers, i)) {
                    cursor.fill(delta, &self.part_end);
                }
            }
            // The newest delta with the smallest key. On a tie, the newer delta wins.
            let mut best: Option<(usize, &[u8])> = None;
            for (i, cursor) in self.deltas.iter().enumerate() {
                let key = match (cursor, delta_of(layers, i)) {
                    (Some(cursor), Some(delta)) => cursor.key(delta),
                    _ => None,
                };
                if let Some(key) = key {
                    if best.is_none_or(|(_, best_key)| key < best_key) {
                        best = Some((i, key));
                    }
                }
            }
            let file_key = self.file.as_ref().and_then(|file| file.key());
            let source = match (file_key, best) {
                (None, None) => None,
                (Some(_), None) => Some(Source::File),
                (None, Some(_)) => Some(Source::Delta),
                (Some(file_key), Some((_, delta_key))) => Some(match file_key.cmp(delta_key) {
                    std::cmp::Ordering::Less => Source::File,
                    std::cmp::Ordering::Equal => Source::Both,
                    std::cmp::Ordering::Greater => Source::Delta,
                }),
            };
            let best = best.map(|(i, _)| i);
            let Some(source) = source else {
                if self.next_part()? {
                    continue;
                }
                return Ok(Step::End);
            };
            if source == Source::File {
                let row = match (self.file.as_mut(), &self.layers.file) {
                    (Some(cursor), Some(file)) => cursor.take(file)?,
                    _ => return Ok(Step::End),
                };
                return Ok(match row.value {
                    ValueDeletable::Merge(_) => Step::Resolve(row.key),
                    _ => Step::Row(row),
                });
            }
            if source == Source::Both {
                if let (Some(cursor), Some(file)) = (self.file.as_mut(), &self.layers.file) {
                    cursor.skip(file)?;
                }
            }
            let layers = &*self.layers;
            let found = best.and_then(|i| {
                let delta = delta_of(layers, i)?;
                Some((i, delta, self.deltas[i].as_ref()?.version(delta)?))
            });
            let Some((b, delta, version)) = found else {
                return Ok(Step::End);
            };
            // Older deltas that hold the same key lose.
            for (i, cursor) in self.deltas.iter_mut().enumerate().skip(b + 1) {
                if let (Some(cursor), Some(older)) = (cursor, delta_of(layers, i)) {
                    if cursor.key(older) == Some(version.key) {
                        cursor.next_key(older);
                    }
                }
            }
            // Copy out only what the scan returns.
            let step = match version.kind {
                Kind::Value => Some(Step::Row(to_row(&version))),
                Kind::Tombstone => None,
                Kind::Merge => Some(Step::Resolve(Bytes::copy_from_slice(version.key))),
            };
            if let Some(cursor) = self.deltas[b].as_mut() {
                cursor.next_key(delta);
            }
            if let Some(step) = step {
                return Ok(step);
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
        if let (Some(cursor), Some(file)) = (self.file.as_mut(), &self.layers.file) {
            cursor.seek(file, next_key)?;
        }
        let layers = &*self.layers;
        for (i, cursor) in self.deltas.iter_mut().enumerate() {
            if let (Some(cursor), Some(delta)) = (cursor, delta_of(layers, i)) {
                cursor.seek(delta, next_key);
            }
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
    use crate::range_cache::delta::DeltaMemory;
    use crate::range_cache::file::{RangeCachePageSource, RangeFile, RangeFileWriter};
    use crate::read_trace::ReadTrace;
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
            let delta = Delta::new(&DeltaMemory::detached());
            let mut writer = delta.writer();
            for (k, kind, seq) in versions.iter().filter(|(k, _, _)| keys(*k)) {
                writer.append(&entry(*k, *kind, *seq));
            }
            drop(writer);
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
                let all = Delta::new(&DeltaMemory::detached());
                let mut writer = all.writer();
                for ((k, seq), kind) in &history {
                    writer.append(&history_entry(*k, *kind, *seq));
                }
                drop(writer);
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
                    let delta = Delta::new(&DeltaMemory::detached());
                    let mut writer = delta.writer();
                    for ((k, seq), kind) in &history {
                        if *seq > low && *seq <= high {
                            writer.append(&history_entry(*k, *kind, *seq));
                        }
                    }
                    drop(writer);
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
                        crate::range_cache::Found::Value(kv) => Some(kv),
                        crate::range_cache::Found::Absent => None,
                        crate::range_cache::Found::Merge => crate::range_cache::Pinned {
                            layers: layers.clone(),
                            seq,
                        }
                        .get(&key(k), Some(merge.clone()))
                        .await
                        .unwrap(),
                    };
                    assert_eq!(got, want, "key {k} at {seq}");
                }
            });
        }
    }
}
