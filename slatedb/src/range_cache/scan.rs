//! The scan path of the range cache.
//!
//! [`CachedScan`] walks the pinned layers of cached ranges directly (design 01 section 5.2): a
//! cursor on the base, and one on each buffer and run that can hold a version visible at the
//! pinned sequence number. Buffers and runs can hold several versions of a key; their cursors
//! skip the versions above the sequence number. For each key the scan takes the newest visible
//! version from the newest source that holds the key (buffers, then runs, then the base), and it
//! skips tombstones.
//!
//! A scan can span adjacent cached ranges. It then walks the parts in key order, all pinned at
//! the same sequence number. The parts hold disjoint keys, so the scan is their concatenation.
//!
//! A merge operand needs the merge operator and the older versions of its key. For such a key,
//! the scan reads that key alone through the normal iterator stack, at the same sequence number.
//! No other row needs the normal stack: cached ranges hold no expiring rows, and a base holds
//! only values and base-less merge operands.
//!
//! The cursors hold positions, not references: they read their buffer or file through the layers
//! that the scan holds. So a scan takes one reference count, on its layers, and none per source.

use std::collections::VecDeque;
use std::io;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::ops::RangeBounds;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use super::buffer::{Cursor, Kind, Version};
use super::delta::{past_end, to_row, Delta};
use super::{Layers, Run};
use crate::bytes_range::BytesRange;
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::merge_operator::MergeOperatorType;
use crate::range_cache::file::{RangeFile, RangeFileCursor};
use crate::types::{RowEntry, ValueDeletable};

/// Buffer `i` of `layers`, in the order of [`Layers::buffers`]: 0 is the active buffer.
fn buffer_of(layers: &Layers, i: usize) -> &Delta {
    match i {
        0 => &layers.active,
        _ => &layers.sealed[i - 1],
    }
}

/// A cursor over the versions of a buffer that are visible at one sequence number. It is a
/// position in the buffer: the scan reads it through the layers that it holds, so the cursor
/// takes no reference of its own and reads the buffer in place.
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

    /// Moves to the newest visible version of the next key, unless the cursor is at one. A buffer
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

    /// Leaves `key` if the cursor is at one of its versions, visible or not.
    fn skip_key(&mut self, delta: &Delta, key: &[u8]) {
        if self.done {
            return;
        }
        let buffer = delta.buffer();
        let mut moved = false;
        while buffer.version(&self.pos).is_some_and(|v| v.key == key) {
            buffer.advance(&mut self.pos);
            moved = true;
        }
        if moved {
            self.peeked = false;
        }
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

/// A cursor over the versions of a run that are visible at one sequence number. Like
/// [`DeltaCursor`], it reads the run's file through the layers that the scan holds.
struct RunCursor {
    cursor: RangeFileCursor,
    seq: u64,
    /// The cursor is at the newest visible version of its key, which the scan has not used yet.
    peeked: bool,
}

impl RunCursor {
    fn new(run: &Run, start: Bound<&[u8]>, end: Bound<Bytes>, seq: u64) -> io::Result<Self> {
        Ok(Self {
            cursor: RangeFileCursor::with_bounds(&run.file, start, end)?,
            seq,
            peeked: false,
        })
    }

    /// Moves to the newest visible version of the next key, unless the cursor is at one.
    fn fill(&mut self, file: &RangeFile) -> io::Result<()> {
        while !self.peeked {
            match self.cursor.seq() {
                None => return Ok(()),
                Some(seq) if seq <= self.seq => self.peeked = true,
                Some(_) => self.cursor.skip(file)?,
            }
        }
        Ok(())
    }

    fn key(&self) -> Option<&[u8]> {
        if self.peeked {
            self.cursor.key()
        } else {
            None
        }
    }

    /// The version that `fill` found, then leaves its key.
    fn take(&mut self, file: &RangeFile) -> io::Result<RowEntry> {
        let row = self.cursor.take(file)?;
        self.peeked = false;
        self.skip_key(file, &row.key)?;
        Ok(row)
    }

    /// Leaves `key` if the cursor is at one of its versions, visible or not.
    fn skip_key(&mut self, file: &RangeFile, key: &[u8]) -> io::Result<()> {
        let mut moved = false;
        while self.cursor.key() == Some(key) {
            self.cursor.skip(file)?;
            moved = true;
        }
        if moved {
            self.peeked = false;
        }
        Ok(())
    }

    /// Moves forward to the first key at or after `key`. It never moves backwards.
    fn seek(&mut self, file: &RangeFile, key: &[u8]) -> io::Result<()> {
        if self.cursor.key().is_some_and(|current| current < key) {
            self.cursor.seek(file, key)?;
            self.peeked = false;
        }
        Ok(())
    }
}

/// A cursor over one buffer or run of the current part, with its place in the layers. A run's
/// cursor is large (it holds a page copy and the page's row offsets), so it is boxed.
enum Source {
    Buffer(usize, DeltaCursor),
    Run(usize, Box<RunCursor>),
}

impl Source {
    fn fill(&mut self, layers: &Layers, end: &Bound<Bytes>) -> io::Result<()> {
        match self {
            Source::Buffer(i, cursor) => {
                cursor.fill(buffer_of(layers, *i), end);
                Ok(())
            }
            Source::Run(i, cursor) => cursor.fill(&layers.runs[*i].file),
        }
    }

    /// The key of the version that `fill` found.
    fn key<'a>(&'a self, layers: &'a Layers) -> Option<&'a [u8]> {
        match self {
            Source::Buffer(i, cursor) => cursor.key(buffer_of(layers, *i)),
            Source::Run(_, cursor) => cursor.key(),
        }
    }

    fn skip_key(&mut self, layers: &Layers, key: &[u8]) -> io::Result<()> {
        match self {
            Source::Buffer(i, cursor) => {
                cursor.skip_key(buffer_of(layers, *i), key);
                Ok(())
            }
            Source::Run(i, cursor) => cursor.skip_key(&layers.runs[*i].file, key),
        }
    }

    fn seek(&mut self, layers: &Layers, key: &[u8]) -> io::Result<()> {
        match self {
            Source::Buffer(i, cursor) => {
                cursor.seek(buffer_of(layers, *i), key);
                Ok(())
            }
            Source::Run(i, cursor) => cursor.seek(&layers.runs[*i].file, key),
        }
    }
}

/// The key of the version that a step takes, borrowed from a buffer or held by the run's row.
enum StepKey<'a> {
    Buffer(&'a [u8]),
    Run(Bytes),
}

impl StepKey<'_> {
    fn as_bytes(&self) -> &[u8] {
        match self {
            StepKey::Buffer(key) => key,
            StepKey::Run(key) => key,
        }
    }
}

/// Returns true if a run can hold a key of `[start, end)`.
fn run_meets(run: &Run, start: Bound<&[u8]>, end: &Bound<Bytes>) -> bool {
    let after_start = match start {
        Unbounded => true,
        Included(start) => run.max_key.as_ref() >= start,
        Excluded(start) => run.max_key.as_ref() > start,
    };
    after_start && !past_end(end, &run.min_key)
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
    /// A cursor over the base of the current layers, which it reads through `layers`.
    base: Option<RangeFileCursor>,
    /// A cursor for each buffer and run of the current layers that can hold a visible version,
    /// newest first.
    sources: Vec<Source>,
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
            base: None,
            sources: Vec::new(),
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
        // The cursors read the files of the pinned layers, so they take no reference to them.
        self.base = match &layers.base {
            Some(file) => Some(RangeFileCursor::with_bounds(file, start, end.clone())?),
            None => None,
        };
        let seq = self.seq;
        self.sources.clear();
        // A buffer that is empty now holds no version at or below `seq`: the writer puts each
        // version in its buffer before it commits it, and `seq` was loaded before this call (see
        // Appendix B of RFC 0040). A buffer or run whose versions are all above `seq` holds
        // nothing for the scan either. So the scan skips them.
        for (i, delta) in layers.buffers().enumerate() {
            if !delta.is_empty() && delta.lo() < seq {
                self.sources
                    .push(Source::Buffer(i, DeltaCursor::new(delta, start, seq)));
            }
        }
        for (i, run) in layers.runs.iter().enumerate() {
            if run.lo < seq && run.file.rows() > 0 && run_meets(run, start, &end) {
                let cursor = RunCursor::new(run, start, end.clone(), seq)?;
                self.sources.push(Source::Run(i, Box::new(cursor)));
            }
        }
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
            for source in self.sources.iter_mut() {
                source.fill(layers, &self.part_end)?;
            }
            // The newest source with the smallest key. On a tie, the newer source wins.
            let mut best: Option<(usize, &[u8])> = None;
            for (i, source) in self.sources.iter().enumerate() {
                if let Some(key) = source.key(layers) {
                    if best.is_none_or(|(_, best_key)| key < best_key) {
                        best = Some((i, key));
                    }
                }
            }
            let base_key = self.base.as_ref().and_then(|base| base.key());
            let order = match (base_key, best) {
                (None, None) => None,
                (Some(_), None) => Some(std::cmp::Ordering::Less),
                (None, Some(_)) => Some(std::cmp::Ordering::Greater),
                (Some(base_key), Some((_, key))) => Some(base_key.cmp(key)),
            };
            let best = best.map(|(i, _)| i);
            let Some(order) = order else {
                if self.next_part()? {
                    continue;
                }
                return Ok(Step::End);
            };
            if order == std::cmp::Ordering::Less {
                let row = match (self.base.as_mut(), &self.layers.base) {
                    (Some(cursor), Some(file)) => cursor.take(file)?,
                    _ => return Ok(Step::End),
                };
                return Ok(match row.value {
                    ValueDeletable::Merge(_) => Step::Resolve(row.key),
                    _ => Step::Row(row),
                });
            }
            if order == std::cmp::Ordering::Equal {
                if let (Some(cursor), Some(file)) = (self.base.as_mut(), &layers.base) {
                    cursor.skip(file)?;
                }
            }
            let Some(b) = best else {
                return Ok(Step::End);
            };
            // The key of the version to take. Older sources that hold it lose.
            let key = match &self.sources[b] {
                Source::Buffer(i, cursor) => cursor.key(buffer_of(layers, *i)).map(StepKey::Buffer),
                Source::Run(_, cursor) => cursor.cursor.key_bytes().map(StepKey::Run),
            };
            let Some(key) = key else {
                return Ok(Step::End);
            };
            for (j, source) in self.sources.iter_mut().enumerate() {
                if j != b {
                    source.skip_key(layers, key.as_bytes())?;
                }
            }
            // Copy out only what the scan returns.
            let step = match &mut self.sources[b] {
                Source::Buffer(i, cursor) => {
                    let delta = buffer_of(layers, *i);
                    let step = cursor
                        .version(delta)
                        .and_then(|version| match version.kind {
                            Kind::Value => Some(Step::Row(to_row(&version))),
                            Kind::Tombstone => None,
                            Kind::Merge => Some(Step::Resolve(Bytes::copy_from_slice(version.key))),
                        });
                    cursor.skip_key(delta, key.as_bytes());
                    step
                }
                Source::Run(i, cursor) => {
                    let row = cursor.take(&layers.runs[*i].file)?;
                    match row.value {
                        ValueDeletable::Value(_) => Some(Step::Row(row)),
                        ValueDeletable::Tombstone => None,
                        ValueDeletable::Merge(_) => Some(Step::Resolve(row.key)),
                    }
                }
            };
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
        if let (Some(cursor), Some(file)) = (self.base.as_mut(), &self.layers.base) {
            cursor.seek(file, next_key)?;
        }
        let layers = &*self.layers;
        for source in self.sources.iter_mut() {
            source.seek(layers, next_key)?;
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
    use crate::range_cache::file::{RangeCachePageSource, RangeFileWriter};
    use crate::read_trace::ReadTrace;
    use crate::test_utils::StringConcatMergeOperator;
    use proptest::prelude::*;
    use std::collections::BTreeMap;
    use std::ops::Bound::{self, Unbounded};
    use std::sync::atomic::Ordering;

    /// The sequence number of the base of the test layers. Runs, sealed buffers and the active
    /// buffer hold the versions above it, in that order, up to `MAX_SEQ`.
    const BASE_SEQ: u64 = 100;
    const MAX_SEQ: u64 = 300;

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
    /// such times a tombstone under a merge operand never changes the result. The base drops
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

    /// Where the versions above the base go. The cuts split `(BASE_SEQ, MAX_SEQ]` into
    /// intervals: the first `runs` are runs, the last one is the active buffer, and the ones in
    /// between are sealed buffers.
    #[derive(Clone, Debug)]
    struct Shape {
        cuts: Vec<u64>,
        runs: usize,
        /// How run `i` is written, by the three bits at `3 * i`: 0 straight from its versions,
        /// 1 as a flush of two sealed buffers that split its interval, 2 as a minor merge of two
        /// runs that split it, 3 as a flush-merge of a buffer with a run, 4 and 5 as minor merges
        /// of three and four runs (cascades of merges by similar size), 6 as a flush-merge of a
        /// buffer with two runs, 7 as a flush-merge of two buffers with two runs.
        writes: u32,
    }

    /// The sealed buffers and the runs, newest first, whose merge writes a run of `how`.
    fn merge_inputs(how: u32) -> (usize, usize) {
        match how {
            1 => (2, 0),
            2 => (0, 2),
            3 => (1, 1),
            4 => (0, 3),
            5 => (0, 4),
            6 => (1, 2),
            _ => (2, 2),
        }
    }

    fn shape() -> impl Strategy<Value = Shape> {
        proptest::collection::btree_set(BASE_SEQ + 1..MAX_SEQ, 0..10).prop_flat_map(|cuts| {
            let cuts: Vec<u64> = cuts.into_iter().collect();
            let most = cuts.len().min(*crate::range_cache::MAX_RUNS_RANGE.end());
            (Just(cuts), 0..=most, any::<u32>()).prop_map(|(cuts, runs, writes)| Shape {
                cuts,
                runs,
                writes,
            })
        })
    }

    /// One version per key and sequence number, as the writer applies them.
    type History = BTreeMap<(u8, u64), Kind>;

    /// Layers with only an active buffer that holds every version of `history`.
    fn all_of(history: &History, make: fn(u8, Kind, u64) -> RowEntry) -> Arc<Layers> {
        let all = Delta::for_test(&DeltaMemory::detached(), 0);
        let mut writer = all.writer();
        for ((k, seq), kind) in history {
            writer.append(&make(*k, *kind, *seq));
        }
        drop(writer);
        Arc::new(Layers {
            active: Arc::new(all),
            sealed: Vec::new(),
            runs: Vec::new(),
            base: None,
            base_seq: 0,
            covered_to: 0,
            end: None,
        })
    }

    /// Builds layers of `shape` from the versions of `history` whose keys pass `keys`: a base
    /// with the state at `BASE_SEQ`, written the way a load or a major merge writes it, then
    /// runs and buffers with every version of their intervals.
    #[allow(clippy::too_many_arguments)]
    async fn layers_of(
        dir: &std::path::Path,
        name: &str,
        history: &History,
        make: fn(u8, Kind, u64) -> RowEntry,
        shape: &Shape,
        page_source: RangeCachePageSource,
        keys: impl Fn(u8) -> bool,
        end: Option<Bytes>,
    ) -> Arc<Layers> {
        let history: History = history
            .iter()
            .filter(|((k, _), _)| keys(*k))
            .map(|(k, v)| (*k, *v))
            .collect();
        let merge: MergeOperatorType = Arc::new(StringConcatMergeOperator);
        let disk = crate::range_cache::file::FileBudget::detached();
        let mut writer = RangeFileWriter::create(dir, &format!("{name}-base"), 512).unwrap();
        let mut rows = crate::range_cache::stack_iterator(
            &all_of(&history, make),
            BASE_SEQ,
            BytesRange::unbounded(),
            Some(merge),
        )
        .await
        .unwrap();
        while let Some(row) = rows.next_entry().await.unwrap() {
            writer.push(&row).unwrap();
        }
        let path = writer.finish().unwrap();
        let base = RangeFile::open(path, page_source, disk.clone()).unwrap();

        // The versions of `(lo, hi]`, key ascending and newest first.
        let versions = |lo: u64, hi: u64| {
            let mut out: Vec<RowEntry> = history
                .iter()
                .filter(|((_, seq), _)| *seq > lo && *seq <= hi)
                .map(|((k, seq), kind)| make(*k, *kind, *seq))
                .collect();
            out.sort_by(|a, b| a.key.cmp(&b.key).then(b.seq.cmp(&a.seq)));
            out
        };
        let bounds: Vec<u64> = std::iter::once(BASE_SEQ)
            .chain(shape.cuts.iter().copied())
            .chain(std::iter::once(MAX_SEQ))
            .collect();
        let intervals = bounds.len() - 1;
        let file_of = |file_name: String, lo: u64, hi: u64| {
            let mut writer = RangeFileWriter::create_run(dir, &file_name, 512).unwrap();
            for row in versions(lo, hi) {
                writer.push(&row).unwrap();
            }
            let path = writer.finish().unwrap();
            let file = RangeFile::open(path, page_source, disk.clone()).unwrap();
            Run::new(file, lo, hi).unwrap()
        };
        let buffer_of = |lo: u64, hi: u64| {
            let delta = Delta::for_test(&DeltaMemory::detached(), lo);
            let mut writer = delta.writer();
            for row in versions(lo, hi) {
                writer.append(&row);
            }
            drop(writer);
            delta
        };
        let mut runs = Vec::new();
        for i in 0..shape.runs {
            let (lo, hi) = (bounds[i], bounds[i + 1]);
            let run = match (shape.writes >> (3 * i)) & 7 {
                0 => file_of(format!("{name}-run{i}"), lo, hi),
                how => {
                    // Inputs that split the interval into parts, written by the merge of
                    // flushes, flush-merges and minor merges. Part `j`, newest first, covers
                    // `(edges[parts - j - 1], edges[parts - j]]`; the newest parts are the
                    // sealed buffers, as in a flush-merge. Parts can be empty.
                    let (buffer_parts, run_parts) = merge_inputs(how);
                    let parts = buffer_parts + run_parts;
                    let edges: Vec<u64> = (0..=parts)
                        .map(|k| lo + (hi - lo) * k as u64 / parts as u64)
                        .collect();
                    let part = |j: usize| (edges[parts - j - 1], edges[parts - j]);
                    let sealed: Vec<(Delta, u64)> = (0..buffer_parts)
                        .map(|j| {
                            let (l, h) = part(j);
                            (buffer_of(l, h), h)
                        })
                        .collect();
                    let merged: Vec<Run> = (buffer_parts..parts)
                        .map(|j| {
                            let (l, h) = part(j);
                            file_of(format!("{name}-run{i}-part{j}"), l, h)
                        })
                        .collect();
                    let buffers: Vec<(&Delta, u64)> = sealed.iter().map(|(d, s)| (d, *s)).collect();
                    let inputs: Vec<&Run> = merged.iter().collect();
                    let mut writer =
                        RangeFileWriter::create_run(dir, &format!("{name}-run{i}"), 512).unwrap();
                    let written =
                        crate::range_cache::merge::merge_versions(&mut writer, &buffers, &inputs)
                            .unwrap();
                    assert_eq!(written as usize, versions(lo, hi).len());
                    let path = writer.finish().unwrap();
                    let file = RangeFile::open(path, page_source, disk.clone()).unwrap();
                    Run::new(file, lo, hi).unwrap()
                }
            };
            runs.push(Arc::new(run));
        }
        runs.reverse();
        let buffer = |lo: u64, hi: u64| {
            let delta = Delta::for_test(&DeltaMemory::detached(), lo);
            let mut writer = delta.writer();
            for row in versions(lo, hi) {
                writer.append(&row);
            }
            drop(writer);
            Arc::new(delta)
        };
        let mut sealed = Vec::new();
        for i in shape.runs..intervals - 1 {
            let delta = buffer(bounds[i], bounds[i + 1]);
            delta.sealed_at().store(bounds[i + 1], Ordering::Release);
            sealed.push(delta);
        }
        sealed.reverse();
        let layers = Layers {
            active: buffer(bounds[intervals - 1], MAX_SEQ),
            sealed,
            runs,
            base: Some(Arc::new(base)),
            base_seq: BASE_SEQ,
            covered_to: bounds[shape.runs],
            end,
        };
        layers.check(*crate::range_cache::MAX_RUNS_RANGE.end());
        Arc::new(layers)
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

    fn page_source(mmap: bool) -> RangeCachePageSource {
        if mmap {
            RangeCachePageSource::Mmap
        } else {
            RangeCachePageSource::PositionalRead
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 300,
            // About one generated range in four is empty and rejected.
            max_global_rejects: 1_000_000,
            ..ProptestConfig::default()
        })]

        /// The direct scan returns exactly what the normal iterator stack returns over the same
        /// layers at the same sequence number, also after a seek. The layers have a base, up to
        /// eight runs, sealed buffers and the active buffer. With `split`, the scan walks two
        /// adjacent ranges, split at that key, and the normal stack reads the unsplit layers.
        #[test]
        fn should_match_the_normal_iterator_stack(
            history in proptest::collection::btree_map((0u8..40, 1..=MAX_SEQ), kind(), 0..150),
            shape in shape(),
            seq in BASE_SEQ..=MAX_SEQ,
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
                let source = page_source(mmap);
                let layers = layers_of(dir.path(), "f", &history, entry, &shape, source, |_| true, None).await;
                let merge: MergeOperatorType = Arc::new(StringConcatMergeOperator);
                let mut expected =
                    crate::range_cache::stack_iterator(&layers, seq, range.clone(), Some(merge.clone()))
                        .await
                        .unwrap();
                // A split scan starts in the part that holds its start.
                let parts = match split {
                    None => vec![layers.clone()],
                    Some(at) => {
                        let high = layers_of(dir.path(), "high", &history, entry, &shape, source, |k| k >= at, None).await;
                        let starts_low = match RangeBounds::start_bound(&range) {
                            Unbounded => true,
                            Included(k) | Excluded(k) => *k < key(at),
                        };
                        if starts_low {
                            let low = layers_of(dir.path(), "low", &history, entry, &shape, source, |k| k < at, Some(key(at))).await;
                            vec![low, high]
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

        /// Cached reads at any sequence number from the base's to the newest return the state
        /// of the whole history at that number, as the normal path over all versions would.
        ///
        /// The history is one table of all versions. The base holds its state at `BASE_SEQ`,
        /// written the way a load or a major merge writes it; up to eight runs, sealed buffers and
        /// the active buffer hold the versions after the base. Unlike the test above, which reads
        /// the same layers both ways, the expected answer here does not come from the layers.
        /// Keys have many versions, so a run often holds several versions of a key, across its
        /// pages, and a read at an older sequence number stops at a version below the newest.
        #[test]
        fn should_match_the_history_at_any_read_seq(
            history in proptest::collection::btree_map((0u8..24, 1..=MAX_SEQ), kind(), 0..200),
            shape in shape(),
            seq in BASE_SEQ..=MAX_SEQ,
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
                let merge: MergeOperatorType = Arc::new(StringConcatMergeOperator);
                let all = all_of(&history, history_entry);
                let layers = layers_of(dir.path(), "f", &history, history_entry, &shape, page_source(mmap), |_| true, None).await;

                // Scans. Compare what a caller sees. The entry kinds can differ: the base drops
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
                    assert_eq!(got, want, "key {k} at {seq} in {shape:?}");
                }
            });
        }
    }
}
