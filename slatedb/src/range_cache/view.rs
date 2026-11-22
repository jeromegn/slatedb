//! The delta view of a cached range: one sorted view over the versions of all its runs, so that a
//! scan opens one cursor however many runs the range holds, and runs never merge with each other
//! to keep scans fast (the idea of REMIX, Zhong et al., FAST '21). Every range that holds runs has
//! a view, and a scan reads the runs only through it.
//!
//! A view is immutable. It lists the versions of the runs of the range, in the range's keys, that
//! no chunk of the range's base has absorbed (design 04 I12; for a range of one chunk that was
//! never split, every version of every run; a range split from another shares its runs with the
//! other children of the split, and its view lists only its own keys) in the order of a run (key
//! ascending, then sequence number descending) as a *locator*: the run's slot and the row's
//! offset in the run's file. Runs hold
//! disjoint intervals of sequence numbers, so the versions of all runs have one total order, and
//! the versions of one key are adjacent, newest first. A run's rows appear in the view in the
//! order of its file.
//!
//! Each entry has, in four arrays of the heap:
//!
//! - one selector byte: the slot of the run (6 bits, so at most [`VIEW_SLOTS`] runs) and a flag
//!   that the entry is the first, newest, version of its key in the view. A cursor moves from key
//!   to key by the flag, and from run to run without a key comparison.
//! - the row's offset in its run file (4 bytes; run files must be smaller than 4 GiB).
//! - for the first version of a key, the length of the prefix that it shares with the previous
//!   key of the view, capped at [`LCP_CAP`], and its byte after that prefix (front coding, 2
//!   bytes). A search compares a key with these two bytes and the length it shares with the
//!   previous key, and reads a key from its file only when they do not decide.
//!
//! Every [`SEGMENT`] entries or so, the first version of a key starts a segment, and its whole key,
//! after the prefix that every key of the view shares, is an *anchor* in memory. A seek is one
//! binary search over the anchors and a walk of the front-coded bytes of one segment, so it reads
//! no file in the common case: whole keys are kept, not a fixed number of bytes, so keys with long
//! shared prefixes compare as well as short ones.
//!
//! A flush builds the view with its new run from the view of the range (see
//! [`DeltaView::with_run`]): the run's writer collects the keys and offsets of its rows as it
//! writes them, so the build does not read the new run; it walks the old view by its anchors and
//! front coding, copies the old entries between the points where new keys go, and reads an old
//! key only where the front coding does not decide an order. A major merge builds the view
//! without the runs it absorbed, and a flush at the run cap the view without the newest runs that
//! it merged into its run (see [`DeltaView::without`]); dropping runs needs no key at all except
//! for new anchors. A split gives each child the view of its keys (see [`DeltaView::restrict`]),
//! which copies the entries between two seeks the same way. The batch writer publishes the view in
//! the same store as the runs, so a reader's view always lists exactly the runs of its layers.
//!
//! A view holds nothing that the run files do not: it can be built again from them alone, as
//! `DeltaView::from_runs` does in tests.

use std::borrow::Cow;
use std::cmp::Ordering as Cmp;
use std::io;
use std::ops::Bound;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::file::{RowLocators, RowSpan};
use super::Run;

/// The most runs that one view holds: the slot field of a selector has 6 bits.
pub(crate) const VIEW_SLOTS: usize = 64;
/// The slot bits of a selector.
const SLOT: u8 = 0x3f;
/// The selector flag of the first, newest, version of a key.
const FIRST: u8 = 0x80;
/// The entries of a segment that a build aims for. A new key starts a segment after this many
/// entries, an old key after twice this many (it needs a key read).
pub(super) const SEGMENT: usize = 16;
/// The largest stored shared-prefix length. The stored value means "at least this many".
const LCP_CAP: usize = 255;

fn corrupt(msg: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("range cache view: {msg}"),
    )
}

fn lcp(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Compares `key` with the concatenation `a ++ b`, and returns the order and the length of the
/// prefix that they share.
fn cmp_split(key: &[u8], a: &[u8], b: &[u8]) -> (Cmp, usize) {
    let head = &key[..key.len().min(a.len())];
    let l = lcp(head, a);
    if l < a.len() {
        if l < head.len() {
            return (head[l].cmp(&a[l]), l);
        }
        // `key` is a proper prefix of `a`.
        return (Cmp::Less, l);
    }
    let tail = &key[a.len()..];
    let m = lcp(tail, b);
    let order = match (tail.get(m), b.get(m)) {
        (Some(x), Some(y)) => x.cmp(y),
        (Some(_), None) => Cmp::Greater,
        (None, Some(_)) => Cmp::Less,
        (None, None) => Cmp::Equal,
    };
    (order, a.len() + m)
}

/// The memory of all views of one cache, from their build until their last holder drops them.
#[derive(Debug, Default)]
pub(crate) struct ViewMemory(AtomicU64);

impl ViewMemory {
    pub(crate) fn bytes(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// What a view build did, and how long it took: the cost of a view next to the merges of runs
/// that it replaces.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BuildCost {
    /// Rows of the new run that the build added, from the locators of its writer.
    pub(crate) rows_added: u64,
    /// Keys of entries of the old view, read from their run files by locator.
    pub(crate) keys_read: u64,
    /// The bytes of those reads: each old key read's row header and key.
    pub(crate) bytes_read: u64,
    /// Wall time of the build, on one thread that does nothing else. The caller times it.
    pub(crate) nanos: u64,
}

/// The delta view of one range. See the module documentation.
pub(crate) struct DeltaView {
    /// The runs of the view by slot.
    slots: Vec<Option<Arc<Run>>>,
    sel: Vec<u8>,
    off: Vec<u32>,
    lcp: Vec<u8>,
    byte: Vec<u8>,
    /// The prefix that every key of the view shares.
    prefix: Vec<u8>,
    /// The entry of each anchor, ascending. Entry 0 is always the first.
    anchor_at: Vec<u32>,
    /// `anchor_keys[anchor_end[j - 1]..anchor_end[j]]` is the key of anchor `j` after `prefix`.
    anchor_end: Vec<u32>,
    anchor_keys: Vec<u8>,
    /// The smallest `lo` of a run of the view that holds a row, or `u64::MAX`: a scan at `seq`
    /// can see a version of the view only if this is below `seq`.
    lowest: u64,
    memory: Arc<ViewMemory>,
    /// The heap bytes of the view, counted in `memory` and in the metadata budget.
    bytes: u64,
}

impl std::fmt::Debug for DeltaView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaView")
            .field("entries", &self.sel.len())
            .field("anchors", &self.anchor_at.len())
            .field("runs", &self.runs().count())
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Drop for DeltaView {
    fn drop(&mut self) {
        self.memory.0.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

impl DeltaView {
    /// A view of no run, to add a range's first run to.
    pub(crate) fn empty(memory: &Arc<ViewMemory>) -> Self {
        Out::new(Vec::new(), Vec::new(), 0).finish(memory)
    }

    /// The number of versions in the view.
    pub(crate) fn len(&self) -> usize {
        self.sel.len()
    }

    /// The heap bytes of the view.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The runs of the view, in slot order.
    pub(crate) fn runs(&self) -> impl Iterator<Item = &Arc<Run>> {
        self.slots.iter().flatten()
    }

    /// True if the view holds `run`.
    pub(crate) fn holds(&self, run: &Arc<Run>) -> bool {
        self.runs().any(|r| Arc::ptr_eq(r, run))
    }

    /// True if a version of the view can be visible at `seq`.
    pub(crate) fn has_visible(&self, seq: u64) -> bool {
        self.lowest < seq
    }

    /// The run of entry `i`.
    pub(crate) fn run_of(&self, i: usize) -> &Run {
        self.slot_run(self.sel[i] & SLOT)
    }

    fn slot_run(&self, slot: u8) -> &Run {
        self.slots[slot as usize]
            .as_deref()
            .expect("a view entry of an empty slot")
    }

    /// The slot of entry `i`.
    pub(crate) fn slot_of(&self, i: usize) -> u8 {
        self.sel[i] & SLOT
    }

    /// The run in `slot`.
    pub(crate) fn run_in(&self, slot: u8) -> &Run {
        self.slot_run(slot)
    }

    fn is_first(&self, i: usize) -> bool {
        self.sel[i] & FIRST != 0
    }

    /// The offset of entry `i`'s row in its run file.
    pub(crate) fn offset(&self, i: usize) -> u64 {
        self.off[i] as u64
    }

    /// The first entry after `i` that starts a key, or the end.
    pub(crate) fn next_key(&self, i: usize) -> usize {
        let mut j = i + 1;
        while j < self.sel.len() && self.sel[j] & FIRST == 0 {
            j += 1;
        }
        j
    }

    /// The header of entry `i`'s row.
    pub(crate) fn row(&self, i: usize) -> io::Result<RowSpan> {
        self.run_of(i).file.row_at(self.offset(i))
    }

    /// The key of entry `i`, read from its run file.
    pub(crate) fn key(&self, i: usize) -> io::Result<Cow<'_, [u8]>> {
        let span = self.row(i)?;
        self.run_of(i).file.bytes_at(span.key.0, span.key.1)
    }

    /// The last entry among the `window` entries after entry `i` whose row is in entry `i`'s run
    /// and page, which ends at file offset `page_end`; the search stops before an entry of a
    /// `others + 1`-th run other than entry `i`'s. A run's rows are in the view in file order, so
    /// the rows of that run and page from entry `i`'s to the one returned are all entries of that
    /// stretch, and a copy of the file between them holds every row that the stretch takes from
    /// the page (see the `scan` module). `i` if there is none.
    pub(crate) fn page_mates(
        &self,
        i: usize,
        page_end: u64,
        window: usize,
        others: usize,
    ) -> usize {
        let slot = self.sel[i] & SLOT;
        let mut seen = [0u8; 8];
        let mut distinct = 0;
        let mut last = i;
        for j in i + 1..(i + 1 + window).min(self.sel.len()) {
            let s = self.sel[j] & SLOT;
            if s == slot {
                if self.off[j] as u64 >= page_end {
                    break;
                }
                last = j;
            } else if !seen[..distinct].contains(&s) {
                if distinct >= others.min(seen.len()) {
                    break;
                }
                seen[distinct] = s;
                distinct += 1;
            }
        }
        last
    }

    fn anchor_suffix(&self, j: usize) -> &[u8] {
        let start = match j {
            0 => 0,
            _ => self.anchor_end[j - 1] as usize,
        };
        &self.anchor_keys[start..self.anchor_end[j] as usize]
    }

    /// The end of segment `j`: the entry of the next anchor, or the end of the view.
    fn segment_end(&self, j: usize) -> usize {
        self.anchor_at
            .get(j + 1)
            .map_or(self.sel.len(), |&a| a as usize)
    }

    /// The first entry whose key is at or after `start`, which starts a key, or the end. One
    /// binary search over the anchors and a walk of one segment's front coding; it reads a key
    /// from a file only where the front coding does not decide.
    pub(crate) fn seek(&self, start: Bound<&[u8]>) -> io::Result<usize> {
        let (target, strict) = match start {
            Bound::Unbounded => return Ok(0),
            Bound::Included(k) => (k, false),
            Bound::Excluded(k) => (k, true),
        };
        if self.sel.is_empty() {
            return Ok(0);
        }
        // Every key starts with the prefix: a target outside it is before or after all of them.
        let p = self.prefix.len();
        let head = &target[..target.len().min(p)];
        match head.cmp(&self.prefix[..head.len()]) {
            Cmp::Less => return Ok(0),
            Cmp::Greater => return Ok(self.sel.len()),
            // A target that is a proper prefix of the prefix is before every key.
            Cmp::Equal if target.len() < p => return Ok(0),
            Cmp::Equal => {}
        }
        let tail = &target[p..];
        // The anchors before the target: below it, or at most it for an excluded start.
        let (mut lo, mut hi) = (0, self.anchor_at.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let before = match self.anchor_suffix(mid).cmp(tail) {
                Cmp::Less => true,
                Cmp::Equal => strict,
                Cmp::Greater => false,
            };
            if before {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            return Ok(0);
        }
        let s = lo - 1;
        let anchor = self.anchor_suffix(s);
        let mut m = p + lcp(anchor, tail);
        let at = self.anchor_at[s] as usize;
        let end = self.segment_end(s);
        // The anchor is below the target, or equal to it for an excluded start: then the next key
        // is the answer.
        let mut equal = strict && anchor == tail;
        for i in at + 1..end {
            if !self.is_first(i) {
                continue;
            }
            if equal {
                return Ok(i);
            }
            match self.coded_cmp(i, m, target)? {
                Coded::Less(l) => m = l,
                Coded::Equal if strict => equal = true,
                Coded::Equal | Coded::Greater(..) => return Ok(i),
            }
        }
        Ok(end)
    }

    /// Compares the key of entry `i`, which starts a key, with `target`, given that the previous
    /// key of the view is below `target` and shares `m` bytes with it. Reads the key only when the
    /// front coding does not decide.
    fn coded_cmp(&self, i: usize, m: usize, target: &[u8]) -> io::Result<Coded> {
        self.coded_cmp_counted(i, m, target, &mut BuildCost::default())
    }

    fn coded_cmp_counted(
        &self,
        i: usize,
        m: usize,
        target: &[u8],
        cost: &mut BuildCost,
    ) -> io::Result<Coded> {
        let l = self.lcp[i] as usize;
        let b = self.byte[i];
        // The previous key `prev` is below `target` and shares `m` bytes with it, and this key is
        // above `prev` and shares `l` bytes with it, then has byte `b`.
        let decided = if l == LCP_CAP && m >= LCP_CAP {
            None
        } else if l < m {
            // This key leaves `prev` where `prev` and `target` still agree, upwards.
            Some(Coded::Greater(l, b))
        } else if l > m {
            // This key agrees with `prev` where `prev` is below `target`.
            Some(Coded::Less(m))
        } else {
            match target.get(m) {
                Some(t) if b < *t => Some(Coded::Less(m)),
                Some(t) if b > *t => Some(Coded::Greater(m, b)),
                _ => None,
            }
        };
        if let Some(decided) = decided {
            return Ok(decided);
        }
        let key = self.key(i)?;
        cost.keys_read += 1;
        cost.bytes_read += (key.len() + 21) as u64;
        let l = lcp(&key, target);
        Ok(match key.as_ref().cmp(target) {
            Cmp::Less => Coded::Less(l),
            Cmp::Equal => Coded::Equal,
            Cmp::Greater => Coded::Greater(l, key[l]),
        })
    }

    /// Builds the view of `runs` (newest first) from their files alone, reading every key. Flushes
    /// and merges build views incrementally; tests check those against this build, which is what
    /// a range opened from its files would use.
    #[cfg(test)]
    pub(crate) fn from_runs(
        runs: &[Arc<Run>],
        memory: &Arc<ViewMemory>,
        cost: &mut BuildCost,
    ) -> io::Result<Self> {
        Self::from_runs_where(runs, memory, cost, |_, _| true)
    }

    /// Builds the view of the rows of `runs` (newest first) for which `keep(run, key)` holds, from
    /// their files alone: for tests that build a view without the versions that a chunk absorbed
    /// (design 04 I12).
    #[cfg(test)]
    pub(crate) fn from_runs_where(
        runs: &[Arc<Run>],
        memory: &Arc<ViewMemory>,
        cost: &mut BuildCost,
        keep: impl Fn(&Run, &[u8]) -> bool,
    ) -> io::Result<Self> {
        if runs.len() > VIEW_SLOTS {
            return Err(corrupt("too many runs"));
        }
        // (key, seq, slot, offset) of every row. Runs hold disjoint intervals, so the order of
        // (key ascending, seq descending) has no ties.
        let mut rows: Vec<(Vec<u8>, u64, u8, u32)> = Vec::new();
        for (slot, run) in runs.iter().enumerate() {
            check_size(run)?;
            run.file.for_each_row(|offset, key, seq| {
                if keep(run, key) {
                    rows.push((key.to_vec(), seq, slot as u8, offset as u32));
                }
                Ok(())
            })?;
            cost.bytes_read += run.file.data_bytes();
        }
        cost.rows_added += rows.len() as u64;
        rows.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let prefix = match (rows.first(), rows.last()) {
            (Some(first), Some(last)) => first.0[..lcp(&first.0, &last.0)].to_vec(),
            _ => Vec::new(),
        };
        let slots = runs.iter().map(|r| Some(r.clone())).collect();
        let mut out = Out::new(prefix, slots, rows.len());
        let mut prev: Option<&[u8]> = None;
        for (key, _, slot, offset) in &rows {
            match prev {
                Some(p) if p == key.as_slice() => out.push(*slot, *offset, 0, 0),
                _ => {
                    let (l, b) = coding(prev, key);
                    if out.wants_anchor() {
                        out.anchor(&key[out.prefix.len()..]);
                    }
                    out.push(*slot | FIRST, *offset, l, b);
                }
            }
            prev = Some(key);
        }
        Ok(out.finish(memory))
    }

    /// The view after a flush: `was`, the range's view (`None` for a range without runs), without
    /// `replaced`, the newest runs that a flush at the run cap merged into its run (none for a
    /// plain flush), and with `run`, the flush's run, whose writer collected `rows`.
    pub(crate) fn after_flush(
        was: Option<&DeltaView>,
        replaced: &[Arc<Run>],
        run: &Arc<Run>,
        rows: &RowLocators,
        memory: &Arc<ViewMemory>,
        cost: &mut BuildCost,
    ) -> io::Result<Self> {
        let kept;
        let base = match was {
            Some(view) if !replaced.is_empty() => {
                kept = view.without(replaced, cost)?;
                kept.as_ref()
            }
            Some(view) => Some(view),
            None if replaced.is_empty() => None,
            None => return Err(corrupt("a flush replaced runs of a range without a view")),
        };
        match base {
            Some(view) => view.with_run(run, rows, cost),
            None => DeltaView::empty(memory).with_run(run, rows, cost),
        }
    }

    /// The view of this view's runs and `run`, a new run newer than all of them, in a free slot.
    /// `rows` are the run's keys and row offsets, as its writer collected them. It walks this
    /// view by its anchors and front coding, copying the old entries between the points where new
    /// keys go; it reads an old key from its file only where the front coding does not decide,
    /// and to start a long segment.
    pub(crate) fn with_run(
        &self,
        run: &Arc<Run>,
        new: &RowLocators,
        cost: &mut BuildCost,
    ) -> io::Result<Self> {
        check_size(run)?;
        if new.rows() as u64 != run.file.rows() {
            return Err(corrupt("row locators of another run"));
        }
        let slot = (0..VIEW_SLOTS)
            .find(|&s| self.slots.get(s).is_none_or(|r| r.is_none()))
            .ok_or_else(|| corrupt("no free slot"))?;
        let mut slots = self.slots.clone();
        if slots.len() <= slot {
            slots.resize(slot + 1, None);
        }
        slots[slot] = Some(run.clone());
        cost.rows_added += new.rows() as u64;
        let (Some(first), Some(last)) = (new.first(), new.last()) else {
            // An empty run adds a slot and no entry.
            let mut out = Out::new(self.prefix.clone(), slots, self.len());
            self.copy_all(&mut out, &self.prefix);
            return Ok(out.finish(&self.memory));
        };
        let prefix = if self.sel.is_empty() {
            first[..lcp(first, last)].to_vec()
        } else {
            let keep = lcp(&self.prefix, first).min(lcp(&self.prefix, last));
            self.prefix[..keep].to_vec()
        };
        let mut out = Out::new(prefix, slots, self.len() + new.rows());
        let mut merge = Merge::new(self, &mut out, cost);
        for g in 0..new.groups() {
            merge.insert(new.key(g), new.offsets(g), slot as u8)?;
        }
        merge.finish()?;
        Ok(out.finish(&self.memory))
    }

    /// The view without `dropped`, runs of the range: the oldest runs, which a major merge folded
    /// into a new base, or the newest ones, which a flush at the run cap merged into its run. A
    /// key keeps the versions of the other runs, in order, or loses all of them. All versions of a
    /// key have its bytes, so a key that stays keeps its front coding, and the coding of a key
    /// after dropped keys follows from theirs: no key is read, except to start a segment whose
    /// anchor was dropped. Returns `None` if no run remains.
    pub(crate) fn without(
        &self,
        dropped: &[Arc<Run>],
        cost: &mut BuildCost,
    ) -> io::Result<Option<Self>> {
        self.kept(dropped, 0..self.len(), cost)
    }

    /// The view of a range split from this view's range (design 04 section 3.2): the versions of
    /// the keys from `start` to `end` (`None`: no end), without `dropped`, the oldest runs that no
    /// chunk of the new range needs. Two seeks find the keys, and the versions between them are
    /// copied as [`DeltaView::without`] copies them: a key reads its file only to start a segment.
    /// Returns `None` if no run remains. A run that remains keeps its slot, also if it holds no
    /// version in the keys.
    pub(crate) fn restrict(
        &self,
        start: &[u8],
        end: Option<&[u8]>,
        dropped: &[Arc<Run>],
        cost: &mut BuildCost,
    ) -> io::Result<Option<Self>> {
        let from = self.seek(Bound::Included(start))?;
        let to = match end {
            Some(end) => self.seek(Bound::Included(end))?,
            None => self.len(),
        };
        self.kept(dropped, from..to.max(from), cost)
    }

    /// The view of the entries `entries` (a range that starts and ends at the first version of a
    /// key, or at the end) less the versions of `dropped`. See [`DeltaView::without`].
    fn kept(
        &self,
        dropped: &[Arc<Run>],
        entries: std::ops::Range<usize>,
        cost: &mut BuildCost,
    ) -> io::Result<Option<Self>> {
        let mut slots = self.slots.clone();
        let mut dead = [false; VIEW_SLOTS];
        for run in dropped {
            let slot = slots
                .iter()
                .position(|r| r.as_ref().is_some_and(|r| Arc::ptr_eq(r, run)))
                .ok_or_else(|| corrupt("a dropped run that the view does not hold"))?;
            dead[slot] = true;
            slots[slot] = None;
        }
        while slots.last().is_some_and(|r| r.is_none()) {
            slots.pop();
        }
        if slots.is_empty() {
            return Ok(None);
        }
        let live = |i: usize| !dead[self.slot_of(i) as usize];
        let kept = entries.clone().filter(|&i| live(i)).count();
        let mut out = Out::new(self.prefix.clone(), slots, kept);
        // The smallest shared length of the dropped keys since the last kept key, and the byte of
        // the last key that has it (see `chain`).
        let mut gone: Option<(u8, u8)> = None;
        // The first anchor at or after the first entry.
        let mut anchor = self
            .anchor_at
            .partition_point(|&a| (a as usize) < entries.start);
        let mut i = entries.start;
        while i < entries.end {
            // Entry `i` starts a key; its versions end at `end`.
            let end = self.next_key(i);
            let is_anchor = self.anchor_at.get(anchor).is_some_and(|&a| a as usize == i);
            if is_anchor {
                anchor += 1;
            }
            let Some(first) = (i..end).find(|&j| live(j)) else {
                gone = Some(chain(gone, self.lcp[i], self.byte[i]));
                i = end;
                continue;
            };
            let (l, b) = match gone.take() {
                Some(d) => chain(Some(d), self.lcp[i], self.byte[i]),
                None => (self.lcp[i], self.byte[i]),
            };
            let (l, b) = if out.sel.is_empty() { (0, 0) } else { (l, b) };
            if is_anchor {
                out.anchor(self.anchor_suffix(anchor - 1));
            } else if out.must_anchor() {
                let key = self.key(i)?;
                cost.keys_read += 1;
                cost.bytes_read += (key.len() + 21) as u64;
                out.anchor(&key[out.prefix.len()..]);
            }
            out.push((self.sel[first] & SLOT) | FIRST, self.off[first], l, b);
            for j in first + 1..end {
                if live(j) {
                    out.push(self.sel[j] & SLOT, self.off[j], 0, 0);
                }
            }
            i = end;
        }
        Ok(Some(out.finish(&self.memory)))
    }

    /// Copies every entry to `out`, whose prefix is a prefix of this view's.
    fn copy_all(&self, out: &mut Out, prefix: &[u8]) {
        let mut anchor = 0;
        for i in 0..self.len() {
            if self.anchor_at.get(anchor).is_some_and(|&a| a as usize == i) {
                out.anchor_split(&self.prefix[prefix.len()..], self.anchor_suffix(anchor));
                anchor += 1;
            }
            out.push(self.sel[i], self.off[i], self.lcp[i], self.byte[i]);
        }
    }

    /// The entries as (run, offset, first of its key), for tests that compare views by content.
    #[cfg(test)]
    pub(crate) fn entries(&self) -> Vec<(usize, u64, bool)> {
        (0..self.len())
            .map(|i| {
                (
                    std::ptr::from_ref(self.run_of(i)).addr(),
                    self.offset(i),
                    self.is_first(i),
                )
            })
            .collect()
    }

    /// Checks every invariant of the view against its files: the order of the entries, the
    /// flags, the front coding, the prefix, the anchors, and that the view lists every row of
    /// its runs once.
    #[cfg(test)]
    pub(crate) fn check(&self) -> io::Result<()> {
        self.check_in(&[], None)
    }

    /// [`DeltaView::check`] for the view of a range from `start` to `end` that a split made: the
    /// view lists exactly the rows of its runs in those keys.
    #[cfg(test)]
    pub(crate) fn check_in(&self, start: &[u8], end: Option<&[u8]>) -> io::Result<()> {
        let mut prev: Option<(Vec<u8>, u64)> = None;
        let mut anchor = 0;
        let mut rows = 0u64;
        let inside = |key: &[u8]| key >= start && end.is_none_or(|end| key < end);
        for i in 0..self.len() {
            let span = self.row(i)?;
            let key = self.key(i)?.to_vec();
            assert!(inside(&key), "entry {i} outside the range's keys");
            assert!(
                key.starts_with(&self.prefix),
                "entry {i} outside the prefix"
            );
            let run = self.run_of(i);
            assert!(
                span.seq > run.lo && span.seq <= run.hi,
                "entry {i} outside its run"
            );
            let first = match &prev {
                None => true,
                Some((k, s)) => {
                    assert!(
                        k < &key || (k == &key && *s > span.seq),
                        "entry {i} out of order"
                    );
                    k != &key
                }
            };
            assert_eq!(self.is_first(i), first, "the first flag of entry {i}");
            if first {
                if let Some((k, _)) = &prev {
                    let l = lcp(k, &key);
                    assert_eq!(self.lcp[i] as usize, l.min(LCP_CAP), "lcp of entry {i}");
                    if l < LCP_CAP {
                        assert_eq!(self.byte[i], key[l], "byte of entry {i}");
                    }
                }
            }
            if self.anchor_at.get(anchor).is_some_and(|&a| a as usize == i) {
                assert!(first, "anchor {anchor} inside a key");
                assert_eq!(
                    self.anchor_suffix(anchor),
                    &key[self.prefix.len()..],
                    "anchor {anchor}"
                );
                anchor += 1;
            }
            prev = Some((key, span.seq));
            rows += 1;
        }
        assert_eq!(anchor, self.anchor_at.len(), "anchors past the entries");
        assert!(self.len() == 0 || self.anchor_at.first() == Some(&0));
        let mut expected = 0u64;
        for run in self.runs() {
            run.file.for_each_row(|_, key, _| {
                expected += u64::from(inside(key));
                Ok(())
            })?;
        }
        assert_eq!(rows, expected, "the view does not list every row once");
        Ok(())
    }
}

/// The result of comparing a key with a target by front coding: below it, sharing this many
/// bytes; equal; above it, sharing this many bytes, then this byte of the key.
enum Coded {
    Less(usize),
    Equal,
    Greater(usize, u8),
}

/// The front coding of `key` after `prev`: the length of their shared prefix, capped, and the
/// byte of `key` after it.
fn coding(prev: Option<&[u8]>, key: &[u8]) -> (u8, u8) {
    match prev {
        None => (0, 0),
        Some(prev) => {
            let l = lcp(prev, key);
            (l.min(LCP_CAP) as u8, key.get(l).copied().unwrap_or(0))
        }
    }
}

/// The front coding of a key after a run of dropped keys: for sorted keys `a < b < c`, the
/// prefix that `a` and `c` share is the shorter of what `a` and `b` and what `b` and `c` share, and
/// `c`'s byte after it is the byte of the last key in the chain that has the shortest.
fn chain(dropped: Option<(u8, u8)>, l: u8, b: u8) -> (u8, u8) {
    match dropped {
        Some((dl, db)) if dl < l => (dl, db),
        _ => (l, b),
    }
}

/// Run files must be smaller than 4 GiB, so that a row offset fits in 32 bits.
fn check_size(run: &Run) -> io::Result<()> {
    if run.file.file_len() > u32::MAX as u64 {
        return Err(corrupt("a run file of 4 GiB or more"));
    }
    Ok(())
}

/// A view being built, entry by entry.
struct Out {
    slots: Vec<Option<Arc<Run>>>,
    sel: Vec<u8>,
    off: Vec<u32>,
    lcp: Vec<u8>,
    byte: Vec<u8>,
    prefix: Vec<u8>,
    anchor_at: Vec<u32>,
    anchor_end: Vec<u32>,
    anchor_keys: Vec<u8>,
    /// Entries since the last anchor.
    since: usize,
}

impl Out {
    fn new(prefix: Vec<u8>, slots: Vec<Option<Arc<Run>>>, entries: usize) -> Self {
        Self {
            slots,
            sel: Vec::with_capacity(entries),
            off: Vec::with_capacity(entries),
            lcp: Vec::with_capacity(entries),
            byte: Vec::with_capacity(entries),
            prefix,
            anchor_at: Vec::with_capacity(entries / SEGMENT + 1),
            anchor_end: Vec::with_capacity(entries / SEGMENT + 1),
            anchor_keys: Vec::new(),
            since: 0,
        }
    }

    /// True where a new key, whose bytes are at hand, should start a segment.
    fn wants_anchor(&self) -> bool {
        self.sel.is_empty() || self.since >= SEGMENT
    }

    /// True where an old key should start a segment, though it needs a key read.
    fn must_anchor(&self) -> bool {
        self.sel.is_empty() || self.since >= 2 * SEGMENT
    }

    /// Starts a segment at the next entry, whose key after the prefix is `suffix`.
    fn anchor(&mut self, suffix: &[u8]) {
        self.anchor_split(&[], suffix);
    }

    /// Starts a segment at the next entry, whose key after the prefix is `a ++ b`.
    fn anchor_split(&mut self, a: &[u8], b: &[u8]) {
        self.anchor_at.push(self.sel.len() as u32);
        self.anchor_keys.extend_from_slice(a);
        self.anchor_keys.extend_from_slice(b);
        self.anchor_end.push(self.anchor_keys.len() as u32);
        self.since = 0;
    }

    fn push(&mut self, sel: u8, off: u32, lcp: u8, byte: u8) {
        self.sel.push(sel);
        self.off.push(off);
        self.lcp.push(lcp);
        self.byte.push(byte);
        self.since += 1;
    }

    fn finish(self, memory: &Arc<ViewMemory>) -> DeltaView {
        let bytes = (self.sel.capacity()
            + self.off.capacity() * 4
            + self.lcp.capacity()
            + self.byte.capacity()
            + self.prefix.capacity()
            + self.anchor_at.capacity() * 4
            + self.anchor_end.capacity() * 4
            + self.anchor_keys.capacity()
            + self.slots.capacity() * size_of::<Option<Arc<Run>>>()
            + size_of::<DeltaView>()) as u64;
        memory.0.fetch_add(bytes, Ordering::Relaxed);
        let lowest = self
            .slots
            .iter()
            .flatten()
            .filter(|r| r.file.rows() > 0)
            .map(|r| r.lo)
            .min()
            .unwrap_or(u64::MAX);
        DeltaView {
            slots: self.slots,
            sel: self.sel,
            off: self.off,
            lcp: self.lcp,
            byte: self.byte,
            prefix: self.prefix,
            anchor_at: self.anchor_at,
            anchor_end: self.anchor_end,
            anchor_keys: self.anchor_keys,
            lowest,
            memory: memory.clone(),
            bytes,
        }
    }
}

/// What the merge emitted last.
enum Last<'n> {
    Nothing,
    /// An old key, copied with its versions. The walk knows what it shares with the target.
    Old,
    /// A new key (possibly merged with an old key of the same bytes).
    New(&'n [u8]),
}

/// The merge of a new run's keys into an old view (see [`DeltaView::with_run`]).
///
/// The old entries before `p` have been emitted. `has_prev` says whether an old key comes
/// before `p`; it is below every new key still to come, and `prev_lcp` is what it shares with the
/// last new key inserted (or the last target compared). `fix` is the front coding that the old key
/// at `p` takes when a new key was inserted right before it.
struct Merge<'v, 'o, 'c, 'n> {
    old: &'v DeltaView,
    out: &'o mut Out,
    cost: &'c mut BuildCost,
    p: usize,
    /// The first old anchor at or after `p`.
    a: usize,
    has_prev: bool,
    /// What the old key before `p` shares with `last_target`.
    prev_lcp: usize,
    last_target: Option<&'n [u8]>,
    last: Last<'n>,
    fix: Option<(u8, u8)>,
}

impl<'v, 'o, 'c, 'n> Merge<'v, 'o, 'c, 'n> {
    fn new(old: &'v DeltaView, out: &'o mut Out, cost: &'c mut BuildCost) -> Self {
        Self {
            old,
            out,
            cost,
            p: 0,
            a: 0,
            has_prev: false,
            prev_lcp: 0,
            last_target: None,
            last: Last::Nothing,
            fix: None,
        }
    }

    /// The full key of old anchor `j`, as the old prefix and the anchor's suffix.
    fn anchor_cmp(&self, j: usize, key: &[u8]) -> (Cmp, usize) {
        let (order, l) = cmp_split(key, &self.old.prefix, self.old.anchor_suffix(j));
        // `cmp_split` orders `key` against the anchor; turn it into the anchor against `key`.
        (order.reverse(), l)
    }

    /// Emits the old key at `p` with its versions, and moves `p` past it.
    fn emit_old_key(&mut self) -> io::Result<()> {
        let old = self.old;
        let i = self.p;
        let is_anchor = old.anchor_at.get(self.a).is_some_and(|&a| a as usize == i);
        let (l, b) = self.fix.take().unwrap_or((old.lcp[i], old.byte[i]));
        let (l, b) = if self.out.sel.is_empty() {
            (0, 0)
        } else {
            (l, b)
        };
        if is_anchor {
            let rebase = &old.prefix[self.out.prefix.len()..];
            self.out.anchor_split(rebase, old.anchor_suffix(self.a));
            self.a += 1;
        } else if self.out.must_anchor() {
            let key = old.key(i)?;
            self.cost.keys_read += 1;
            self.cost.bytes_read += (key.len() + 21) as u64;
            self.out.anchor(&key[self.out.prefix.len()..]);
        }
        self.out.push(old.sel[i], old.off[i], l, b);
        let end = old.next_key(i);
        for j in i + 1..end {
            self.out.push(old.sel[j], old.off[j], 0, 0);
        }
        self.p = end;
        self.has_prev = true;
        self.last = Last::Old;
        Ok(())
    }

    /// Emits the versions of the new key `key` before the old entry at `p`. `prev_lcp` is what the
    /// last emitted old key shares with it, if `last` is old.
    fn emit_new_key(&mut self, key: &'n [u8], offsets: &[u32], slot: u8, prev_lcp: usize) {
        let (l, b) = match self.last {
            Last::Nothing => (0, 0),
            Last::Old => (
                prev_lcp.min(LCP_CAP) as u8,
                key.get(prev_lcp).copied().unwrap_or(0),
            ),
            Last::New(prev) => coding(Some(prev), key),
        };
        if self.out.wants_anchor() {
            self.out.anchor(&key[self.out.prefix.len()..]);
        }
        let mut sel = slot | FIRST;
        for &offset in offsets {
            self.out.push(sel, offset, l, b);
            sel = slot;
        }
        self.last = Last::New(key);
    }

    /// Inserts the versions of the new key `key` (offsets newest first) in the merge.
    fn insert(&mut self, key: &'n [u8], offsets: &[u32], slot: u8) -> io::Result<()> {
        let old = self.old;
        let n = old.len();
        // What the old key before `p` shares with `key`: for sorted `prev < last < key`, the
        // shorter of what `prev` shares with `last` and what `last` shares with `key`.
        let mut m = match self.last_target {
            Some(last) => self.prev_lcp.min(lcp(last, key)),
            None => 0,
        };
        // Skip whole segments: the last anchor at or after `p` whose key is at most `key`.
        let mut jump = None;
        let mut j = self.a;
        while j < old.anchor_at.len() {
            let (order, l) = self.anchor_cmp(j, key);
            if order == Cmp::Greater {
                break;
            }
            jump = Some((j, order, l));
            j += 1;
        }
        if let Some((j, order, l)) = jump {
            let at = old.anchor_at[j] as usize;
            while self.p < at {
                self.emit_old_key()?;
            }
            debug_assert_eq!(self.a, j);
            if order == Cmp::Equal {
                return self.merge_at(key, offsets, slot);
            }
            // The anchor is below `key`.
            self.emit_old_key()?;
            m = l;
        }
        // Walk the front coding up to the next anchor, whose key is above `key`.
        let limit = old.anchor_at.get(self.a).map_or(n, |&a| a as usize);
        while self.p < limit {
            debug_assert!(self.has_prev, "a walk without a previous key");
            match old.coded_cmp_counted(self.p, m, key, self.cost)? {
                Coded::Less(l) => {
                    self.emit_old_key()?;
                    m = l;
                }
                Coded::Equal => return self.merge_at(key, offsets, slot),
                Coded::Greater(l, b) => {
                    return self.insert_before(key, offsets, slot, m, l, b);
                }
            }
        }
        if self.p >= n {
            // After every old key.
            self.emit_new_key(key, offsets, slot, m);
            self.prev_lcp = m;
            self.last_target = Some(key);
            return Ok(());
        }
        // Before the next anchor, whose key is above `key`.
        let (order, l) = self.anchor_cmp(self.a, key);
        debug_assert_eq!(order, Cmp::Greater);
        let b = cmp_split_byte(&old.prefix, old.anchor_suffix(self.a), l);
        self.insert_before(key, offsets, slot, m, l, b)
    }

    /// Inserts `key` before the old key at `p`, which is above it, shares `l` bytes with it and
    /// then has byte `b`. The old key before `p`, if any, shares `m` bytes with `key`.
    fn insert_before(
        &mut self,
        key: &'n [u8],
        offsets: &[u32],
        slot: u8,
        m: usize,
        l: usize,
        b: u8,
    ) -> io::Result<()> {
        self.emit_new_key(key, offsets, slot, m);
        self.fix = Some((l.min(LCP_CAP) as u8, b));
        self.prev_lcp = m;
        self.last_target = Some(key);
        Ok(())
    }

    /// Merges `key` with the old key at `p`, which has the same bytes: the new versions go first,
    /// and the old ones follow as older versions of the key.
    fn merge_at(&mut self, key: &'n [u8], offsets: &[u32], slot: u8) -> io::Result<()> {
        let old = self.old;
        let i = self.p;
        let is_anchor = old.anchor_at.get(self.a).is_some_and(|&a| a as usize == i);
        // The old key's place in the order goes to the new key; a pending fix was for the old
        // key as the first of its key, which it no longer is.
        self.fix = None;
        let (l, b) = match self.last {
            Last::Nothing => (0, 0),
            // The last emitted key is the old key before `p`, so the old key's own coding is the
            // new key's: they have the same bytes.
            Last::Old => (old.lcp[i], old.byte[i]),
            Last::New(prev) => coding(Some(prev), key),
        };
        if is_anchor || self.out.wants_anchor() {
            self.out.anchor(&key[self.out.prefix.len()..]);
        }
        if is_anchor {
            self.a += 1;
        }
        let mut sel = slot | FIRST;
        for &offset in offsets {
            self.out.push(sel, offset, l, b);
            sel = slot;
        }
        let end = old.next_key(i);
        for j in i..end {
            self.out.push(old.sel[j] & SLOT, old.off[j], 0, 0);
        }
        self.p = end;
        self.has_prev = true;
        // The old key before the new `p` is this key.
        self.prev_lcp = key.len();
        self.last_target = Some(key);
        self.last = Last::New(key);
        Ok(())
    }

    /// Copies the old entries after the last new key.
    fn finish(mut self) -> io::Result<()> {
        while self.p < self.old.len() {
            self.emit_old_key()?;
        }
        Ok(())
    }
}

/// The byte at `l` of the concatenation `a ++ b`.
fn cmp_split_byte(a: &[u8], b: &[u8], l: usize) -> u8 {
    if l < a.len() {
        a[l]
    } else {
        b.get(l - a.len()).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_cache::file::{FileBudget, RangeCachePageSource, RangeFile, RangeFileWriter};
    use crate::types::{RowEntry, ValueDeletable};
    use proptest::prelude::*;

    /// Writes a run of `(lo, hi]` with `versions`, (key, seq), in any order, with a writer that
    /// collects its row locators, as a flush does. Checks that they equal the locators read back
    /// from the file.
    fn run(
        dir: &std::path::Path,
        name: &str,
        lo: u64,
        hi: u64,
        mut versions: Vec<(Vec<u8>, u64)>,
        page_size: usize,
    ) -> (Arc<Run>, RowLocators) {
        versions.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut writer = RangeFileWriter::create_run(dir, name, page_size)
            .unwrap()
            .with_locators();
        for (key, seq) in versions {
            let value = ValueDeletable::Value(bytes::Bytes::from(format!("v{seq}")));
            writer
                .push(&RowEntry::new(key.into(), value, seq, None, None))
                .unwrap();
        }
        let (path, locators) = writer.finish_with_locators().unwrap();
        let file =
            RangeFile::open(path, RangeCachePageSource::Mmap, FileBudget::detached()).unwrap();
        assert_eq!(locators, RowLocators::read(&file).unwrap());
        (Arc::new(Run::new(file, lo, hi).unwrap()), locators)
    }

    /// Keys of the shapes that matter: short ones, ones that are prefixes of others, the empty
    /// key, and long ones that share a long prefix and differ late (also past the 255-byte cap).
    fn key_strategy() -> impl Strategy<Value = Vec<u8>> {
        prop_oneof![
            proptest::collection::vec(0u8..4, 0..4),
            (0u8..4, 0u16..64).prop_map(|(a, n)| {
                let mut k = b"tpcc/order_line/".to_vec();
                k.extend_from_slice(&[0, 0, 0, 7, a]);
                k.extend_from_slice(&n.to_be_bytes());
                k
            }),
            (0u8..3, 0u8..4).prop_map(|(a, b)| {
                let mut k = vec![b'x'; 300];
                k.push(a);
                k.push(b);
                k
            }),
        ]
    }

    /// The versions of a run, (key, seq), and its interval.
    type Spec = (Vec<(Vec<u8>, u64)>, u64, u64);

    /// The expected entries of `runs`: every version, key ascending then newest first, as
    /// (key, seq).
    fn expected(runs: &[Spec]) -> Vec<(Vec<u8>, u64)> {
        let mut all: Vec<(Vec<u8>, u64)> = runs.iter().flat_map(|r| r.0.clone()).collect();
        all.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        all
    }

    fn keys_of(view: &DeltaView) -> Vec<(Vec<u8>, u64)> {
        (0..view.len())
            .map(|i| (view.key(i).unwrap().to_vec(), view.row(i).unwrap().seq))
            .collect()
    }

    /// Brute force: the first entry at or after `start`.
    fn brute_seek(entries: &[(Vec<u8>, u64)], start: &Bound<Vec<u8>>) -> usize {
        entries
            .iter()
            .position(|(k, _)| match start {
                Bound::Unbounded => true,
                Bound::Included(s) => k >= s,
                Bound::Excluded(s) => k > s,
            })
            .unwrap_or(entries.len())
    }

    fn check_seeks(view: &DeltaView, entries: &[(Vec<u8>, u64)], targets: &[Vec<u8>]) {
        let mut probes: Vec<Vec<u8>> = targets.to_vec();
        probes.extend(entries.iter().map(|(k, _)| k.clone()));
        for t in &probes {
            for bound in [
                Bound::Included(t.clone()),
                Bound::Excluded(t.clone()),
                Bound::Unbounded,
            ] {
                let got = view.seek(bound.as_ref().map(|k| k.as_slice())).unwrap();
                assert_eq!(got, brute_seek(entries, &bound), "seek {bound:?}");
            }
        }
    }

    /// Checks `page_mates` from every entry against its definition.
    fn check_page_mates(view: &DeltaView, page_size: u64) {
        for i in 0..view.len() {
            for (window, others) in [(64, 3), (4, 0), (16, 1)] {
                let page_end = view.offset(i) - view.offset(i) % page_size + page_size;
                let j = view.page_mates(i, page_end, window, others);
                assert!(j >= i && j <= i + window);
                assert_eq!(view.slot_of(j), view.slot_of(i));
                assert!(view.offset(j) < page_end);
                let mut seen: Vec<u8> = (i + 1..j)
                    .map(|k| view.slot_of(k))
                    .filter(|s| *s != view.slot_of(i))
                    .collect();
                seen.sort_unstable();
                seen.dedup();
                assert!(seen.len() <= others, "{} other runs before {j}", seen.len());
                // The next entry of the same run and page is not within reach.
                let next = (j + 1..view.len().min(i + 1 + window))
                    .find(|&k| view.slot_of(k) == view.slot_of(i));
                if let Some(k) = next {
                    let mut between: Vec<u8> = (i + 1..k)
                        .map(|m| view.slot_of(m))
                        .filter(|s| *s != view.slot_of(i))
                        .collect();
                    between.sort_unstable();
                    between.dedup();
                    assert!(view.offset(k) >= page_end || between.len() > others);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 200, ..ProptestConfig::default() })]

        /// Views built incrementally, run by run as flushes build them, with runs dropped as a
        /// major merge drops them, and after a flush at the run cap, which replaces the newest
        /// runs with its own, list the same entries as a view built from the run files alone,
        /// keep every invariant, and seek like a brute-force search.
        #[test]
        fn should_build_views_that_match_their_runs(
            batches in proptest::collection::vec(
                proptest::collection::btree_set(key_strategy(), 0..40), 1..7),
            versions in 1u64..3,
            absorbed in 0usize..3,
            replaced in 1usize..3,
            flushed in proptest::collection::btree_set(key_strategy(), 0..20),
            targets in proptest::collection::vec(key_strategy(), 0..20),
            page_size in prop_oneof![Just(512usize), Just(4096usize)],
            split_keys in (key_strategy(), proptest::option::of(key_strategy())),
            split_drops in 0usize..3,
        ) {
            let dir = tempfile::tempdir().unwrap();
            let memory = Arc::new(ViewMemory::default());
            // Runs oldest first; each holds `versions` versions of its keys.
            let mut seq = 0;
            let mut specs = Vec::new();
            for batch in &batches {
                let lo = seq;
                let mut rows = Vec::new();
                for key in batch {
                    for _ in 0..versions {
                        seq += 1;
                        rows.push((key.clone(), seq));
                    }
                }
                seq += 1;
                specs.push((rows, lo, seq));
            }
            let runs: Vec<(Arc<Run>, RowLocators)> = specs
                .iter()
                .enumerate()
                .map(|(i, (rows, lo, hi))| run(dir.path(), &format!("r{i}"), *lo, *hi, rows.clone(), page_size))
                .collect();
            // Incremental: the oldest run on an empty view, then each newer one.
            let mut cost = BuildCost::default();
            let mut view = DeltaView::empty(&memory);
            for (r, rows) in &runs {
                view = view.with_run(r, rows, &mut cost).unwrap();
                view.check().unwrap();
            }
            let newest_first: Vec<Arc<Run>> = runs.iter().rev().map(|(r, _)| r.clone()).collect();
            let scratch = DeltaView::from_runs(&newest_first, &memory, &mut cost).unwrap();
            prop_assert_eq!(keys_of(&view), keys_of(&scratch));
            prop_assert_eq!(view.entries(), scratch.entries());
            prop_assert_eq!(keys_of(&view), expected(&specs));
            check_seeks(&view, &expected(&specs), &targets);
            check_page_mates(&view, page_size as u64);

            // A major merge absorbs the oldest runs.
            let absorbed = absorbed.min(runs.len());
            let gone: Vec<Arc<Run>> = runs[..absorbed].iter().rev().map(|(r, _)| r.clone()).collect();
            match view.without(&gone, &mut cost).unwrap() {
                None => prop_assert_eq!(absorbed, runs.len()),
                Some(left) => {
                    left.check().unwrap();
                    let want = expected(&specs[absorbed..]);
                    prop_assert_eq!(keys_of(&left), want.clone());
                    check_seeks(&left, &want, &targets);
                }
            }

            // A split (design 04 section 3.2): the view of the keys from `lo` to `hi` of a child,
            // without the oldest runs, which no chunk of the child needs.
            let (lo, hi) = split_keys;
            let hi = hi.filter(|hi| *hi > lo);
            let inside = |k: &[u8]| k >= lo.as_slice() && hi.as_ref().is_none_or(|hi| k < hi.as_slice());
            let drops = split_drops.min(runs.len());
            let gone: Vec<Arc<Run>> = runs[..drops].iter().rev().map(|(r, _)| r.clone()).collect();
            match view.restrict(&lo, hi.as_deref(), &gone, &mut cost).unwrap() {
                None => prop_assert_eq!(drops, runs.len()),
                Some(part) => {
                    part.check_in(&lo, hi.as_deref()).unwrap();
                    let want: Vec<(Vec<u8>, u64)> = expected(&specs[drops..])
                        .into_iter()
                        .filter(|(k, _)| inside(k))
                        .collect();
                    prop_assert_eq!(keys_of(&part), want.clone());
                    let kept: Vec<Arc<Run>> = runs[drops..].iter().rev().map(|(r, _)| r.clone()).collect();
                    let scratch = DeltaView::from_runs_where(&kept, &memory, &mut cost, |_, k| inside(k)).unwrap();
                    prop_assert_eq!(part.entries(), scratch.entries());
                    prop_assert_eq!(part.runs().count(), runs.len() - drops);
                    check_seeks(&part, &want, &targets);
                    check_page_mates(&part, page_size as u64);
                    // A child's flush adds a run of its own keys to the restricted view.
                    let mut rows: Vec<(Vec<u8>, u64)> = Vec::new();
                    let mut at = seq;
                    for key in flushed.iter().filter(|k| inside(k)) {
                        at += 1;
                        rows.push((key.clone(), at));
                    }
                    let (child_run, child_rows) = run(dir.path(), "child", seq, at + 1, rows.clone(), page_size);
                    let grown = part.with_run(&child_run, &child_rows, &mut cost).unwrap();
                    grown.check_in(&lo, hi.as_deref()).unwrap();
                    let mut want_specs = specs[drops..].to_vec();
                    want_specs.push((rows, seq, at + 1));
                    let want: Vec<(Vec<u8>, u64)> = expected(&want_specs)
                        .into_iter()
                        .filter(|(k, _)| inside(k))
                        .collect();
                    prop_assert_eq!(keys_of(&grown), want.clone());
                    check_seeks(&grown, &want, &targets);
                }
            }

            // A flush at the run cap: its run holds its own versions and those of the newest
            // runs, which it replaces.
            let replaced = replaced.min(runs.len());
            let kept = runs.len() - replaced;
            let lo = specs[kept].1;
            let mut rows: Vec<(Vec<u8>, u64)> = specs[kept..].iter().flat_map(|s| s.0.clone()).collect();
            for key in &flushed {
                seq += 1;
                rows.push((key.clone(), seq));
            }
            seq += 1;
            let (merged, merged_rows) = run(dir.path(), "merged", lo, seq, rows.clone(), page_size);
            let replaces: Vec<Arc<Run>> = runs[kept..].iter().rev().map(|(r, _)| r.clone()).collect();
            let after = DeltaView::after_flush(Some(&view), &replaces, &merged, &merged_rows, &memory, &mut cost).unwrap();
            after.check().unwrap();
            let mut remaining: Vec<Arc<Run>> = vec![merged.clone()];
            remaining.extend(runs[..kept].iter().rev().map(|(r, _)| r.clone()));
            let rebuilt = DeltaView::from_runs(&remaining, &memory, &mut cost).unwrap();
            prop_assert_eq!(after.entries(), rebuilt.entries());
            let mut want_specs = specs[..kept].to_vec();
            want_specs.push((rows, lo, seq));
            let want = expected(&want_specs);
            prop_assert_eq!(keys_of(&after), want.clone());
            check_seeks(&after, &want, &targets);
            check_page_mates(&after, page_size as u64);
            prop_assert_eq!(after.runs().count(), kept + 1);
            prop_assert!(after.holds(&merged) && replaces.iter().all(|r| !after.holds(r)));

            drop((view, scratch, after, rebuilt));
            prop_assert_eq!(memory.bytes(), 0);
        }
    }

    /// A flush onto a range without a view builds its first view; a flush that replaces runs
    /// needs the view that holds them.
    #[test]
    fn should_build_a_first_view_and_reject_replacing_without_one() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Arc::new(ViewMemory::default());
        let mut cost = BuildCost::default();
        let keys: Vec<(Vec<u8>, u64)> = (0..100u32)
            .map(|i| (format!("key{i:06}").into_bytes(), i as u64 + 1))
            .collect();
        let (r, rows) = run(dir.path(), "a", 0, 100, keys, 512);
        let first = DeltaView::after_flush(None, &[], &r, &rows, &memory, &mut cost).unwrap();
        first.check().unwrap();
        assert_eq!(first.len(), 100);
        assert_eq!(cost.rows_added, 100);
        assert!(first.has_visible(1) && !first.has_visible(0));
        assert!(DeltaView::after_flush(
            None,
            std::slice::from_ref(&r),
            &r,
            &rows,
            &memory,
            &mut cost
        )
        .is_err());
        // Locators of another run.
        let (other, other_rows) = run(dir.path(), "b", 100, 101, vec![(b"k".to_vec(), 101)], 512);
        assert!(first.with_run(&other, &rows, &mut cost).is_err());
        let both = first.with_run(&other, &other_rows, &mut cost).unwrap();
        both.check().unwrap();
        assert_eq!(
            both.without(&[other, r], &mut cost)
                .unwrap()
                .map(|v| v.len()),
            None
        );
    }

    #[test]
    fn should_count_its_memory_until_it_drops() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Arc::new(ViewMemory::default());
        let keys: Vec<(Vec<u8>, u64)> = (0..1000u32)
            .map(|i| (format!("key{i:06}").into_bytes(), i as u64 + 1))
            .collect();
        let (r, _) = run(dir.path(), "a", 0, 1000, keys, 4096);
        let mut cost = BuildCost::default();
        let view = DeltaView::from_runs(&[r], &memory, &mut cost).unwrap();
        assert!(view.bytes() >= 7 * 1000);
        assert_eq!(memory.bytes(), view.bytes());
        drop(view);
        assert_eq!(memory.bytes(), 0);
    }
}
