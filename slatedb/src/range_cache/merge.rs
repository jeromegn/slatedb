//! Lossless merges of versions into one run file: the flush of several sealed buffers, and the
//! flush at the run cap, which merges buffers with the newest runs (design 01 section 7.3). A merge
//! reads its inputs within the keys of its range, which after a split share buffers and runs with
//! the other children of the split.
//!
//! The inputs hold disjoint intervals of sequence numbers, each in the order of a run (key
//! ascending, then sequence number descending). So their versions have one total order with no
//! ties, and the output is every input version in that order: tombstones and merge operands
//! stay, and nothing is folded. A binary heap of the inputs by their next version picks each
//! output row with `log k` comparisons, so a flush of a long backlog of buffers stays linear in
//! its rows.

use std::cmp::Ordering;
use std::io;
use std::ops::{Bound, RangeBounds};

use bytes::Bytes;

use super::buffer::{Buffer, Cursor};
use super::delta::{past_end, Delta};
use super::file::{RangeFile, RangeFileCursor, RangeFileWriter};
use super::Run;
use crate::bytes_range::BytesRange;

/// One input of a merge, at its next version.
enum Source<'a> {
    /// A sealed buffer and its seal: versions above the seal belong to no run (they are rows of
    /// a batch that a removal sealed out), so the merge skips them. Versions at or past `end` are
    /// outside the merge's keys.
    Buffer {
        buffer: &'a Buffer,
        cursor: Cursor,
        sealed_at: u64,
        end: Bound<Bytes>,
    },
    Run {
        file: &'a RangeFile,
        cursor: Box<RangeFileCursor>,
    },
}

impl<'a> Source<'a> {
    fn buffer(delta: &'a Delta, sealed_at: u64, keys: &BytesRange) -> Self {
        let buffer = delta.buffer();
        let start = RangeBounds::start_bound(keys).map(|key| key.as_ref());
        let mut source = Source::Buffer {
            buffer,
            cursor: buffer.seek(start),
            sealed_at,
            end: RangeBounds::end_bound(keys).cloned(),
        };
        source.settle();
        source
    }

    fn run(run: &'a Run, keys: &BytesRange) -> io::Result<Self> {
        Ok(Source::Run {
            file: &run.file,
            cursor: Box::new(RangeFileCursor::new(&run.file, keys)?),
        })
    }

    /// Skips a buffer's versions above its seal.
    fn settle(&mut self) {
        if let Source::Buffer {
            buffer,
            cursor,
            sealed_at,
            ..
        } = self
        {
            while buffer.version(cursor).is_some_and(|v| v.seq > *sealed_at) {
                buffer.advance(cursor);
            }
        }
    }

    /// The key and sequence number of the next version, or `None` at the end.
    fn head(&self) -> Option<(&[u8], u64)> {
        match self {
            Source::Buffer {
                buffer,
                cursor,
                end,
                ..
            } => buffer
                .version(cursor)
                .filter(|v| !past_end(end, v.key))
                .map(|v| (v.key, v.seq)),
            Source::Run { cursor, .. } => Some((cursor.key()?, cursor.seq()?)),
        }
    }

    /// Writes the next version and moves past it.
    fn emit(&mut self, writer: &mut RangeFileWriter) -> io::Result<()> {
        match self {
            Source::Buffer { buffer, cursor, .. } => {
                if let Some(v) = buffer.version(cursor) {
                    writer.push_version(v.key, v.seq, v.kind, v.create_ts, v.value)?;
                    buffer.advance(cursor);
                }
                self.settle();
            }
            Source::Run { file, cursor } => {
                let row = cursor.take(file)?;
                writer.push(&row)?;
            }
        }
        Ok(())
    }
}

/// Returns true if the next version of `a` comes before the next version of `b`: a smaller key,
/// or the same key and a greater sequence number. An exhausted source comes last.
fn before(a: &Source<'_>, b: &Source<'_>) -> bool {
    match (a.head(), b.head()) {
        (Some((ka, sa)), Some((kb, sb))) => match ka.cmp(kb) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => sa > sb,
        },
        (Some(_), None) => true,
        _ => false,
    }
}

/// Restores the heap order below `at`.
fn sift_down(heap: &mut [usize], sources: &[Source<'_>], mut at: usize) {
    loop {
        let (left, right) = (2 * at + 1, 2 * at + 2);
        let mut first = at;
        if left < heap.len() && before(&sources[heap[left]], &sources[heap[first]]) {
            first = left;
        }
        if right < heap.len() && before(&sources[heap[right]], &sources[heap[first]]) {
            first = right;
        }
        if first == at {
            return;
        }
        heap.swap(at, first);
        at = first;
    }
}

/// Writes every version of `buffers` (each up to its seal) and `runs` to `writer`, in run order.
/// The inputs must hold disjoint intervals of sequence numbers. Returns the versions written.
#[cfg(test)]
pub(super) fn merge_versions(
    writer: &mut RangeFileWriter,
    buffers: &[(&Delta, u64)],
    runs: &[&Run],
) -> io::Result<u64> {
    merge_versions_in(writer, buffers, runs, &BytesRange::unbounded())
}

/// Writes every version of `buffers` (each up to its seal) and `runs` whose key is in `keys` to
/// `writer`, in run order. The inputs must hold disjoint intervals of sequence numbers. Returns
/// the versions written. A range's flush passes its own keys: after a split, the range shares its
/// sealed buffers and runs with the other children of the split, which hold their keys too
/// (design 04 X27).
pub(super) fn merge_versions_in(
    writer: &mut RangeFileWriter,
    buffers: &[(&Delta, u64)],
    runs: &[&Run],
    keys: &BytesRange,
) -> io::Result<u64> {
    let mut sources: Vec<Source<'_>> = Vec::with_capacity(buffers.len() + runs.len());
    for (delta, sealed_at) in buffers {
        sources.push(Source::buffer(delta, *sealed_at, keys));
    }
    for run in runs {
        sources.push(Source::run(run, keys)?);
    }
    let mut heap: Vec<usize> = (0..sources.len())
        .filter(|i| sources[*i].head().is_some())
        .collect();
    for at in (0..heap.len() / 2).rev() {
        sift_down(&mut heap, &sources, at);
    }
    let mut written = 0;
    while let Some(&top) = heap.first() {
        sources[top].emit(writer)?;
        written += 1;
        if sources[top].head().is_none() {
            let last = heap.pop().expect("a heap with a root");
            if !heap.is_empty() {
                heap[0] = last;
            }
        }
        sift_down(&mut heap, &sources, 0);
    }
    Ok(written)
}
