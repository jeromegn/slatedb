//! Where the delta of a range lands in its base (design 04 sections 4.3 and 4.5): the measurement
//! of `h`, the share of the base where the delta lands, that each major merge makes while it walks
//! the base and the runs that it absorbs in key order.
//!
//! The merge reads every absorbed version once. A [`Measured`] source wraps the merge's iterator
//! over one absorbed run within one chunk. For each version that it passes on, it finds the page of
//! the chunk's file whose keys hold the version's key, with one search of the file's fence index on
//! the heap, and adds the version's key and value bytes to the *slice* of the file that holds the
//! page: slices are virtual pieces of 1 MiB of the file's data, by file offset. A version below the
//! first key of the file or above its last key, as an append is, lands in no slice and counts as
//! *outside*. After the merge, [`SliceTally::spread`] sums the 1 MiB slices into 8 MiB slices (eight
//! each, as both cut at file offsets) and computes for both sizes:
//!
//! - the bytes of the slices that hold at least one absorbed version (*touched*);
//! - the bytes of the fewest densest slices (absorbed bytes per slice byte) whose absorbed bytes,
//!   with the outside bytes, reach half of all absorbed bytes (*half*). That is what a partial
//!   major merge that rewrites the densest chunks of the slice's size until it absorbs half of the
//!   range's delta would rewrite, the delta outside the base's keys being written as new chunks
//!   without a rewrite (design 04 section 4.3).
//!
//! Nothing is written or read for the slices, and they need not start at a row. The cost is one
//! fence search and one add per absorbed version on the merge's blocking thread; reads pay nothing.
//! With one chunk per range, the slices cut the whole base file.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use super::file::{RangeFile, RangeFileIterator};
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::types::{RowEntry, ValueDeletable};

/// The slice sizes, smallest first. Each is a multiple of the first.
pub(super) const SLICE_BYTES: [u64; 2] = [1 << 20, 8 << 20];

/// Where the delta of one major merge landed, or of several summed. See the module documentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Spread {
    /// The data bytes of the base files: what the slices cut.
    pub(crate) base_bytes: u64,
    /// Key and value bytes of the absorbed versions inside the base files' keys.
    pub(crate) inside_bytes: u64,
    /// Key and value bytes of the absorbed versions outside them.
    pub(crate) outside_bytes: u64,
    /// By slice size, as [`SLICE_BYTES`].
    pub(crate) slices: [SliceSpread; 2],
}

/// The touched and half bytes of one slice size. See the module documentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SliceSpread {
    pub(crate) touched_bytes: u64,
    pub(crate) half_bytes: u64,
}

impl Spread {
    /// Adds the spread of another chunk of the same merge.
    pub(super) fn add(&mut self, other: &Spread) {
        self.base_bytes += other.base_bytes;
        self.inside_bytes += other.inside_bytes;
        self.outside_bytes += other.outside_bytes;
        for (mine, theirs) in self.slices.iter_mut().zip(&other.slices) {
            mine.touched_bytes += theirs.touched_bytes;
            mine.half_bytes += theirs.half_bytes;
        }
    }
}

/// The absorbed bytes of each 1 MiB slice of one base file, and outside its keys, which the
/// merge's [`Measured`] sources add to. They run on the merge's thread, one after another, so the
/// adds never contend.
pub(super) struct SliceTally {
    base: Arc<RangeFile>,
    /// The last key of the base file, `None` if it holds no row.
    last: Option<Bytes>,
    slices: Box<[AtomicU64]>,
    outside: AtomicU64,
}

impl SliceTally {
    /// A tally for `base`. It reads the base file's last page, for its last key.
    pub(super) fn new(base: Arc<RangeFile>) -> io::Result<Arc<Self>> {
        let last = base.last_key()?;
        let slices = base.data_len().div_ceil(SLICE_BYTES[0]) as usize;
        Ok(Arc::new(Self {
            base,
            last,
            slices: (0..slices).map(|_| AtomicU64::new(0)).collect(),
            outside: AtomicU64::new(0),
        }))
    }

    /// Adds `bytes` of a version of `key` to the slice that holds `key`, or to the outside bytes.
    fn add(&self, key: &[u8], bytes: u64) {
        let slice = match &self.last {
            Some(last) if key <= last.as_ref() => self
                .base
                .page_offset_of(key)
                .and_then(|offset| self.slices.get((offset / SLICE_BYTES[0]) as usize)),
            _ => None,
        };
        slice
            .unwrap_or(&self.outside)
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Where the added versions landed.
    pub(super) fn spread(&self) -> Spread {
        let slices: Vec<u64> = self
            .slices
            .iter()
            .map(|s| s.load(Ordering::Relaxed))
            .collect();
        spread_of(
            self.base.data_len(),
            &slices,
            self.outside.load(Ordering::Relaxed),
        )
    }
}

/// The spread of the absorbed bytes `small` of each slice of the first size of a base of
/// `base_bytes` data bytes, with `outside` bytes outside its keys.
fn spread_of(base_bytes: u64, small: &[u64], outside: u64) -> Spread {
    let inside: u64 = small.iter().sum();
    let total = inside + outside;
    let mut slices = [SliceSpread::default(); 2];
    for (spread, &size) in slices.iter_mut().zip(&SLICE_BYTES) {
        let per = (size / SLICE_BYTES[0]) as usize;
        // (absorbed bytes, slice bytes) of each slice that holds a version.
        let mut held: Vec<(u64, u64)> = small
            .chunks(per)
            .enumerate()
            .map(|(i, group)| {
                let start = i as u64 * size;
                (
                    group.iter().sum(),
                    base_bytes.saturating_sub(start).min(size),
                )
            })
            .filter(|(absorbed, _)| *absorbed > 0)
            .collect();
        spread.touched_bytes = held.iter().map(|(_, bytes)| bytes).sum();
        // Densest first: `a` before `b` if a.0 / a.1 > b.0 / b.1. Equal densities keep key order.
        held.sort_by(|a, b| {
            (u128::from(b.0) * u128::from(a.1)).cmp(&(u128::from(a.0) * u128::from(b.1)))
        });
        let mut absorbed = outside;
        for (delta, bytes) in held {
            if 2 * absorbed >= total {
                break;
            }
            absorbed += delta;
            spread.half_bytes += bytes;
        }
    }
    Spread {
        base_bytes,
        inside_bytes: inside,
        outside_bytes: outside,
        slices,
    }
}

/// The key and value bytes of a version.
fn version_bytes(entry: &RowEntry) -> u64 {
    let value = match &entry.value {
        ValueDeletable::Value(v) | ValueDeletable::Merge(v) => v.len(),
        ValueDeletable::Tombstone => 0,
    };
    (entry.key.len() + value) as u64
}

/// A source of a major merge: the iterator over one absorbed run within one chunk, which adds each
/// version that it returns to the chunk's [`SliceTally`].
pub(super) struct Measured {
    inner: RangeFileIterator,
    tally: Arc<SliceTally>,
}

impl Measured {
    pub(super) fn new(inner: RangeFileIterator, tally: Arc<SliceTally>) -> Self {
        Self { inner, tally }
    }
}

#[async_trait]
impl RowEntryIterator for Measured {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        self.inner.init().await
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        let entry = self.inner.next().await?;
        if let Some(entry) = &entry {
            self.tally.add(&entry.key, version_bytes(entry));
        }
        Ok(entry)
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        self.inner.seek(next_key).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes_range::BytesRange;
    use crate::range_cache::file::{FileBudget, RangeCachePageSource, RangeFileWriter};

    const MIB: u64 = 1 << 20;

    #[test]
    fn should_find_the_touched_and_densest_half_of_the_slices() {
        // Twenty 1 MiB slices; the last is half full. Delta in slices 0 (10), 1 (30), 9 (60) and
        // 19 (40, on half a MiB, so the densest).
        let mut small = vec![0u64; 20];
        small[0] = 10;
        small[1] = 30;
        small[9] = 60;
        small[19] = 40;
        let spread = spread_of(19 * MIB + MIB / 2, &small, 0);
        assert_eq!((spread.inside_bytes, spread.outside_bytes), (140, 0));
        let [one, eight] = spread.slices;
        assert_eq!(one.touched_bytes, 3 * MIB + MIB / 2);
        // Densest: 19 (80 per MiB), then 9 (60): 100 of 140 is past half.
        assert_eq!(one.half_bytes, MIB / 2 + MIB);
        // 8 MiB slices: [0, 8) holds 40, [8, 16) holds 60, [16, 19.5) holds 40 on 3.5 MiB.
        assert_eq!(eight.touched_bytes, 19 * MIB + MIB / 2);
        // Densest: the last (40 / 3.5), then [8, 16) (60 / 8): 100 of 140.
        assert_eq!(eight.half_bytes, 3 * MIB + MIB / 2 + 8 * MIB);

        // Delta outside the base's keys counts as absorbed without a rewrite.
        let spread = spread_of(19 * MIB + MIB / 2, &small, 140);
        assert_eq!(spread.slices[0].half_bytes, 0);
        let spread = spread_of(19 * MIB + MIB / 2, &small, 100);
        // 100 of 240 outside: the densest slice (40) brings it to half.
        assert_eq!(spread.slices[0].half_bytes, MIB / 2);
        // No delta at all.
        let spread = spread_of(4 * MIB, &[0; 4], 0);
        assert_eq!(spread.slices, [SliceSpread::default(); 2]);
    }

    /// A base of 6,000 rows of about 500 bytes (a few MiB), and versions inside it, below its
    /// first key and above its last: each lands in the slice of its key's page, or outside.
    #[tokio::test]
    async fn should_add_each_absorbed_version_to_the_slice_of_its_key() {
        let dir = tempfile::tempdir().unwrap();
        let key = |i: usize| Bytes::from(format!("key{i:06}"));
        let value = vec![b'v'; 500];
        let mut writer = RangeFileWriter::create(dir.path(), "base", 4096).unwrap();
        for i in 100..6100 {
            writer
                .push(&RowEntry::new_value(&key(i), &value, 1))
                .unwrap();
        }
        let path = writer.finish().unwrap();
        let base = Arc::new(
            RangeFile::open(path, RangeCachePageSource::Mmap, FileBudget::detached()).unwrap(),
        );
        assert!(base.data_len() > 2 * MIB && base.data_len() < 8 * MIB);
        // One version every 100 keys of the base, two below it and three above it.
        let mut run = RangeFileWriter::create_run(dir.path(), "run", 4096).unwrap();
        let mut expected = vec![0u64; base.data_len().div_ceil(MIB) as usize];
        let mut outside = 0;
        let picks = (0..100)
            .step_by(50)
            .chain((100..6100).step_by(100))
            .chain([7000, 7001, 7002]);
        for i in picks {
            let version = RowEntry::new_value(&key(i), b"x", 5);
            run.push(&version).unwrap();
            let bytes = key(i).len() as u64 + 1;
            if (100..6100).contains(&i) {
                let offset = base.page_offset_of(&key(i)).unwrap();
                expected[(offset / MIB) as usize] += bytes;
            } else {
                outside += bytes;
            }
        }
        let run = Arc::new(
            RangeFile::open(
                run.finish().unwrap(),
                RangeCachePageSource::Mmap,
                FileBudget::detached(),
            )
            .unwrap(),
        );
        let tally = SliceTally::new(base.clone()).unwrap();
        let mut source = Measured::new(
            RangeFileIterator::new(run, BytesRange::unbounded()),
            tally.clone(),
        );
        source.init().await.unwrap();
        while source.next().await.unwrap().is_some() {}
        let slices: Vec<u64> = tally
            .slices
            .iter()
            .map(|s| s.load(Ordering::Relaxed))
            .collect();
        assert_eq!(slices, expected);
        let spread = tally.spread();
        assert_eq!(spread.base_bytes, base.data_len());
        assert_eq!(spread.inside_bytes, expected.iter().sum::<u64>());
        assert_eq!(spread.outside_bytes, outside);
        assert_eq!(outside, 5 * (key(0).len() as u64 + 1));
        // Every 1 MiB slice holds versions, and one 8 MiB slice holds them all.
        let [one, eight] = spread.slices;
        assert!(expected.iter().all(|&bytes| bytes > 0));
        assert_eq!(one.touched_bytes, base.data_len());
        assert!(one.half_bytes > 0 && one.half_bytes < base.data_len());
        assert_eq!(
            eight,
            SliceSpread {
                touched_bytes: base.data_len(),
                half_bytes: base.data_len(),
            }
        );
    }
}
