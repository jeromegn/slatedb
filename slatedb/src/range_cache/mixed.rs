//! Scans that cross cached and uncached key ranges.
//!
//! A mixed scan splits its range into parts in key order. A cached part reads the pinned layers
//! of one cached range with a [`CachedScan`]. A gap reads through the normal read path, from a
//! database state view that holds every version at the scan's sequence number, with that
//! sequence number as `max_seq` (see `DbInner::range_cache_scan` for how the view is chosen).
//! All parts read at the same sequence number, so the scan is the concatenation of their
//! answers.
//!
//! The scan opens each part when it first reaches it. A gap costs a normal-path iterator, which
//! can read from object storage, so a scan that ends early, or seeks past a gap, never opens it.
//! When the scan opens a gap that no ready cached range holds, it samples the read for automatic
//! promotion, as a normal-path scan is sampled. It counts each gap that it opens once, by why the
//! part is a gap (`fallback_count{op=gap}`).

use std::collections::VecDeque;
use std::ops::Bound::{self, Excluded, Included, Unbounded};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use super::scan::CachedScan;
use super::stats::{Fallback, FallbackOp, ReadCount};
use super::{scan_start, Layers, RangeCache};
use crate::bytes_range::BytesRange;
use crate::config::ScanOptions;
use crate::db_iter::DbIterator;
use crate::db_state::DbStateView;
use crate::error::SlateDBError;
use crate::iter::RowEntryIterator;
use crate::merge_operator::MergeOperatorType;
use crate::reader::{Reader, ScanContext};
use crate::types::RowEntry;

/// What a part of a mixed scan reads.
pub(crate) enum PartKind {
    /// The pinned layers of a cached range that follow the sequence number rules.
    Cached(Arc<Layers>),
    /// The normal read path, because no range holds the part (`uncovered`), its range loads
    /// (`loading`), or its range holds it but cannot serve it at the scan's sequence number
    /// (`too_old`, `sealed`).
    Gap(Fallback),
}

impl PartKind {
    /// True for a gap that automatic promotion samples: promotion has nothing to gain for a part
    /// that a cached range holds.
    fn samples(reason: Fallback) -> bool {
        matches!(reason, Fallback::Uncovered | Fallback::Loading)
    }
}

/// A part of a mixed scan: a non-empty key range inside the scan range.
pub(crate) struct Part {
    pub(crate) kind: PartKind,
    pub(crate) start: Bound<Bytes>,
    pub(crate) end: Bound<Bytes>,
}

/// What a mixed scan needs to open its gaps through the normal read path.
pub(crate) struct GapSource {
    pub(crate) reader: Reader,
    /// A state view that holds every version at the scan's sequence number.
    pub(crate) view: DbStateView,
    pub(crate) options: ScanOptions,
    pub(crate) prefix: Option<Bytes>,
    pub(crate) cache: Arc<RangeCache>,
}

/// The part that the scan reads now.
enum Open {
    None,
    // Boxed, as both are large and the scan opens each part once.
    Cached(Box<CachedScan>),
    Gap(Box<DbIterator>),
}

/// A scan over cached parts and gaps. See the module documentation.
pub(crate) struct MixedScan {
    current: Open,
    /// The bounds of the current part.
    current_start: Bound<Bytes>,
    current_end: Bound<Bytes>,
    /// The parts that the scan has not opened yet, in key order.
    parts: VecDeque<Part>,
    seq: u64,
    /// The scan covers one key. See [`CachedScan::set_point`].
    point: bool,
    merge_operator: Option<MergeOperatorType>,
    gaps: GapSource,
}

/// Returns true if a part that ends at `end` holds no key at or after `key`.
fn ends_before(end: &Bound<Bytes>, key: &[u8]) -> bool {
    match end {
        Unbounded => false,
        Included(end) => end.as_ref() < key,
        Excluded(end) => end.as_ref() <= key,
    }
}

/// Returns true if a part that starts at `start` can hold a key below `key`.
fn starts_before(start: &Bound<Bytes>, key: &[u8]) -> bool {
    match start {
        Unbounded => true,
        Included(start) | Excluded(start) => start.as_ref() < key,
    }
}

impl MixedScan {
    pub(crate) fn new(
        parts: Vec<Part>,
        seq: u64,
        point: bool,
        merge_operator: Option<MergeOperatorType>,
        gaps: GapSource,
    ) -> Self {
        Self {
            current: Open::None,
            current_start: Unbounded,
            current_end: Unbounded,
            parts: parts.into(),
            seq,
            point,
            merge_operator,
            gaps,
        }
    }

    /// Opens the next part if it is cached. Returns false if there is none, or if it is a gap,
    /// which needs an await.
    fn open_cached(&mut self) -> Result<bool, SlateDBError> {
        if !matches!(
            self.parts.front(),
            Some(Part {
                kind: PartKind::Cached(_),
                ..
            })
        ) {
            return Ok(false);
        }
        let Some(part) = self.parts.pop_front() else {
            return Ok(false);
        };
        let PartKind::Cached(layers) = part.kind else {
            return Ok(false);
        };
        let Some(range) = BytesRange::try_new(part.start.clone(), part.end.clone()) else {
            // A seek left nothing of the part.
            return Ok(true);
        };
        let mut scan =
            CachedScan::new(vec![layers], self.seq, &range, self.merge_operator.clone())?;
        scan.set_point(self.point);
        self.current = Open::Cached(Box::new(scan));
        self.current_start = part.start;
        self.current_end = part.end;
        Ok(true)
    }

    /// Opens the next part. Returns false after the last part.
    async fn open_next(&mut self) -> Result<bool, SlateDBError> {
        if self.open_cached()? {
            return Ok(true);
        }
        let Some(part) = self.parts.pop_front() else {
            return Ok(false);
        };
        let PartKind::Gap(reason) = part.kind else {
            return Ok(true);
        };
        let Some(range) = BytesRange::try_new(part.start.clone(), part.end.clone()) else {
            return Ok(true);
        };
        let gaps = &self.gaps;
        let sampled = if PartKind::samples(reason) {
            gaps.cache.sample()
        } else {
            None
        };
        gaps.cache.hits.count(ReadCount::ScanGap);
        gaps.cache.count_fallback(FallbackOp::Gap, reason);
        let iter = gaps
            .reader
            .scan_with_options(
                range.clone(),
                &gaps.options,
                ScanContext {
                    db_state: &gaps.view,
                    write_batch_iter: None,
                    max_seq: Some(self.seq),
                    prefix: gaps.prefix.clone(),
                },
            )
            .await?;
        if let Some(sampled) = sampled {
            gaps.cache.record(sampled, &scan_start(&range));
        }
        self.current = Open::Gap(Box::new(iter));
        self.current_start = part.start;
        self.current_end = part.end;
        Ok(true)
    }
}

#[async_trait]
impl RowEntryIterator for MixedScan {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        loop {
            match &mut self.current {
                Open::Cached(scan) => {
                    if let Some(entry) = scan.next().await? {
                        return Ok(Some(entry));
                    }
                }
                Open::Gap(iter) => {
                    if let Some(entry) = iter.next_entry().await? {
                        return Ok(Some(entry));
                    }
                }
                Open::None => {
                    if !self.open_next().await? {
                        return Ok(None);
                    }
                    continue;
                }
            }
            self.current = Open::None;
        }
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        if !matches!(self.current, Open::None) {
            if ends_before(&self.current_end, next_key) {
                self.current = Open::None;
            } else if !starts_before(&self.current_start, next_key) {
                // The open part starts at or after `next_key`, so it is there already. This
                // happens when `next_sync` opened a cached part and stopped before its first
                // row.
                return Ok(());
            }
        }
        match &mut self.current {
            Open::Cached(scan) => scan.seek(next_key).await,
            Open::Gap(iter) => iter.seek_entry(next_key).await,
            Open::None => {
                // Skip the parts that end before `next_key` without opening them, and start
                // the next one at `next_key`.
                while self
                    .parts
                    .front()
                    .is_some_and(|part| ends_before(&part.end, next_key))
                {
                    self.parts.pop_front();
                }
                if let Some(part) = self.parts.front_mut() {
                    if starts_before(&part.start, next_key) {
                        part.start = Included(Bytes::copy_from_slice(next_key));
                    }
                }
                Ok(())
            }
        }
    }

    fn next_sync(&mut self) -> Option<Result<Option<RowEntry>, SlateDBError>> {
        loop {
            match &mut self.current {
                Open::Cached(scan) => match scan.next_sync()? {
                    Ok(Some(entry)) => return Some(Ok(Some(entry))),
                    Ok(None) => self.current = Open::None,
                    Err(e) => return Some(Err(e)),
                },
                // A gap reads through the normal path, which needs an await.
                Open::Gap(_) => return None,
                Open::None => {
                    if self.parts.is_empty() {
                        return Some(Ok(None));
                    }
                    match self.open_cached() {
                        Ok(true) => continue,
                        Ok(false) => return None,
                        Err(e) => return Some(Err(e)),
                    }
                }
            }
        }
    }
}
