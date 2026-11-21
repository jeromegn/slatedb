//! Metrics of the range cache.
//!
//! The counters of the read path (`get_count`, `scan_count`, `mixed_scan_count`,
//! `scan_gap_count`, `mixed_scan_retry_count`, `pin_retry_count` and `unusable_count`) are
//! counted on each thread first and added to the shared counters every few thousand reads or
//! about every 10 ms, so a read writes no counter that other threads write. They are exact only
//! after such a fold. See [`crate::RangeCacheTuning::stats_fold_events`].

use slatedb_common::metrics::{CounterFn, GaugeFn, MetricsRecorderHelper};
use std::sync::Arc;

macro_rules! range_cache_stat_name {
    ($suffix:expr) => {
        concat!("slatedb.range_cache.", $suffix)
    };
}

/// Point reads that the range cache served. The `result` label is `hit` (a value), `absent`
/// (a definite miss) or `merge` (a merge operand that the cache resolved).
pub const GET_COUNT: &str = range_cache_stat_name!("get_count");
/// Scans that the range cache served completely.
pub const SCAN_COUNT: &str = range_cache_stat_name!("scan_count");
/// Scans that cached ranges served in part: the parts that no usable cached range holds, the
/// gaps, come from the normal read path.
pub const MIXED_SCAN_COUNT: &str = range_cache_stat_name!("mixed_scan_count");
/// Gaps of mixed scans that the normal read path opened. A scan opens a gap only when it
/// reaches it.
pub const SCAN_GAP_COUNT: &str = range_cache_stat_name!("scan_gap_count");
/// Attempts to plan a mixed scan at the latest state that a memtable freeze made the scan
/// repeat. After a few, the scan uses the normal read path.
pub const MIXED_SCAN_RETRY_COUNT: &str = range_cache_stat_name!("mixed_scan_retry_count");
/// Attempts of a read to use a ready cached range that failed the sequence number rules and that
/// the read tried again. The `op` label is `get` or `scan` (a scan that cached ranges hold
/// completely). The `reason` label is `too_old` (the read's sequence number is below the range
/// file's) or `sealed` (it is above the sequence number at which the cache stopped applying
/// writes to the active delta: a freeze or a removal happened during the attempt).
pub const PIN_RETRY_COUNT: &str = range_cache_stat_name!("pin_retry_count");
/// Reads that a ready cached range holds but that used the normal read path, because the range
/// failed the sequence number rules. The `reason` label is the reason of the last attempt, as for
/// `pin_retry_count`. The `op` label is `get` and `scan` (after the last attempt), or
/// `scan_part` (a part of a mixed scan that the normal read path then reads as a gap).
pub const UNUSABLE_COUNT: &str = range_cache_stat_name!("unusable_count");
/// Ranges that the cache loaded.
pub const LOAD_COUNT: &str = range_cache_stat_name!("load_count");
/// Bytes of range cache files that loads wrote.
pub const LOAD_BYTES: &str = range_cache_stat_name!("load_bytes");
/// Rebuilds of range cache files.
pub const REBUILD_COUNT: &str = range_cache_stat_name!("rebuild_count");
/// Bytes of range cache files that rebuilds wrote. Divide by the bytes of the writes to cached
/// ranges to get the local write amplification.
pub const REBUILD_BYTES: &str = range_cache_stat_name!("rebuild_bytes");
/// Ranges that the cache removed. The `reason` label is `evicted`, `explicit`,
/// `expiring_write`, `error`, `closed` or `not_worth_it` (an automatic range whose reads did not
/// pay for a rebuild).
pub const REMOVE_COUNT: &str = range_cache_stat_name!("remove_count");
/// The number of cached ranges, loading ranges included.
pub const RANGE_COUNT: &str = range_cache_stat_name!("range_count");
/// The bytes of open range cache files.
pub const FILE_BYTES: &str = range_cache_stat_name!("file_bytes");
/// The estimated bytes of the active and frozen deltas of cached ranges. The cache updates it
/// when ranges change and, with automatic promotion, at each evaluation.
pub const DELTA_BYTES: &str = range_cache_stat_name!("delta_bytes");
/// The memory that the deltas hold, exactly: every chunk of every delta from its allocation until
/// the last holder of the delta (its range, a scan, a merge resolution or a rebuild) drops it. It
/// changes when a delta allocates a chunk or drops. A delta's memory is larger than its share of
/// `delta_bytes` by its index and its last chunk's free room.
pub const DELTA_ALLOCATED_BYTES: &str = range_cache_stat_name!("delta_allocated_bytes");

// Automatic promotion. The cache registers these metrics only when automatic promotion is on.

/// Normal-path reads that automatic promotion sampled. The `result` label is `recorded` or
/// `dropped` (the queue to the promoter was full).
pub const SAMPLE_COUNT: &str = range_cache_stat_name!("sample_count");
/// Candidate ranges that the promoter evaluated. The `source` label is `sst_index` (bounds from
/// the index of an SST), `scan` (bounds from a bounded scan) or `remembered` (a rejected
/// candidate, evaluated again without a new scan).
pub const CANDIDATE_COUNT: &str = range_cache_stat_name!("candidate_count");
/// Candidates that the promoter admitted. The `result` label is `free` (it fit in the budget) or
/// `evicting` (it needed eviction and beat the ranges that it displaces).
pub const ADMISSION_COUNT: &str = range_cache_stat_name!("admission_count");
/// Candidates that the promoter rejected. The `reason` label is `cold` (too few sampled reads),
/// `not_reused` (its reads did not span enough time, such as a single pass), `too_large`,
/// `no_room` (only explicitly cached ranges could make room), `not_denser` (it did not beat the
/// ranges that it would evict by the margin) or `backoff` (a recent failed promotion).
pub const REJECTION_COUNT: &str = range_cache_stat_name!("rejection_count");
/// Automatic ranges that finished loading.
pub const PROMOTION_COUNT: &str = range_cache_stat_name!("promotion_count");
/// Admitted candidates that overlap a range that the cache removed recently.
pub const REPROMOTION_COUNT: &str = range_cache_stat_name!("repromotion_count");

/// Why the cache removed a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RemoveReason {
    /// The disk budget was full.
    Evicted,
    /// A call to `uncache_range` or `clear_range_cache`.
    Explicit,
    /// A write with an expire time touched the range.
    ExpiringWrite,
    /// A load, a rebuild or a read of the range failed.
    Error,
    /// The database closed or failed.
    Closed,
    /// The delta of an automatic range outgrew its threshold, and the reads that the range
    /// served since its file was written did not pay for a rebuild.
    NotWorthIt,
}

impl RemoveReason {
    /// Returns true if a later promotion of the range should wait. Space and explicit removals
    /// do not count as failures.
    pub(crate) fn is_failure(self) -> bool {
        matches!(
            self,
            RemoveReason::ExpiringWrite | RemoveReason::Error | RemoveReason::NotWorthIt
        )
    }
}

/// Why a ready cached range failed the sequence number rules for a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unfit {
    /// The read's sequence number is below the sequence number of the range's file.
    TooOld,
    /// The read's sequence number is above the sequence number at which the active delta was
    /// sealed.
    Sealed,
}

/// The kind of read that pins cached ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinOp {
    /// A point read.
    Get,
    /// A scan that cached ranges hold completely.
    Scan,
    /// One cached part of a mixed scan.
    ScanPart,
}

/// A counter of the read path. A read adds to it on its own thread, and the thread adds its
/// counts to the shared counter at fold points. See the `hits` module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadCount {
    GetHit,
    GetAbsent,
    GetMerge,
    Scan,
    MixedScan,
    ScanGap,
    MixedScanRetry,
    RetryGetTooOld,
    RetryGetSealed,
    RetryScanTooOld,
    RetryScanSealed,
    UnusableGetTooOld,
    UnusableGetSealed,
    UnusableScanTooOld,
    UnusableScanSealed,
    UnusablePartTooOld,
    UnusablePartSealed,
}

/// The number of [`ReadCount`] variants.
pub(crate) const READ_COUNTS: usize = 17;

impl ReadCount {
    /// Every variant, in the order of their discriminants.
    pub(crate) const ALL: [ReadCount; READ_COUNTS] = [
        ReadCount::GetHit,
        ReadCount::GetAbsent,
        ReadCount::GetMerge,
        ReadCount::Scan,
        ReadCount::MixedScan,
        ReadCount::ScanGap,
        ReadCount::MixedScanRetry,
        ReadCount::RetryGetTooOld,
        ReadCount::RetryGetSealed,
        ReadCount::RetryScanTooOld,
        ReadCount::RetryScanSealed,
        ReadCount::UnusableGetTooOld,
        ReadCount::UnusableGetSealed,
        ReadCount::UnusableScanTooOld,
        ReadCount::UnusableScanSealed,
        ReadCount::UnusablePartTooOld,
        ReadCount::UnusablePartSealed,
    ];

    /// The index of the counter in the arrays of the `hits` module.
    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// A failed attempt of `op` that the read tries again. A part of a mixed scan is never
    /// tried again, so `op` is `Get` or `Scan`.
    pub(crate) fn retry(op: PinOp, reason: Unfit) -> Self {
        match (op, reason) {
            (PinOp::Get, Unfit::TooOld) => ReadCount::RetryGetTooOld,
            (PinOp::Get, Unfit::Sealed) => ReadCount::RetryGetSealed,
            (PinOp::Scan | PinOp::ScanPart, Unfit::TooOld) => ReadCount::RetryScanTooOld,
            (PinOp::Scan | PinOp::ScanPart, Unfit::Sealed) => ReadCount::RetryScanSealed,
        }
    }

    /// A read of `op` that used the normal read path because of `reason`.
    pub(crate) fn unusable(op: PinOp, reason: Unfit) -> Self {
        match (op, reason) {
            (PinOp::Get, Unfit::TooOld) => ReadCount::UnusableGetTooOld,
            (PinOp::Get, Unfit::Sealed) => ReadCount::UnusableGetSealed,
            (PinOp::Scan, Unfit::TooOld) => ReadCount::UnusableScanTooOld,
            (PinOp::Scan, Unfit::Sealed) => ReadCount::UnusableScanSealed,
            (PinOp::ScanPart, Unfit::TooOld) => ReadCount::UnusablePartTooOld,
            (PinOp::ScanPart, Unfit::Sealed) => ReadCount::UnusablePartSealed,
        }
    }

    /// The metric name and labels of the counter.
    fn metric(self) -> (&'static str, &'static [(&'static str, &'static str)]) {
        match self {
            ReadCount::GetHit => (GET_COUNT, &[("result", "hit")]),
            ReadCount::GetAbsent => (GET_COUNT, &[("result", "absent")]),
            ReadCount::GetMerge => (GET_COUNT, &[("result", "merge")]),
            ReadCount::Scan => (SCAN_COUNT, &[]),
            ReadCount::MixedScan => (MIXED_SCAN_COUNT, &[]),
            ReadCount::ScanGap => (SCAN_GAP_COUNT, &[]),
            ReadCount::MixedScanRetry => (MIXED_SCAN_RETRY_COUNT, &[]),
            ReadCount::RetryGetTooOld => (PIN_RETRY_COUNT, &[("op", "get"), ("reason", "too_old")]),
            ReadCount::RetryGetSealed => (PIN_RETRY_COUNT, &[("op", "get"), ("reason", "sealed")]),
            ReadCount::RetryScanTooOld => {
                (PIN_RETRY_COUNT, &[("op", "scan"), ("reason", "too_old")])
            }
            ReadCount::RetryScanSealed => {
                (PIN_RETRY_COUNT, &[("op", "scan"), ("reason", "sealed")])
            }
            ReadCount::UnusableGetTooOld => {
                (UNUSABLE_COUNT, &[("op", "get"), ("reason", "too_old")])
            }
            ReadCount::UnusableGetSealed => {
                (UNUSABLE_COUNT, &[("op", "get"), ("reason", "sealed")])
            }
            ReadCount::UnusableScanTooOld => {
                (UNUSABLE_COUNT, &[("op", "scan"), ("reason", "too_old")])
            }
            ReadCount::UnusableScanSealed => {
                (UNUSABLE_COUNT, &[("op", "scan"), ("reason", "sealed")])
            }
            ReadCount::UnusablePartTooOld => (
                UNUSABLE_COUNT,
                &[("op", "scan_part"), ("reason", "too_old")],
            ),
            ReadCount::UnusablePartSealed => {
                (UNUSABLE_COUNT, &[("op", "scan_part"), ("reason", "sealed")])
            }
        }
    }
}

/// Registers the counters of the read path, indexed by [`ReadCount::index`].
pub(crate) fn read_counters(recorder: &MetricsRecorderHelper) -> [Arc<dyn CounterFn>; READ_COUNTS] {
    ReadCount::ALL.map(|count| {
        let (name, labels) = count.metric();
        recorder.counter(name).labels(labels).register()
    })
}

/// Why the promoter rejected a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RejectReason {
    Cold,
    NotReused,
    TooLarge,
    NoRoom,
    NotDenser,
    Backoff,
}

/// Where the bounds of a candidate came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateSource {
    SstIndex,
    Scan,
    Remembered,
}

/// Metrics of automatic promotion.
pub(crate) struct AutoStats {
    pub(crate) samples_recorded: Arc<dyn CounterFn>,
    pub(crate) samples_dropped: Arc<dyn CounterFn>,
    candidates_sst_index: Arc<dyn CounterFn>,
    candidates_scan: Arc<dyn CounterFn>,
    candidates_remembered: Arc<dyn CounterFn>,
    pub(crate) admissions_free: Arc<dyn CounterFn>,
    pub(crate) admissions_evicting: Arc<dyn CounterFn>,
    rejected: [Arc<dyn CounterFn>; 6],
    pub(crate) promotions: Arc<dyn CounterFn>,
    pub(crate) repromotions: Arc<dyn CounterFn>,
}

impl AutoStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> Self {
        let labeled =
            |name, label, value| recorder.counter(name).labels(&[(label, value)]).register();
        let rejected = |reason| labeled(REJECTION_COUNT, "reason", reason);
        Self {
            samples_recorded: labeled(SAMPLE_COUNT, "result", "recorded"),
            samples_dropped: labeled(SAMPLE_COUNT, "result", "dropped"),
            candidates_sst_index: labeled(CANDIDATE_COUNT, "source", "sst_index"),
            candidates_scan: labeled(CANDIDATE_COUNT, "source", "scan"),
            candidates_remembered: labeled(CANDIDATE_COUNT, "source", "remembered"),
            admissions_free: labeled(ADMISSION_COUNT, "result", "free"),
            admissions_evicting: labeled(ADMISSION_COUNT, "result", "evicting"),
            rejected: [
                rejected("cold"),
                rejected("not_reused"),
                rejected("too_large"),
                rejected("no_room"),
                rejected("not_denser"),
                rejected("backoff"),
            ],
            promotions: recorder.counter(PROMOTION_COUNT).register(),
            repromotions: recorder.counter(REPROMOTION_COUNT).register(),
        }
    }

    pub(crate) fn rejected(&self, reason: RejectReason) {
        let index = match reason {
            RejectReason::Cold => 0,
            RejectReason::NotReused => 1,
            RejectReason::TooLarge => 2,
            RejectReason::NoRoom => 3,
            RejectReason::NotDenser => 4,
            RejectReason::Backoff => 5,
        };
        self.rejected[index].increment(1);
    }

    pub(crate) fn candidate(&self, source: CandidateSource) {
        match source {
            CandidateSource::SstIndex => &self.candidates_sst_index,
            CandidateSource::Scan => &self.candidates_scan,
            CandidateSource::Remembered => &self.candidates_remembered,
        }
        .increment(1);
    }
}

/// The metrics of the range cache, except the counters of the read path. Those go through the
/// `hits` module. See [`read_counters`].
pub(crate) struct RangeCacheStats {
    pub(crate) loads: Arc<dyn CounterFn>,
    pub(crate) load_bytes: Arc<dyn CounterFn>,
    pub(crate) rebuilds: Arc<dyn CounterFn>,
    pub(crate) rebuild_bytes: Arc<dyn CounterFn>,
    removed_evicted: Arc<dyn CounterFn>,
    removed_explicit: Arc<dyn CounterFn>,
    removed_expiring_write: Arc<dyn CounterFn>,
    removed_error: Arc<dyn CounterFn>,
    removed_closed: Arc<dyn CounterFn>,
    removed_not_worth_it: Arc<dyn CounterFn>,
    pub(crate) ranges: Arc<dyn GaugeFn>,
    pub(crate) file_bytes: Arc<dyn GaugeFn>,
    pub(crate) delta_bytes: Arc<dyn GaugeFn>,
}

impl RangeCacheStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> Self {
        let removed = |reason| {
            recorder
                .counter(REMOVE_COUNT)
                .labels(&[("reason", reason)])
                .register()
        };
        Self {
            loads: recorder.counter(LOAD_COUNT).register(),
            load_bytes: recorder.counter(LOAD_BYTES).register(),
            rebuilds: recorder.counter(REBUILD_COUNT).register(),
            rebuild_bytes: recorder.counter(REBUILD_BYTES).register(),
            removed_evicted: removed("evicted"),
            removed_explicit: removed("explicit"),
            removed_expiring_write: removed("expiring_write"),
            removed_error: removed("error"),
            removed_closed: removed("closed"),
            removed_not_worth_it: removed("not_worth_it"),
            ranges: recorder.gauge(RANGE_COUNT).register(),
            file_bytes: recorder.gauge(FILE_BYTES).register(),
            delta_bytes: recorder.gauge(DELTA_BYTES).register(),
        }
    }

    pub(crate) fn removed(&self, reason: RemoveReason) {
        let counter = match reason {
            RemoveReason::Evicted => &self.removed_evicted,
            RemoveReason::Explicit => &self.removed_explicit,
            RemoveReason::ExpiringWrite => &self.removed_expiring_write,
            RemoveReason::Error => &self.removed_error,
            RemoveReason::Closed => &self.removed_closed,
            RemoveReason::NotWorthIt => &self.removed_not_worth_it,
        };
        counter.increment(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn should_list_each_read_count_once_in_order() {
        for (i, count) in ReadCount::ALL.iter().enumerate() {
            assert_eq!(count.index(), i, "{count:?}");
        }
        let metrics: BTreeSet<_> = ReadCount::ALL.iter().map(|c| c.metric()).collect();
        assert_eq!(metrics.len(), READ_COUNTS, "two read counts share a metric");
    }

    #[test]
    fn should_map_retries_and_fallbacks_to_their_op_and_reason() {
        let cases = [
            (PinOp::Get, Unfit::TooOld, "get", "too_old"),
            (PinOp::Get, Unfit::Sealed, "get", "sealed"),
            (PinOp::Scan, Unfit::TooOld, "scan", "too_old"),
            (PinOp::Scan, Unfit::Sealed, "scan", "sealed"),
        ];
        for (op, reason, op_label, reason_label) in cases {
            let labels = [("op", op_label), ("reason", reason_label)];
            assert_eq!(
                ReadCount::retry(op, reason).metric(),
                (PIN_RETRY_COUNT, &labels[..])
            );
            assert_eq!(
                ReadCount::unusable(op, reason).metric(),
                (UNUSABLE_COUNT, &labels[..])
            );
        }
        assert_eq!(
            ReadCount::unusable(PinOp::ScanPart, Unfit::Sealed).metric(),
            (
                UNUSABLE_COUNT,
                &[("op", "scan_part"), ("reason", "sealed")][..]
            )
        );
    }
}
