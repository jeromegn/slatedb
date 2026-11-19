//! Metrics of the range cache.

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
/// Scans that the range cache served.
pub const SCAN_COUNT: &str = range_cache_stat_name!("scan_count");
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

pub(crate) struct RangeCacheStats {
    pub(crate) get_hits: Arc<dyn CounterFn>,
    pub(crate) get_absent: Arc<dyn CounterFn>,
    pub(crate) get_merges: Arc<dyn CounterFn>,
    pub(crate) scans: Arc<dyn CounterFn>,
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
        let get = |result| {
            recorder
                .counter(GET_COUNT)
                .labels(&[("result", result)])
                .register()
        };
        let removed = |reason| {
            recorder
                .counter(REMOVE_COUNT)
                .labels(&[("reason", reason)])
                .register()
        };
        Self {
            get_hits: get("hit"),
            get_absent: get("absent"),
            get_merges: get("merge"),
            scans: recorder.counter(SCAN_COUNT).register(),
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
