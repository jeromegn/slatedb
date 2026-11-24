//! Metrics for the range cache.
//!
//! Read-path counters (`get_count`, `scan_count`, `mixed_scan_count`, `scan_gap_count`,
//! `mixed_scan_retry_count`, `pin_retry_count`, `unusable_count`, and `fallback_count`) accumulate
//! per thread and fold into shared counters periodically. See
//! [`crate::RangeCacheTuning::stats_fold_events`].

use slatedb_common::metrics::{CounterFn, GaugeFn, MetricsRecorderHelper};
use std::sync::Arc;

macro_rules! range_cache_stat_name {
    ($suffix:expr) => {
        concat!("slatedb.range_cache.", $suffix)
    };
}

/// Point reads served by the range cache. `result` is `hit`, `absent`, or `merge`.
pub const GET_COUNT: &str = range_cache_stat_name!("get_count");
/// Scans served completely by the range cache.
pub const SCAN_COUNT: &str = range_cache_stat_name!("scan_count");
/// Scans served in part by cached ranges and in part by the normal read path.
pub const MIXED_SCAN_COUNT: &str = range_cache_stat_name!("mixed_scan_count");
/// Uncached gaps opened by mixed scans on the normal read path.
pub const SCAN_GAP_COUNT: &str = range_cache_stat_name!("scan_gap_count");
/// Mixed-scan planning retries caused by a concurrent memtable freeze.
pub const MIXED_SCAN_RETRY_COUNT: &str = range_cache_stat_name!("mixed_scan_retry_count");
/// Cached ranges pinned lazily by snapshot and transaction scans.
pub const LAZY_PIN_COUNT: &str = range_cache_stat_name!("lazy_pin_count");
/// Retried attempts to pin a ready cached range. `op` is `get` or `scan`; `reason` is `too_old`
/// or `sealed`.
pub const PIN_RETRY_COUNT: &str = range_cache_stat_name!("pin_retry_count");
/// Reads that fell back to the normal read path because a cached range failed sequence rules.
/// `op` is `get`, `scan`, or `scan_part`; `reason` is `too_old` or `sealed`.
pub const UNUSABLE_COUNT: &str = range_cache_stat_name!("unusable_count");
/// Reads that used the normal read path instead of the range cache, counted once by `op` (`get`,
/// `scan`, or `gap`) and `reason` (`uncovered`, `loading`, `too_old`, `sealed`, `error`, `frozen`,
/// `descending`, `remote`, `dirty`, `filter`, or `traced`).
pub const FALLBACK_COUNT: &str = range_cache_stat_name!("fallback_count");
/// Ranges loaded into the cache.
pub const LOAD_COUNT: &str = range_cache_stat_name!("load_count");
/// Bytes of range cache files written by loads.
pub const LOAD_BYTES: &str = range_cache_stat_name!("load_bytes");
/// Loads rejected before starting. `reason` is `memory`.
pub const LOAD_REJECTED_COUNT: &str = range_cache_stat_name!("load_rejected_count");
/// Major merges that rebuilt a range base from its base and runs.
pub const REBUILD_COUNT: &str = range_cache_stat_name!("rebuild_count");
/// Bytes of base files written by major merges.
pub const REBUILD_BYTES: &str = range_cache_stat_name!("rebuild_bytes");
/// Active write buffers sealed. `reason` is `full` or `pressure`.
pub const SEAL_COUNT: &str = range_cache_stat_name!("seal_count");
/// Flushes of sealed buffers to run files.
pub const FLUSH_COUNT: &str = range_cache_stat_name!("flush_count");
/// Bytes of run files written by plain flushes.
pub const FLUSH_BYTES: &str = range_cache_stat_name!("flush_bytes");
/// Flushes that merged the newest runs at `max_runs` to stay within the run cap.
pub const FLUSH_MERGE_COUNT: &str = range_cache_stat_name!("flush_merge_count");
/// Bytes of run files written by flush-merges.
pub const FLUSH_MERGE_BYTES: &str = range_cache_stat_name!("flush_merge_bytes");
/// Flushed runs or merged bases rejected because the range state changed concurrently.
pub const INSTALL_REJECTED_COUNT: &str = range_cache_stat_name!("install_rejected_count");
/// Zero-copy range splits. `when` is `maintenance` or `attach`.
pub const SPLIT_COUNT: &str = range_cache_stat_name!("split_count");
/// Total child ranges created by splits.
pub const SPLIT_RANGE_COUNT: &str = range_cache_stat_name!("split_range_count");
/// Splits rejected because the range state changed before install.
pub const SPLIT_REJECTED_COUNT: &str = range_cache_stat_name!("split_rejected_count");
/// Merges of two adjacent quiet ranges into one without copying file data.
pub const QUIET_MERGE_COUNT: &str = range_cache_stat_name!("quiet_merge_count");
/// Quiet merges rejected because either range changed before install.
pub const QUIET_MERGE_REJECTED_COUNT: &str = range_cache_stat_name!("quiet_merge_rejected_count");
/// Ranges removed from the cache. `reason` is `evicted`, `explicit`, `expiring_write`, `error`,
/// `closed`, `not_worth_it`, or `overflow`.
pub const REMOVE_COUNT: &str = range_cache_stat_name!("remove_count");
/// Cached ranges currently indexed, including loading ranges.
pub const RANGE_COUNT: &str = range_cache_stat_name!("range_count");
/// Disk bytes of open range cache files.
pub const FILE_BYTES: &str = range_cache_stat_name!("file_bytes");
/// Disk bytes of unique base files across cached ranges (shared files counted once).
pub const BASE_FILE_BYTES: &str = range_cache_stat_name!("base_file_bytes");
/// Disk bytes of unique run files across cached ranges (shared files counted once).
pub const RUN_FILE_BYTES: &str = range_cache_stat_name!("run_file_bytes");
/// Estimated bytes of active and sealed write buffers across cached ranges.
pub const DELTA_BYTES: &str = range_cache_stat_name!("delta_bytes");
/// Heap metadata bytes of open files, delta views, and cached range layer sets.
pub const META_BYTES: &str = range_cache_stat_name!("meta_bytes");
/// Exact heap bytes allocated by live write buffers (chunks and filters). Never exceeds
/// `max_recent_bytes`.
pub const DELTA_ALLOCATED_BYTES: &str = range_cache_stat_name!("delta_allocated_bytes");
/// Heap bytes of all live delta views. Also included in `meta_bytes`.
pub const VIEW_BYTES: &str = range_cache_stat_name!("view_bytes");

/// Normal-path reads sampled for automatic promotion. `result` is `recorded` or `dropped`.
pub const SAMPLE_COUNT: &str = range_cache_stat_name!("sample_count");
/// Candidate ranges evaluated by automatic promotion. `source` is `sst_index`, `scan`, or
/// `remembered`.
pub const CANDIDATE_COUNT: &str = range_cache_stat_name!("candidate_count");
/// Candidates admitted by automatic promotion. `result` is `free` or `evicting`.
pub const ADMISSION_COUNT: &str = range_cache_stat_name!("admission_count");
/// Candidates rejected by automatic promotion. `reason` is `cold`, `not_reused`, `too_large`,
/// `no_room`, `not_denser`, or `backoff`.
pub const REJECTION_COUNT: &str = range_cache_stat_name!("rejection_count");
/// Automatic ranges that completed loading.
pub const PROMOTION_COUNT: &str = range_cache_stat_name!("promotion_count");
/// Admitted candidates overlapping a recently removed range.
pub const REPROMOTION_COUNT: &str = range_cache_stat_name!("repromotion_count");

/// Why the cache removed a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RemoveReason {
    /// Evicted to stay within disk or metadata budgets.
    Evicted,
    /// Removed by `uncache_range` or `clear_range_cache`.
    Explicit,
    /// Touched by a write with an expire time.
    ExpiringWrite,
    /// Removed after a load, merge, or read error.
    Error,
    /// Removed when the database closed or failed.
    Closed,
    /// Removed because reads since the last base build did not pay for a major merge.
    NotWorthIt,
    /// Removed to stay within the write buffer memory cap.
    Overflow,
}

impl RemoveReason {
    /// Returns true if a later promotion of the range should back off.
    pub(crate) fn is_failure(self) -> bool {
        matches!(
            self,
            RemoveReason::ExpiringWrite | RemoveReason::Error | RemoveReason::NotWorthIt
        )
    }
}

/// Why a ready cached range failed sequence number checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unfit {
    /// The read sequence number is below a needed chunk's `base_seq`.
    TooOld,
    /// The read sequence number is above the active buffer seal.
    Sealed,
}

/// Why a read or mixed-scan gap used the normal read path. See [`FALLBACK_COUNT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fallback {
    Uncovered,
    Loading,
    TooOld,
    Sealed,
    Error,
    Frozen,
    Descending,
    Remote,
    Dirty,
    Filter,
    Traced,
}

impl From<Unfit> for Fallback {
    fn from(reason: Unfit) -> Self {
        match reason {
            Unfit::TooOld => Fallback::TooOld,
            Unfit::Sealed => Fallback::Sealed,
        }
    }
}

impl Fallback {
    /// Priority order when choosing one fallback reason for an unserved scan.
    pub(crate) fn rank(self) -> u8 {
        match self {
            Fallback::TooOld => 0,
            Fallback::Sealed => 1,
            Fallback::Loading => 2,
            _ => 3,
        }
    }
}

/// Operation label for [`FALLBACK_COUNT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FallbackOp {
    Get,
    Scan,
    Gap,
}

/// The kind of read pinning cached ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinOp {
    Get,
    Scan,
    ScanPart,
}

/// A read-path counter accumulated per thread and folded into shared metrics.
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
    FallbackGetUncovered,
    FallbackGetLoading,
    FallbackGetTooOld,
    FallbackGetSealed,
    FallbackGetError,
    FallbackGetRemote,
    FallbackGetDirty,
    FallbackGetFilter,
    FallbackGetTraced,
    FallbackScanUncovered,
    FallbackScanLoading,
    FallbackScanTooOld,
    FallbackScanSealed,
    FallbackScanFrozen,
    FallbackScanDescending,
    FallbackScanRemote,
    FallbackScanDirty,
    FallbackScanFilter,
    FallbackScanTraced,
    FallbackGapUncovered,
    FallbackGapLoading,
    FallbackGapTooOld,
    FallbackGapSealed,
    /// Cached range pinned lazily by a snapshot or transaction scan.
    LazyPin,
}

/// Number of [`ReadCount`] variants.
pub(crate) const READ_COUNTS: usize = 41;

impl ReadCount {
    /// Every variant in discriminant order.
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
        ReadCount::FallbackGetUncovered,
        ReadCount::FallbackGetLoading,
        ReadCount::FallbackGetTooOld,
        ReadCount::FallbackGetSealed,
        ReadCount::FallbackGetError,
        ReadCount::FallbackGetRemote,
        ReadCount::FallbackGetDirty,
        ReadCount::FallbackGetFilter,
        ReadCount::FallbackGetTraced,
        ReadCount::FallbackScanUncovered,
        ReadCount::FallbackScanLoading,
        ReadCount::FallbackScanTooOld,
        ReadCount::FallbackScanSealed,
        ReadCount::FallbackScanFrozen,
        ReadCount::FallbackScanDescending,
        ReadCount::FallbackScanRemote,
        ReadCount::FallbackScanDirty,
        ReadCount::FallbackScanFilter,
        ReadCount::FallbackScanTraced,
        ReadCount::FallbackGapUncovered,
        ReadCount::FallbackGapLoading,
        ReadCount::FallbackGapTooOld,
        ReadCount::FallbackGapSealed,
        ReadCount::LazyPin,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// Returns true if this read counts as a scan rather than a get on the range.
    pub(crate) fn is_scan(self) -> bool {
        matches!(
            self,
            ReadCount::Scan | ReadCount::MixedScan | ReadCount::LazyPin
        )
    }

    /// Maps `(op, reason)` to its fallback counter, or `None` for impossible combinations.
    pub(crate) fn fallback(op: FallbackOp, reason: Fallback) -> Option<Self> {
        use Fallback as F;
        use FallbackOp as O;
        Some(match (op, reason) {
            (O::Get, F::Uncovered) => ReadCount::FallbackGetUncovered,
            (O::Get, F::Loading) => ReadCount::FallbackGetLoading,
            (O::Get, F::TooOld) => ReadCount::FallbackGetTooOld,
            (O::Get, F::Sealed) => ReadCount::FallbackGetSealed,
            (O::Get, F::Error) => ReadCount::FallbackGetError,
            (O::Get, F::Remote) => ReadCount::FallbackGetRemote,
            (O::Get, F::Dirty) => ReadCount::FallbackGetDirty,
            (O::Get, F::Filter) => ReadCount::FallbackGetFilter,
            (O::Get, F::Traced) => ReadCount::FallbackGetTraced,
            (O::Scan, F::Uncovered) => ReadCount::FallbackScanUncovered,
            (O::Scan, F::Loading) => ReadCount::FallbackScanLoading,
            (O::Scan, F::TooOld) => ReadCount::FallbackScanTooOld,
            (O::Scan, F::Sealed) => ReadCount::FallbackScanSealed,
            (O::Scan, F::Frozen) => ReadCount::FallbackScanFrozen,
            (O::Scan, F::Descending) => ReadCount::FallbackScanDescending,
            (O::Scan, F::Remote) => ReadCount::FallbackScanRemote,
            (O::Scan, F::Dirty) => ReadCount::FallbackScanDirty,
            (O::Scan, F::Filter) => ReadCount::FallbackScanFilter,
            (O::Scan, F::Traced) => ReadCount::FallbackScanTraced,
            (O::Gap, F::Uncovered) => ReadCount::FallbackGapUncovered,
            (O::Gap, F::Loading) => ReadCount::FallbackGapLoading,
            (O::Gap, F::TooOld) => ReadCount::FallbackGapTooOld,
            (O::Gap, F::Sealed) => ReadCount::FallbackGapSealed,
            _ => return None,
        })
    }
    pub(crate) fn retry(op: PinOp, reason: Unfit) -> Self {
        match (op, reason) {
            (PinOp::Get, Unfit::TooOld) => ReadCount::RetryGetTooOld,
            (PinOp::Get, Unfit::Sealed) => ReadCount::RetryGetSealed,
            (PinOp::Scan | PinOp::ScanPart, Unfit::TooOld) => ReadCount::RetryScanTooOld,
            (PinOp::Scan | PinOp::ScanPart, Unfit::Sealed) => ReadCount::RetryScanSealed,
        }
    }

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
            ReadCount::FallbackGetUncovered => fallback_labels("get", "uncovered"),
            ReadCount::FallbackGetLoading => fallback_labels("get", "loading"),
            ReadCount::FallbackGetTooOld => fallback_labels("get", "too_old"),
            ReadCount::FallbackGetSealed => fallback_labels("get", "sealed"),
            ReadCount::FallbackGetError => fallback_labels("get", "error"),
            ReadCount::FallbackGetRemote => fallback_labels("get", "remote"),
            ReadCount::FallbackGetDirty => fallback_labels("get", "dirty"),
            ReadCount::FallbackGetFilter => fallback_labels("get", "filter"),
            ReadCount::FallbackGetTraced => fallback_labels("get", "traced"),
            ReadCount::FallbackScanUncovered => fallback_labels("scan", "uncovered"),
            ReadCount::FallbackScanLoading => fallback_labels("scan", "loading"),
            ReadCount::FallbackScanTooOld => fallback_labels("scan", "too_old"),
            ReadCount::FallbackScanSealed => fallback_labels("scan", "sealed"),
            ReadCount::FallbackScanFrozen => fallback_labels("scan", "frozen"),
            ReadCount::FallbackScanDescending => fallback_labels("scan", "descending"),
            ReadCount::FallbackScanRemote => fallback_labels("scan", "remote"),
            ReadCount::FallbackScanDirty => fallback_labels("scan", "dirty"),
            ReadCount::FallbackScanFilter => fallback_labels("scan", "filter"),
            ReadCount::FallbackScanTraced => fallback_labels("scan", "traced"),
            ReadCount::FallbackGapUncovered => fallback_labels("gap", "uncovered"),
            ReadCount::FallbackGapLoading => fallback_labels("gap", "loading"),
            ReadCount::FallbackGapTooOld => fallback_labels("gap", "too_old"),
            ReadCount::FallbackGapSealed => fallback_labels("gap", "sealed"),
            ReadCount::LazyPin => (LAZY_PIN_COUNT, &[]),
        }
    }
}

/// Returns the metric name and label slice for `fallback_count`.
fn fallback_labels(
    op: &'static str,
    reason: &'static str,
) -> (&'static str, &'static [(&'static str, &'static str)]) {
    macro_rules! labels {
        ($($op:literal, $reason:literal;)*) => {
            match (op, reason) {
                $(($op, $reason) => &[("op", $op), ("reason", $reason)],)*
                _ => &[],
            }
        };
    }
    let labels: &'static [(&'static str, &'static str)] = labels! {
        "get", "uncovered"; "get", "loading"; "get", "too_old"; "get", "sealed"; "get", "error";
        "get", "remote"; "get", "dirty"; "get", "filter"; "get", "traced";
        "scan", "uncovered"; "scan", "loading"; "scan", "too_old"; "scan", "sealed";
        "scan", "frozen"; "scan", "descending"; "scan", "remote"; "scan", "dirty";
        "scan", "filter"; "scan", "traced";
        "gap", "uncovered"; "gap", "loading"; "gap", "too_old"; "gap", "sealed";
    };
    (FALLBACK_COUNT, labels)
}

/// Registers read-path counters indexed by [`ReadCount::index`].
pub(crate) fn read_counters(recorder: &MetricsRecorderHelper) -> [Arc<dyn CounterFn>; READ_COUNTS] {
    ReadCount::ALL.map(|count| {
        let (name, labels) = count.metric();
        recorder.counter(name).labels(labels).register()
    })
}

/// Why automatic promotion rejected a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RejectReason {
    Cold,
    NotReused,
    TooLarge,
    NoRoom,
    NotDenser,
    Backoff,
}

/// Source of a candidate's key bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateSource {
    SstIndex,
    Scan,
    Remembered,
}

/// Automatic promotion metrics.
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

/// Range cache metrics outside the per-thread read counters.
pub(crate) struct RangeCacheStats {
    pub(crate) loads: Arc<dyn CounterFn>,
    pub(crate) load_bytes: Arc<dyn CounterFn>,
    pub(crate) loads_rejected_memory: Arc<dyn CounterFn>,
    pub(crate) rebuilds: Arc<dyn CounterFn>,
    pub(crate) rebuild_bytes: Arc<dyn CounterFn>,
    pub(crate) seals_full: Arc<dyn CounterFn>,
    pub(crate) seals_pressure: Arc<dyn CounterFn>,
    pub(crate) flushes: Arc<dyn CounterFn>,
    pub(crate) flush_bytes: Arc<dyn CounterFn>,
    pub(crate) flush_merges: Arc<dyn CounterFn>,
    pub(crate) flush_merge_bytes: Arc<dyn CounterFn>,
    pub(crate) installs_rejected: Arc<dyn CounterFn>,
    pub(crate) splits_maintenance: Arc<dyn CounterFn>,
    pub(crate) splits_attach: Arc<dyn CounterFn>,
    pub(crate) split_ranges: Arc<dyn CounterFn>,
    pub(crate) splits_rejected: Arc<dyn CounterFn>,
    pub(crate) quiet_merges: Arc<dyn CounterFn>,
    pub(crate) quiet_merges_rejected: Arc<dyn CounterFn>,
    removed_evicted: Arc<dyn CounterFn>,
    removed_explicit: Arc<dyn CounterFn>,
    removed_expiring_write: Arc<dyn CounterFn>,
    removed_error: Arc<dyn CounterFn>,
    removed_closed: Arc<dyn CounterFn>,
    removed_not_worth_it: Arc<dyn CounterFn>,
    removed_overflow: Arc<dyn CounterFn>,
    pub(crate) ranges: Arc<dyn GaugeFn>,
    pub(crate) file_bytes: Arc<dyn GaugeFn>,
    pub(crate) base_file_bytes: Arc<dyn GaugeFn>,
    pub(crate) run_file_bytes: Arc<dyn GaugeFn>,
    pub(crate) delta_bytes: Arc<dyn GaugeFn>,
    pub(crate) meta_bytes: Arc<dyn GaugeFn>,
    pub(crate) view_bytes: Arc<dyn GaugeFn>,
}

impl RangeCacheStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> Self {
        let labeled =
            |name, label, value| recorder.counter(name).labels(&[(label, value)]).register();
        let removed = |reason| labeled(REMOVE_COUNT, "reason", reason);
        Self {
            loads: recorder.counter(LOAD_COUNT).register(),
            load_bytes: recorder.counter(LOAD_BYTES).register(),
            loads_rejected_memory: labeled(LOAD_REJECTED_COUNT, "reason", "memory"),
            rebuilds: recorder.counter(REBUILD_COUNT).register(),
            rebuild_bytes: recorder.counter(REBUILD_BYTES).register(),
            seals_full: labeled(SEAL_COUNT, "reason", "full"),
            seals_pressure: labeled(SEAL_COUNT, "reason", "pressure"),
            flushes: recorder.counter(FLUSH_COUNT).register(),
            flush_bytes: recorder.counter(FLUSH_BYTES).register(),
            flush_merges: recorder.counter(FLUSH_MERGE_COUNT).register(),
            flush_merge_bytes: recorder.counter(FLUSH_MERGE_BYTES).register(),
            installs_rejected: recorder.counter(INSTALL_REJECTED_COUNT).register(),
            splits_maintenance: labeled(SPLIT_COUNT, "when", "maintenance"),
            splits_attach: labeled(SPLIT_COUNT, "when", "attach"),
            split_ranges: recorder.counter(SPLIT_RANGE_COUNT).register(),
            splits_rejected: recorder.counter(SPLIT_REJECTED_COUNT).register(),
            quiet_merges: recorder.counter(QUIET_MERGE_COUNT).register(),
            quiet_merges_rejected: recorder.counter(QUIET_MERGE_REJECTED_COUNT).register(),
            removed_evicted: removed("evicted"),
            removed_explicit: removed("explicit"),
            removed_expiring_write: removed("expiring_write"),
            removed_error: removed("error"),
            removed_closed: removed("closed"),
            removed_not_worth_it: removed("not_worth_it"),
            removed_overflow: removed("overflow"),
            ranges: recorder.gauge(RANGE_COUNT).register(),
            file_bytes: recorder.gauge(FILE_BYTES).register(),
            base_file_bytes: recorder.gauge(BASE_FILE_BYTES).register(),
            run_file_bytes: recorder.gauge(RUN_FILE_BYTES).register(),
            delta_bytes: recorder.gauge(DELTA_BYTES).register(),
            meta_bytes: recorder.gauge(META_BYTES).register(),
            view_bytes: recorder.gauge(VIEW_BYTES).register(),
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
            RemoveReason::Overflow => &self.removed_overflow,
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

    #[test]
    fn should_map_each_possible_fallback_to_its_own_counter() {
        use Fallback as F;
        let reasons = [
            (F::Uncovered, "uncovered"),
            (F::Loading, "loading"),
            (F::TooOld, "too_old"),
            (F::Sealed, "sealed"),
            (F::Error, "error"),
            (F::Frozen, "frozen"),
            (F::Descending, "descending"),
            (F::Remote, "remote"),
            (F::Dirty, "dirty"),
            (F::Filter, "filter"),
            (F::Traced, "traced"),
        ];
        let ops = [
            (FallbackOp::Get, "get"),
            (FallbackOp::Scan, "scan"),
            (FallbackOp::Gap, "gap"),
        ];
        let mut counters = BTreeSet::new();
        for (op, op_label) in ops {
            for (reason, reason_label) in reasons {
                let possible = match op {
                    FallbackOp::Get => !matches!(reason, F::Frozen | F::Descending),
                    FallbackOp::Scan => reason != F::Error,
                    FallbackOp::Gap => {
                        matches!(reason, F::Uncovered | F::Loading | F::TooOld | F::Sealed)
                    }
                };
                let Some(count) = ReadCount::fallback(op, reason) else {
                    assert!(!possible, "{op_label} {reason_label}");
                    continue;
                };
                assert!(possible, "{op_label} {reason_label}");
                let labels = [("op", op_label), ("reason", reason_label)];
                assert_eq!(count.metric(), (FALLBACK_COUNT, &labels[..]));
                assert!(counters.insert(count.index()));
            }
        }
        assert_eq!(counters.len(), 23);
    }
}
