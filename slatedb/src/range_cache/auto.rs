//! Automatic promotion: stage 2 of RFC 0040.
//!
//! The read path samples normal-path reads that the cache could have served (point reads and
//! scan starts in no cached range), times each one with the database clock, and sends the key
//! and the time to a bounded queue. The [`Promoter`], a background task on the database's task
//! executor, does everything else:
//!
//! - It keeps a windowed reservoir of samples: the window is four heat half-lives, cut into 32
//!   slots, and each slot keeps a uniform random subset of its samples and their count.
//! - At each evaluation it draws seed keys from recent samples, weighted by cost. For a seed in
//!   no cached range, it forms a candidate of about `target_range_bytes` around the key: from the
//!   block index of the SST that holds the key in the largest sorted run, or else from a bounded
//!   forward and reverse scan. The candidate stops at the neighbouring cached ranges and extends
//!   to them when it reaches them, so cached ranges stay adjacent.
//! - Admission estimates the candidate's sampled reads and its normal-path cost per second from
//!   the reservoir, decayed with the heat half-life. See [`Promoter::admit`].
//! - A rejected candidate is remembered for a while, so a later seed in it is evaluated again
//!   from the reservoir without a new index read or scan.
//! - An admitted candidate loads in its own task, so the promoter keeps reading samples during
//!   the load. Until the load ends, the promoter evaluates no candidate.
//!
//! The cache side (in the parent module) keeps the heat of cached ranges, evicts automatic ranges
//! by read density, removes an automatic range whose reads do not pay for a rebuild, and
//! remembers removed ranges (ghosts) and failed promotions (backoff).

use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use log::debug;
use rand::RngExt;
use slatedb_common::DbRand;
use tokio::task::JoinHandle;

use super::stats::{CandidateSource, RejectReason};
use super::{Origin, RangeCache, RangeCacheTuning};
use crate::bytes_range::BytesRange;
use crate::config::ScanOptions;
use crate::db::DbInner;
use crate::dispatcher::{MessageHandler, MessageTickerDef};
use crate::error::SlateDBError;
use crate::iter::IterationOrder;
use crate::read_trace::ReadTrace;
use crate::reader::ScanContext;
use crate::types::ValueDeletable;
use crate::utils::spawn_bg_task;

/// The name of the promoter's task on the database's task executor.
pub(crate) const RANGE_CACHE_PROMOTER_TASK_NAME: &str = "range_cache_promoter";
/// The name of the task that loads an admitted candidate.
const RANGE_CACHE_PROMOTION_TASK_NAME: &str = "range_cache_promotion";

/// Options for automatic promotion. See [`super::RangeCacheOptions::auto_promote`].
///
/// Create it with [`AutoPromoteOptions::default`] and set the fields to change.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AutoPromoteOptions {
    /// The approximate size of an automatically cached range, in bytes of SST data.
    pub target_range_bytes: u64,
    /// Sample one normal-path read in this many among the reads that the cache could have
    /// served. 1 samples every read.
    pub sample_one_in: u32,
    /// The half-life of the read rates of cached ranges and candidates. Longer half-lives make
    /// promotion and eviction react more slowly to changes in the read pattern.
    pub heat_half_life: Duration,
}

impl Default for AutoPromoteOptions {
    fn default() -> Self {
        Self {
            target_range_bytes: 64 * 1024 * 1024,
            sample_one_in: 64,
            heat_half_life: Duration::from_secs(10),
        }
    }
}

/// The settings that automatic promotion runs with: [`AutoPromoteOptions`] and the promotion
/// fields of [`super::RangeCacheTuning`], which document them.
#[derive(Clone, Debug)]
pub(crate) struct AutoSettings {
    pub(crate) sample_one_in: u32,
    pub(crate) target_range_bytes: u64,
    pub(crate) evaluate_interval: Duration,
    pub(crate) max_candidates_per_evaluation: usize,
    pub(crate) min_evidence: u32,
    pub(crate) full_evidence: u32,
    pub(crate) min_reuse_span: Duration,
    pub(crate) heat_half_life: Duration,
    pub(crate) admission_margin: f64,
    pub(crate) ghost_ttl: Duration,
    pub(crate) candidate_ttl: Duration,
    pub(crate) candidate_recheck: Duration,
    pub(crate) backoff_base: Duration,
    pub(crate) reservoir_capacity: usize,
    pub(crate) queue_capacity: usize,
    pub(crate) min_read_cost: Duration,
    pub(crate) rebuild_cost_ns_per_byte: f64,
}

impl AutoSettings {
    pub(crate) fn new(options: &AutoPromoteOptions, tuning: &RangeCacheTuning) -> Self {
        Self {
            sample_one_in: options.sample_one_in,
            target_range_bytes: options.target_range_bytes,
            evaluate_interval: tuning.evaluate_interval,
            max_candidates_per_evaluation: tuning.max_candidates_per_evaluation,
            min_evidence: tuning.min_evidence,
            full_evidence: tuning.full_evidence,
            min_reuse_span: tuning.min_reuse_span,
            heat_half_life: options.heat_half_life,
            admission_margin: tuning.admission_margin,
            ghost_ttl: tuning.ghost_ttl,
            candidate_ttl: tuning.candidate_ttl,
            candidate_recheck: tuning.candidate_recheck,
            backoff_base: tuning.backoff_base,
            reservoir_capacity: tuning.reservoir_capacity,
            queue_capacity: tuning.queue_capacity,
            min_read_cost: tuning.min_read_cost,
            rebuild_cost_ns_per_byte: tuning.rebuild_cost_ns_per_byte,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), SlateDBError> {
        let invalid = |msg: &str| {
            Err(SlateDBError::InvalidConfiguration(format!(
                "range cache auto_promote: {msg}"
            )))
        };
        if self.target_range_bytes == 0 {
            return invalid("target_range_bytes must be positive");
        }
        if self.evaluate_interval.is_zero() || self.heat_half_life.is_zero() {
            return invalid("evaluate_interval and heat_half_life must be positive");
        }
        if self.admission_margin < 1.0 {
            return invalid("admission_margin must be at least 1");
        }
        if self.full_evidence < self.min_evidence {
            return invalid("full_evidence must be at least min_evidence");
        }
        Ok(())
    }
}

/// A sampled normal-path read.
#[derive(Debug)]
pub(crate) struct Sample {
    pub(crate) key: Bytes,
    /// The time that the read took, at least `min_read_cost`.
    pub(crate) cost_ns: u64,
    pub(crate) at_ms: i64,
}

#[derive(Debug)]
pub(crate) enum PromoterMessage {
    Sample(Sample),
    Evaluate,
}

/// The number of slots of the reservoir window.
const SLOTS: i64 = 32;
/// Seeds come from this many of the newest slots.
const SEED_SLOTS: usize = 8;
/// Bytes that a range cache file spends on a row besides its key and value, for size estimates
/// from scans.
const ROW_OVERHEAD: u64 = 32;

struct Slot {
    start_ms: i64,
    /// The samples that arrived in the slot.
    arrivals: u64,
    /// A uniform random subset of them.
    kept: Vec<Sample>,
}

impl Slot {
    /// How many arrivals each kept sample stands for.
    fn scale(&self) -> f64 {
        self.arrivals as f64 / self.kept.len().max(1) as f64
    }
}

/// Recent samples over a window of four heat half-lives. Each of its 32 slots keeps at most
/// `per_slot` samples, chosen by reservoir sampling, so the reservoir spans the whole window at
/// any read rate.
struct Reservoir {
    slots: VecDeque<Slot>,
    slot_ms: i64,
    per_slot: usize,
    /// Rates never count time before the promoter started.
    created_ms: i64,
}

/// What the reservoir knows about the reads of a key range.
#[derive(Debug)]
struct Evidence {
    /// The estimated sampled reads in the range.
    samples: f64,
    /// The kept samples in the range.
    kept: usize,
    first_ms: i64,
    last_ms: i64,
    /// Estimated normal-path reads per second, decayed.
    read_rate: f64,
    /// Estimated normal-path nanoseconds per second, decayed.
    cost_rate: f64,
}

impl Reservoir {
    fn new(options: &AutoSettings, now_ms: i64) -> Self {
        let window_ms = 4 * options.heat_half_life.as_millis() as i64;
        Self {
            slots: VecDeque::new(),
            slot_ms: (window_ms / SLOTS).max(1),
            per_slot: (options.reservoir_capacity / SLOTS as usize).max(8),
            created_ms: now_ms,
        }
    }

    fn window_ms(&self) -> i64 {
        self.slot_ms * SLOTS
    }

    fn add(&mut self, sample: Sample, rand: &DbRand) {
        let start_ms = sample.at_ms - sample.at_ms.rem_euclid(self.slot_ms);
        if self
            .slots
            .back()
            .is_none_or(|last| last.start_ms < start_ms)
        {
            self.slots.push_back(Slot {
                start_ms,
                arrivals: 0,
                kept: Vec::new(),
            });
            let oldest = start_ms - (SLOTS - 1) * self.slot_ms;
            while self.slots.front().is_some_and(|s| s.start_ms < oldest) {
                self.slots.pop_front();
            }
        }
        // A sample that arrives late goes to its own slot, if the window still holds it.
        let Some(slot) = self.slots.iter_mut().rev().find(|s| s.start_ms == start_ms) else {
            return;
        };
        slot.arrivals += 1;
        if slot.kept.len() < self.per_slot {
            slot.kept.push(sample);
        } else {
            let pick = rand.rng().random_range(0..slot.arrivals);
            if let Some(kept) = slot.kept.get_mut(pick as usize) {
                *kept = sample;
            }
        }
    }

    fn expire(&mut self, now_ms: i64) {
        let oldest = now_ms - self.window_ms();
        while self
            .slots
            .front()
            .is_some_and(|s| s.start_ms + self.slot_ms <= oldest)
        {
            self.slots.pop_front();
        }
    }

    /// Returns the evidence for `[start, end)` from samples at or after `since_ms`.
    fn evidence(
        &self,
        start: &Bytes,
        end: &Option<Bytes>,
        since_ms: Option<i64>,
        now_ms: i64,
        half_life_ms: f64,
        one_in: f64,
    ) -> Evidence {
        let tau_ms = half_life_ms.max(1.0) / std::f64::consts::LN_2;
        let window_start = (now_ms - self.window_ms())
            .max(self.created_ms)
            .max(since_ms.unwrap_or(i64::MIN));
        let mut evidence = Evidence {
            samples: 0.0,
            kept: 0,
            first_ms: i64::MAX,
            last_ms: i64::MIN,
            read_rate: 0.0,
            cost_rate: 0.0,
        };
        let (mut reads, mut cost) = (0.0, 0.0);
        for slot in &self.slots {
            let scale = slot.scale();
            for sample in &slot.kept {
                if sample.at_ms < window_start
                    || sample.key < *start
                    || end.as_ref().is_some_and(|end| sample.key >= *end)
                {
                    continue;
                }
                let weight = (-((now_ms - sample.at_ms).max(0) as f64) / tau_ms).exp() * scale;
                evidence.samples += scale;
                evidence.kept += 1;
                evidence.first_ms = evidence.first_ms.min(sample.at_ms);
                evidence.last_ms = evidence.last_ms.max(sample.at_ms);
                reads += weight;
                cost += weight * sample.cost_ns as f64;
            }
        }
        // The decayed length of the window: the integral of the weight over it, in seconds.
        let span_ms = ((now_ms - window_start).max(self.slot_ms)) as f64;
        let weight_secs = tau_ms * (1.0 - (-span_ms / tau_ms).exp()) / 1000.0;
        evidence.read_rate = reads * one_in / weight_secs;
        evidence.cost_rate = cost * one_in / weight_secs;
        evidence
    }

    /// Draws a key from the newest samples, weighted by their cost.
    fn pick(&self, rand: &DbRand) -> Option<Bytes> {
        let recent = || self.slots.iter().rev().take(SEED_SLOTS);
        let weight = |slot: &Slot, sample: &Sample| slot.scale() * sample.cost_ns as f64;
        let total: f64 = recent()
            .flat_map(|slot| slot.kept.iter().map(move |s| weight(slot, s)))
            .sum();
        if total <= 0.0 {
            return None;
        }
        let mut target = rand.rng().random_range(0.0..total);
        let mut last = None;
        for slot in recent() {
            for sample in &slot.kept {
                let w = weight(slot, sample);
                if target < w {
                    return Some(sample.key.clone());
                }
                target -= w;
                last = Some(&sample.key);
            }
        }
        last.cloned()
    }
}

/// A key range that the promoter considers caching.
#[derive(Clone, Debug)]
struct Candidate {
    start: Bytes,
    end: Option<Bytes>,
    /// The estimated bytes of its data.
    bytes: u64,
}

/// A candidate that admission rejected.
struct Rejected {
    end: Option<Bytes>,
    bytes: u64,
    found_ms: i64,
    checked_ms: i64,
}

enum Remembered {
    /// No remembered candidate holds the key.
    No,
    /// One does, but it was evaluated too recently.
    TooSoon,
    Yes(Candidate),
}

/// What admission decided.
enum Verdict {
    Admit { evicting: bool },
    Reject(RejectReason),
}

/// The background task of automatic promotion. See the module documentation.
pub(crate) struct Promoter {
    db: Arc<DbInner>,
    cache: Arc<RangeCache>,
    options: AutoSettings,
    reservoir: Reservoir,
    /// Rejected candidates by start key.
    rejected: BTreeMap<Bytes, Rejected>,
    last_evaluation_ms: Option<i64>,
    avg_read_cost_ns: f64,
    /// The load of the last admitted candidate. It runs as its own task, so the promoter keeps
    /// taking samples from the queue while it runs. Only one load runs at a time: admission
    /// counts only the files of loaded ranges against the disk budget.
    load: Option<JoinHandle<Result<(), SlateDBError>>>,
}

impl Promoter {
    pub(crate) fn new(db: Arc<DbInner>, cache: Arc<RangeCache>) -> Self {
        let options = cache
            .auto
            .as_ref()
            .expect("the promoter runs only with automatic promotion")
            .options
            .clone();
        let now = cache.now_ms();
        Self {
            reservoir: Reservoir::new(&options, now),
            avg_read_cost_ns: options.min_read_cost.as_nanos() as f64,
            db,
            cache,
            options,
            rejected: BTreeMap::new(),
            last_evaluation_ms: None,
            load: None,
        }
    }

    fn stats(&self) -> &super::stats::AutoStats {
        &self
            .cache
            .auto
            .as_ref()
            .expect("the promoter runs only with automatic promotion")
            .stats
    }

    async fn evaluate(&mut self, now: i64) {
        if self.db.check_closed().is_err() {
            return;
        }
        self.reservoir.expire(now);
        let ttl_ms = self.options.candidate_ttl.as_millis() as i64;
        self.rejected.retain(|_, r| now - r.found_ms < ttl_ms);
        if let Some(auto) = &self.cache.auto {
            auto.avg_read_cost_ns.store(
                self.avg_read_cost_ns as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.cache.update_gauges();
        if self.load.as_ref().is_some_and(|load| !load.is_finished()) {
            return;
        }

        let max = self.options.max_candidates_per_evaluation;
        let mut evaluated = 0;
        let mut scanned = false;
        for _ in 0..max * 4 {
            if evaluated >= max {
                break;
            }
            let Some(seed) = self.reservoir.pick(&self.db.rand) else {
                break;
            };
            if self.cache.covers(&seed) {
                continue;
            }
            if self.cache.backed_off(&seed) {
                self.stats().rejected(RejectReason::Backoff);
                continue;
            }
            let candidate = match self.remembered(&seed, now) {
                Remembered::TooSoon => continue,
                Remembered::Yes(candidate) => {
                    self.stats().candidate(CandidateSource::Remembered);
                    candidate
                }
                Remembered::No => match self.form(&seed, !scanned).await {
                    Ok(Some((candidate, source))) => {
                        scanned |= source == CandidateSource::Scan;
                        self.stats().candidate(source);
                        candidate
                    }
                    Ok(None) => continue,
                    Err(e) => {
                        debug!("range cache candidate failed [error={:?}]", e);
                        continue;
                    }
                },
            };
            evaluated += 1;
            if self.decide(candidate, now) {
                break;
            }
        }
    }

    /// Admits `candidate` and starts its load, or remembers it as rejected. Returns true if it
    /// started a load.
    fn decide(&mut self, candidate: Candidate, now: i64) -> bool {
        let ghost = self.cache.ghost_since(&candidate.start, &candidate.end);
        let evidence = self.reservoir.evidence(
            &candidate.start,
            &candidate.end,
            ghost,
            now,
            self.options.heat_half_life.as_millis() as f64,
            self.options.sample_one_in.max(1) as f64,
        );
        match self.admit(&candidate, &evidence) {
            Verdict::Reject(reason) => {
                self.stats().rejected(reason);
                let found_ms = self
                    .rejected
                    .get(&candidate.start)
                    .map_or(now, |r| r.found_ms);
                self.rejected.insert(
                    candidate.start.clone(),
                    Rejected {
                        end: candidate.end.clone(),
                        bytes: candidate.bytes,
                        found_ms,
                        checked_ms: now,
                    },
                );
                false
            }
            Verdict::Admit { evicting } => {
                self.rejected.remove(&candidate.start);
                if evicting {
                    self.stats().admissions_evicting.increment(1);
                } else {
                    self.stats().admissions_free.increment(1);
                }
                if ghost.is_some() {
                    self.stats().repromotions.increment(1);
                }
                let read_cost_ns = if evidence.read_rate > 0.0 {
                    evidence.cost_rate / evidence.read_rate
                } else {
                    self.avg_read_cost_ns
                };
                let origin = Origin::Auto {
                    read_cost_ns: (read_cost_ns as u64).max(1),
                    read_rate: evidence.read_rate,
                };
                let end = candidate.end.clone().map_or(Unbounded, Excluded);
                let range = BytesRange::new(Included(candidate.start.clone()), end);
                let db = self.db.clone();
                let cache = self.cache.clone();
                self.load = Some(spawn_bg_task(
                    RANGE_CACHE_PROMOTION_TASK_NAME.to_string(),
                    &self.cache.handle,
                    |_| {},
                    async move {
                        match db.cache_range_with(range, origin).await {
                            Ok(()) => {
                                if let Some(auto) = &cache.auto {
                                    auto.stats.promotions.increment(1);
                                }
                            }
                            Err(e) => debug!("range cache promotion failed [error={:?}]", e),
                        }
                        Ok(())
                    },
                ));
                true
            }
        }
    }

    /// Admission of `candidate`:
    ///
    /// 1. It needs at least the evidence threshold of sampled reads (and two kept samples). The
    ///    threshold rises from `min_evidence` to `full_evidence` as the budget fills from half
    ///    to full. After a recent removal of an overlapping range, only later samples count.
    /// 2. Its samples must span at least `min_reuse_span`.
    /// 3. If it fits in the disk budget, it is admitted.
    /// 4. Otherwise its density (normal-path cost per second per byte) must be more than
    ///    `admission_margin` times the density of every automatic range that eviction would
    ///    remove, lowest density first, to make room. Explicit ranges are never displaced.
    fn admit(&self, candidate: &Candidate, evidence: &Evidence) -> Verdict {
        let budget = self.cache.options.max_disk_bytes;
        let attached = self.cache.attached_bytes();
        let fill = attached as f64 / budget.max(1) as f64;
        let rise = ((fill - 0.5) / 0.5).clamp(0.0, 1.0);
        let threshold = self.options.min_evidence as f64
            + (self.options.full_evidence - self.options.min_evidence) as f64 * rise;
        if evidence.kept < 2 || evidence.samples < threshold {
            return Verdict::Reject(RejectReason::Cold);
        }
        let span_ms = evidence.last_ms - evidence.first_ms;
        if span_ms < self.options.min_reuse_span.as_millis() as i64 {
            return Verdict::Reject(RejectReason::NotReused);
        }
        if candidate.bytes > budget {
            return Verdict::Reject(RejectReason::TooLarge);
        }
        if attached + candidate.bytes <= budget {
            return Verdict::Admit { evicting: false };
        }
        let need = attached + candidate.bytes - budget;
        let Some(victims) = self.cache.victim_density(need) else {
            return Verdict::Reject(RejectReason::NoRoom);
        };
        let density = evidence.cost_rate / candidate.bytes.max(1) as f64;
        if density > self.options.admission_margin * victims {
            Verdict::Admit { evicting: true }
        } else {
            Verdict::Reject(RejectReason::NotDenser)
        }
    }

    /// Finds a remembered candidate that holds `key`.
    fn remembered(&mut self, key: &Bytes, now: i64) -> Remembered {
        let recheck_ms = self.options.candidate_recheck.as_millis() as i64;
        let Some((start, rejected)) = self
            .rejected
            .range_mut::<Bytes, _>((Unbounded, Included(key)))
            .next_back()
        else {
            return Remembered::No;
        };
        if rejected.end.as_ref().is_some_and(|end| key >= end) {
            return Remembered::No;
        }
        if now - rejected.checked_ms < recheck_ms {
            return Remembered::TooSoon;
        }
        rejected.checked_ms = now;
        Remembered::Yes(Candidate {
            start: start.clone(),
            end: rejected.end.clone(),
            bytes: rejected.bytes,
        })
    }

    /// Forms a candidate around `key`, from an SST index, or else, if `may_scan`, from bounded
    /// scans.
    async fn form(
        &self,
        key: &Bytes,
        may_scan: bool,
    ) -> Result<Option<(Candidate, CandidateSource)>, SlateDBError> {
        let (below, above) = self.cache.neighbours(key);
        if let Some(candidate) = self.sst_index_candidate(key, &below, &above).await? {
            return Ok(Some((candidate, CandidateSource::SstIndex)));
        }
        if !may_scan {
            return Ok(None);
        }
        let candidate = self.scan_candidate(key, &below, &above).await?;
        Ok(Some((candidate, CandidateSource::Scan)))
    }

    /// About `target_range_bytes` of blocks around `key` in the SST of the largest sorted run
    /// that holds it, then whole neighbouring SSTs if that SST is too small. The bounds are
    /// block boundaries, so no data is read except the SST index. It never crosses `below` or
    /// `above`.
    async fn sst_index_candidate(
        &self,
        key: &Bytes,
        below: &Bytes,
        above: &Option<Bytes>,
    ) -> Result<Option<Candidate>, SlateDBError> {
        let run = {
            let view = self.db.state.read().view();
            let core = view.state.core();
            // Segmented databases have one tree per segment. They use bounded scans.
            if core.segment_extractor_name.is_some() {
                return Ok(None);
            }
            match core
                .tree
                .compacted
                .iter()
                .max_by_key(|sr| sr.estimate_size())
            {
                Some(run) => run.clone(),
                None => return Ok(None),
            }
        };
        let views = run.sst_views();
        let Some(sst) = run.find_last_sst_with_range_covering_key(key) else {
            return Ok(None);
        };
        let view = &views[sst];
        let index = self
            .db
            .table_store
            .read_index(
                &view.sst,
                true,
                Some(Bytes::new()),
                &ReadTrace::none(),
                None,
            )
            .await?;
        let metas = index.borrow().block_meta();
        let n = metas.len();
        if n == 0 {
            return Ok(None);
        }
        let first_key = |i: usize| metas.get(i).first_key().bytes();
        let avg = view.sst.estimate_size() / n as u64;
        let size = |i: usize| {
            if i + 1 < n {
                metas
                    .get(i + 1)
                    .offset()
                    .saturating_sub(metas.get(i).offset())
            } else {
                avg
            }
        };
        // A block can join if it is not wholly below `below` or at or above `above`.
        let above_key = |k: &[u8]| above.as_ref().is_none_or(|a| k < a.as_ref());
        let block_ok_low = |i: usize| i + 1 >= n || first_key(i + 1) > below.as_ref();
        let target = self.options.target_range_bytes;
        // The last block whose first key is at or below `key`, by binary search.
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if first_key(mid) <= key.as_ref() {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let hot = lo.saturating_sub(1);
        let (mut low, mut high, mut bytes) = (hot, hot + 1, size(hot));
        while bytes < target {
            let can_low = low > 0 && block_ok_low(low - 1);
            let can_high = high < n && above_key(first_key(high));
            if can_low && (!can_high || hot - low <= high - 1 - hot) {
                low -= 1;
                bytes += size(low);
            } else if can_high {
                bytes += size(high);
                high += 1;
            } else {
                break;
            }
        }
        let mut start = if low > 0 {
            Bytes::copy_from_slice(first_key(low))
        } else {
            view.compacted_effective_start_key().clone()
        };
        let mut end = if high < n {
            Some(Bytes::copy_from_slice(first_key(high)))
        } else {
            views
                .get(sst + 1)
                .map(|v| v.compacted_effective_start_key().clone())
        };
        // Whole neighbouring SSTs, while the SST that holds the key is used up on that side.
        let (mut left, mut right) = (sst, sst + 1);
        while bytes < target {
            let can_left = low == 0
                && left > 0
                && views[left].compacted_effective_start_key().as_ref() > below.as_ref();
            let can_right = high == n
                && right < views.len()
                && above_key(views[right].compacted_effective_start_key());
            if can_left && (!can_right || sst - left <= right - 1 - sst) {
                left -= 1;
                bytes += views[left].estimate_size();
                start = views[left].compacted_effective_start_key().clone();
            } else if can_right {
                bytes += views[right].estimate_size();
                right += 1;
                end = views
                    .get(right)
                    .map(|v| v.compacted_effective_start_key().clone());
            } else {
                break;
            }
        }
        // At the ends of the run, the candidate reaches the ends of the key space.
        if low == 0 && left == 0 {
            start = Bytes::new();
        }
        if high == n && right == views.len() {
            end = None;
        }
        if start < *below {
            start = below.clone();
        }
        if let Some(above) = above {
            if end.as_ref().is_none_or(|end| end > above) {
                end = Some(above.clone());
            }
        }
        let holds_key = start <= *key && end.as_ref().is_none_or(|end| key < end);
        Ok(holds_key.then_some(Candidate { start, end, bytes }))
    }

    /// About `target_range_bytes` of rows around `key`: half from a forward scan, half from a
    /// reverse scan, each stopping at the neighbouring cached range.
    async fn scan_candidate(
        &self,
        key: &Bytes,
        below: &Bytes,
        above: &Option<Bytes>,
    ) -> Result<Candidate, SlateDBError> {
        let half = (self.options.target_range_bytes / 2).max(1);
        let db_state = self.db.state.read().view();
        let options = self.cache.options.tuning.bulk_scan_options();
        let context = || ScanContext {
            db_state: &db_state,
            write_batch_iter: None,
            max_seq: None,
            prefix: None,
        };
        let row_bytes = |key: &Bytes, value: &ValueDeletable| {
            let value = match value {
                ValueDeletable::Value(v) | ValueDeletable::Merge(v) => v.len(),
                _ => 0,
            };
            (key.len() + value) as u64 + ROW_OVERHEAD
        };

        let forward = BytesRange::new(
            Included(key.clone()),
            above.clone().map_or(Unbounded, Excluded),
        );
        let mut iter = self
            .db
            .reader
            .scan_with_options(forward, &options, context())
            .await?;
        let mut end = above.clone();
        let mut bytes = 0;
        while let Some(entry) = iter.next_entry().await? {
            if bytes >= half {
                end = Some(Bytes::copy_from_slice(&entry.key));
                break;
            }
            bytes += row_bytes(&entry.key, &entry.value);
        }

        let mut start = below.clone();
        if key > below {
            let reverse = ScanOptions {
                order: IterationOrder::Descending,
                ..options.clone()
            };
            let range = BytesRange::new(Included(below.clone()), Excluded(key.clone()));
            let mut iter = self
                .db
                .reader
                .scan_with_options(range, &reverse, context())
                .await?;
            let mut reverse_bytes = 0;
            start = key.clone();
            let mut exhausted = true;
            while let Some(entry) = iter.next_entry().await? {
                if reverse_bytes >= half {
                    exhausted = false;
                    break;
                }
                reverse_bytes += row_bytes(&entry.key, &entry.value);
                start = Bytes::copy_from_slice(&entry.key);
            }
            if exhausted {
                start = below.clone();
            }
            bytes += reverse_bytes;
        }
        Ok(Candidate { start, end, bytes })
    }
}

#[async_trait]
impl MessageHandler<PromoterMessage> for Promoter {
    fn tickers(&mut self) -> Vec<MessageTickerDef<PromoterMessage>> {
        vec![MessageTickerDef::new(
            self.options.evaluate_interval,
            Box::new(|| PromoterMessage::Evaluate),
        )]
    }

    async fn handle(&mut self, message: PromoterMessage) -> Result<(), SlateDBError> {
        if let PromoterMessage::Sample(sample) = message {
            self.avg_read_cost_ns += (sample.cost_ns as f64 - self.avg_read_cost_ns) / 1024.0;
            self.reservoir.add(sample, &self.db.rand);
        }
        // Evaluate on samples too: under heavy load the dispatcher favours queued messages
        // over ticks.
        let now = self.cache.now_ms();
        let interval_ms = self.options.evaluate_interval.as_millis() as i64;
        if self
            .last_evaluation_ms
            .is_none_or(|last| now - last >= interval_ms)
        {
            self.last_evaluation_ms = Some(now);
            self.evaluate(now).await;
        }
        // Promotion failures are counted and logged. They never fail the database.
        Ok(())
    }

    async fn cleanup(
        &mut self,
        _messages: BoxStream<'async_trait, PromoterMessage>,
        _result: Result<(), SlateDBError>,
    ) -> Result<(), SlateDBError> {
        // The database is closed now, so a running load stops at its next check. Wait for it,
        // as the promoter waited for its loads before they ran as their own task.
        if let Some(load) = self.load.take() {
            if let Err(e) = load.await {
                debug!("range cache promotion task failed [error={:?}]", e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(key: &str, at_ms: i64) -> Sample {
        Sample {
            key: Bytes::copy_from_slice(key.as_bytes()),
            cost_ns: 1000,
            at_ms,
        }
    }

    fn options() -> AutoSettings {
        let options = AutoPromoteOptions {
            heat_half_life: Duration::from_secs(8),
            ..AutoPromoteOptions::default()
        };
        let tuning = RangeCacheTuning {
            reservoir_capacity: 32 * 8,
            ..RangeCacheTuning::default()
        };
        AutoSettings::new(&options, &tuning)
    }

    #[test]
    fn should_keep_a_bounded_sample_of_each_slot() {
        let rand = DbRand::new(7);
        let mut reservoir = Reservoir::new(&options(), 0);
        // One slot is a second long: 4 x 8 s over 32 slots.
        for i in 0..1000 {
            reservoir.add(sample(&format!("k{:03}", i % 100), 10), &rand);
        }
        assert_eq!(reservoir.slots.len(), 1);
        assert_eq!(reservoir.slots[0].kept.len(), 8);
        assert_eq!(reservoir.slots[0].arrivals, 1000);
        // The estimate scales the kept samples back to the arrivals.
        let all = reservoir.evidence(&Bytes::new(), &None, None, 1000, 8000.0, 1.0);
        assert_eq!(all.samples, 1000.0);
        assert_eq!(all.kept, 8);
    }

    #[test]
    fn should_forget_samples_outside_the_window() {
        let rand = DbRand::new(7);
        let mut reservoir = Reservoir::new(&options(), 0);
        reservoir.add(sample("a", 0), &rand);
        reservoir.add(sample("b", 31_000), &rand);
        assert_eq!(reservoir.slots.len(), 2);
        reservoir.add(sample("c", 32_000), &rand);
        assert_eq!(reservoir.slots.len(), 2, "the slot of `a` left the window");
        reservoir.expire(100_000);
        assert!(reservoir.slots.is_empty());
    }

    #[test]
    fn should_estimate_rates_and_spans_of_a_range() {
        let rand = DbRand::new(7);
        let mut reservoir = Reservoir::new(&options(), 0);
        // Ten reads a second of keys in [b, c) for 10 s, sampled one in one.
        for ms in (0..10_000).step_by(100) {
            reservoir.add(sample("b1", ms), &rand);
        }
        let b = Bytes::from_static(b"b");
        let c = Some(Bytes::from_static(b"c"));
        assert_eq!(
            reservoir
                .evidence(&c.clone().unwrap(), &None, None, 10_000, 8000.0, 1.0)
                .kept,
            0
        );
        let evidence = reservoir.evidence(&b, &c, None, 10_000, 8000.0, 1.0);
        assert_eq!(evidence.kept, 80, "8 per one-second slot");
        assert_eq!(evidence.samples, 100.0);
        assert!(evidence.last_ms - evidence.first_ms >= 9_000);
        assert!(
            (evidence.read_rate - 10.0).abs() < 1.5,
            "{}",
            evidence.read_rate
        );
        assert!((evidence.cost_rate - 10_000.0).abs() < 1_500.0);
        // Only samples after a removal count.
        let fresh = reservoir.evidence(&b, &c, Some(9_000), 10_000, 8000.0, 1.0);
        assert_eq!(fresh.samples, 10.0);
        assert!(fresh.first_ms >= 9_000);
    }
}
