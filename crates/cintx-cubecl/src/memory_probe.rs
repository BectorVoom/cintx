//! Device allocation accounting for batched runs (`def2_speed_memory_optimization_plan.md` M6).
//!
//! A batched evaluation's memory cost has two halves, and only one of them is
//! knowable from the host without asking the backend:
//!
//! - **Planned bytes** — the output buffer, the quartet/shape/factor tables and
//!   the G-tensor scratch slab. Their sizes are computed by the same
//!   expressions the dispatch uses (`out_len * 8`, `g_slab_stride(g_size) *
//!   slots * 8`), so they are exact and free to record.
//! - **Backend residency** — what the allocator is actually holding, including
//!   pages it retains for reuse and padding the plan does not model.
//!   `ComputeClient::memory_usage` answers that, but it is a *blocking* submit:
//!   it drains the stream to read the server's allocator state.
//!
//! Charging every production batch for that drain would be self-defeating for a
//! plan whose next workstream overlaps dispatch with readback (S4), so the
//! residency half is opt-in under `CINTX_BATCH_MEMORY_PROFILE`, exactly as the
//! host-transform split is opt-in under `CINTX_HOST_TRANSFORM_PROFILE`:
//!
//! - **off** (the default): the planned-bytes fields are populated — they cost
//!   nothing — and `device_bytes_in_use_peak` / `device_allocs_added` stay `0`.
//! - **on**: the two residency fields are populated from a `memory_usage` call
//!   before the run and after each dispatch, and the run is serialized at each
//!   of those points.
//!
//! So a residency number is read off a profiling run and never compared against
//! an unprofiled one; the planned bytes are comparable across both.

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use std::sync::OnceLock;

/// Whether `CINTX_BATCH_MEMORY_PROFILE` asks for backend residency sampling.
///
/// Read once per process: a batch consults this per dispatch, and the drain a
/// `memory_usage` call forces is precisely what must not happen by accident.
#[must_use]
pub fn residency_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CINTX_BATCH_MEMORY_PROFILE")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    })
}

/// Running device-allocation totals for one batched evaluation.
///
/// Construct one per batch, charge each dispatch's planned allocations, sample
/// residency where the profile is enabled, and fold the totals into the batch's
/// [`crate::kernels::two_electron::BatchExecutionStats`] with
/// [`Self::store_into`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceMemoryProbe {
    /// Largest single Cartesian output buffer any one dispatch allocated.
    ///
    /// The *peak* rather than the sum, because a dispatch's output is read back
    /// and dropped before the next one allocates — under the current serial
    /// pipeline. S4's overlap changes that, which is why this is recorded now:
    /// the number that moves is the evidence the overlap cost memory.
    out_bytes_peak: usize,
    /// G-tensor scratch summed over every dispatch of this run.
    ///
    /// The sum, not the peak, because today each dispatch allocates its own
    /// slab inside `launch()` and frees it after. M4 makes this one allocation
    /// for the whole batch, and the difference between this and
    /// [`Self::g_slab_bytes_peak`] is exactly what M4 removes.
    g_slab_bytes_total: usize,
    /// Largest single G-tensor slab allocation.
    g_slab_bytes_peak: usize,
    /// Bytes of quartet/shape/factor/constant tables uploaded per dispatch.
    table_bytes_total: usize,
    /// Distinct device allocations this run requested, planned ones only.
    planned_allocs: u64,
    /// Peak `bytes_in_use` the backend reported, `0` unless profiling.
    bytes_in_use_peak: u64,
    /// Active allocations the backend gained over the run, `0` unless profiling.
    allocs_added: u64,
    /// `number_allocs` at the first sample, for the [`Self::allocs_added`] delta.
    allocs_baseline: u64,
    /// Has a baseline sample been taken?
    baselined: bool,
}

impl DeviceMemoryProbe {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Charge one dispatch's Cartesian output buffer.
    pub fn charge_output(&mut self, bytes: usize) {
        self.out_bytes_peak = self.out_bytes_peak.max(bytes);
        self.planned_allocs += 1;
    }

    /// Charge one dispatch's G-tensor scratch slab.
    pub fn charge_g_slab(&mut self, bytes: usize) {
        self.g_slab_bytes_total += bytes;
        self.g_slab_bytes_peak = self.g_slab_bytes_peak.max(bytes);
        self.planned_allocs += 1;
    }

    /// Charge `count` table uploads totalling `bytes`.
    pub fn charge_tables(&mut self, bytes: usize, count: u64) {
        self.table_bytes_total += bytes;
        self.planned_allocs += count;
    }

    /// Sample the backend's allocator, if the profile is enabled.
    ///
    /// The first call establishes the baseline the allocation delta is measured
    /// from; every call updates the residency peak. A backend that cannot
    /// report is skipped rather than failing the batch — a probe must not be a
    /// new way for an evaluation to error.
    pub fn sample<R: Runtime>(&mut self, client: &ComputeClient<R>) {
        if !residency_profiling_enabled() {
            return;
        }
        let Ok(usage) = client.memory_usage() else {
            return;
        };
        if !self.baselined {
            self.allocs_baseline = usage.number_allocs;
            self.baselined = true;
        }
        self.bytes_in_use_peak = self.bytes_in_use_peak.max(usage.bytes_in_use);
        self.allocs_added = self
            .allocs_added
            .max(usage.number_allocs.saturating_sub(self.allocs_baseline));
    }

    /// Fold another probe's totals into this one.
    ///
    /// A batch is evaluated chunk by chunk (M1) and each chunk carries its own
    /// ledger, so the run's numbers are these combined: peaks take the maximum
    /// because chunks do not coexist, totals add because every chunk's
    /// allocation really happened.
    pub fn merge(&mut self, other: &Self) {
        self.out_bytes_peak = self.out_bytes_peak.max(other.out_bytes_peak);
        self.g_slab_bytes_total += other.g_slab_bytes_total;
        self.g_slab_bytes_peak = self.g_slab_bytes_peak.max(other.g_slab_bytes_peak);
        self.table_bytes_total += other.table_bytes_total;
        self.planned_allocs += other.planned_allocs;
        self.bytes_in_use_peak = self.bytes_in_use_peak.max(other.bytes_in_use_peak);
        self.allocs_added = self.allocs_added.max(other.allocs_added);
    }

    /// Fold these totals into a batch's statistics.
    pub fn store_into(&self, stats: &mut crate::kernels::two_electron::BatchExecutionStats) {
        stats.device_out_bytes_peak = self.out_bytes_peak;
        stats.device_g_slab_bytes_total = self.g_slab_bytes_total;
        stats.device_g_slab_bytes_peak = self.g_slab_bytes_peak;
        stats.device_table_bytes_total = self.table_bytes_total;
        stats.device_planned_allocs = self.planned_allocs;
        stats.device_bytes_in_use_peak = self.bytes_in_use_peak;
        stats.device_allocs_added = self.allocs_added;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_accumulate_as_peak_and_total() {
        let mut probe = DeviceMemoryProbe::new();
        probe.charge_output(100);
        probe.charge_output(40);
        probe.charge_g_slab(1_000);
        probe.charge_g_slab(3_000);
        probe.charge_tables(64, 3);

        let mut stats = crate::kernels::two_electron::BatchExecutionStats::default();
        probe.store_into(&mut stats);

        // Output is a peak: the two dispatches never hold their buffers at once.
        assert_eq!(stats.device_out_bytes_peak, 100);
        // Scratch is both, because today it is allocated per dispatch (M4).
        assert_eq!(stats.device_g_slab_bytes_total, 4_000);
        assert_eq!(stats.device_g_slab_bytes_peak, 3_000);
        assert_eq!(stats.device_table_bytes_total, 64);
        assert_eq!(stats.device_planned_allocs, 2 + 2 + 3);
    }

    /// Residency stays zero without the env var, whatever else was charged —
    /// the property that keeps an unprofiled run free of `memory_usage` drains.
    #[test]
    fn residency_is_inert_unless_profiling() {
        let mut probe = DeviceMemoryProbe::new();
        probe.charge_output(8);
        let mut stats = crate::kernels::two_electron::BatchExecutionStats::default();
        probe.store_into(&mut stats);
        if !residency_profiling_enabled() {
            assert_eq!(stats.device_bytes_in_use_peak, 0);
            assert_eq!(stats.device_allocs_added, 0);
        }
    }
}
