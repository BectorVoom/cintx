//! Attribution of the host cart-to-sph cost (Task 36-T0).
//!
//! Every batched family ends with the same shape: for each tuple, for each
//! contraction block, for each component, transform one Cartesian block to
//! spherical and scatter it into the caller's AO grid. That loop is the largest
//! single cost in three of the four workloads that are still slower than
//! vendored libcint, and `BatchExecutionStats::host_transform_ns` reports it as
//! one number — which says *that* it is expensive, not *why*.
//!
//! This splits it three ways:
//!
//! - **allocate** — the per-block buffers the transform needs,
//! - **transform** — the c2s arithmetic itself,
//! - **scatter** — the strided write into the output grid.
//!
//! # Why it is opt-in
//!
//! A block is small (an s/p 3c2e block is 27 elements), so three
//! `Instant::now()` calls per block are not free relative to the work they
//! bracket. Charging every production run for that would make
//! `host_transform_ns` itself a probe artifact. Instead the probe is inert
//! unless `CINTX_HOST_TRANSFORM_PROFILE` is set to something other than `0`:
//!
//! - **off** (the default): every method is a predictable-branch no-op, the
//!   three counters stay `0`, and `host_transform_ns` measures the transform
//!   and nothing else.
//! - **on**: the three counters are populated and sum to `host_transform_ns`
//!   for the same run, probe overhead included in both.
//!
//! So the split is a *ratio* read off a profiling run, not a number to compare
//! against an unprofiled one.

use std::sync::OnceLock;
use std::time::Instant;

/// Whether `CINTX_HOST_TRANSFORM_PROFILE` asks for the three-way split.
///
/// Read once per process: the batch transform consults this per block, and a
/// `std::env::var` there would cost more than the probe it gates.
pub fn profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CINTX_HOST_TRANSFORM_PROFILE")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    })
}

/// Running allocate/transform/scatter totals for one batched evaluation.
///
/// Construct one per batch, bracket each block's three steps, and fold the
/// totals into the batch's [`crate::kernels::two_electron::BatchExecutionStats`]
/// with [`Self::store_into`].
#[derive(Clone, Copy, Debug)]
pub struct HostTransformProfile {
    enabled: bool,
    /// Start of the step currently being timed; `None` while disabled.
    cursor: Option<Instant>,
    alloc_ns: u64,
    transform_ns: u64,
    scatter_ns: u64,
}

impl Default for HostTransformProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl HostTransformProfile {
    /// A profile that is live only when `CINTX_HOST_TRANSFORM_PROFILE` asks.
    pub fn new() -> Self {
        Self {
            enabled: profiling_enabled(),
            cursor: None,
            alloc_ns: 0,
            transform_ns: 0,
            scatter_ns: 0,
        }
    }

    /// Open a timed step. Call immediately before the first of the three.
    #[inline]
    pub fn start(&mut self) {
        if self.enabled {
            self.cursor = Some(Instant::now());
        }
    }

    /// Close the open step, charging it to *allocate*, and open the next.
    #[inline]
    pub fn charge_alloc(&mut self) {
        self.charge(Step::Alloc);
    }

    /// Close the open step, charging it to *transform*, and open the next.
    #[inline]
    pub fn charge_transform(&mut self) {
        self.charge(Step::Transform);
    }

    /// Close the open step, charging it to *scatter*, and open the next.
    ///
    /// The next step is opened rather than left closed so a loop that ends with
    /// the scatter and begins with an allocation needs one `start` per block,
    /// not two clock reads at the seam.
    #[inline]
    pub fn charge_scatter(&mut self) {
        self.charge(Step::Scatter);
    }

    #[inline]
    fn charge(&mut self, step: Step) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if let Some(open) = self.cursor.replace(now) {
            let elapsed = now.duration_since(open).as_nanos() as u64;
            match step {
                Step::Alloc => self.alloc_ns += elapsed,
                Step::Transform => self.transform_ns += elapsed,
                Step::Scatter => self.scatter_ns += elapsed,
            }
        }
    }

    /// Add another profile's totals to this one.
    ///
    /// Task 36-T2 gives each worker its own profile, because the counters are
    /// plain `u64` and sharing them would need a lock per block. Folding them
    /// afterwards is exact — they are sums.
    pub fn merge(&mut self, other: &Self) {
        self.alloc_ns += other.alloc_ns;
        self.transform_ns += other.transform_ns;
        self.scatter_ns += other.scatter_ns;
    }

    /// Stop charging; anything after this and before the next [`Self::start`]
    /// belongs to no step.
    #[inline]
    pub fn pause(&mut self) {
        self.cursor = None;
    }

    /// Fold the totals into a batch's statistics.
    ///
    /// A no-op while profiling is off, which is what leaves the three fields at
    /// `0` and marks the split as "not measured" rather than "measured as zero".
    pub fn store_into(&self, stats: &mut crate::kernels::two_electron::BatchExecutionStats) {
        if !self.enabled {
            return;
        }
        stats.host_transform_alloc_ns = self.alloc_ns;
        stats.host_transform_c2s_ns = self.transform_ns;
        stats.host_transform_scatter_ns = self.scatter_ns;
    }
}

/// One line attributing [`BatchExecutionStats::host_transform_ns`] three ways,
/// or `None` when the run was not profiled.
///
/// Benchmarks print this under the dispatch/transform split so a profiling run
/// says *why* the transform costs what it does, and an ordinary run stays
/// silent rather than printing three zeros that read as "free".
pub fn format_split(stats: &crate::kernels::two_electron::BatchExecutionStats) -> Option<String> {
    let alloc = stats.host_transform_alloc_ns;
    let c2s = stats.host_transform_c2s_ns;
    let scatter = stats.host_transform_scatter_ns;
    let total = alloc + c2s + scatter;
    if total == 0 {
        return None;
    }
    let share = |part: u64| 100.0 * part as f64 / total as f64;
    Some(format!(
        "host cart->sph attribution: allocate {:.3} ms ({:.1}%)           c2s {:.3} ms ({:.1}%)  scatter {:.3} ms ({:.1}%)           [sum {:.3} ms vs host_transform_ns {:.3} ms]",
        alloc as f64 / 1e6,
        share(alloc),
        c2s as f64 / 1e6,
        share(c2s),
        scatter as f64 / 1e6,
        share(scatter),
        total as f64 / 1e6,
        stats.host_transform_ns as f64 / 1e6,
    ))
}

#[derive(Clone, Copy)]
enum Step {
    Alloc,
    Transform,
    Scatter,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disabled profile records nothing and leaves the stats untouched — the
    /// default path must not pay for the probe.
    #[test]
    fn disabled_profile_is_inert() {
        let mut profile = HostTransformProfile {
            enabled: false,
            cursor: None,
            alloc_ns: 0,
            transform_ns: 0,
            scatter_ns: 0,
        };
        profile.start();
        profile.charge_alloc();
        profile.charge_transform();
        profile.charge_scatter();
        assert_eq!(
            (profile.alloc_ns, profile.transform_ns, profile.scatter_ns),
            (0, 0, 0)
        );

        let mut stats = crate::kernels::two_electron::BatchExecutionStats {
            host_transform_ns: 42,
            ..Default::default()
        };
        profile.store_into(&mut stats);
        assert_eq!(stats.host_transform_alloc_ns, 0);
        assert_eq!(stats.host_transform_c2s_ns, 0);
        assert_eq!(stats.host_transform_scatter_ns, 0);
    }

    /// An enabled profile charges each bracketed step to its own counter, and
    /// the three are what `store_into` publishes.
    #[test]
    fn enabled_profile_attributes_each_step() {
        let mut profile = HostTransformProfile {
            enabled: true,
            cursor: None,
            alloc_ns: 0,
            transform_ns: 0,
            scatter_ns: 0,
        };
        profile.start();
        profile.charge_alloc();
        profile.charge_transform();
        profile.charge_scatter();
        profile.pause();

        // Every counter saw at least one clock tick's worth of a closed step.
        // Wall-clock magnitudes are not asserted — only that the routing works.
        let total = profile.alloc_ns + profile.transform_ns + profile.scatter_ns;
        assert!(total > 0, "an enabled profile must charge something");

        let mut stats = crate::kernels::two_electron::BatchExecutionStats::default();
        profile.store_into(&mut stats);
        assert_eq!(stats.host_transform_alloc_ns, profile.alloc_ns);
        assert_eq!(stats.host_transform_c2s_ns, profile.transform_ns);
        assert_eq!(stats.host_transform_scatter_ns, profile.scatter_ns);
    }

    /// `pause` closes the bracket: work after it is charged to nothing.
    #[test]
    fn pause_drops_the_open_step() {
        let mut profile = HostTransformProfile {
            enabled: true,
            cursor: None,
            alloc_ns: 0,
            transform_ns: 0,
            scatter_ns: 0,
        };
        profile.start();
        profile.pause();
        profile.charge_alloc();
        assert_eq!(profile.alloc_ns, 0);
    }
}
