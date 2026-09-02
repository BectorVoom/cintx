//! Autotuned launch geometry, cached across process runs.
//!
//! This is the `tuning.rs` of `docs/design/cubecl_speed_optimization_plan.md`
//! Phase 6, and it implements the two techniques the CubeCL manual's
//! *Autotune Optimization* chapter describes:
//!
//! 1. **Persistent caching.** CubeCL's [`TuneCache`] writes the winning
//!    candidate per key to disk and reloads it at the first tuned dispatch, so
//!    the benchmarking pass is a one-off cost rather than a per-process one.
//!    [`install_runtime_config`] is where that cache location and the tuning
//!    level are pinned.
//! 2. **Early-stop criteria via intra-group priorities.** Every candidate cube
//!    width is registered in one [`TuneGroup`](cubecl::tune::TuneGroup) with a
//!    priority closure that returns `-1` for any width this device or this
//!    workload cannot honour — past `max_units_per_cube`, past the scratch
//!    budget, past the parallel width the work can actually fill, or not a
//!    whole number of planes. A pruned candidate is dropped from the tuning
//!    plan before it is ever compiled or launched, which is what keeps the cold
//!    pass bounded on a wide candidate list.
//!
//! # What is tuned, and what is not
//!
//! Only **launch geometry** — how many units a cube has. Every batched kernel
//! in this crate covers its whole index space through a grid-stride or
//! lane-strided loop, so the cube width changes how the work is divided and
//! never which values are produced. That property is what makes autotuning
//! admissible here at all: the design's exit gate is that *tuning never changes
//! results*, and it holds by construction rather than by measurement.
//!
//! The angular-momentum shape scalars, the Rys order, and the HRR branch stay
//! comptime and are part of the key, not of the search space.
//!
//! # Bounds
//!
//! Cold-start work and cache growth are both bounded on purpose:
//!
//! - [`MIN_TUNE_ITEMS`] — dispatches below this are launched on the heuristic
//!   geometry without tuning, because the benchmark would cost more than the
//!   dispatch it is trying to improve.
//! - [`tune_sample_items`] — the benchmark runs on a bounded prefix of the work
//!   list. The winning candidate is then executed on the *full* list, so a
//!   truncated benchmark costs accuracy in the ranking, never in the result.
//! - [`MAX_TUNED_KEYS`] — a hard ceiling on distinct keys tuned per process.
//!   Past it, new keys fall back to the heuristic rather than growing the cache
//!   without limit.
//!
//! # The default follows the decomposition, and why
//!
//! Tuning is on for the **cooperative** (planed-backend) decomposition and off
//! for the **per-unit** (host-runtime) one. That is not a compromise: the two
//! were measured with different instruments and gave opposite answers.
//!
//! On the ROCm gfx1151 device, where CubeCL ranks candidates by *device
//! timestamp*, `balanced` beat the heuristic on every def2 work list —
//! H2O/def2-SVP 33.3 -> 32.4 ms (1.03x), SO2/def2-SVP 369 -> 256 ms (1.44x),
//! H2O/def2-TZVP 194 -> 151 ms (1.28x) — with bit-identical values
//! (`def2_rocm_extended_and_tuning`). On the 16-core CPU-runtime dev host,
//! where the ranking is host wall clock, it lost. The measurement below is that
//! CPU result, and it is why the per-unit default stayed where it was.
//!
//! `CINTX_AUTOTUNE=off|balanced|extensive` and [`set_policy`] override both
//! defaults.
//!
//! On the 16-core CPU-runtime dev host, a 4096-quartet
//! `s`/`p`/`d` batch (8 launch classes) measured, in release, as the median of
//! 25 repeats:
//!
//! | policy | first call | median batch |
//! |---|---|---|
//! | `off` | 13.0 s, 15.2 s | 75.7 ms, 81.5 ms |
//! | `balanced`, cache warm | 12.5 s, 18.2 s | 88.8 ms, 273 ms |
//!
//! The tuner's own profiled samples ranked most of its picks *ahead* of the
//! heuristic — and end-to-end they were not faster. Two things explain the gap
//! and neither is fixable from here: the CubeCL CPU runtime's profiled timings
//! carry a spread of 2x between a candidate's min and its median, which is
//! wider than the differences being ranked; and the wall clock on this host
//! moves by more than that between identical runs. A cold pass also costs a
//! separate JIT compilation per surviving candidate — `CubeDim` is part of a
//! kernel's compiled identity — which is where the ~65 s cold first call in an
//! earlier run went.
//!
//! So the per-unit default stays on the geometry that is already measured. The
//! prediction that closed that paragraph — "turn it on where the ranking is a
//! device-timestamp profile rather than host wall clock, and where the
//! cooperative arm has a real plane-width search to do" — is the one the ROCm
//! rows above went and checked, which is why the cooperative default is now
//! `balanced` rather than an invitation.
//!
//! # Environment
//!
//! - `CINTX_AUTOTUNE=off|balanced|extensive` — tuning policy, or
//!   [`set_policy`] to choose one programmatically. Either overrides the
//!   per-decomposition default; `off` is the pure heuristic path. `balanced` and `extensive` map onto CubeCL's
//!   [`AutotuneLevel`], which controls how coarsely
//!   [`anchor`](cubecl::tune::anchor) buckets the workload fields of the key,
//!   and therefore how many distinct keys exist.
//! - `CINTX_AUTOTUNE_CACHE=<dir>` — where the persistent cache lives. Unset,
//!   CubeCL's own default applies: the `target` directory of the nearest cargo
//!   project *above the current working directory*. That default is therefore
//!   cwd-dependent — an application launched from elsewhere retunes — so any
//!   deployment that wants a stable cache should set this.
//!
//! A `cubecl.toml` found from the current directory upwards, and any `CUBECL_*`
//! variable, both take precedence: this module only fills in fields the caller
//! has not already spoken for, and never overwrites a configuration that
//! another crate installed first.
//!
//! # Known deviation from the Phase 6 exit gate
//!
//! The design plan asks that *the selected variant beats or matches the safe
//! default beyond the configured noise threshold*. CubeCL 0.10 exposes no such
//! threshold: its tuner ranks candidates by `0.8 * min + 0.2 * median`, inflated
//! by the coefficient of variation, and keeps the best score outright. Two
//! candidates that are indistinguishable within run-to-run noise can therefore
//! resolve either way.
//!
//! The exposure is bounded rather than eliminated. The heuristic geometry is
//! itself one of the measured candidates — usually equal to one of the
//! [`CANDIDATE_CUBE_WIDTHS`] — so a noisy tie is a tie between two geometries
//! that were both measured on the real kernel, not a jump into a different
//! regime; the pruning rules keep the plainly-wrong widths out of the ranking
//! entirely. `CINTX_AUTOTUNE=off` is the escape hatch if a caller needs the
//! heuristic geometry to be the one that runs, guaranteed.

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use cubecl::client::ComputeClient;
use cubecl::tune::{AutotuneKey, anchor};
use serde::{Deserialize, Serialize};

use crate::plane::LaunchHardware;

/// Version of the *meaning* of a [`LaunchGeometryKey`] and of the candidate
/// list it is searched over.
///
/// It is part of the key, so bumping it retires every persisted entry: do that
/// whenever a field changes meaning, a candidate is added or removed, or a
/// kernel's decomposition changes what a cube width buys. CubeCL's own
/// checksum only covers the candidate *names*, which is not enough — a kernel
/// rewrite can invalidate a measurement without renaming anything.
pub const TUNING_SCHEMA_VERSION: u32 = 1;

/// Hard ceiling on distinct keys tuned in one process.
///
/// The workload fields of the key are anchored to powers of a base, so a real
/// molecule produces tens of keys, not thousands. This ceiling is the backstop
/// for a caller that sweeps shapes: past it, dispatches keep running on the
/// heuristic geometry instead of paying an unbounded cold-start cost and
/// writing an unbounded cache.
pub const MAX_TUNED_KEYS: usize = 128;

/// Work below which a dispatch is never tuned.
///
/// The benchmark is 3 warmup runs plus 10 samples per surviving candidate. For
/// a handful of shell tuples that is strictly more work than the dispatch
/// itself, and the geometry barely matters at that size anyway.
pub const MIN_TUNE_ITEMS: usize = 64;

/// Shortest work-list prefix a benchmark pass may run on.
pub const TUNE_SAMPLE_MIN_ITEMS: usize = 64;

/// Longest work-list prefix a benchmark pass may run on.
pub const TUNE_SAMPLE_MAX_ITEMS: usize = 1024;

/// Per-pass work budget, in whatever unit the caller's `per_item_work` proxy
/// counts — G-tensor elements, for every family tuned so far.
///
/// This is what makes the prefix work-aware instead of a flat item count, and
/// the reason is the one bias a truncated benchmark really has: a candidate
/// that spends more on dispatch (more units to wake, a wider grid) is
/// under-rewarded when the prefix is too short to amortize that cost, so a
/// short prefix systematically favours narrow cubes. Sizing the prefix so every
/// pass does about the same *work* removes that bias where it bites — cheap
/// classes, where one item is nowhere near enough to pay for a thread — and
/// costs nothing where it does not: for an expensive class a single item
/// already amortizes the dispatch, so a short prefix ranks it correctly.
pub const TUNE_SAMPLE_WORK_BUDGET: usize = 1 << 18;

/// How long a prefix of the work list to benchmark on.
///
/// The kernels read their item count from a scalar and walk exactly that many
/// rows, so a truncated run is the same kernel on the same shapes with a
/// shorter list — same specialization, same extents, same decomposition.
///
/// `per_item_work` is a coarse per-item cost proxy (the G-tensor size, for the
/// integral families); it only has to be right to an order of magnitude.
#[must_use]
pub fn tune_sample_items(items: usize, per_item_work: usize) -> usize {
    let by_work = (TUNE_SAMPLE_WORK_BUDGET / per_item_work.max(1))
        .clamp(TUNE_SAMPLE_MIN_ITEMS, TUNE_SAMPLE_MAX_ITEMS);
    items.min(by_work)
}

/// The cube widths a launch-geometry search may choose between.
///
/// The list is deliberately wider than any single backend can use: the
/// [priority closure](cube_width_priority) prunes it down to the widths that
/// are viable for the key at hand, and a pruned width is never compiled.
/// `1` matters on CPU backends (one unit is an OS thread); the plane multiples
/// matter on GPU ones.
pub const CANDIDATE_CUBE_WIDTHS: [u32; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// How much autotuning the caller is willing to pay for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutotunePolicy {
    /// No tuning: every dispatch uses its heuristic geometry. The default —
    /// see the module docs for the measurement that decided that.
    Off,
    /// Tune, with the key buckets CubeCL calls `balanced`.
    Balanced,
    /// Tune, with finer key buckets — more keys, more cold-start passes, a
    /// closer fit per workload.
    Extensive,
}

/// The policy every dispatch consults, as a `u8` so it can be both lazily
/// initialized from the environment and overridden by [`set_policy`].
static POLICY: AtomicU8 = AtomicU8::new(POLICY_UNSET);

const POLICY_UNSET: u8 = 0;
const POLICY_OFF: u8 = 1;
const POLICY_BALANCED: u8 = 2;
const POLICY_EXTENSIVE: u8 = 3;

impl AutotunePolicy {
    /// Parse a `CINTX_AUTOTUNE` value. Unrecognized values are rejected so a
    /// typo cannot silently change the policy.
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Some(Self::Off),
            "balanced" | "1" | "on" | "true" => Some(Self::Balanced),
            "extensive" | "2" => Some(Self::Extensive),
            _ => None,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Off => POLICY_OFF,
            Self::Balanced => POLICY_BALANCED,
            Self::Extensive => POLICY_EXTENSIVE,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            POLICY_OFF => Some(Self::Off),
            POLICY_BALANCED => Some(Self::Balanced),
            POLICY_EXTENSIVE => Some(Self::Extensive),
            _ => None,
        }
    }

    /// Does this policy tune at all?
    #[must_use]
    pub fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// The policy the caller asked for, if any.
///
/// `None` means neither `CINTX_AUTOTUNE` nor [`set_policy`] has spoken, and the
/// per-decomposition default in [`policy_for`] applies. An unrecognized
/// environment value is reported and treated as unset rather than erroring:
/// this is a performance knob, and no result depends on it.
#[must_use]
pub fn configured_policy() -> Option<AutotunePolicy> {
    if let Some(policy) = AutotunePolicy::from_code(POLICY.load(Ordering::Relaxed)) {
        return Some(policy);
    }
    match std::env::var("CINTX_AUTOTUNE") {
        Ok(value) if !value.is_empty() => match AutotunePolicy::parse(&value) {
            Some(parsed) => {
                // A racing reader may resolve the same value concurrently; both
                // store the same code, so the race is benign.
                POLICY.store(parsed.code(), Ordering::Relaxed);
                Some(parsed)
            }
            None => {
                tracing::warn!(
                    value = %value,
                    "unrecognized CINTX_AUTOTUNE value; using the per-backend default"
                );
                None
            }
        },
        _ => None,
    }
}

/// The policy in force for a dispatch with this decomposition.
///
/// # Why the default is per decomposition
///
/// The decomposition *is* the backend question the measurement turns on. A
/// [`Decomposition::PerUnit`] dispatch runs on a host runtime whose units are OS
/// threads and whose profiled timings are host wall clock; a
/// [`Decomposition::Cooperative`] one runs on a backend with hardware planes,
/// where CubeCL ranks candidates by device timestamp. Those are different
/// measurement instruments, and they gave opposite answers:
///
/// | decomposition | workload | `off` | `balanced` | |
/// |---|---|---|---|---|
/// | per-unit (16-core CPU) | 4096 quartets, 8 classes | 75.7 / 81.5 ms | 88.8 / 273 ms | slower |
/// | cooperative (ROCm gfx1151) | H2O/def2-SVP, 3081 quartets | 33.3 ms | 32.4 ms | 1.03x |
/// | cooperative (ROCm gfx1151) | SO2/def2-SVP, 22 155 quartets | 369 ms | 256 ms | **1.44x** |
/// | cooperative (ROCm gfx1151) | H2O/def2-TZVP, 18 145 quartets | 194 ms | 151 ms | **1.28x** |
///
/// The GPU rows are `def2_rocm_extended_and_tuning`'s, and they came with
/// bit-identical values — the kernel covers the same index space at every
/// geometry, so a tuned launch buys speed and never results.
///
/// So the default follows the evidence rather than splitting the difference:
/// tuning is on where its ranking is trustworthy and measurably wins, and off
/// where the module's own CPU measurement said it does not. `CINTX_AUTOTUNE`
/// and [`set_policy`] override both.
#[must_use]
pub fn policy_for(decomposition: Decomposition) -> AutotunePolicy {
    configured_policy().unwrap_or(match decomposition {
        Decomposition::PerUnit => AutotunePolicy::Off,
        Decomposition::Cooperative => AutotunePolicy::Balanced,
    })
}

/// The process-wide policy, for questions that name no decomposition.
///
/// Diagnostics and the CubeCL level install use this. It reports the configured
/// policy when there is one and [`DEFAULT_POLICY`] otherwise; the decision that
/// actually gates a dispatch is [`policy_for`].
#[must_use]
pub fn policy() -> AutotunePolicy {
    configured_policy().unwrap_or(DEFAULT_POLICY)
}

/// The policy reported when nothing is configured and no decomposition is
/// named. Not the value a cooperative dispatch gets — see [`policy_for`].
pub const DEFAULT_POLICY: AutotunePolicy = AutotunePolicy::Off;

/// Choose the tuning policy programmatically, overriding `CINTX_AUTOTUNE`.
///
/// Takes effect for dispatches that follow; it does not re-tune or discard what
/// is already measured. Call it before the first tuned dispatch if the intent
/// is to configure a process, rather than to change its mind halfway.
pub fn set_policy(policy: AutotunePolicy) {
    POLICY.store(policy.code(), Ordering::Relaxed);
}

/// Install this crate's autotune configuration into the CubeCL global config.
///
/// Idempotent, and deliberately non-destructive:
///
/// - If any crate has already set or read the global configuration, it is left
///   exactly as it is. (`CubeClRuntimeConfig::set` *panics* in that case, which
///   is not an acceptable failure mode for a library, so the storage slot is
///   filled directly instead.)
/// - The base is whatever `cubecl.toml` / `CUBECL_*` would have produced, so a
///   caller that configured CubeCL by file or environment keeps that
///   configuration; only the fields this crate owns are filled in on top.
///
/// Call it before the first tuned dispatch — [`should_tune`] does.
pub fn install_runtime_config() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, autotune::AutotuneLevel};

        let storage = CubeClRuntimeConfig::storage();
        let mut slot = storage.lock();
        if slot.is_some() {
            // Someone else owns the configuration; adopt it as-is.
            return;
        }

        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();

        // `CUBECL_AUTOTUNE_LEVEL` is applied by `override_from_env` above and
        // wins over the cintx policy, which is why this runs after it.
        if std::env::var("CUBECL_AUTOTUNE_LEVEL").is_err() {
            config.autotune.level = match policy() {
                AutotunePolicy::Extensive => AutotuneLevel::Extensive,
                // `Off` never reaches a tuned dispatch, so the level it maps to
                // is irrelevant — and a cooperative dispatch under the
                // per-decomposition default arrives here wanting `Balanced`
                // anyway.
                AutotunePolicy::Off | AutotunePolicy::Balanced => AutotuneLevel::Balanced,
            };
        }

        if let Ok(dir) = std::env::var("CINTX_AUTOTUNE_CACHE")
            && !dir.is_empty()
        {
            config.autotune.cache = cubecl::config::cache::CacheConfig::File(dir.into());
        }

        *slot = Some(std::sync::Arc::new(config));
    });
}

/// Which family's launch geometry a key describes.
///
/// Part of the key rather than of the tuner name so that one persistent cache
/// file covers the whole crate and a `family` never collides with another's
/// measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TunedFamily {
    /// `int2e`-shaped shell-quartet batches.
    TwoE,
    /// `int1e`-shaped shell-pair batches.
    OneE,
}

impl fmt::Display for TunedFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TwoE => formatter.write_str("2e"),
            Self::OneE => formatter.write_str("1e"),
        }
    }
}

/// How a dispatch divides its work, which decides what a cube width *is*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Decomposition {
    /// One work item per unit: the cube width is the thread count, and each
    /// unit owns a private scratch slab.
    PerUnit,
    /// One work item per cube, its inner block split across the cube's lanes:
    /// the cube width is the cooperative width, and the scratch slab is per
    /// cube and independent of the width.
    Cooperative,
}

impl fmt::Display for Decomposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PerUnit => formatter.write_str("per-unit"),
            Self::Cooperative => formatter.write_str("cooperative"),
        }
    }
}

/// The coarse workload-and-device description a geometry measurement is
/// attributed to.
///
/// Following the design plan: tuned *by device fingerprint and coarse workload
/// key, not exact dimensions*. The workload fields are anchored to powers of a
/// base by [`anchor`], so neighbouring problem sizes share a measurement, and
/// the hardware fields are carried explicitly because the
/// [priority closure](cube_width_priority) that prunes unviable candidates only
/// ever sees the key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LaunchGeometryKey {
    /// [`TUNING_SCHEMA_VERSION`] at the time the entry was written.
    pub schema: u32,
    /// Which family this dispatch belongs to.
    pub family: TunedFamily,
    /// How the dispatch divides its work.
    pub decomposition: Decomposition,
    /// The kernel's comptime specialization axis — the Rys order, for every
    /// family tuned so far. Two dispatches that specialize differently compile
    /// to different programs and cannot share a measurement.
    pub specialization: u32,
    /// Anchored work-item count (quartets, pairs, triples).
    pub items: u32,
    /// Anchored cooperative width: the inner block length one item's work is
    /// split across. Meaningless for [`Decomposition::PerUnit`], where it is
    /// still carried so a key stays comparable across the two arms.
    pub block_len: u32,
    /// Anchored per-slot scratch cost in bytes — what bounds how many slots
    /// fit in the dispatch's scratch budget.
    pub slot_bytes: u32,
    /// Plane (warp / wavefront / subgroup) width; `1` on CPU-like backends.
    pub plane_dim: u32,
    /// Hardware ceiling on units in one cube.
    pub max_units_per_cube: u32,
    /// Independent hardware execution contexts.
    pub parallel_units: u32,
}

impl fmt::Display for LaunchGeometryKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{}/v{}/spec{}/items{}/block{}/slot{}B/plane{}",
            self.family,
            self.decomposition,
            self.schema,
            self.specialization,
            self.items,
            self.block_len,
            self.slot_bytes,
            self.plane_dim,
        )
    }
}

impl AutotuneKey for LaunchGeometryKey {}

impl LaunchGeometryKey {
    /// Build a key for one dispatch.
    ///
    /// `items`, `block_len` and `slot_bytes` are anchored here rather than by
    /// the caller, so every family buckets them the same way and the
    /// granularity follows the configured [`AutotunePolicy`].
    #[must_use]
    pub fn new(
        family: TunedFamily,
        hardware: &LaunchHardware,
        decomposition: Decomposition,
        specialization: u32,
        items: usize,
        block_len: u32,
        slot_bytes: usize,
    ) -> Self {
        // `anchor` reads the CubeCL global configuration (that is where the
        // level driving its bucket base lives) and caches the level in a static
        // on first use. The configuration must therefore be installed before
        // the first key is ever built, not merely before the first tune.
        install_runtime_config();
        Self {
            schema: TUNING_SCHEMA_VERSION,
            family,
            decomposition,
            specialization,
            items: anchor(items, Some(1 << 24), Some(1), None) as u32,
            block_len: anchor(block_len as usize, Some(1 << 16), Some(1), None) as u32,
            slot_bytes: anchor(slot_bytes, Some(1 << 30), Some(64), None) as u32,
            plane_dim: hardware.plane_dim,
            max_units_per_cube: hardware.max_units_per_cube,
            parallel_units: hardware.parallel_units,
        }
    }

    /// How many private scratch slots `budget_bytes` affords at this key's
    /// per-slot cost.
    ///
    /// `slot_bytes` is anchored *upwards*, so this under-estimates the slot
    /// count — the pruning it feeds stays on the safe side of the budget.
    #[must_use]
    pub fn max_slots(&self, budget_bytes: usize) -> usize {
        (budget_bytes / (self.slot_bytes.max(1) as usize)).max(1)
    }
}

/// Intra-group priority for one candidate cube width — the early-stop rule.
///
/// Returns `-1` to prune the candidate out of the tuning plan entirely (it is
/// never compiled, never launched, and never contributes to cold-start cost),
/// and `1` for a width worth measuring.
///
/// A width is pruned when it is:
///
/// - past this device's `max_units_per_cube`;
/// - past the scratch budget, in the per-unit arm where every unit owns a slab;
/// - past the parallelism the work can fill — more threads than items, or than
///   the device has execution contexts;
/// - not a whole number of planes, or more than one plane past the cooperative
///   width the kernel can actually use, in the cooperative arm.
///
/// `budget_bytes` is the family's scratch ceiling; it is a parameter because
/// each family owns its own.
#[must_use]
pub fn cube_width_priority(key: &LaunchGeometryKey, width: u32, budget_bytes: usize) -> i8 {
    if width == 0 || width > key.max_units_per_cube {
        return -1;
    }
    match key.decomposition {
        Decomposition::PerUnit => {
            // One slab per unit, and the units are the only parallel axis.
            if width as usize > key.max_slots(budget_bytes)
                || width > key.parallel_units
                || width > key.items
            {
                return -1;
            }
            1
        }
        Decomposition::Cooperative => {
            let plane = key.plane_dim.max(1);
            // A partial plane is idle silicon on every backend that has planes.
            if !width.is_multiple_of(plane) {
                return -1;
            }
            // Lanes past the inner block length have no element to take, so at
            // most the plane-rounded block length is worth measuring.
            let useful = key.block_len.div_ceil(plane).max(1) * plane;
            if width > useful {
                return -1;
            }
            1
        }
    }
}

/// Should this dispatch be tuned, or launched on its heuristic geometry?
///
/// Enforces the policy, the [`MIN_TUNE_ITEMS`] work floor and the
/// [`MAX_TUNED_KEYS`] cardinality bound, and installs the CubeCL configuration
/// on the way through so the persistent cache is in place before the first
/// tuner is constructed.
#[must_use]
pub fn should_tune(key: &LaunchGeometryKey, items: usize) -> bool {
    if !policy_for(key.decomposition).enabled() || items < MIN_TUNE_ITEMS {
        return false;
    }
    install_runtime_config();

    let mut seen = tuned_keys().lock().expect("tuned key set poisoned");
    if seen.contains(key) {
        return true;
    }
    if seen.len() >= MAX_TUNED_KEYS {
        tracing::debug!(
            key = %key,
            bound = MAX_TUNED_KEYS,
            "launch-geometry key budget exhausted; using the heuristic geometry"
        );
        return false;
    }
    seen.insert(key.clone());
    true
}

/// The distinct keys this process has admitted for tuning.
fn tuned_keys() -> &'static Mutex<HashSet<LaunchGeometryKey>> {
    static KEYS: OnceLock<Mutex<HashSet<LaunchGeometryKey>>> = OnceLock::new();
    KEYS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// How many distinct launch-geometry keys this process has admitted.
///
/// Diagnostics and tests: the count is the cache cardinality this process can
/// have contributed, and it is bounded by [`MAX_TUNED_KEYS`].
#[must_use]
pub fn tuned_key_count() -> usize {
    tuned_keys().lock().expect("tuned key set poisoned").len()
}

/// A stable, filesystem-safe identity for one device.
///
/// Used as the tuner id, which is also the persistent cache's directory, so it
/// must separate any two devices whose measurements must not be shared and stay
/// byte-identical for the same device across runs. It is an FNV-1a hash of the
/// runtime's reported hardware properties, in the same spirit as
/// [`crate::capability::capability_fingerprint`] — a hasher with fixed
/// constants rather than `DefaultHasher`, whose output std does not promise to
/// keep stable across releases.
#[must_use]
pub fn device_fingerprint<R: cubecl::Runtime>(client: &ComputeClient<R>) -> String {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037_u64;
    const FNV_PRIME: u64 = 1_099_511_628_211_u64;

    let mut hash = OFFSET_BASIS;
    let mut feed = |data: &[u8]| {
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= u64::from(b'\0');
        hash = hash.wrapping_mul(FNV_PRIME);
    };

    feed(std::any::type_name::<R>().as_bytes());
    feed(format!("{:?}", client.properties().hardware).as_bytes());
    feed(&TUNING_SCHEMA_VERSION.to_le_bytes());

    let name: String = R::name(client)
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    format!("{name}-{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_hardware() -> LaunchHardware {
        LaunchHardware {
            plane_dim: 32,
            has_planes: true,
            parallel_units: 40,
            max_units_per_cube: 1024,
            max_cubes_x: 65_535,
        }
    }

    fn cpu_hardware() -> LaunchHardware {
        LaunchHardware {
            plane_dim: 1,
            has_planes: false,
            parallel_units: 8,
            max_units_per_cube: 64,
            max_cubes_x: 1,
        }
    }

    fn cooperative_key() -> LaunchGeometryKey {
        LaunchGeometryKey::new(
            TunedFamily::TwoE,
            &gpu_hardware(),
            Decomposition::Cooperative,
            3,
            4096,
            100,
            8 * 1024,
        )
    }

    fn per_unit_key() -> LaunchGeometryKey {
        LaunchGeometryKey::new(
            TunedFamily::TwoE,
            &cpu_hardware(),
            Decomposition::PerUnit,
            3,
            4096,
            100,
            8 * 1024,
        )
    }

    #[test]
    fn anchored_key_merges_neighbouring_workloads() {
        let hardware = gpu_hardware();
        let build = |items: usize| {
            LaunchGeometryKey::new(
                TunedFamily::TwoE,
                &hardware,
                Decomposition::Cooperative,
                3,
                items,
                100,
                8 * 1024,
            )
        };
        // 4097 and 5000 both anchor up to the same power-of-two bucket, so one
        // measurement serves both — that is the whole point of anchoring.
        assert_eq!(build(4097), build(5000));
        // Sizes an order of magnitude apart must not share a measurement.
        assert_ne!(build(64), build(65_536));
    }

    #[test]
    fn key_separates_specialization_family_and_decomposition() {
        let key = cooperative_key();
        let mut other = key.clone();
        other.specialization += 1;
        assert_ne!(key, other);

        let mut other = key.clone();
        other.family = TunedFamily::OneE;
        assert_ne!(key, other);

        let mut other = key.clone();
        other.decomposition = Decomposition::PerUnit;
        assert_ne!(key, other);
    }

    #[test]
    fn cooperative_priority_prunes_partial_planes_and_idle_lanes() {
        let key = cooperative_key();
        let budget = 256 * 1024 * 1024;
        // A whole number of planes, within the plane-rounded block length.
        assert_eq!(cube_width_priority(&key, 32, budget), 1);
        assert_eq!(cube_width_priority(&key, 128, budget), 1);
        // Not a whole plane.
        assert_eq!(cube_width_priority(&key, 16, budget), -1);
        // Past the plane-rounded block length (128 for block_len 128): idle lanes.
        assert_eq!(cube_width_priority(&key, 256, budget), -1);
    }

    #[test]
    fn cooperative_priority_prunes_past_the_hardware_ceiling() {
        let mut key = cooperative_key();
        key.max_units_per_cube = 64;
        key.block_len = 1024;
        let budget = 256 * 1024 * 1024;
        assert_eq!(cube_width_priority(&key, 64, budget), 1);
        assert_eq!(cube_width_priority(&key, 128, budget), -1);
    }

    #[test]
    fn per_unit_priority_prunes_past_cores_work_and_scratch() {
        let key = per_unit_key();
        let budget = 256 * 1024 * 1024;
        assert_eq!(cube_width_priority(&key, 8, budget), 1);
        // More threads than the backend has execution contexts.
        assert_eq!(cube_width_priority(&key, 16, budget), -1);

        // More threads than there is work to give them.
        let mut small = per_unit_key();
        small.items = 2;
        assert_eq!(cube_width_priority(&small, 8, budget), -1);
        assert_eq!(cube_width_priority(&small, 2, budget), 1);

        // More private scratch than the family's budget affords.
        let mut fat = per_unit_key();
        fat.slot_bytes = 64 * 1024 * 1024;
        assert_eq!(cube_width_priority(&fat, 8, budget), -1);
        assert_eq!(cube_width_priority(&fat, 4, budget), 1);
    }

    #[test]
    fn every_candidate_width_is_decidable_and_at_least_one_survives() {
        let budget = 256 * 1024 * 1024;
        for key in [cooperative_key(), per_unit_key()] {
            let viable = CANDIDATE_CUBE_WIDTHS
                .iter()
                .filter(|&&width| cube_width_priority(&key, width, budget) >= 0)
                .count();
            assert!(
                viable > 0,
                "no viable candidate width survived pruning for {key}"
            );
            assert!(
                viable < CANDIDATE_CUBE_WIDTHS.len(),
                "pruning removed nothing for {key}; the early-stop rule is inert"
            );
        }
    }

    #[test]
    fn policy_codes_round_trip() {
        // The global is deliberately not exercised here: these unit tests share
        // a process with kernel tests, and flipping the process-wide policy on
        // would hand one of them a tuning pass it did not ask for. The
        // `set_policy` path itself is covered by
        // `tests/tuned_geometry_parity.rs`, which owns the policy under a guard.
        for policy in [
            AutotunePolicy::Off,
            AutotunePolicy::Balanced,
            AutotunePolicy::Extensive,
        ] {
            assert_eq!(AutotunePolicy::from_code(policy.code()), Some(policy));
        }
        assert_eq!(AutotunePolicy::from_code(POLICY_UNSET), None);
        assert!(!DEFAULT_POLICY.enabled(), "tuning ships opt-in");
    }

    #[test]
    fn policy_parsing_covers_the_documented_spellings() {
        assert_eq!(AutotunePolicy::parse("off"), Some(AutotunePolicy::Off));
        assert_eq!(AutotunePolicy::parse("OFF"), Some(AutotunePolicy::Off));
        assert_eq!(
            AutotunePolicy::parse("balanced"),
            Some(AutotunePolicy::Balanced)
        );
        assert_eq!(
            AutotunePolicy::parse("extensive"),
            Some(AutotunePolicy::Extensive)
        );
        assert_eq!(AutotunePolicy::parse("sideways"), None);
        assert!(!AutotunePolicy::Off.enabled());
        assert!(AutotunePolicy::Balanced.enabled());
    }

    #[test]
    fn sample_prefix_is_work_aware_and_bounded() {
        // A cheap class gets the long prefix: a short one would not amortize
        // the cost of waking a wide cube, and would rank narrow cubes too high.
        assert_eq!(tune_sample_items(100_000, 1), TUNE_SAMPLE_MAX_ITEMS);
        // An expensive class gets the short one: one item already amortizes the
        // dispatch, and a long prefix would only make the tuning pass costly.
        assert_eq!(tune_sample_items(100_000, 1 << 20), TUNE_SAMPLE_MIN_ITEMS);
        // Never more work than the caller actually has.
        assert_eq!(tune_sample_items(7, 1), 7);
        // The budget divides: 2^18 / 2^9 = 512, inside both bounds.
        assert_eq!(tune_sample_items(100_000, 1 << 9), 512);
    }

    #[test]
    fn key_round_trips_through_the_persistent_cache_encoding() {
        let key = cooperative_key();
        let encoded = serde_json::to_string(&key).expect("key serializes");
        let decoded: LaunchGeometryKey = serde_json::from_str(&encoded).expect("key deserializes");
        assert_eq!(key, decoded);
    }
}
