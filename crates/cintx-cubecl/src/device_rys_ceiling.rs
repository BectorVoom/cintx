//! Per-backend device Rys ceiling and the FMA-fusion probe that gates it
//! (Phase 33, task 33-05 scaffolding).
//!
//! # Why the ceiling is per backend
//!
//! The device Rys kernels serve `nroots <= 5` with the polynomial-fit
//! `rys_root{1..5}` family. Beyond that the Wheeler/Jacobi path takes over, and
//! from `nroots >= 8` it runs in **double-double** arithmetic — pairs of `f64`
//! carrying ~106 bits of mantissa, built on Dekker/Knuth error-free transforms.
//!
//! Those transforms are the whole reason the extended path is accurate, and
//! they are fragile in a specific way: `two_prod(a, b)` computes the product's
//! rounding error as `fma(a, b, -(a*b))`, which is the exact error **only if
//! the backend lowers `fma` to a true fused multiply-add** — one rounding, no
//! intermediate rounding of `a*b`. A backend that pre-rounds the product
//! returns zero for that term, silently degrading every dd operation to plain
//! `f64` while the code still claims 106 bits.
//!
//! That is a silent accuracy loss, not a crash, and it is per backend: each
//! compiles the same `#[cube]` source through a different compiler to a
//! different ISA. Measuring it on one backend says nothing about another —
//! which is exactly why the ceiling cannot be one global constant.
//!
//! # Fail-closed by construction
//!
//! [`device_nroots_ceiling`] returns [`BASE_DEVICE_NROOTS`] unless **all three**
//! hold:
//!
//! 1. the `extended-device-rys` feature is compiled in — off by default;
//! 2. [`fma_fusion_verified`] measured a true fused multiply-add **on that
//!    backend, in this process**; and
//! 3. the calling family has been flipped onto the inline extended entry —
//!    [`RysFamily::runs_extended_rys`], the per-family opt-in of task 33-03.
//!
//! A backend whose probe has not run, or has run and failed, keeps the base
//! ceiling; so does a family nobody has wired up yet. Adding a backend or a
//! family to the build therefore cannot raise its ceiling by accident: the
//! default is the safe value and the raise requires evidence.

use crate::backend::ResolvedBackend;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Rys order the polynomial-fit device kernels (`rys_root{1..5}`) serve.
///
/// Scalar 2e `nroots = (li+lj+lk+ll)/2 + 1`, so this covers an angular-momentum
/// sum of 8 — the whole def2-SVP envelope, and everything below `(ff|ff)`.
pub const BASE_DEVICE_NROOTS: usize = 5;

/// Rys order the Wheeler/Jacobi device path serves once it is unlocked.
///
/// 12 is the vendor's own quadmath ceiling; beyond it libcint has no reference
/// to be compatible with.
pub const EXTENDED_DEVICE_NROOTS: usize = 12;

/// Which backend a probe result belongs to.
///
/// A probe measures a compiler and an ISA, so its result is keyed on the
/// backend arm rather than on the client instance: two clients on the same
/// backend lower `fma` identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProbeTarget {
    Cpu,
    Wgpu,
    Cuda,
    Rocm,
    Metal,
}

impl ProbeTarget {
    fn of(backend: &ResolvedBackend) -> Self {
        match backend {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(_) => Self::Cpu,
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(_, _) => Self::Wgpu,
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(_) => Self::Cuda,
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(_) => Self::Rocm,
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(_, _) => Self::Metal,
        }
    }

    /// Name used in diagnostics.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
        }
    }
}

/// Device probe: `out[i] = fma(a[i], b[i], -(a[i] * b[i]))`.
///
/// Under a true fused multiply-add this is TwoProd's exact error term, which is
/// non-zero whenever the product is not exactly representable. Under a
/// pre-rounded product it is exactly zero.
#[cube(launch)]
fn fma_fusion_probe_kernel(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>) {
    let i = ABSOLUTE_POS;
    if i < a.len() {
        let ai = a[i];
        let bi = b[i];
        let p = ai * bi;
        out[i] = fma(ai, bi, -p);
    }
}

/// Operand pairs whose `f64` product is not exactly representable.
///
/// Chosen so every host error term is non-zero: a probe over exactly
/// representable products would pass on a backend that does not fuse, because
/// both the fused and the pre-rounded answer would be zero.
// Written at full precision on purpose: each operand is a chosen bit pattern,
// and a shorter spelling that happens to round to the same `f64` would hide that
// the choice is deliberate rather than incidental.
#[allow(clippy::excessive_precision)]
const PROBE_A: [f64; 6] = [
    1.000_000_000_931_322_6,
    std::f64::consts::PI,
    1.300_000_000_000_000_1,
    7.123_456_789_012_345,
    0.1,
    0.999_999_999_999_999_8,
];
#[allow(clippy::excessive_precision)]
const PROBE_B: [f64; 6] = [
    1.000_000_000_931_322_6,
    std::f64::consts::E,
    2.699_999_999_999_999_7,
    3.987_654_321_098_765,
    0.3,
    1.000_000_000_000_000_2,
];

/// What one probe run found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FmaProbeResult {
    /// Backend the probe ran on.
    pub target: ProbeTarget,
    /// Did every error term match the host `f64::mul_add` reference bit for bit?
    pub fused: bool,
    /// Operand pairs compared.
    pub pairs: usize,
    /// Pairs whose device error term differed from the host reference.
    pub divergent: usize,
}

fn run_probe<R: Runtime>(client: &ComputeClient<R>, target: ProbeTarget) -> FmaProbeResult {
    let host: Vec<f64> = PROBE_A
        .iter()
        .zip(PROBE_B.iter())
        .map(|(&a, &b)| {
            let p = a * b;
            a.mul_add(b, -p)
        })
        .collect();
    debug_assert!(
        host.iter().any(|e| *e != 0.0),
        "probe operands must produce a non-zero TwoProd error term, or the \
         probe cannot distinguish a fused backend from a pre-rounding one"
    );

    let n = PROBE_A.len();
    let a_h = client.create_from_slice(f64::as_bytes(&PROBE_A));
    let b_h = client.create_from_slice(f64::as_bytes(&PROBE_B));
    let out_h = client.create_from_slice(f64::as_bytes(&vec![0.0_f64; n]));

    fma_fusion_probe_kernel::launch::<R>(
        client,
        crate::plane::single_cube_count(),
        CubeDim::new_1d(n as u32),
        // SAFETY: each buffer is created at exactly `n` elements.
        unsafe { ArrayArg::from_raw_parts(a_h, n) },
        unsafe { ArrayArg::from_raw_parts(b_h, n) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
    );

    let raw = client.read_one_unchecked(out_h);
    let device = f64::from_bytes(&raw);
    let divergent = (0..n)
        .filter(|&i| device[i].to_bits() != host[i].to_bits())
        .count();

    FmaProbeResult {
        target,
        fused: divergent == 0,
        pairs: n,
        divergent,
    }
}

fn dispatch_probe(backend: &ResolvedBackend) -> FmaProbeResult {
    let target = ProbeTarget::of(backend);
    match backend {
        #[cfg(feature = "cpu")]
        ResolvedBackend::Cpu(client) => run_probe::<cubecl::cpu::CpuRuntime>(client, target),
        #[cfg(feature = "wgpu")]
        ResolvedBackend::Wgpu(client, _) => run_probe::<cubecl_wgpu::WgpuRuntime>(client, target),
        #[cfg(feature = "cuda")]
        ResolvedBackend::Cuda(client) => run_probe::<cubecl_cuda::CudaRuntime>(client, target),
        #[cfg(feature = "rocm")]
        ResolvedBackend::Rocm(client) => run_probe::<cubecl_hip::HipRuntime>(client, target),
        #[cfg(feature = "metal")]
        ResolvedBackend::Metal(client, _) => run_probe::<cubecl_wgpu::WgpuRuntime>(client, target),
    }
}

/// Run the FMA-fusion probe on `backend`, or return the cached result.
///
/// The probe is one launch over six operand pairs, so the cost is a launch
/// rather than a computation; it is still cached per backend arm because a
/// compiler's lowering does not change within a process.
pub fn probe_fma_fusion(backend: &ResolvedBackend) -> FmaProbeResult {
    static CACHE: OnceLock<Mutex<BTreeMap<ProbeTarget, FmaProbeResult>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let target = ProbeTarget::of(backend);

    if let Some(cached) = cache.lock().expect("fma probe cache poisoned").get(&target) {
        return *cached;
    }
    let result = dispatch_probe(backend);
    cache
        .lock()
        .expect("fma probe cache poisoned")
        .insert(target, result);
    result
}

/// Does `backend` lower `fma` to a true fused multiply-add?
///
/// This is the precondition for the double-double Wheeler path: see the module
/// note. A `false` here is not a failure to report to the caller — it is the
/// reason the ceiling stays at [`BASE_DEVICE_NROOTS`].
#[must_use]
pub fn fma_fusion_verified(backend: &ResolvedBackend) -> bool {
    probe_fma_fusion(backend).fused
}

/// Which family's launcher is asking for a ceiling.
///
/// # Why the ceiling is per family as well as per backend
///
/// A backend's FMA probe says the *solver* can be trusted there. It says
/// nothing about whether a given family's kernel has been wired to call it:
/// task 33-03 flips families one at a time, each behind its own oracle parity
/// gate, and a family that has not been flipped still has `urys`/`wrys` sized
/// for the five polynomial-fit roots and a launcher whose comptime `match`
/// stops at 5.
///
/// Handing such a family a raised ceiling does not make it slower or less
/// accurate — it makes it *wrong*, silently: the launcher's catch-all arm
/// evaluates a `nroots = 7` class at order 5, or the kernel indexes past a
/// five-element array. Both were observed the first time the ceiling was raised
/// globally.
///
/// So the ceiling is a question about a (backend, family) pair, and asking it
/// requires naming the family. Flipping a family is then one edit in
/// [`Self::runs_extended_rys`] plus its parity gate, and nothing else moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RysFamily {
    /// `int3c2e` — scalar `(mu nu | P)`, both the batch and per-tuple paths.
    Int3c2e,
    /// `int3c2e_ip1` / `int3c2e_ip2` and the rest of the 3c2e derivative set.
    Int3c2eDeriv,
    /// `int2e` and its derivative set.
    Int2e,
    /// `int2c2e` and its derivative set.
    Int2c2e,
    /// Scalar Rys-based one-electron integrals — `int1e_nuc` and `int1e_rinv`.
    Int1e,
    /// The one-electron derivative set: the nuclear gradient, `drinv`, the
    /// second-derivative and GIAO kernels.
    Int1eDeriv,
}

impl RysFamily {
    /// Has this family been flipped onto the inline extended Rys entry
    /// (`math::rys_wheeler::rys_roots_ext_dev`)?
    ///
    /// The list grows one entry per commit in task 33-03, in the order the plan
    /// sets by workload weight: `int3c2e` (which unblocks def2-TZVP + def2/J
    /// RI-J), then `int2e`, `int2c2e`, `int1e_*`. Each entry is added in the
    /// same commit as the parity gate that justifies it.
    #[must_use]
    pub const fn runs_extended_rys(self) -> bool {
        matches!(
            self,
            Self::Int3c2e | Self::Int2e | Self::Int2c2e | Self::Int1e
        )
    }

    /// Name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Int3c2e => "int3c2e",
            Self::Int3c2eDeriv => "int3c2e-deriv",
            Self::Int2e => "int2e",
            Self::Int2c2e => "int2c2e",
            Self::Int1e => "int1e",
            Self::Int1eDeriv => "int1e-deriv",
        }
    }
}

/// The largest Rys order `family`'s device kernels may serve on `backend`.
///
/// Returns [`BASE_DEVICE_NROOTS`] unless **all three** hold: the
/// `extended-device-rys` feature is compiled in, the FMA-fusion probe passed on
/// this backend, and `family` has been flipped onto the inline extended entry
/// ([`RysFamily::runs_extended_rys`]). Each condition is necessary; none is
/// sufficient.
#[must_use]
pub fn device_nroots_ceiling(backend: &ResolvedBackend, family: RysFamily) -> usize {
    if cfg!(feature = "extended-device-rys")
        && family.runs_extended_rys()
        && fma_fusion_verified(backend)
    {
        EXTENDED_DEVICE_NROOTS
    } else {
        BASE_DEVICE_NROOTS
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::backend::ResolvedBackend;
    use cintx_runtime::{BackendIntent, BackendKind};

    fn cpu() -> ResolvedBackend {
        ResolvedBackend::from_intent(&BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        })
        .expect("cpu backend")
    }

    /// The CPU backend fuses — the finding the pre-existing `fma_probe` in
    /// `rys_wheeler.rs` established, restated through the backend-generic entry
    /// point so it covers the same ground the other backends will.
    #[test]
    fn cpu_backend_fuses_fma() {
        let result = probe_fma_fusion(&cpu());
        assert_eq!(result.target, ProbeTarget::Cpu);
        assert_eq!(
            result.divergent, 0,
            "{}/{} probe pairs diverged from the host `f64::mul_add` reference; \
             the double-double TwoProd error term is not exact on this backend",
            result.divergent, result.pairs
        );
        assert!(result.fused);
    }

    /// The probe is cached, so asking twice costs one launch.
    #[test]
    fn probe_result_is_cached_per_backend() {
        let first = probe_fma_fusion(&cpu());
        let second = probe_fma_fusion(&cpu());
        assert_eq!(first, second);
    }

    /// **The fail-closed gate.** Without the `extended-device-rys` feature the
    /// ceiling stays at 5 even though the CPU probe passes — a passing probe is
    /// necessary but not sufficient, because task 33-03 also requires a green
    /// per-family oracle parity test before any family may use the extended
    /// path. If this assertion ever needs changing, that is the signal that the
    /// raise is happening; it must not happen as a side effect.
    #[test]
    fn ceiling_stays_at_the_base_without_the_opt_in_feature() {
        let backend = cpu();
        assert!(
            fma_fusion_verified(&backend),
            "precondition: the CPU probe passes"
        );
        if cfg!(feature = "extended-device-rys") {
            assert_eq!(
                device_nroots_ceiling(&backend, RysFamily::Int3c2e),
                EXTENDED_DEVICE_NROOTS
            );
        } else {
            assert_eq!(
                device_nroots_ceiling(&backend, RysFamily::Int3c2e),
                BASE_DEVICE_NROOTS,
                "a passing probe alone must not raise the ceiling"
            );
        }
    }

    /// The flipped set, spelled out once.
    ///
    /// Task 33-03 grows this list one family per commit, each in the same
    /// commit as the oracle parity gate that justifies it. Asserting the whole
    /// set here means a family cannot join it as a side effect of an edit
    /// somewhere else — and gives the per-family gates one place to read from
    /// instead of each guessing which of its neighbours is still unflipped.
    #[test]
    fn the_flipped_set_is_exactly_the_four_scalar_families() {
        let flipped: Vec<&str> = [
            RysFamily::Int3c2e,
            RysFamily::Int3c2eDeriv,
            RysFamily::Int2e,
            RysFamily::Int2c2e,
            RysFamily::Int1e,
            RysFamily::Int1eDeriv,
        ]
        .into_iter()
        .filter(|f| f.runs_extended_rys())
        .map(RysFamily::name)
        .collect();
        assert_eq!(flipped, ["int3c2e", "int2e", "int2c2e", "int1e"]);
    }

    /// **The per-family gate.** A family that has not been flipped keeps the
    /// base ceiling even with the feature compiled in and the probe passing.
    ///
    /// This is not belt-and-braces. Raising the ceiling globally the first time
    /// made the unflipped `int2e` batch accept an `(f f | f f)` class and
    /// evaluate it through its launcher's catch-all `nroots = 5` arm, and made
    /// the CubeCL optimizer panic on a five-element root array indexed at 6.
    /// A silently wrong answer is the failure mode this test exists to prevent.
    #[test]
    fn an_unflipped_family_keeps_the_base_ceiling() {
        let backend = cpu();
        for family in [
            RysFamily::Int3c2eDeriv,
            RysFamily::Int2e,
            RysFamily::Int2c2e,
            RysFamily::Int1e,
            RysFamily::Int1eDeriv,
        ] {
            if family.runs_extended_rys() {
                continue;
            }
            assert_eq!(
                device_nroots_ceiling(&backend, family),
                BASE_DEVICE_NROOTS,
                "{} has not been flipped onto the inline extended entry, so its \
                 ceiling must stay at {BASE_DEVICE_NROOTS}",
                family.name()
            );
        }
    }
}
