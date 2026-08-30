use cintx_core::PrecisionKind;
use tracing::Span;

pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

/// Which compute backend to target for integral evaluation.
///
/// Per D-03, backend kind is control-plane metadata carried in `ExecutionOptions` and
/// propagated through `WorkspaceQuery` so query/evaluate drift can be detected and
/// rejected with a typed error.
///
/// Phase 16-02 (Wave 1) per-variant cfg gating per D-10. The `Cpu` arm is
/// always present (D-06: cpu is default-on, the feature flag is informational
/// only). `Wgpu`, `Cuda`, `Rocm`, `Metal` are each gated on their own feature
/// flag and the cintx-cubecl feature wiring forwards every backend feature
/// to the matching flag here so the enum stays in lockstep.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum BackendKind {
    /// CPU execution profile. Always available — `cpu` feature is default-on (D-06).
    #[default]
    Cpu,
    /// wgpu-backed CubeCL runtime.
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CUDA-backed CubeCL runtime (compile-only; verification gap per
    /// `.planning/notes/cuda-metal-verification-gap.md`).
    #[cfg(feature = "cuda")]
    Cuda,
    /// ROCm/HIP-backed CubeCL runtime.
    #[cfg(feature = "rocm")]
    Rocm,
    /// Metal — M1 alias for the wgpu runtime on Apple targets. See
    /// `.planning/phases/16-multi-backend-support/16-02-PLAN.md`
    /// `<context_deviation>` block and
    /// `.planning/notes/cuda-metal-verification-gap.md`.
    #[cfg(feature = "metal")]
    Metal,
}

/// Backend selection intent carried through the query/evaluate contract.
///
/// `selector` is an advisory hint that lets callers express device preference
/// (e.g. `"auto"`, `"device:0"`).  The runtime MAY use it for adapter selection
/// but must fail closed if the requested adapter is unavailable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BackendIntent {
    pub backend: BackendKind,
    /// Adapter selection hint.  `"auto"` means the runtime picks the best
    /// available adapter for the resolved backend.
    pub selector: String,
}

impl Default for BackendIntent {
    fn default() -> Self {
        // D-11: backend default flipped from Wgpu to Cpu. Wave 0 (16-01)
        // audited every implicit BackendIntent::default() callsite in
        // 16-01-SUMMARY.md so this flip is mechanically safe.
        Self {
            backend: BackendKind::Cpu,
            selector: "auto".to_owned(),
        }
    }
}

/// A snapshot of the concrete backend adapter capabilities at query time.
///
/// When `query_workspace` runs, the planner records the active adapter's
/// identity and a lightweight fingerprint of its capabilities.  `evaluate`
/// then rejects the plan if the token has drifted (e.g. if the caller swapped
/// adapters between query and evaluate), satisfying D-08.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendCapabilityToken {
    /// Human-readable adapter identifier (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub adapter_name: String,
    /// Backend API string (e.g. `"cpu"`, `"wgpu"`, `"cuda"`, `"rocm"`, `"metal"`).
    pub backend_api: String,
    /// Lightweight capability fingerprint.  Must change whenever device
    /// features or limits relevant to execution differ.
    pub capability_fingerprint: u64,
}

impl Default for BackendCapabilityToken {
    fn default() -> Self {
        // RESEARCH §8.7: backend_api must flip to "cpu" alongside the
        // BackendIntent::default() flip so drift detection at workspace.rs
        // (planning_matches) stays symmetric.
        Self {
            adapter_name: String::new(),
            backend_api: "cpu".to_owned(),
            capability_fingerprint: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionOptions {
    pub memory_limit_bytes: Option<usize>,
    pub trace_span: Option<Span>,
    pub chunk_size_override: Option<usize>,
    pub profile_label: Option<&'static str>,
    /// Backend selection intent that must remain stable across query/evaluate.
    pub backend_intent: BackendIntent,
    /// Capability token snapshotted at query time; compared at evaluate time.
    pub backend_capability_token: BackendCapabilityToken,
    /// F12/STG/YP zeta parameter. When set, populates `operator_env_params.f12_zeta`
    /// on the `ExecutionPlan` for F12-family operators.
    /// Must be non-zero for F12 calls (validated by `validate_f12_env_params` before kernel launch).
    pub f12_zeta: Option<f64>,
    /// Rinv origin for iprinv operators (env[4..6] in the raw API).
    /// When set, populates `operator_env_params.rinv_orig` on the `ExecutionPlan`.
    /// Must be present for any operator whose name contains "iprinv"
    /// (validated by `validate_rinv_orig_env_params` before kernel launch).
    pub rinv_orig: Option<[f64; 3]>,
    /// Common (gauge) origin (env[1..3] in the raw API), used by moment/GIAO families.
    /// When set, populates `operator_env_params.common_orig` on the `ExecutionPlan`.
    /// Unset (None) defaults to [0,0,0] (libcint reads unset env as zero); finiteness-
    /// validated by `validate_common_orig_env_params`, NOT presence-validated (D-01).
    pub common_orig: Option<[f64; 3]>,
    /// Range-separation parameter ω (env[8], `PTR_RANGE_OMEGA` in the raw API).
    ///
    /// libcint's sign convention, which is also the one
    /// `pyscf_pbc_df::traits::JkOpts::omega` and
    /// `pyscf_pbc_df::rsdf_builder::omega` already use, so no second convention
    /// enters the workspace:
    ///
    /// * `Some(ω)` with `ω > 0` — long range, `erf(ω r₁₂)/r₁₂`
    /// * `Some(ω)` with `ω < 0` — short range, `erfc(|ω| r₁₂)/r₁₂`
    /// * `None` or `Some(0.0)` — full Coulomb `1/r₁₂` (the default)
    ///
    /// Range separation is NOT a distinct integral symbol: libcint has no
    /// `int2e_sr_*` and PySCF never asks for one. When set, this populates
    /// `operator_env_params.range_omega` on the `ExecutionPlan`, and
    /// `query_workspace` folds the short-range Rys-root doubling
    /// ([`crate::range_omega::nrys_roots_for`]) into the workspace request.
    ///
    /// Accepted for the three scalar Coulomb operators (`int2e`, `int3c2e`,
    /// `int2c2e`) and their `ip`-family gradient/Hessian rows; every other
    /// operator rejects a set value with `UnsupportedApi` rather than silently
    /// evaluating the full-range kernel. The admitted set is exactly what
    /// [`crate::range_omega::derivative_headroom`] knows the `ng[]` raises for,
    /// because those raises size the workspace (see
    /// [`crate::validator::validate_range_omega_env_params`]).
    pub range_omega: Option<f64>,
    /// AO symmetry packing requested by the caller. Phase 18 implements `S1` only;
    /// every other variant returns `FacadeError::UnsupportedAoSymmetry` from
    /// `SessionRequest::query_workspace`. `None` is the default and is treated as
    /// `Some(AoSymmetry::S1)`.
    pub aosym: Option<cintx_core::AoSymmetry>,
    /// Output precision for this evaluation.
    ///
    /// `F64` (default) = byte-identical to libcint 6.1.3 (D-08/D-12).
    /// `F32` = opt-in, ~1e-4 rtol oracle gate (D-09), unlocks non-`SHADER_F64`
    /// wgpu adapters (D-10 / PREC-06).
    ///
    /// `PrecisionKind` derives `Default` as `F64`, so `ExecutionOptions::default()`
    /// produces the f64 path unchanged — existing call sites are unaffected.
    pub precision: PrecisionKind,
}

impl ExecutionOptions {
    pub fn with_default_limits() -> Self {
        Self {
            memory_limit_bytes: Some(DEFAULT_MEMORY_LIMIT_BYTES),
            ..Self::default()
        }
    }

    pub const fn default_memory_limit_bytes() -> usize {
        DEFAULT_MEMORY_LIMIT_BYTES
    }

    pub fn effective_memory_limit_bytes(&self, required_bytes: usize) -> usize {
        self.memory_limit_bytes.unwrap_or(required_bytes)
    }
}
