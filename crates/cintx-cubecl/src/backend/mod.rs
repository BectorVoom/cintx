//! CubeCL backend resolution: wraps ComputeClient handles behind a single enum.
//!
//! `ResolvedBackend` holds the live client for whichever compute backend the
//! executor was configured to use. This avoids generics on `CubeClExecutor`
//! (required because `BackendExecutor` is used as `&dyn BackendExecutor`).
//!
//! Phase 16-02 (Wave 1) introduces per-variant `#[cfg(feature = "...")]`
//! gating on both [`ResolvedBackend`] and [`BackendKind`] (CONTEXT D-10).
//! Supported feature flags: `cpu` (default-on, D-06), `wgpu`, `cuda`, `rocm`,
//! `metal` (M1: aliased to the wgpu runtime — see `<context_deviation>` in
//! `.planning/phases/16-multi-backend-support/16-02-PLAN.md`).
//!
//! cuda and metal arms ship as compile-only this phase — see
//! `.planning/notes/cuda-metal-verification-gap.md` for the verification
//! risk-accept that applies to those modules.

#[cfg(feature = "cpu")]
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "rocm")]
pub mod rocm_backend;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

use cintx_core::cintxRsError;
use cintx_runtime::{BackendIntent, BackendKind};

/// A resolved, live compute-client handle for one of the supported backends.
///
/// Each arm is gated by its own `#[cfg(feature = "...")]` (D-10). The Metal
/// arm carries the same `WgpuRuntime` client as Wgpu under the M1 alias —
/// the typed identity is preserved so capability fingerprints and error
/// diagnostics can distinguish the two even though the GPU client is shared.
pub enum ResolvedBackend {
    /// CPU backend client (default-on; `cpu` feature gates the cubecl/cpu
    /// runtime crate).
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
    /// wgpu GPU backend client, paired with the adapter's feature names for
    /// capability checks (e.g. SHADER_F64 gating).
    #[cfg(feature = "wgpu")]
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
    /// CUDA backend client. Compile-only this phase — see
    /// `.planning/notes/cuda-metal-verification-gap.md`.
    #[cfg(feature = "cuda")]
    Cuda(cubecl::client::ComputeClient<cubecl_cuda::CudaRuntime>),
    /// ROCm backend client (cubecl-hip). Runtime-verifiable on the dev host.
    #[cfg(feature = "rocm")]
    Rocm(cubecl::client::ComputeClient<cubecl_hip::HipRuntime>),
    /// Metal — M1 alias: dispatches through `cubecl_wgpu::WgpuRuntime` on
    /// Apple targets. See `.planning/notes/cuda-metal-verification-gap.md`.
    #[cfg(feature = "metal")]
    Metal(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>, Vec<String>),
}

impl ResolvedBackend {
    /// Returns the wgpu/metal feature list if this is a Wgpu or Metal backend,
    /// or an empty slice for CPU/CUDA/ROCm (which expose features through
    /// other channels).
    pub fn wgpu_features(&self) -> &[String] {
        match self {
            #[cfg(feature = "cpu")]
            ResolvedBackend::Cpu(_) => &[],
            #[cfg(feature = "wgpu")]
            ResolvedBackend::Wgpu(_, features) => features,
            #[cfg(feature = "cuda")]
            ResolvedBackend::Cuda(_) => &[],
            #[cfg(feature = "rocm")]
            ResolvedBackend::Rocm(_) => &[],
            #[cfg(feature = "metal")]
            ResolvedBackend::Metal(_, features) => features,
        }
    }
}

impl ResolvedBackend {
    /// Construct a `ResolvedBackend` from a `BackendIntent`.
    ///
    /// Calls the appropriate backend helper to bootstrap and validate the
    /// compute client. Returns `UnsupportedApi` when the requested backend
    /// cannot be initialised. The five-arm cfg-gated dispatch matches the
    /// `BackendKind` enum exhaustively per active feature combination.
    pub fn from_intent(intent: &BackendIntent) -> Result<Self, cintxRsError> {
        match &intent.backend {
            BackendKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let client = cpu_backend::resolve_cpu_client()?;
                    Ok(ResolvedBackend::Cpu(client))
                }
                #[cfg(not(feature = "cpu"))]
                Err(cintxRsError::UnsupportedApi {
                    requested: "cpu-backend:feature-not-enabled".to_owned(),
                })
            }
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => {
                let report = crate::runtime_bootstrap::bootstrap_wgpu_runtime(intent)?;
                let features = report.snapshot.features.clone();
                let client = wgpu_backend::resolve_wgpu_client(intent)?;
                Ok(ResolvedBackend::Wgpu(client, features))
            }
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => {
                let client = cuda_backend::resolve_cuda_client()?;
                Ok(ResolvedBackend::Cuda(client))
            }
            #[cfg(feature = "rocm")]
            BackendKind::Rocm => {
                let client = rocm_backend::resolve_rocm_client()?;
                Ok(ResolvedBackend::Rocm(client))
            }
            #[cfg(feature = "metal")]
            BackendKind::Metal => {
                // M1 alias: Metal dispatches through the wgpu runtime.
                // See .planning/notes/cuda-metal-verification-gap.md for the
                // compile-only risk-accept.
                tracing::info!("BackendKind::Metal selected; dispatching via cubecl-wgpu");
                let report = crate::runtime_bootstrap::bootstrap_wgpu_runtime(intent)?;
                let features = report.snapshot.features.clone();
                let client = wgpu_backend::resolve_wgpu_client(intent)?;
                Ok(ResolvedBackend::Metal(client, features))
            }
        }
    }
}

/// The set of all backend names recognized by `CINTX_BACKEND` (D-02).
///
/// Used by `resolve_backend_kind` to distinguish "recognized but not compiled"
/// (returns `BackendNotCompiled` per D-01) from "unrecognized value" (returns
/// `InvalidEnvParam` per D-02).
pub const KNOWN_BACKEND_NAMES: &[&str] = &["cpu", "wgpu", "cuda", "rocm", "metal"];

/// The compile-time-cfg'd set of backend names that this build can resolve.
///
/// Populated from the active `#[cfg(feature = "...")]` arms; never re-uses
/// the static set so the `BackendNotCompiled` error variant carries an
/// accurate `compiled_in` field per D-01.
pub const COMPILED_IN_BACKENDS: &[&str] = &[
    #[cfg(feature = "cpu")]
    "cpu",
    #[cfg(feature = "wgpu")]
    "wgpu",
    #[cfg(feature = "cuda")]
    "cuda",
    #[cfg(feature = "rocm")]
    "rocm",
    #[cfg(feature = "metal")]
    "metal",
];

/// Public introspection helper: the set of backend names that `CINTX_BACKEND`
/// can resolve to in this build.
///
/// Returns a static slice reflecting the active feature combination at
/// compile time. Useful for diagnostic surfaces and for downstream callers
/// that want to enumerate backends without depending on the
/// `cintx-cubecl::backend::COMPILED_IN_BACKENDS` const directly.
pub fn compiled_backends() -> &'static [&'static str] {
    COMPILED_IN_BACKENDS
}

/// Read the `CINTX_BACKEND` environment variable and return the corresponding
/// `BackendKind`. Defaults to `BackendKind::default()` (= `Cpu` per D-11)
/// when the variable is absent or empty.
///
/// Recognized values (case-sensitive): `"cpu"`, `"wgpu"`, `"cuda"`, `"rocm"`,
/// `"metal"`.
///
/// # Errors
///
/// - Returns `cintxRsError::BackendNotCompiled` (D-01) when the value names
///   a known backend that was not compiled into this build.
/// - Returns `cintxRsError::InvalidEnvParam` (D-02) when the value is not a
///   recognized backend name.
///
/// No silent fallback — D-03 mandates a fallible chokepoint.
pub fn resolve_backend_kind() -> Result<BackendKind, cintxRsError> {
    match std::env::var("CINTX_BACKEND").as_deref() {
        Err(_) | Ok("") => Ok(BackendKind::default()),
        Ok("cpu") => Ok(BackendKind::Cpu),
        #[cfg(feature = "wgpu")]
        Ok("wgpu") => Ok(BackendKind::Wgpu),
        #[cfg(feature = "cuda")]
        Ok("cuda") => Ok(BackendKind::Cuda),
        #[cfg(feature = "rocm")]
        Ok("rocm") => Ok(BackendKind::Rocm),
        #[cfg(feature = "metal")]
        Ok("metal") => Ok(BackendKind::Metal),
        // Compiled-out backends — D-01 hard error.
        Ok(name) if KNOWN_BACKEND_NAMES.contains(&name) => {
            Err(cintxRsError::BackendNotCompiled {
                requested: name.to_owned(),
                compiled_in: COMPILED_IN_BACKENDS
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
            })
        }
        // Unrecognized — D-02 error.
        Ok(other) => Err(cintxRsError::InvalidEnvParam {
            param: "CINTX_BACKEND",
            reason: format!(
                "unrecognized value {:?}; recognized: {:?}",
                other, KNOWN_BACKEND_NAMES
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn resolved_backend_from_intent_selects_cpu_arm() {
        let intent = BackendIntent {
            backend: BackendKind::Cpu,
            selector: "auto".to_owned(),
        };
        let backend = ResolvedBackend::from_intent(&intent);
        assert!(backend.is_ok(), "CPU backend should initialise successfully");
        assert!(matches!(backend.unwrap(), ResolvedBackend::Cpu(_)));
    }

    #[test]
    fn compiled_backends_includes_cpu() {
        let names = compiled_backends();
        assert!(
            names.contains(&"cpu"),
            "cpu must be in compiled_backends when default features are on"
        );
    }

    // The full BACK-05 contract suite (env-unset → Cpu, empty-string → Cpu,
    // unrecognized → InvalidEnvParam, recognized-not-compiled →
    // BackendNotCompiled, recognized-compiled → that kind) lands in Task 2 of
    // this plan along with the in-tree env-mutex guard. Keeping the test mod
    // intentionally minimal here so Task 1's commit covers only the
    // structural surface.
}
