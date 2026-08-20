//! CubeCL backend foundations and family launch plumbing.
//!
//! Phase 16-02 (Wave 1): supported Cargo backend feature flags are
//! `cpu` (default-on), `wgpu`, `cuda`, `rocm`, `metal`. cuda and metal arms
//! ship as compile-only this phase — see
//! `.planning/notes/cuda-metal-verification-gap.md` for the verification
//! risk-accept that applies to those modules.

pub mod backend;
mod batch_pilot;
pub mod capability;
pub mod executor;
#[path = "kernels/mod.rs"]
pub mod kernels;
pub mod math;
pub mod resident_cache;
#[cfg(feature = "wgpu")]
pub mod runtime_bootstrap;
pub mod specialization;
pub mod transfer;
pub mod transform;

pub use backend::{ResolvedBackend, compiled_backends};
pub use batch_pilot::{EriSsssInput, OverlapSsInput, PilotOutputArenaStats};
pub use capability::{
    CapabilityReason, WgpuCapabilitySnapshot, WgpuPreflightReport, capability_fingerprint,
};
pub use executor::{
    BackendCache, CUBECL_RUNTIME_PROFILE, CubeClExecutor, check_shader_f64_in_features,
};
pub use resident_cache::{DeviceResidentCache, ResidentCache};
#[cfg(feature = "wgpu")]
pub use runtime_bootstrap::bootstrap_wgpu_runtime;
pub use specialization::{ComponentRank, SpecializationKey};
pub use transfer::{TransferPlan, TransferWorkspaceBuffers};

#[cfg(test)]
mod tests {
    #[test]
    fn exports_compile() {
        #[allow(unused_imports)]
        use super::{
            BackendCache, CUBECL_RUNTIME_PROFILE, CapabilityReason, CubeClExecutor,
            DeviceResidentCache, ResolvedBackend, TransferPlan, TransferWorkspaceBuffers,
            WgpuCapabilitySnapshot, WgpuPreflightReport, backend, capability,
            capability_fingerprint, check_shader_f64_in_features, compiled_backends, executor,
            kernels, resident_cache, specialization, transfer, transform,
        };
        #[cfg(feature = "wgpu")]
        #[allow(unused_imports)]
        use super::{bootstrap_wgpu_runtime, runtime_bootstrap};
    }
}
