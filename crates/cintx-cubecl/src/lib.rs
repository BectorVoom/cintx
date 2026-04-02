//! CubeCL backend foundations and family launch plumbing.

pub mod backend;
pub mod capability;
pub mod executor;
#[path = "kernels/mod.rs"]
pub mod kernels;
pub mod resident_cache;
pub mod runtime_bootstrap;
pub mod specialization;
pub mod transfer;
pub mod transform;

pub use backend::ResolvedBackend;
pub use capability::{
    CapabilityReason, WgpuCapabilitySnapshot, WgpuPreflightReport, capability_fingerprint,
};
pub use executor::{CUBECL_RUNTIME_PROFILE, CubeClExecutor};
pub use resident_cache::{DeviceResidentCache, ResidentCache};
pub use runtime_bootstrap::bootstrap_wgpu_runtime;
pub use specialization::{ComponentRank, SpecializationKey};
pub use transfer::{TransferPlan, TransferWorkspaceBuffers};

#[cfg(test)]
mod tests {
    #[test]
    fn exports_compile() {
        #[allow(unused_imports)]
        use super::{
            CUBECL_RUNTIME_PROFILE, CapabilityReason, CubeClExecutor, DeviceResidentCache,
            ResolvedBackend, TransferPlan, TransferWorkspaceBuffers, WgpuCapabilitySnapshot,
            WgpuPreflightReport, backend, bootstrap_wgpu_runtime, capability,
            capability_fingerprint, executor, kernels, resident_cache, runtime_bootstrap,
            specialization, transfer, transform,
        };
    }
}
