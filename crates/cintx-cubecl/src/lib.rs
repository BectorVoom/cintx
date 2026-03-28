//! CubeCL backend foundations and family launch plumbing.

pub mod executor;
#[path = "kernels/mod.rs"]
pub mod kernels;
pub mod resident_cache;
pub mod specialization;
pub mod transfer;
pub mod transform;

pub use executor::{CUBECL_RUNTIME_PROFILE, CubeClExecutor};
pub use resident_cache::{DeviceResidentCache, ResidentCache};
pub use specialization::{ComponentRank, SpecializationKey};
pub use transfer::{TransferPlan, TransferWorkspaceBuffers};

#[cfg(test)]
mod tests {
    #[test]
    fn exports_compile() {
        #[allow(unused_imports)]
        use super::{
            CUBECL_RUNTIME_PROFILE, CubeClExecutor, DeviceResidentCache, TransferPlan,
            TransferWorkspaceBuffers, executor, kernels, resident_cache, specialization, transfer,
            transform,
        };
    }
}
