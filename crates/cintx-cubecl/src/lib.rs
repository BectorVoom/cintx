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
pub mod device_rys_ceiling;
pub mod executor;
#[path = "kernels/mod.rs"]
pub mod kernels;
pub mod math;
pub mod plane;
pub mod precision_ratio;
pub mod resident_cache;
#[cfg(feature = "wgpu")]
pub mod runtime_bootstrap;
pub mod shared_memory;
pub mod specialization;
pub mod transfer;
pub mod transform;

pub use backend::{ResolvedBackend, compiled_backends};
pub use batch_pilot::{EriSsssInput, OverlapSsInput, PilotOutputArenaStats};
pub use capability::{
    CapabilityReason, WgpuCapabilitySnapshot, WgpuPreflightReport, capability_fingerprint,
};
pub use device_rys_ceiling::{
    BASE_DEVICE_NROOTS, EXTENDED_DEVICE_NROOTS, FmaProbeResult, ProbeTarget, RysFamily,
    device_nroots_ceiling, fma_fusion_verified, probe_fma_fusion,
};
pub use executor::{
    BackendCache, CUBECL_RUNTIME_PROFILE, CubeClExecutor, check_shader_f64_in_features,
};
pub use kernels::center_2c2e::{
    TwoC2eBatchOutput, evaluate_2c2e_pair_batch, evaluate_2c2e_pair_batch_resident,
};
pub use kernels::center_3c1e::{
    ThreeC1eBatchOutput, evaluate_3c1e_triple_batch, evaluate_3c1e_triple_batch_resident,
};
pub use kernels::center_3c2e::{
    ThreeC2eBatchOutput, ThreeC2eDerivBatchOutput, ThreeC2eDerivFamily,
    evaluate_3c2e_deriv_triple_batch, evaluate_3c2e_deriv_triple_batch_resident,
    evaluate_3c2e_triple_batch, evaluate_3c2e_triple_batch_resident,
    evaluate_3c2e_triple_batch_resident_with, evaluate_3c2e_triple_batch_with,
};
pub use kernels::one_electron::{
    BatchAtom, OneEBatchOutput, OneEDerivBatchOutput, OneEDerivOperator, OneEOperator,
    evaluate_1e_deriv_pair_batch, evaluate_1e_deriv_pair_batch_resident, evaluate_1e_pair_batch,
    evaluate_1e_pair_batch_resident, evaluate_1e_pair_batch_resident_with,
    evaluate_1e_pair_batch_with,
};
pub use kernels::two_electron::{
    BatchExecutionStats as TwoEBatchStats, BatchOptions, BatchShell, ResidentBasis,
    ResidentTwoEBasis, TwoEBatchOptions, TwoEBatchOutput, evaluate_2e_quartet_batch,
    evaluate_2e_quartet_batch_resident, evaluate_2e_quartet_batch_with,
};
pub use plane::{
    DEFAULT_PLANE_DIM, STANDARD_PLANE_ALIGNED_CUBE_DIM, backend_plane_cube_dim,
    cooperative_cube_dim, cube_count_1d, cube_count_2d, cube_count_3d, linear_grid_cube_count,
    occupancy_launch_geometry, plane_aligned_cube_dim, plane_aligned_cube_dim_2d,
    plane_aligned_cube_dim_3d, plane_cooperative_launch_geometry, planes_per_cube, runtime_is_cpu,
    single_cube_count, standard_plane_cube_dim, tiled_grid_cube_count_2d, tiled_grid_cube_count_3d,
};
pub use precision_ratio::{PrecisionRatio, measure_precision_ratio};
pub use resident_cache::{DeviceResidentCache, ResidentCache};
#[cfg(feature = "wgpu")]
pub use runtime_bootstrap::bootstrap_wgpu_runtime;
pub use shared_memory::{
    CapacityClass, FallbackReason, SharedLayout, SharedMemoryMetrics, SharedVariant,
    calc_1e_layout, calc_2c2e_layout, calc_2e_layout, calc_3c1e_layout, calc_3c2e_layout,
    calc_4c1e_layout, calc_ecp_type2_layout, calc_f12_layout, calc_math_layout, calc_sigma_layout,
    generate_layout_catalog, validate_shared_layout_bounds,
};
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
            kernels, resident_cache, shared_memory, specialization, transfer, transform,
        };
        #[cfg(feature = "wgpu")]
        #[allow(unused_imports)]
        use super::{bootstrap_wgpu_runtime, runtime_bootstrap};
    }
}
