//! Stub launch functions for unstable-source API families.
//!
//! Phase 14 Wave 1 stubs — all 5 unstable-source dispatch families return
//! `UnsupportedApi` until Wave 2 plans implement the real kernels.
//!
//! Families covered:
//!   - origi: origin-displaced r^n one-electron integrals (cint1e_a.c)
//!   - grids: grid-point integrals with NGRIDS env parameter (cint1e_grids.c)
//!   - breit: Breit spinor-only two-electron integrals (breit.c)
//!   - origk: origin-k-displaced three-center one-electron integrals (cint3c1e_a.c)
//!   - ssc: spin-spin contact three-center two-electron integral (cint3c2e.c)

use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats};

/// Stub for origi family (int1e_r2_origi, int1e_r4_origi, ip2 derivatives).
/// Implementation pending in Phase 14 Wave 2.
pub fn launch_origi(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "origi: stub — implementation pending".to_owned(),
    })
}

/// Stub for grids family (int1e_grids and derivative variants).
/// Implementation pending in Phase 14 Wave 2.
pub fn launch_grids(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "grids: stub — implementation pending".to_owned(),
    })
}

/// Stub for breit family (int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor).
/// Implementation pending in Phase 14 Wave 2.
pub fn launch_breit(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "breit: stub — implementation pending".to_owned(),
    })
}

/// Stub for origk family (int3c1e_r2/r4/r6_origk and ip1 derivatives).
/// Implementation pending in Phase 14 Wave 2.
pub fn launch_origk(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "origk: stub — implementation pending".to_owned(),
    })
}

/// Stub for ssc family (int3c2e_sph_ssc).
/// Implementation pending in Phase 14 Wave 2.
pub fn launch_ssc(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _spec: &SpecializationKey,
    _output: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "ssc: stub — implementation pending".to_owned(),
    })
}
