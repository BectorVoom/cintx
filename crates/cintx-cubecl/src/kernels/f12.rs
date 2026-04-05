//! F12/STG/YP kernel entry points.
//!
//! This is a stub for Plan 13-01. The full kernel implementation is in Plan 13-02.
//! Currently returns UnsupportedApi so callers get a clear error instead of a panic.

use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats};

/// Launch stub for F12/STG/YP family integrals.
///
/// Returns `UnsupportedApi` until Plan 13-02 implements the real kernel.
pub fn launch_f12(
    _backend: &ResolvedBackend,
    _plan: &ExecutionPlan<'_>,
    _specialization: &SpecializationKey,
    _staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    Err(cintxRsError::UnsupportedApi {
        requested: "f12 kernel not yet implemented (Plan 13-02)".to_owned(),
    })
}
