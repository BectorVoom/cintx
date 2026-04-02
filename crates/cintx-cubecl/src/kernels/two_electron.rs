use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats};

pub fn launch_two_electron(
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "2e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_2e",
            detail: format!(
                "canonical_family mismatch for 2e launch: {}",
                specialization.canonical_family()
            ),
        });
    }
    // Stub: staging remains zeros; real kernel implementation comes in Phase 9/10.
    // Suppress unused variable warning until real kernel uses backend.
    let _ = backend;

    let staging_bytes = staging.len() * std::mem::size_of::<f64>();
    Ok(ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: staging_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: staging_bytes,
        not0: i32::from(!staging.is_empty()),
        fallback_reason: plan.workspace.fallback_reason,
    })
}
