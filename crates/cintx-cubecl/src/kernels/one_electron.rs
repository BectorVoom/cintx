use crate::specialization::SpecializationKey;
use crate::transfer::TransferPlan;
use cintx_core::cintxRsError;
use cintx_runtime::{ExecutionPlan, ExecutionStats};

pub fn launch_one_electron(
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    transfer: &TransferPlan,
) -> Result<ExecutionStats, cintxRsError> {
    if specialization.canonical_family() != "1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_1e",
            detail: format!(
                "canonical_family mismatch: expected 1e, got {}",
                specialization.canonical_family()
            ),
        });
    }
    transfer.ensure_output_contract()?;
    let staging = transfer.stage_output_buffer()?;

    Ok(ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes: transfer.workspace_bytes,
        chunk_count: 1,
        planned_batches: 1,
        transfer_bytes: transfer.transfer_bytes,
        not0: i32::from(!staging.is_empty()),
        fallback_reason: plan.workspace.fallback_reason,
    })
}
