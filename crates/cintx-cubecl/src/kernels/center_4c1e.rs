use crate::specialization::SpecializationKey;
use crate::transfer::TransferPlan;
use cintx_core::{Representation, cintxRsError};
use cintx_runtime::{ExecutionPlan, ExecutionStats};

fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("outside Validated4C1E ({reason})"),
    }
}

fn ensure_validated_4c1e(
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
) -> Result<(), cintxRsError> {
    if specialization.canonical_family() != "4c1e" {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_center_4c1e",
            detail: format!(
                "canonical_family mismatch for 4c1e launch: {}",
                specialization.canonical_family()
            ),
        });
    }

    if !matches!(
        plan.representation,
        Representation::Cart | Representation::Spheric
    ) {
        return Err(validated_4c1e_error("representation must be cart/sph"));
    }
    if !plan.descriptor.entry.component_rank.trim().is_empty()
        && plan.descriptor.entry.component_rank != "scalar"
    {
        return Err(validated_4c1e_error("component rank must be scalar"));
    }
    if plan
        .shells
        .as_slice()
        .iter()
        .any(|shell| shell.ang_momentum > 4)
    {
        return Err(validated_4c1e_error("max(l)>4"));
    }

    Ok(())
}

pub fn launch_center_4c1e(
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    transfer: &TransferPlan,
) -> Result<ExecutionStats, cintxRsError> {
    ensure_validated_4c1e(plan, specialization)?;
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
