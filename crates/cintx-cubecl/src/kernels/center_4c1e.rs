use crate::backend::ResolvedBackend;
use crate::specialization::SpecializationKey;
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
    backend: &ResolvedBackend,
    plan: &ExecutionPlan<'_>,
    specialization: &SpecializationKey,
    staging: &mut [f64],
) -> Result<ExecutionStats, cintxRsError> {
    ensure_validated_4c1e(plan, specialization)?;
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
