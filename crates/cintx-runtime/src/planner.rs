use crate::options::ExecutionOptions;
use crate::validator::{ValidatedShellTuple, validate_shell_tuple};
use crate::workspace::{
    ChunkPlanner, DEFAULT_ALIGNMENT_BYTES, WorkspaceAllocator, WorkspaceQuery, WorkspaceRequest,
};
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple, cintxRsError};
use cintx_ops::resolver::{OperatorDescriptor, Resolver, ResolverError};
use tracing::{debug, info_span};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionStats {
    pub workspace_bytes: usize,
    pub required_workspace_bytes: usize,
    pub peak_workspace_bytes: usize,
    pub chunk_count: usize,
    pub planned_batches: usize,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan<'a> {
    pub basis: &'a BasisSet,
    pub descriptor: &'a OperatorDescriptor,
    pub representation: Representation,
    pub shells: ValidatedShellTuple,
    pub workspace: &'a WorkspaceQuery,
}

impl<'a> ExecutionPlan<'a> {
    pub fn new(
        op: OperatorId,
        rep: Representation,
        basis: &'a BasisSet,
        shells: ShellTuple,
        workspace: &'a WorkspaceQuery,
    ) -> Result<Self, cintxRsError> {
        let descriptor = Resolver::descriptor(op).map_err(|err| map_resolver_error(op, err))?;
        let shells = validate_shell_tuple(descriptor, rep, basis, &shells)?;
        let expected_request = estimate_workspace_request(descriptor, basis, &shells)?;
        if workspace.request() != expected_request {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "planner",
                detail: "workspace query does not match the validated shell tuple".to_owned(),
            });
        }
        Ok(Self {
            basis,
            descriptor,
            representation: rep,
            shells,
            workspace,
        })
    }
}

pub fn query_workspace(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    opts: &ExecutionOptions,
) -> Result<WorkspaceQuery, cintxRsError> {
    let _parent = opts.trace_span.as_ref().map(tracing::Span::enter);
    let span = info_span!(
        "query_workspace",
        operator = %op,
        representation = %rep,
        profile = opts.profile_label.unwrap_or("default")
    );
    let _entered = span.enter();

    let descriptor = Resolver::descriptor(op).map_err(|err| map_resolver_error(op, err))?;
    let validated = validate_shell_tuple(descriptor, rep, basis, &shells)?;
    let request = estimate_workspace_request(descriptor, basis, &validated)?;
    let chunk_plan = ChunkPlanner::from_options(opts).plan(&request)?;
    let chunk_count = chunk_plan.chunks.len();
    let bytes = chunk_plan
        .chunks
        .iter()
        .map(|chunk| chunk.bytes)
        .max()
        .unwrap_or(request.required_bytes);
    let fallback_reason = chunk_plan.fallback_reason;
    let chunks = chunk_plan.chunks;

    debug!(
        family = descriptor.family(),
        operator_name = descriptor.operator_name(),
        required_bytes = request.required_bytes,
        peak_chunk_bytes = bytes,
        chunk_count,
        fallback_reason = fallback_reason.unwrap_or("none"),
        "workspace query planned"
    );

    Ok(WorkspaceQuery {
        bytes,
        alignment: request.alignment,
        required_bytes: request.required_bytes,
        chunk_count,
        work_units: request.work_units,
        min_chunk_bytes: request.min_chunk_bytes,
        fallback_reason,
        chunks,
        memory_limit_bytes: opts.memory_limit_bytes,
        chunk_size_override: opts.chunk_size_override,
    })
}

pub fn evaluate(
    plan: ExecutionPlan<'_>,
    opts: &ExecutionOptions,
    allocator: &mut dyn WorkspaceAllocator,
) -> Result<ExecutionStats, cintxRsError> {
    let _parent = opts.trace_span.as_ref().map(tracing::Span::enter);
    let span = info_span!(
        "evaluate",
        operator = plan.descriptor.operator_name(),
        family = plan.descriptor.family(),
        representation = %plan.representation,
        profile = opts.profile_label.unwrap_or("default")
    );
    let _entered = span.enter();

    if !plan.workspace.planning_matches(opts) {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "evaluate",
            detail: "execution options do not match the query_workspace contract".to_owned(),
        });
    }

    let mut peak_workspace_bytes = 0usize;

    for chunk in &plan.workspace.chunks {
        debug!(
            chunk_index = chunk.index,
            chunk_bytes = chunk.bytes,
            chunk_work_units = chunk.work_unit_count,
            fallback_reason = plan.workspace.fallback_reason.unwrap_or("none"),
            "executing planned chunk"
        );
        let buffer = allocator.try_alloc(chunk.bytes, plan.workspace.alignment)?;
        peak_workspace_bytes = peak_workspace_bytes.max(buffer.len());
        allocator.release(buffer);
    }

    Ok(ExecutionStats {
        workspace_bytes: plan.workspace.bytes,
        required_workspace_bytes: plan.workspace.required_bytes,
        peak_workspace_bytes,
        chunk_count: plan.workspace.chunks.len(),
        planned_batches: plan
            .workspace
            .chunks
            .iter()
            .map(|chunk| chunk.work_unit_count)
            .sum(),
        fallback_reason: plan.workspace.fallback_reason,
    })
}

fn estimate_workspace_request(
    descriptor: &OperatorDescriptor,
    basis: &BasisSet,
    shells: &ValidatedShellTuple,
) -> Result<WorkspaceRequest, cintxRsError> {
    let component_multiplier = parse_component_multiplier(descriptor.entry.component_rank)?;
    let output_bytes = shells
        .output_elements()
        .checked_mul(component_multiplier)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "workspace_estimator",
            detail: "output byte estimate overflowed usize".to_owned(),
        })?;
    let basis_bytes = basis
        .meta()
        .total_ao
        .checked_mul(descriptor.entry.arity as usize)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>() * 2))
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "workspace_estimator",
            detail: "basis byte estimate overflowed usize".to_owned(),
        })?;
    let shell_bytes = shells
        .total_ao()
        .checked_mul(std::mem::size_of::<f64>() * 4)
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "workspace_estimator",
            detail: "shell byte estimate overflowed usize".to_owned(),
        })?;
    let required_bytes = output_bytes
        .checked_add(basis_bytes)
        .and_then(|value| value.checked_add(shell_bytes))
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "workspace_estimator",
            detail: "workspace byte estimate overflowed usize".to_owned(),
        })?;
    let work_units = shells.work_units();
    let min_chunk_bytes = required_bytes
        .div_ceil(work_units)
        .max(DEFAULT_ALIGNMENT_BYTES);

    Ok(WorkspaceRequest {
        required_bytes,
        alignment: DEFAULT_ALIGNMENT_BYTES,
        work_units,
        min_chunk_bytes,
    })
}

fn parse_component_multiplier(component_rank: &str) -> Result<usize, cintxRsError> {
    let trimmed = component_rank.trim();
    if trimmed.is_empty() {
        return Ok(1);
    }

    let mut count = 1usize;
    let mut found = false;
    for segment in trimmed.split(|ch: char| !ch.is_ascii_digit()) {
        if segment.is_empty() {
            continue;
        }
        let value = segment
            .parse::<usize>()
            .map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "component_rank",
                detail: format!("failed to parse component rank {component_rank:?}"),
            })?;
        count = count
            .checked_mul(value)
            .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                from: "component_rank",
                detail: format!("component rank overflow for {component_rank:?}"),
            })?;
        found = true;
    }

    if found {
        Ok(count)
    } else {
        Err(cintxRsError::ChunkPlanFailed {
            from: "component_rank",
            detail: format!("component rank {component_rank:?} has no numeric dimensions"),
        })
    }
}

fn map_resolver_error(op: OperatorId, err: ResolverError) -> cintxRsError {
    match err {
        ResolverError::MissingOperatorId(_) => cintxRsError::UnsupportedApi {
            requested: op.to_string(),
        },
        ResolverError::UnsupportedRepresentation {
            family,
            operator,
            representation,
        } => cintxRsError::UnsupportedRepresentation {
            operator: format!("{family}/{operator}"),
            representation,
        },
        other => cintxRsError::ChunkPlanFailed {
            from: "resolver",
            detail: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::HostWorkspaceAllocator;
    use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell};
    use std::sync::Arc;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis(rep: Representation) -> (BasisSet, ShellTuple) {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let shell_a = Arc::new(
            Shell::try_new(0, 1, 1, 2, 0, rep, arc_f64(&[1.0]), arc_f64(&[1.0, 0.5])).unwrap(),
        );
        let shell_b = Arc::new(
            Shell::try_new(0, 1, 1, 2, 0, rep, arc_f64(&[0.8]), arc_f64(&[0.7, 0.3])).unwrap(),
        );

        let basis = BasisSet::try_new(
            atoms,
            Arc::from(vec![shell_a.clone(), shell_b.clone()].into_boxed_slice()),
        )
        .unwrap();
        let shells = ShellTuple::try_from_iter([shell_a, shell_b]).unwrap();
        (basis, shells)
    }

    #[test]
    fn query_workspace_honors_memory_limit() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            ..ExecutionOptions::default()
        };

        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        )
        .expect("workspace query should succeed");

        assert!(query.required_bytes > query.bytes);
        assert!(query.bytes <= 192);
        assert!(query.chunk_count > 1);

        let plan = ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .expect("plan should build");
        let mut allocator = HostWorkspaceAllocator::default();
        let stats = evaluate(plan, &opts, &mut allocator).expect("evaluation should succeed");

        assert_eq!(stats.workspace_bytes, query.bytes);
        assert_eq!(stats.chunk_count, query.chunk_count);
        assert_eq!(stats.fallback_reason, Some("memory_limit"));
        assert_eq!(stats.planned_batches, query.work_units);
        assert!(allocator.allocations() >= query.chunk_count);
    }

    #[test]
    fn query_workspace_reports_unreachable_limit() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(32),
            ..ExecutionOptions::default()
        };

        let err = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &opts,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            cintxRsError::MemoryLimitExceeded {
                requested: _,
                limit: 32,
            }
        ));
    }

    #[test]
    fn evaluate_rejects_query_workspace_contract_drift() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let query_opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            ..ExecutionOptions::default()
        };
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            &query_opts,
        )
        .expect("workspace query should succeed");
        let plan = ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .expect("plan should build");

        let eval_opts = ExecutionOptions {
            memory_limit_bytes: Some(256),
            ..ExecutionOptions::default()
        };
        let mut allocator = HostWorkspaceAllocator::default();
        let err = evaluate(plan, &eval_opts, &mut allocator).unwrap_err();

        assert!(matches!(
            err,
            cintxRsError::ChunkPlanFailed {
                from: "evaluate",
                ..
            }
        ));
    }
}
