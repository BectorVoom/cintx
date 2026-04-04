use crate::dispatch::{
    BackendExecutor, DispatchDecision, ExecutionIo, OutputOwnership,
};
use crate::metrics::{ExecutionStats, RunMetrics};
use crate::options::ExecutionOptions;
use crate::scheduler::schedule_chunks;
use crate::validator::{ValidatedShellTuple, validate_shell_tuple};
use crate::workspace::{
    ChunkInfo, ChunkPlanner, DEFAULT_ALIGNMENT_BYTES, WorkspaceAllocator, WorkspaceQuery,
    WorkspaceRequest,
};
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple, cintxRsError};
use cintx_ops::resolver::{OperatorDescriptor, Resolver, ResolverError};
use tracing::{debug, info_span};

#[cfg(test)]
use crate::dispatch::{DispatchFamily, WorkspaceBytes};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputLayoutMetadata {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub staging_elements: usize,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan<'a> {
    pub basis: &'a BasisSet,
    pub descriptor: &'a OperatorDescriptor,
    pub representation: Representation,
    pub shells: ValidatedShellTuple,
    pub workspace: &'a WorkspaceQuery,
    pub dispatch: DispatchDecision,
    pub component_count: usize,
    pub output_layout: OutputLayoutMetadata,
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

        let dispatch = DispatchDecision::from_manifest_family(descriptor.family())?;
        let component_count = parse_component_multiplier(descriptor.entry.component_rank)?;
        let output_layout = build_output_layout(&shells, rep, component_count)?;

        Ok(Self {
            basis,
            descriptor,
            representation: rep,
            shells,
            workspace,
            dispatch,
            component_count,
            output_layout,
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
        profile = opts.profile_label.unwrap_or("default"),
        backend = ?opts.backend_intent.backend,
        backend_selector = %opts.backend_intent.selector,
        capability_fingerprint = opts.backend_capability_token.capability_fingerprint,
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
        backend_intent: opts.backend_intent.clone(),
        backend_capability_token: opts.backend_capability_token.clone(),
    })
}

pub fn evaluate(
    plan: ExecutionPlan<'_>,
    opts: &ExecutionOptions,
    allocator: &mut dyn WorkspaceAllocator,
    executor: &dyn BackendExecutor,
) -> Result<ExecutionStats, cintxRsError> {
    let _parent = opts.trace_span.as_ref().map(tracing::Span::enter);
    let span = info_span!(
        "evaluate",
        operator = plan.descriptor.operator_name(),
        family = plan.descriptor.family(),
        representation = %plan.representation,
        dispatch_family = ?plan.dispatch.family,
        profile = opts.profile_label.unwrap_or("default")
    );
    let _entered = span.enter();

    if !plan.workspace.planning_matches(opts) {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "evaluate",
            detail: "backend contract drift detected: execution options do not match the query_workspace contract (memory_limit, chunk_size_override, backend_intent, or backend_capability_token changed)".to_owned(),
        });
    }

    if plan.dispatch.final_write != OutputOwnership::CompatFinalWrite {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "evaluate",
            detail: "planner dispatch must preserve CompatFinalWrite ownership".to_owned(),
        });
    }
    plan.dispatch.ensure_output_contract()?;

    if !executor.supports(&plan) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "{}/{}/{}",
                plan.descriptor.family(),
                plan.descriptor.operator_name(),
                plan.representation
            ),
        });
    }

    let backend_workspace = executor.query_workspace(&plan)?.get();
    if backend_workspace > plan.workspace.bytes {
        return Err(cintxRsError::MemoryLimitExceeded {
            requested: backend_workspace,
            limit: plan.workspace.bytes,
        });
    }

    let schedule = schedule_chunks(plan.workspace);
    let mut metrics = RunMetrics::default();

    for chunk in schedule.chunks() {
        debug!(
            chunk_index = chunk.index,
            chunk_bytes = chunk.bytes,
            chunk_work_units = chunk.work_unit_count,
            dispatch_family = ?plan.dispatch.family,
            output_contract = "staging-only",
            fallback_reason = plan.workspace.fallback_reason.unwrap_or("none"),
            "executing runtime-owned scheduled chunk"
        );

        let mut workspace = allocator.try_alloc(chunk.bytes, plan.workspace.alignment)?;
        let mut staging = try_alloc_staging(staging_elements_for_chunk(&plan, chunk)?)?;

        {
            let mut io =
                ExecutionIo::new(chunk, staging.as_mut_slice(), &mut workspace, plan.dispatch)?;
            io.ensure_output_contract()?;

            let backend_stats = executor.execute(&plan, &mut io)?;
            io.ensure_output_contract()?;

            metrics.observe_transfer_bytes(io.transfer_bytes());
            metrics.observe_not0(io.not0());
            metrics.merge_backend_stats(&backend_stats);
        }

        metrics.observe_chunk(chunk, workspace.len());
        allocator.release(workspace);
    }

    Ok(metrics.finish(plan.workspace))
}

fn build_output_layout(
    shells: &ValidatedShellTuple,
    representation: Representation,
    component_count: usize,
) -> Result<OutputLayoutMetadata, cintxRsError> {
    let extents: Vec<usize> = shells
        .as_slice()
        .iter()
        .map(|shell| shell.ao_per_shell())
        .collect();
    let base_elements = extents
        .iter()
        .try_fold(1usize, |acc, extent| acc.checked_mul(*extent))
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "layout",
            detail: "output extent product overflowed usize".to_owned(),
        })?;
    let complex_multiplier = if matches!(representation, Representation::Spinor) {
        2usize
    } else {
        1usize
    };
    let staging_elements = base_elements
        .checked_mul(component_count)
        .and_then(|value| value.checked_mul(complex_multiplier))
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "layout",
            detail: "staging element count overflowed usize".to_owned(),
        })?;

    Ok(OutputLayoutMetadata {
        extents,
        component_axis_leading: true,
        complex_interleaved: complex_multiplier == 2,
        staging_elements,
    })
}

fn staging_elements_for_chunk(
    plan: &ExecutionPlan<'_>,
    chunk: &ChunkInfo,
) -> Result<usize, cintxRsError> {
    let total_units = plan.workspace.work_units.max(1);
    let start = chunk.work_unit_start.min(total_units);
    let end = chunk
        .work_unit_start
        .checked_add(chunk.work_unit_count)
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "layout",
            detail: "chunk work unit range overflowed usize".to_owned(),
        })?
        .min(total_units);

    let prefix = plan.output_layout.staging_elements.saturating_mul(start) / total_units;
    let suffix = plan.output_layout.staging_elements.saturating_mul(end) / total_units;
    Ok(suffix.saturating_sub(prefix).max(1))
}

fn try_alloc_staging(elements: usize) -> Result<Vec<f64>, cintxRsError> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
    let mut staging = Vec::new();
    staging
        .try_reserve_exact(elements)
        .map_err(|_| cintxRsError::HostAllocationFailed { bytes })?;
    staging.resize(elements, 0.0);
    Ok(staging)
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

    #[derive(Debug, Default)]
    struct MockBackend {
        supports: bool,
    }

    impl BackendExecutor for MockBackend {
        fn supports(&self, _plan: &ExecutionPlan<'_>) -> bool {
            self.supports
        }

        fn query_workspace(
            &self,
            plan: &ExecutionPlan<'_>,
        ) -> Result<WorkspaceBytes, cintxRsError> {
            Ok(WorkspaceBytes(plan.workspace.bytes))
        }

        fn execute(
            &self,
            plan: &ExecutionPlan<'_>,
            io: &mut ExecutionIo<'_>,
        ) -> Result<ExecutionStats, cintxRsError> {
            let transfer_bytes = {
                let staging = io.staging_output();
                if let Some(first) = staging.first_mut() {
                    *first = 1.0;
                }
                staging.len().saturating_mul(std::mem::size_of::<f64>())
            };
            let not0 = io.chunk().work_unit_count as i32;
            let peak_workspace_bytes = io.workspace().len();

            io.record_transfer_bytes(transfer_bytes);
            io.record_not0(not0);

            Ok(ExecutionStats {
                workspace_bytes: plan.workspace.bytes,
                required_workspace_bytes: plan.workspace.required_bytes,
                peak_workspace_bytes,
                chunk_count: 1,
                planned_batches: io.chunk().work_unit_count,
                transfer_bytes,
                not0,
                fallback_reason: plan.workspace.fallback_reason,
            })
        }
    }

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
        assert_eq!(plan.dispatch.family, DispatchFamily::OneElectron);
        assert_eq!(plan.dispatch.final_write, OutputOwnership::CompatFinalWrite);

        let mut allocator = HostWorkspaceAllocator::default();
        let backend = MockBackend { supports: true };
        let stats =
            evaluate(plan, &opts, &mut allocator, &backend).expect("evaluation should succeed");

        assert_eq!(stats.workspace_bytes, query.bytes);
        assert_eq!(stats.chunk_count, query.chunk_count);
        assert_eq!(stats.fallback_reason, Some("memory_limit"));
        assert_eq!(stats.planned_batches, query.work_units);
        assert!(stats.transfer_bytes > 0);
        assert!(stats.not0 > 0);
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
        let backend = MockBackend { supports: true };
        let err = evaluate(plan, &eval_opts, &mut allocator, &backend).unwrap_err();

        assert!(matches!(
            err,
            cintxRsError::ChunkPlanFailed {
                from: "evaluate",
                ..
            }
        ));
    }

    #[test]
    fn evaluate_rejects_dispatch_paths_that_skip_compat_final_write() {
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
        let mut plan = ExecutionPlan::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &query,
        )
        .expect("plan should build");
        plan.dispatch.final_write = OutputOwnership::BackendStagingOnly;

        let mut allocator = HostWorkspaceAllocator::default();
        let backend = MockBackend { supports: true };
        let err = evaluate(plan, &opts, &mut allocator, &backend).unwrap_err();

        assert!(matches!(
            err,
            cintxRsError::ChunkPlanFailed {
                from: "evaluate",
                ..
            }
        ));
    }

    #[test]
    fn query_workspace_records_backend_contract_metadata() {
        use crate::options::{BackendCapabilityToken, BackendIntent, BackendKind};

        let (basis, shells) = sample_basis(Representation::Cart);
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            backend_intent: BackendIntent {
                backend: BackendKind::Wgpu,
                selector: "device:0".to_owned(),
            },
            backend_capability_token: BackendCapabilityToken {
                adapter_name: "Test GPU".to_owned(),
                backend_api: "wgpu".to_owned(),
                capability_fingerprint: 12345,
            },
            ..ExecutionOptions::default()
        };

        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &opts,
        )
        .expect("workspace query should succeed");

        assert_eq!(
            query.backend_intent, opts.backend_intent,
            "query must carry backend_intent from opts"
        );
        assert_eq!(
            query.backend_capability_token, opts.backend_capability_token,
            "query must carry backend_capability_token from opts"
        );
    }

    #[test]
    fn evaluate_rejects_query_workspace_backend_intent_drift() {
        use crate::options::{BackendIntent, BackendKind};

        let (basis, shells) = sample_basis(Representation::Cart);
        let query_opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            backend_intent: BackendIntent {
                backend: BackendKind::Wgpu,
                selector: "auto".to_owned(),
            },
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

        // Drift: different backend kind at evaluate time
        let eval_opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            backend_intent: BackendIntent {
                backend: BackendKind::Cpu,
                selector: "auto".to_owned(),
            },
            ..ExecutionOptions::default()
        };

        let mut allocator = HostWorkspaceAllocator::default();
        let backend = MockBackend { supports: true };
        let err = evaluate(plan, &eval_opts, &mut allocator, &backend).unwrap_err();

        assert!(
            matches!(
                err,
                cintxRsError::ChunkPlanFailed {
                    from: "evaluate",
                    ..
                }
            ),
            "backend intent drift must fail evaluate with ChunkPlanFailed"
        );
    }

    #[test]
    fn evaluate_rejects_query_workspace_backend_capability_token_drift() {
        use crate::options::BackendCapabilityToken;

        let (basis, shells) = sample_basis(Representation::Cart);
        let query_opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            backend_capability_token: BackendCapabilityToken {
                adapter_name: "GPU A".to_owned(),
                backend_api: "wgpu".to_owned(),
                capability_fingerprint: 100,
            },
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

        // Drift: different capability fingerprint at evaluate time
        let eval_opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            backend_capability_token: BackendCapabilityToken {
                adapter_name: "GPU A".to_owned(),
                backend_api: "wgpu".to_owned(),
                capability_fingerprint: 999,
            },
            ..ExecutionOptions::default()
        };

        let mut allocator = HostWorkspaceAllocator::default();
        let backend = MockBackend { supports: true };
        let err = evaluate(plan, &eval_opts, &mut allocator, &backend).unwrap_err();

        assert!(
            matches!(
                err,
                cintxRsError::ChunkPlanFailed {
                    from: "evaluate",
                    ..
                }
            ),
            "capability token drift must fail evaluate with ChunkPlanFailed"
        );
    }
}
