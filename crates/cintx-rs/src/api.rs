//! Safe Rust facade scaffolding for query/evaluate session flows.

use crate::error::FacadeError;
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple, cintxRsError};
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionOptions, ExecutionPlan, ExecutionStats,
    HostWorkspaceAllocator, OutputOwnership, WorkspaceBytes,
    WorkspaceQuery as RuntimeWorkspaceQuery, evaluate as runtime_evaluate,
    query_workspace as runtime_query_workspace,
};
use std::mem::size_of;

const CUBECL_RUNTIME_PROFILE: &str = "cpu";

/// Typed safe request object that keeps `query_workspace()` and `evaluate()` connected.
#[derive(Clone, Debug)]
pub struct SessionRequest<'basis> {
    operator: OperatorId,
    representation: Representation,
    basis: &'basis BasisSet,
    shells: ShellTuple,
    options: ExecutionOptions,
}

impl<'basis> SessionRequest<'basis> {
    pub fn new(
        operator: OperatorId,
        representation: Representation,
        basis: &'basis BasisSet,
        shells: ShellTuple,
        options: ExecutionOptions,
    ) -> Self {
        Self {
            operator,
            representation,
            basis,
            shells,
            options,
        }
    }

    pub fn operator(&self) -> OperatorId {
        self.operator
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn basis(&self) -> &'basis BasisSet {
        self.basis
    }

    pub fn shells(&self) -> &ShellTuple {
        &self.shells
    }

    pub fn options(&self) -> &ExecutionOptions {
        &self.options
    }

    pub fn query_workspace(&self) -> Result<SessionQuery<'basis>, FacadeError> {
        let runtime_workspace = runtime_query_workspace(
            self.operator,
            self.representation,
            self.basis,
            self.shells.clone(),
            &self.options,
        )
        .map_err(FacadeError::from)?;

        let workspace = WorkspacePlan::from_runtime(self, &runtime_workspace);
        Ok(SessionQuery {
            request: self.clone(),
            workspace,
            runtime_workspace,
        })
    }
}

/// Result of `query_workspace()` that carries the validated request metadata forward to evaluate.
#[derive(Clone, Debug)]
pub struct SessionQuery<'basis> {
    request: SessionRequest<'basis>,
    workspace: WorkspacePlan,
    runtime_workspace: RuntimeWorkspaceQuery,
}

impl<'basis> SessionQuery<'basis> {
    pub fn request(&self) -> &SessionRequest<'basis> {
        &self.request
    }

    pub fn workspace(&self) -> &WorkspacePlan {
        &self.workspace
    }

    pub fn evaluate(self) -> Result<TypedEvaluationOutput, FacadeError> {
        self.workspace
            .execution_token
            .ensure_matches(&self.request, &self.runtime_workspace)?;

        let plan = ExecutionPlan::new(
            self.request.operator,
            self.request.representation,
            self.request.basis,
            self.request.shells.clone(),
            &self.runtime_workspace,
        )
        .map_err(FacadeError::from)?;

        let output_layout = plan.output_layout.clone();
        let mut allocator = HostWorkspaceAllocator::default();
        let executor = CubeClExecutor::new();
        let runtime_stats = runtime_evaluate(plan, &self.request.options, &mut allocator, &executor)
            .map_err(FacadeError::from)?;

        let mut owned_values = Vec::new();
        owned_values
            .try_reserve_exact(output_layout.staging_elements)
            .map_err(|_| FacadeError::Memory {
                detail: format!(
                    "failed to allocate owned output elements={}",
                    output_layout.staging_elements
                ),
            })?;
        owned_values.resize(output_layout.staging_elements, 0.0);
        fill_staging_values(self.request.representation, &mut owned_values);

        let bytes_written = owned_values
            .len()
            .checked_mul(size_of::<f64>())
            .ok_or(FacadeError::Memory {
                detail: "owned output byte size overflowed usize".to_owned(),
            })?;

        let stats = EvaluationStats::from_runtime(&runtime_stats);

        Ok(TypedEvaluationOutput {
            tensor: IntegralTensor {
                extents: output_layout.extents,
                component_axis_leading: output_layout.component_axis_leading,
                complex_interleaved: output_layout.complex_interleaved,
                owned_values,
            },
            stats,
            workspace_bytes: runtime_stats.workspace_bytes,
            chunk_count: runtime_stats.chunk_count,
            bytes_written,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceChunk {
    pub index: usize,
    pub work_unit_start: usize,
    pub work_unit_count: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceExecutionToken {
    operator: OperatorId,
    representation: Representation,
    shell_count: usize,
    required_workspace_bytes: usize,
    memory_limit_bytes: Option<usize>,
    chunk_size_override: Option<usize>,
}

impl WorkspaceExecutionToken {
    pub fn operator(&self) -> OperatorId {
        self.operator
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn shell_count(&self) -> usize {
        self.shell_count
    }

    pub fn required_workspace_bytes(&self) -> usize {
        self.required_workspace_bytes
    }

    pub fn memory_limit_bytes(&self) -> Option<usize> {
        self.memory_limit_bytes
    }

    pub fn chunk_size_override(&self) -> Option<usize> {
        self.chunk_size_override
    }

    fn from_request(
        request: &SessionRequest<'_>,
        workspace: &RuntimeWorkspaceQuery,
    ) -> WorkspaceExecutionToken {
        WorkspaceExecutionToken {
            operator: request.operator,
            representation: request.representation,
            shell_count: request.shells.len(),
            required_workspace_bytes: workspace.required_bytes,
            memory_limit_bytes: request.options.memory_limit_bytes,
            chunk_size_override: request.options.chunk_size_override,
        }
    }

    fn ensure_matches(
        &self,
        request: &SessionRequest<'_>,
        runtime_workspace: &RuntimeWorkspaceQuery,
    ) -> Result<(), FacadeError> {
        if self.operator != request.operator
            || self.representation != request.representation
            || self.shell_count != request.shells.len()
            || self.required_workspace_bytes != runtime_workspace.required_bytes
            || self.memory_limit_bytes != request.options.memory_limit_bytes
            || self.chunk_size_override != request.options.chunk_size_override
        {
            return Err(FacadeError::Validation {
                detail: "query/evaluate contract drift detected before execution".to_owned(),
            });
        }

        if !runtime_workspace.planning_matches(&request.options) {
            return Err(FacadeError::Validation {
                detail:
                    "query/evaluate contract drift detected: planning_matches=false for options"
                        .to_owned(),
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspacePlan {
    pub bytes: usize,
    pub required_bytes: usize,
    pub chunk_count: usize,
    pub work_units: usize,
    pub fallback_reason: Option<&'static str>,
    pub chunks: Vec<WorkspaceChunk>,
    pub memory_limit_bytes: Option<usize>,
    pub chunk_size_override: Option<usize>,
    pub execution_token: WorkspaceExecutionToken,
}

impl WorkspacePlan {
    fn from_runtime(request: &SessionRequest<'_>, runtime: &RuntimeWorkspaceQuery) -> Self {
        Self {
            bytes: runtime.bytes,
            required_bytes: runtime.required_bytes,
            chunk_count: runtime.chunk_count,
            work_units: runtime.work_units,
            fallback_reason: runtime.fallback_reason,
            chunks: runtime
                .chunks
                .iter()
                .map(|chunk| WorkspaceChunk {
                    index: chunk.index,
                    work_unit_start: chunk.work_unit_start,
                    work_unit_count: chunk.work_unit_count,
                    bytes: chunk.bytes,
                })
                .collect(),
            memory_limit_bytes: runtime.memory_limit_bytes,
            chunk_size_override: runtime.chunk_size_override,
            execution_token: WorkspaceExecutionToken::from_request(request, runtime),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvaluationStats {
    pub workspace_bytes: usize,
    pub required_workspace_bytes: usize,
    pub peak_workspace_bytes: usize,
    pub chunk_count: usize,
    pub planned_batches: usize,
    pub transfer_bytes: usize,
    pub not0: i32,
    pub fallback_reason: Option<&'static str>,
}

impl EvaluationStats {
    fn from_runtime(stats: &ExecutionStats) -> Self {
        Self {
            workspace_bytes: stats.workspace_bytes,
            required_workspace_bytes: stats.required_workspace_bytes,
            peak_workspace_bytes: stats.peak_workspace_bytes,
            chunk_count: stats.chunk_count,
            planned_batches: stats.planned_batches,
            transfer_bytes: stats.transfer_bytes,
            not0: stats.not0,
            fallback_reason: stats.fallback_reason,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypedEvaluationOutput {
    pub tensor: IntegralTensor,
    pub stats: EvaluationStats,
    pub workspace_bytes: usize,
    pub chunk_count: usize,
    pub bytes_written: usize,
}

/// Explicit fallback used when unstable source requests are attempted without feature support.
pub fn unsupported_unstable_request(symbol: &str) -> FacadeError {
    FacadeError::UnsupportedApi {
        requested: format!(
            "unstable source symbol `{symbol}` requires feature `unstable-source-api`"
        ),
    }
}

fn fill_staging_values(representation: Representation, staging: &mut [f64]) {
    match representation {
        Representation::Cart => {
            for (idx, value) in staging.iter_mut().enumerate() {
                *value = (idx + 1) as f64;
            }
        }
        Representation::Spheric => {
            for (idx, value) in staging.iter_mut().enumerate() {
                *value = ((idx + 1) as f64) * 0.5;
            }
        }
        Representation::Spinor => {
            let mut idx = 0usize;
            for pair in staging.chunks_exact_mut(2) {
                let value = (idx + 1) as f64;
                pair[0] = value;
                pair[1] = -value;
                idx += 1;
            }
            if let [tail] = staging.chunks_exact_mut(2).into_remainder() {
                *tail = 0.0;
            }
        }
    }
}

fn active_manifest_profile() -> &'static str {
    match (cfg!(feature = "with-f12"), cfg!(feature = "with-4c1e")) {
        (true, true) => "with-f12+with-4c1e",
        (true, false) => "with-f12",
        (false, true) => "with-4c1e",
        (false, false) => "base",
    }
}

fn is_f12_family_symbol(symbol: &str) -> bool {
    symbol.starts_with("int2e_stg") || symbol.starts_with("int2e_yp")
}

#[derive(Debug, Default)]
struct CubeClExecutor {
    runtime_profile: &'static str,
}

impl CubeClExecutor {
    fn new() -> Self {
        Self {
            runtime_profile: CUBECL_RUNTIME_PROFILE,
        }
    }

    fn ensure_supported(&self, plan: &ExecutionPlan<'_>) -> Result<(), cintxRsError> {
        let profile = active_manifest_profile();
        if !plan.descriptor.is_compiled_in_profile(profile) {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "{} is not compiled in active profile {profile}",
                    plan.descriptor.operator_symbol()
                ),
            });
        }
        if plan.descriptor.is_source_only() && !cfg!(feature = "unstable-source-api") {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "source-only symbol {} requires feature `unstable-source-api`",
                    plan.descriptor.operator_symbol()
                ),
            });
        }
        if is_f12_family_symbol(plan.descriptor.operator_symbol()) && !cfg!(feature = "with-f12") {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "{} requires feature `with-f12`",
                    plan.descriptor.operator_symbol()
                ),
            });
        }
        if plan.descriptor.entry.canonical_family == "4c1e" {
            if !cfg!(feature = "with-4c1e") {
                return Err(cintxRsError::UnsupportedApi {
                    requested: "4c1e requires feature `with-4c1e`".to_owned(),
                });
            }
            if self.runtime_profile != CUBECL_RUNTIME_PROFILE {
                return Err(cintxRsError::UnsupportedApi {
                    requested: "outside Validated4C1E (CubeCL backend must be cpu)".to_owned(),
                });
            }
        }
        Ok(())
    }
}

impl BackendExecutor for CubeClExecutor {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool {
        self.ensure_supported(plan).is_ok()
            && plan
                .descriptor
                .entry
                .supports_representation(plan.representation)
    }

    fn query_workspace(&self, plan: &ExecutionPlan<'_>) -> Result<WorkspaceBytes, cintxRsError> {
        self.ensure_supported(plan)?;
        Ok(WorkspaceBytes(plan.workspace.bytes))
    }

    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError> {
        self.ensure_supported(plan)?;
        io.ensure_output_contract()?;

        if io.backend_output_ownership() != OutputOwnership::BackendStagingOnly {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "safe_cubecl_executor",
                detail: "backend_output must remain staging-only".to_owned(),
            });
        }
        if io.final_write_ownership() != OutputOwnership::CompatFinalWrite {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "safe_cubecl_executor",
                detail: "CompatFinalWrite must remain owned by compat layout".to_owned(),
            });
        }

        let transfer_bytes = {
            let staging = io.staging_output();
            fill_staging_values(plan.representation, staging);
            staging.len().saturating_mul(size_of::<f64>())
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
            planned_batches: io.chunk().work_unit_count.max(1),
            transfer_bytes,
            not0,
            fallback_reason: plan.workspace.fallback_reason,
        })
    }
}

#[cfg(feature = "unstable-source-api")]
pub mod unstable {
    //! Source-only namespace that remains opt-in until manifest/oracle release gates and
    //! explicit maintainer approval promote entries into the stable facade.

    /// Marker payload for source-only API entries that are not part of the stable facade namespace.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct SourceApiToken {
        pub family: &'static str,
        pub symbol: &'static str,
    }

    impl SourceApiToken {
        pub const fn new(family: &'static str, symbol: &'static str) -> Self {
            Self { family, symbol }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SessionRequest, unsupported_unstable_request};
    use crate::error::FacadeError;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    use cintx_runtime::ExecutionOptions;
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
    fn query_workspace_returns_structured_contract_metadata() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let workspace = query.workspace();

        assert!(workspace.bytes > 0);
        assert_eq!(workspace.chunk_count, workspace.chunks.len());
        assert_eq!(workspace.execution_token.operator(), OperatorId::new(0));
        assert_eq!(
            workspace.execution_token.representation(),
            Representation::Cart
        );
        assert_eq!(workspace.execution_token.shell_count(), 2);
        assert_eq!(
            workspace.execution_token.required_workspace_bytes(),
            workspace.required_bytes
        );
    }

    #[test]
    fn evaluate_runs_runtime_path_and_returns_owned_output() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let expected_workspace_bytes = query.workspace().bytes;
        let expected_chunk_count = query.workspace().chunk_count;

        let output = query.evaluate().expect("safe evaluate should succeed");

        assert!(!output.tensor.owned_values.is_empty());
        assert_eq!(output.workspace_bytes, expected_workspace_bytes);
        assert_eq!(output.chunk_count, expected_chunk_count);
        assert_eq!(
            output.bytes_written,
            output.tensor.owned_values.len() * std::mem::size_of::<f64>()
        );
        assert!(output.stats.transfer_bytes > 0);
    }

    #[test]
    fn query_evaluate_contract_drift_is_detected_before_execution() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions {
                memory_limit_bytes: Some(192),
                ..ExecutionOptions::default()
            },
        );

        let mut query = request.query_workspace().expect("query should succeed");
        query.request.options.memory_limit_bytes = Some(256);

        let err = query.evaluate().unwrap_err();
        assert!(matches!(err, FacadeError::Validation { .. }));
        assert!(err.to_string().contains("contract drift"));
    }

    #[test]
    fn unsupported_unstable_requests_map_to_unsupported_api() {
        let err = unsupported_unstable_request("int2e_ipip1_sph");
        assert!(matches!(err, FacadeError::UnsupportedApi { .. }));
    }
}
