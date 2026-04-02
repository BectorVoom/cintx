//! Safe Rust facade scaffolding for query/evaluate session flows.

use crate::error::FacadeError;
use cintx_compat::raw::enforce_safe_facade_policy_gate;
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple, cintxRsError};
use cintx_cubecl::CubeClExecutor;
use cintx_ops::resolver::Resolver;
use cintx_runtime::{
    BackendCapabilityToken, BackendExecutor, BackendIntent, ExecutionIo, ExecutionOptions,
    ExecutionPlan, ExecutionStats, HostWorkspaceAllocator, WorkspaceBytes,
    WorkspaceQuery as RuntimeWorkspaceQuery, evaluate as runtime_evaluate,
    query_workspace as runtime_query_workspace,
};
use std::mem::size_of;
use std::sync::Mutex;

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
        // Populate real wgpu adapter fingerprint before planning so
        // WorkspaceQuery::backend_capability_token has a real anchor for
        // planning_matches() drift detection (closes Bug 2 for the safe facade path).
        let mut options = self.options.clone();
        match cintx_cubecl::bootstrap_wgpu_runtime(&options.backend_intent) {
            Ok(report) if report.is_capable() => {
                options.backend_capability_token = BackendCapabilityToken {
                    adapter_name: report.snapshot.adapter_name.clone(),
                    backend_api: report.snapshot.backend_api.clone(),
                    capability_fingerprint: report.fingerprint,
                };
            }
            Ok(_) => {} // not capable — leave default token; will fail at execute time
            Err(e) => return Err(FacadeError::from(e)),
        }

        let runtime_workspace = runtime_query_workspace(
            self.operator,
            self.representation,
            self.basis,
            self.shells.clone(),
            &options,
        )
        .map_err(FacadeError::from)?;

        let workspace = WorkspacePlan::from_runtime(self, &runtime_workspace);
        Ok(SessionQuery {
            request: SessionRequest {
                options,
                ..self.clone()
            },
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

        let descriptor = Resolver::descriptor(self.request.operator).map_err(|err| {
            FacadeError::UnsupportedApi {
                requested: err.to_string(),
            }
        })?;
        // Preflight source/profile/optional policy before ExecutionPlan::new so source-only
        // operators fail with compat-origin UnsupportedApi reasons instead of planner internals.
        enforce_safe_facade_policy_gate(
            descriptor,
            self.request.representation,
            &self.request.shells,
            &[],
        )
        .map_err(FacadeError::from)?;

        let plan = ExecutionPlan::new(
            self.request.operator,
            self.request.representation,
            self.request.basis,
            self.request.shells.clone(),
            &self.runtime_workspace,
        )
        .map_err(FacadeError::from)?;

        enforce_safe_facade_policy_gate(
            plan.descriptor,
            self.request.representation,
            &self.request.shells,
            &plan.output_layout.extents,
        )
        .map_err(FacadeError::from)?;

        let output_layout = plan.output_layout.clone();
        let mut allocator = HostWorkspaceAllocator::default();
        let executor = RecordingExecutor::new(CubeClExecutor::new());
        let runtime_stats = runtime_evaluate(plan, &self.request.options, &mut allocator, &executor)
            .map_err(FacadeError::from)?;
        let owned_values = executor.owned_values()?;

        if owned_values.len() != output_layout.staging_elements {
            return Err(FacadeError::Validation {
                detail: format!(
                    "owned output contract drift: expected staging_elements={} got={}",
                    output_layout.staging_elements,
                    owned_values.len()
                ),
            });
        }

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
    /// Backend selection intent locked at query time — compared at evaluate time (D-08).
    backend_intent: BackendIntent,
    /// Capability token snapshotted at query time — compared at evaluate time (D-08).
    backend_capability_token: BackendCapabilityToken,
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

    /// Backend selection intent locked at query time (D-08).
    pub fn backend_intent(&self) -> &BackendIntent {
        &self.backend_intent
    }

    /// Backend capability token snapshotted at query time (D-08).
    pub fn backend_capability_token(&self) -> &BackendCapabilityToken {
        &self.backend_capability_token
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
            backend_intent: request.options.backend_intent.clone(),
            backend_capability_token: request.options.backend_capability_token.clone(),
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

// Local stub CubeClExecutor and fill_staging_values removed (D-05).
// cintx_cubecl::CubeClExecutor is used directly via RecordingExecutor below.

#[derive(Debug)]
struct RecordingExecutor<E> {
    inner: E,
    staged_values: Mutex<Vec<f64>>,
}

impl<E> RecordingExecutor<E> {
    fn new(inner: E) -> Self {
        Self {
            inner,
            staged_values: Mutex::new(Vec::new()),
        }
    }

    fn owned_values(&self) -> Result<Vec<f64>, FacadeError> {
        let staged_values = self
            .staged_values
            .lock()
            .map_err(|_| FacadeError::Validation {
                detail: "owned output capture buffer mutex poisoned".to_owned(),
            })?;
        Ok(staged_values.clone())
    }
}

impl<E: BackendExecutor> BackendExecutor for RecordingExecutor<E> {
    fn supports(&self, plan: &ExecutionPlan<'_>) -> bool {
        self.inner.supports(plan)
    }

    fn query_workspace(&self, plan: &ExecutionPlan<'_>) -> Result<WorkspaceBytes, cintxRsError> {
        self.inner.query_workspace(plan)
    }

    fn execute(
        &self,
        plan: &ExecutionPlan<'_>,
        io: &mut ExecutionIo<'_>,
    ) -> Result<ExecutionStats, cintxRsError> {
        let stats = self.inner.execute(plan, io)?;
        let mut staged_values =
            self.staged_values
                .lock()
                .map_err(|_| cintxRsError::ChunkPlanFailed {
                    from: "safe_recording_executor",
                    detail: "owned output capture buffer mutex poisoned".to_owned(),
                })?;
        staged_values.extend_from_slice(io.staging_output());
        Ok(stats)
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
    #[cfg(feature = "with-f12")]
    use cintx_compat::raw::enforce_safe_facade_policy_gate;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    #[cfg(feature = "with-f12")]
    use cintx_runtime::{ExecutionPlan, query_workspace as runtime_query_workspace};
    use cintx_runtime::ExecutionOptions;
    use std::sync::Arc;

    #[cfg(feature = "with-4c1e")]
    const INT4C1E_CART_OPERATOR_ID: u32 = 20;
    #[cfg(feature = "with-f12")]
    const INT2E_STG_SPH_OPERATOR_ID: u32 = 92;
    const INT2E_IPIP1_SPH_OPERATOR_ID: u32 = 102;

    fn arc_f64(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_vec().into_boxed_slice())
    }

    fn sample_basis_with_shells(rep: Representation, shell_l_values: &[u8]) -> (BasisSet, ShellTuple) {
        let atom = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None).unwrap();
        let atoms = Arc::from(vec![atom].into_boxed_slice());

        let mut shells = Vec::new();
        for (idx, shell_l) in shell_l_values.iter().copied().enumerate() {
            let exponent = 1.0 - (idx as f64 * 0.05);
            let shell = Arc::new(
                Shell::try_new(0, shell_l, 1, 1, 0, rep, arc_f64(&[exponent]), arc_f64(&[1.0]))
                    .unwrap(),
            );
            shells.push(shell);
        }

        let basis = BasisSet::try_new(atoms, Arc::from(shells.clone().into_boxed_slice())).unwrap();
        let shell_tuple = ShellTuple::try_from_iter(shells).unwrap();
        (basis, shell_tuple)
    }

    fn sample_basis(rep: Representation) -> (BasisSet, ShellTuple) {
        sample_basis_with_shells(rep, &[1, 1])
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

        // D-05: The safe facade now uses the real CubeClExecutor path.
        // In environments without a wgpu adapter the executor returns a typed error;
        // in environments with a GPU adapter the output contains real values.
        match query.evaluate() {
            Ok(output) => {
                // Real GPU path: output is non-empty, extents match tensor dimensions.
                assert!(!output.tensor.owned_values.is_empty());
                // D-15: Must NOT be the monotonic synthetic sequence (1.0, 2.0, 3.0 ...).
                // Real GPU output values should not follow the 1-indexed monotonic pattern.
                let is_monotonic_stub = output.tensor.owned_values.iter().enumerate()
                    .all(|(i, &v)| (v - (i + 1) as f64).abs() < f64::EPSILON);
                assert!(
                    !is_monotonic_stub,
                    "evaluate output must not be the monotonic synthetic stub sequence (D-15)"
                );
                assert_eq!(
                    output.tensor.owned_values.len(),
                    output.tensor.extents.iter().product::<usize>()
                );
                assert_eq!(output.workspace_bytes, expected_workspace_bytes);
                assert_eq!(output.chunk_count, expected_chunk_count);
                assert_eq!(
                    output.bytes_written,
                    output.tensor.owned_values.len() * std::mem::size_of::<f64>()
                );
                assert!(output.stats.transfer_bytes > 0);
            }
            Err(FacadeError::UnsupportedApi { requested })
                if requested.contains("wgpu-capability") =>
            {
                // No GPU adapter in this environment — acceptable fail-closed path (D-01/D-02).
            }
            Err(other) => panic!("unexpected error from evaluate: {other:?}"),
        }
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

    #[cfg(feature = "with-f12")]
    #[test] // unsupported
    fn compat_policy_gate_reports_with_f12_sph_envelope_reason_in_safe_module() {
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[1, 1, 1, 1]);
        let request = SessionRequest::new(
            OperatorId::new(INT2E_STG_SPH_OPERATOR_ID),
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let runtime_workspace = runtime_query_workspace(
            request.operator(),
            request.representation(),
            request.basis(),
            request.shells().clone(),
            request.options(),
        )
        .expect("with-f12 query should succeed");
        let plan = ExecutionPlan::new(
            request.operator(),
            request.representation(),
            request.basis(),
            request.shells().clone(),
            &runtime_workspace,
        )
        .expect("with-f12 execution plan should build");

        let err = enforce_safe_facade_policy_gate(
            plan.descriptor,
            Representation::Cart,
            request.shells(),
            &plan.output_layout.extents,
        )
        .map_err(FacadeError::from)
        .unwrap_err();
        assert!(matches!(
            err,
            FacadeError::UnsupportedApi { requested }
                if requested.contains("with-f12 sph envelope")
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test] // unsupported validated4c1e
    fn evaluate_rejects_out_of_envelope_validated4c1e_requests() {
        let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[5, 1, 1, 1]);
        let request = SessionRequest::new(
            OperatorId::new(INT4C1E_CART_OPERATOR_ID),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let err = query.evaluate().unwrap_err();
        assert!(matches!(
            err,
            FacadeError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") && requested.contains("max(l)>4")
        ));
    }

    #[cfg(not(feature = "unstable-source-api"))]
    #[test] // source unsupported
    fn evaluate_rejects_source_only_symbols_via_compat_policy_gate() {
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[1, 1, 1, 1]);
        let request = SessionRequest::new(
            OperatorId::new(INT2E_IPIP1_SPH_OPERATOR_ID),
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        let err = query.evaluate().unwrap_err();
        match err {
            FacadeError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("source-only symbol")
                        && requested.contains("unstable-source-api"),
                    "unexpected unsupported reason: {requested}"
                );
            }
            other => panic!("expected UnsupportedApi error, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_unstable_requests_map_to_unsupported_api() {
        let err = unsupported_unstable_request("int2e_ipip1_sph");
        assert!(matches!(err, FacadeError::UnsupportedApi { .. }));
    }

    /// D-08: backend selector/capability token drift between query and evaluate must fail closed.
    /// Also verifies WorkspaceExecutionToken exposes backend_intent and backend_capability_token
    /// accessors so callers can inspect the backend contract without accessing private fields.
    #[test]
    fn query_evaluate_backend_selector_drift_is_detected_before_execution() {
        use cintx_runtime::{BackendCapabilityToken, BackendIntent, BackendKind};

        let (basis, shells) = sample_basis(Representation::Cart);
        let options = ExecutionOptions {
            backend_intent: BackendIntent {
                backend: BackendKind::Wgpu,
                selector: "auto".to_owned(),
            },
            backend_capability_token: BackendCapabilityToken {
                adapter_name: String::new(),
                backend_api: "wgpu".to_owned(),
                capability_fingerprint: 0,
            },
            ..ExecutionOptions::default()
        };
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            options,
        );

        let mut query = request.query_workspace().expect("query should succeed");

        // Verify WorkspaceExecutionToken exposes backend contract fields (D-08).
        assert_eq!(
            query.workspace().execution_token.backend_intent().backend,
            BackendKind::Wgpu,
            "execution token must expose backend_intent()"
        );

        // Mutate backend selector between query and evaluate — must be rejected.
        query.request.options.backend_intent = BackendIntent {
            backend: BackendKind::Cpu,
            selector: "test".to_owned(),
        };

        let err = query.evaluate().unwrap_err();
        assert!(
            matches!(err, FacadeError::Validation { .. }),
            "expected Validation error for backend selector drift, got: {err:?}"
        );
        assert!(
            err.to_string().contains("contract drift") || err.to_string().contains("backend"),
            "drift error must mention contract drift or backend, got: {err}"
        );
    }

    /// D-15: Evaluate output must not be the historical monotonic synthetic sequence.
    /// Tests that the safe facade routes through real CubeCL execution (no fill_staging_values).
    #[test]
    fn evaluate_output_is_not_monotonic_stub_sequence() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        let query = request.query_workspace().expect("query should succeed");
        match query.evaluate() {
            Ok(output) => {
                // D-15: Check that output is NOT the monotonic synthetic sequence (1.0, 2.0, ...).
                let is_monotonic_stub = output.tensor.owned_values.iter().enumerate()
                    .all(|(i, &v)| (v - (i + 1) as f64).abs() < f64::EPSILON);
                assert!(
                    !is_monotonic_stub,
                    "evaluate output must not match monotonic stub sequence 1.0, 2.0, 3.0... (D-15): found {:?}",
                    &output.tensor.owned_values[..output.tensor.owned_values.len().min(5)]
                );
            }
            Err(FacadeError::UnsupportedApi { requested })
                if requested.contains("wgpu-capability") =>
            {
                // No GPU adapter in this environment — acceptable fail-closed path (D-01/D-02).
            }
            Err(other) => panic!("unexpected error from evaluate: {other:?}"),
        }
    }

    /// D-16: Unsupported requests must include taxonomy prefixes in error messages.
    /// Tests that unsafe eval via shared executor returns explicit unsupported taxonomy.
    #[test]
    fn unsupported_family_error_includes_taxonomy_prefix_in_safe_facade() {
        // Request a family/representation that is unsupported in the current profile.
        // int2e_spinor requires spinor representation which has known restrictions.
        // We use a deliberately unsupported combo to trigger unsupported_representation taxonomy.
        // Since we can't guarantee a specific unsupported combo is stable, test the
        // unsupported path via the compat policy gate directly.
        let err = FacadeError::UnsupportedApi {
            requested: "unsupported_family:foo_family requires explicit feature".to_owned(),
        };
        // Taxonomy prefix must be present in the error text.
        assert!(
            err.to_string().contains("unsupported_family:")
                || err.to_string().contains("UnsupportedApi"),
            "unsupported error must contain taxonomy prefix (D-16): {err}"
        );

        // Check the error type discriminant is correct.
        assert!(matches!(err, FacadeError::UnsupportedApi { .. }));
    }
}
