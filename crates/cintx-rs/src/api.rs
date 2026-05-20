//! Safe Rust facade scaffolding for query/evaluate session flows.

use crate::error::FacadeError;
use cintx_compat::raw::enforce_safe_facade_policy_gate;
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple};
use cintx_cubecl::CubeClExecutor;
use cintx_ops::resolver::Resolver;
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionOptions, ExecutionPlan, ExecutionStats,
    HostWorkspaceAllocator, WorkspaceAllocator,
    WorkspaceQuery as RuntimeWorkspaceQuery, schedule_chunks,
    query_workspace as runtime_query_workspace,
};
use std::mem::size_of;

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
        // Phase 18 D-04: aosym preflight — only S1 (and None ≡ S1) is implemented.
        // Non-S1 packings return a typed FacadeError::UnsupportedAoSymmetry so callers
        // can pattern-match programmatically. Fails fast before any runtime work.
        if let Some(aosym) = self.options.aosym {
            if aosym != cintx_core::AoSymmetry::S1 {
                return Err(FacadeError::UnsupportedAoSymmetry {
                    requested: aosym.to_string(),
                });
            }
        }

        // Phase 19 D-06: ECP-basis preflight — operator.is_ecp() &&
        // basis.ecp_shells().is_empty() returns FacadeError::MissingEcpBasis.
        // Fails fast before any runtime_query_workspace work, mirroring the
        // aosym preflight above. Resolves the operator symbol via the
        // manifest so the error message names the canonical libcint symbol;
        // falls back to the OperatorId Display impl if the manifest lookup
        // fails (defensive — keeps the safe API from panicking on a missing
        // descriptor).
        if self.operator.is_ecp() && self.basis.ecp_shells().is_empty() {
            let symbol = Resolver::descriptor(self.operator)
                .map(|d| d.operator_symbol().to_string())
                .unwrap_or_else(|_| format!("{}", self.operator));
            return Err(FacadeError::MissingEcpBasis { operator: symbol });
        }

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

        let mut plan = ExecutionPlan::new(
            self.request.operator,
            self.request.representation,
            self.request.basis,
            self.request.shells.clone(),
            &self.runtime_workspace,
        )
        .map_err(FacadeError::from)?;

        // Propagate f12_zeta from ExecutionOptions to operator_env_params (safe API path).
        if let Some(zeta) = self.request.options().f12_zeta {
            plan.operator_env_params.f12_zeta = Some(zeta);
        }

        enforce_safe_facade_policy_gate(
            plan.descriptor,
            self.request.representation,
            &self.request.shells,
            &plan.output_layout.extents,
        )
        .map_err(FacadeError::from)?;

        let output_layout = plan.output_layout.clone();
        let mut allocator = HostWorkspaceAllocator::default();
        let executor = CubeClExecutor::new();

        if !executor.supports(&plan) {
            return Err(FacadeError::UnsupportedApi {
                requested: format!(
                    "{}/{}/{}",
                    plan.descriptor.family(),
                    plan.descriptor.operator_name(),
                    self.request.representation
                ),
            });
        }

        let backend_workspace = executor.query_workspace(&plan)
            .map_err(FacadeError::from)?
            .get();
        if backend_workspace > plan.workspace.bytes {
            return Err(FacadeError::from(cintx_core::cintxRsError::MemoryLimitExceeded {
                requested: backend_workspace,
                limit: plan.workspace.bytes,
            }));
        }

        // Allocate the full staging accumulator owned by the facade, so we can read
        // staging values from executor.execute() directly without RecordingExecutor.
        let staging_elements = output_layout.staging_elements;
        let staging_bytes = staging_elements
            .checked_mul(size_of::<f64>())
            .ok_or(FacadeError::Memory {
                detail: "staging element byte count overflowed usize".to_owned(),
            })?;
        let mut owned_values = Vec::new();
        owned_values
            .try_reserve_exact(staging_elements)
            .map_err(|_| FacadeError::Memory {
                detail: format!("failed to allocate staging buffer of {staging_bytes} bytes"),
            })?;
        owned_values.resize(staging_elements, 0.0f64);

        let schedule = schedule_chunks(&plan.workspace);
        let total_units = plan.workspace.work_units.max(1);
        let mut total_not0: i32 = 0;
        let mut total_transfer_bytes: usize = 0;
        let mut total_peak_workspace_bytes: usize = 0;

        for chunk in schedule.chunks() {
            // Compute staging slice range for this chunk.
            let start = chunk.work_unit_start.min(total_units);
            let end = chunk
                .work_unit_start
                .saturating_add(chunk.work_unit_count)
                .min(total_units);
            let prefix = staging_elements.saturating_mul(start) / total_units;
            let suffix = staging_elements.saturating_mul(end) / total_units;
            let chunk_len = suffix.saturating_sub(prefix).max(1);

            let chunk_staging_bytes = chunk_len.checked_mul(size_of::<f64>()).ok_or(
                FacadeError::Memory {
                    detail: "chunk staging byte count overflowed usize".to_owned(),
                },
            )?;
            let mut chunk_staging = Vec::new();
            chunk_staging
                .try_reserve_exact(chunk_len)
                .map_err(|_| FacadeError::Memory {
                    detail: format!(
                        "failed to allocate chunk staging buffer of {chunk_staging_bytes} bytes"
                    ),
                })?;
            chunk_staging.resize(chunk_len, 0.0f64);

            let mut workspace = allocator
                .try_alloc(chunk.bytes, plan.workspace.alignment)
                .map_err(FacadeError::from)?;

            {
                let mut io = ExecutionIo::new(
                    chunk,
                    &mut chunk_staging,
                    &mut workspace,
                    plan.dispatch,
                )
                .map_err(FacadeError::from)?;
                let chunk_stats = executor.execute(&plan, &mut io).map_err(FacadeError::from)?;
                total_not0 = total_not0.saturating_add(chunk_stats.not0.max(0));
                total_transfer_bytes = total_transfer_bytes
                    .saturating_add(io.transfer_bytes())
                    .saturating_add(chunk_stats.transfer_bytes);
                total_peak_workspace_bytes =
                    total_peak_workspace_bytes.max(io.workspace().len());
            }
            allocator.release(workspace);

            // Copy chunk staging into the appropriate range of the accumulator.
            let dest_end = prefix.saturating_add(chunk_len).min(staging_elements);
            if prefix < dest_end {
                owned_values[prefix..dest_end]
                    .copy_from_slice(&chunk_staging[..dest_end - prefix]);
            }
        }

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

        let chunk_count = schedule_chunks(&plan.workspace).len();
        let runtime_stats = ExecutionStats {
            workspace_bytes: plan.workspace.bytes,
            required_workspace_bytes: plan.workspace.required_bytes,
            peak_workspace_bytes: total_peak_workspace_bytes,
            chunk_count: chunk_count.max(plan.workspace.chunks.len()),
            planned_batches: plan.workspace.chunks.iter().map(|c| c.work_unit_count).sum(),
            transfer_bytes: total_transfer_bytes,
            not0: total_not0,
            fallback_reason: plan.workspace.fallback_reason,
        };

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

/// Owned integral tensor returned by `SessionQuery::evaluate`.
///
/// # Memory layout
///
/// `owned_values` is a dense `Vec<F>` (default `f64`) storing `extents.iter().product()` real
/// values (or 2x that for `Spinor` outputs with `complex_interleaved == true`,
/// where real and imaginary parts alternate in the innermost stride).
///
/// The type parameter `F` is the output float precision (`f64` or `f32`). The default
/// `F = f64` keeps every existing call site compiling unchanged (D-12).
///
/// **AO axis layout** — `extents` lists AO-axis sizes in **shell-tuple order**:
/// `extents[0] = ao_per_shell(shells[0])`, `extents[1] = ao_per_shell(shells[1])`,
/// etc. The per-kernel index ordering inside `owned_values` matches libcint's
/// memory layout for that family:
///
/// - **Arity >= 3** (`int2e_*`, `int3c1e_*`, `int3c2e_*`, `int4c1e_*`): **F-order**
///   (Fortran / column-major) — `extents[0]` is the fastest-varying axis.
///   Byte-identical to vendor libcint output without transposition (verified by
///   the Phase 18 oracle parity sweep in `crates/cintx-oracle/tests/safe_api_arity{3,4}_parity.rs`).
/// - **Arity 2** (`int1e_*`, `int2c2e_*`): row-major within each shell-pair
///   block — `extents[0]` is the slowest-varying axis. The arity-2 oracle parity
///   tests apply the column-major-to-row-major conversion to vendor output before
///   comparison (see `crates/cintx-oracle/tests/safe_api_arity2_parity.rs:280-292`).
///
/// When `component_axis_leading == true` (the planner default), an optional
/// component axis (e.g., for IP/derivative operators) is the slowest-varying
/// axis — placed beyond `extents.len()` shell-tuple axes.
///
/// The arity-aware layout is verified implicitly by the oracle parity sweep
/// (`crates/cintx-oracle/tests/safe_api_arity{2,3,4}_parity.rs`). If the layout
/// silently drifts, the first parity test fails.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IntegralTensor<F = f64> {
    pub extents: Vec<usize>,
    pub component_axis_leading: bool,
    pub complex_interleaved: bool,
    pub owned_values: Vec<F>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypedEvaluationOutput<F = f64> {
    pub tensor: IntegralTensor<F>,
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
    use super::{EvaluationStats, IntegralTensor, SessionRequest, TypedEvaluationOutput, unsupported_unstable_request};
    use crate::error::{FacadeError, FacadeErrorKind};
    #[cfg(feature = "with-f12")]
    use cintx_compat::raw::enforce_safe_facade_policy_gate;
    use cintx_core::ecp::{EcpChannel, EcpShell};
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple};
    #[cfg(feature = "with-f12")]
    use cintx_runtime::{ExecutionPlan, query_workspace as runtime_query_workspace};
    use cintx_runtime::ExecutionOptions;
    use std::sync::Arc;

    #[cfg(feature = "with-4c1e")]
    const INT4C1E_CART_OPERATOR_ID: u32 = 24;
    // Phase 19 Plan 01 inserted four `int1e_ecp_*` rows at OperatorIds 26..=29,
    // ahead of the F12 / source-only blocks; every operator id at or after 26
    // shifts by +4. The values below are derived from
    // crates/cintx-ops/src/generated/api_manifest.rs::OPERATOR_DESCRIPTORS
    // post-regeneration (verified manually: int2e_stg_sph is at position 106,
    // int2e_ipip1_sph is at position 116).
    #[cfg(feature = "with-f12")]
    const INT2E_STG_SPH_OPERATOR_ID: u32 = 106;
    #[cfg(not(feature = "unstable-source-api"))]
    const INT2E_IPIP1_SPH_OPERATOR_ID: u32 = 116;

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
    fn evaluate_returns_deterministic_nonzero_real_values() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),                       // int1e_ovlp_cart
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );

        // Capture expected workspace metadata before consuming the first query.
        let query1 = request.clone().query_workspace().expect("query should succeed");
        let expected_workspace_bytes = query1.workspace().bytes;
        let expected_chunk_count = query1.workspace().chunk_count;

        // First evaluation.
        let output1 = query1.evaluate().expect("safe evaluate should succeed");

        // Second independent evaluation from the same request — idempotency check.
        let query2 = request.query_workspace().expect("query should succeed (2nd)");
        let output2 = query2.evaluate().expect("safe evaluate should succeed (2nd)");

        // (1) Idempotency: real, deterministic kernel must return identical values across runs.
        assert_eq!(
            output1.tensor.owned_values, output2.tensor.owned_values,
            "evaluate must be deterministic across repeated calls"
        );

        // (2) Nonzero: a zero-fill regression must fail this test. For two valid GTO shells the
        //     overlap matrix must contain at least one nonzero element (self-overlap of normalized
        //     GTOs is nonzero on the diagonal).
        let nonzero_count = output1
            .tensor
            .owned_values
            .iter()
            .filter(|&&v| v.abs() > 1e-18)
            .count();
        assert!(
            nonzero_count > 0,
            "evaluate must produce at least one nonzero element; got all-zero owned_values \
             (regression to zero-fill stub?)"
        );

        // (3) Existing extent / byte-count / stats invariants (preserved from prior test).
        assert!(!output1.tensor.owned_values.is_empty());
        assert_eq!(
            output1.tensor.owned_values.len(),
            output1.tensor.extents.iter().product::<usize>(),
        );
        assert_eq!(output1.workspace_bytes, expected_workspace_bytes);
        assert_eq!(output1.chunk_count, expected_chunk_count);
        assert_eq!(
            output1.bytes_written,
            output1.tensor.owned_values.len() * std::mem::size_of::<f64>(),
        );
        assert!(output1.stats.transfer_bytes > 0);
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

    #[test]
    fn aosym_error_path_rejects_non_s1_with_typed_error() {
        use cintx_core::AoSymmetry;
        let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[0, 0]);

        for non_s1 in [AoSymmetry::S2ij, AoSymmetry::S2kl, AoSymmetry::S4, AoSymmetry::S8] {
            let options = ExecutionOptions {
                aosym: Some(non_s1),
                ..Default::default()
            };
            let request = SessionRequest::new(
                OperatorId::new(0),
                Representation::Cart,
                &basis,
                shells.clone(),
                options,
            );
            let err = request
                .query_workspace()
                .expect_err("non-S1 aosym must return UnsupportedAoSymmetry");
            match err {
                FacadeError::UnsupportedAoSymmetry { requested } => {
                    assert_eq!(
                        requested,
                        non_s1.to_string(),
                        "requested field must carry the lowercase pyscf form"
                    );
                }
                other => panic!(
                    "expected UnsupportedAoSymmetry for aosym={non_s1:?}, got {other:?}"
                ),
            }
        }
    }

    #[test]
    fn aosym_none_and_s1_both_succeed_through_query_workspace() {
        use cintx_core::AoSymmetry;
        let (basis, shells) = sample_basis_with_shells(Representation::Cart, &[0, 0]);

        for aosym in [None, Some(AoSymmetry::S1)] {
            let options = ExecutionOptions { aosym, ..Default::default() };
            let request = SessionRequest::new(
                OperatorId::new(0),
                Representation::Cart,
                &basis,
                shells.clone(),
                options,
            );
            request
                .query_workspace()
                .unwrap_or_else(|e| panic!("aosym={aosym:?} must succeed; got {e:?}"));
        }
    }

    // ----------------------------------------------------------------------
    // Phase 19 D-06: FacadeError::MissingEcpBasis variant + query_workspace
    // preflight tests.
    // ----------------------------------------------------------------------

    fn sample_basis_with_ecp(
        rep: Representation,
        shell_l_values: &[u8],
        ecp_count: usize,
    ) -> (BasisSet, ShellTuple) {
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

        let mut ecp_shells: Vec<Arc<EcpShell>> = Vec::new();
        for _ in 0..ecp_count {
            ecp_shells.push(Arc::new(
                EcpShell::try_new(
                    0,
                    EcpChannel::Local,
                    0,
                    1,
                    1,
                    0,
                    arc_f64(&[0.5]),
                    arc_f64(&[1.0]),
                )
                .unwrap(),
            ));
        }
        let ecp_arc = Arc::from(ecp_shells.into_boxed_slice());

        let basis =
            BasisSet::try_new_with_ecp(atoms, Arc::from(shells.clone().into_boxed_slice()), ecp_arc)
                .unwrap();
        let shell_tuple = ShellTuple::try_from_iter(shells).unwrap();
        (basis, shell_tuple)
    }

    #[test]
    fn facade_error_missing_ecp_basis_carries_kind_and_operator() {
        // Test 1+2 (variant exists, kind() arm wired).
        let err = FacadeError::MissingEcpBasis {
            operator: "int1e_ecp_sph".to_owned(),
        };
        assert_eq!(err.kind(), FacadeErrorKind::MissingEcpBasis);
        match &err {
            FacadeError::MissingEcpBasis { operator } => {
                assert_eq!(operator, "int1e_ecp_sph");
            }
            other => panic!("expected MissingEcpBasis variant, got {other:?}"),
        }
    }

    #[test]
    fn facade_error_missing_ecp_basis_display_message_matches_pattern() {
        // Test 3: Display message matches the documented contract.
        let err = FacadeError::MissingEcpBasis {
            operator: "int1e_ecp_sph".to_owned(),
        };
        let rendered = format!("{err}");
        assert!(
            rendered.contains("operator 'int1e_ecp_sph'")
                && rendered.contains("requires ECP basis")
                && rendered.contains("BasisSet::ecp_shells()")
                && rendered.contains("is empty"),
            "unexpected Display rendering: {rendered}",
        );
    }

    #[test]
    fn query_workspace_returns_missing_ecp_basis_for_ecp_op_without_ecp_shells() {
        // Test 4: ECP operator + ECP-less basis → MissingEcpBasis preflight
        // fires before any runtime_query_workspace work.
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[0, 0]);
        assert!(basis.ecp_shells().is_empty(), "fixture must have no ECP");
        let request = SessionRequest::new(
            OperatorId::INT1E_ECP_SPH,
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let err = request
            .query_workspace()
            .expect_err("ECP op without ECP basis must fail preflight");
        match err {
            FacadeError::MissingEcpBasis { operator } => {
                assert_eq!(operator, "int1e_ecp_sph");
            }
            other => panic!("expected MissingEcpBasis, got {other:?}"),
        }
    }

    #[test]
    fn query_workspace_does_not_return_missing_ecp_basis_for_non_ecp_operators() {
        // Test 5: non-ECP operator must not trip the MissingEcpBasis gate
        // regardless of whether ecp_shells is empty.
        let (basis, shells) = sample_basis_with_shells(Representation::Spheric, &[0, 0]);
        assert!(basis.ecp_shells().is_empty());
        let request = SessionRequest::new(
            // int1e_ovlp_sph — not an ECP operator
            OperatorId::new(1),
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        // query_workspace may succeed or return a non-MissingEcpBasis error
        // (e.g. UnsupportedApi for some configurations), but it must NEVER
        // return MissingEcpBasis for a non-ECP operator.
        match request.query_workspace() {
            Ok(_) => {}
            Err(FacadeError::MissingEcpBasis { .. }) => {
                panic!("non-ECP operator must not return MissingEcpBasis");
            }
            Err(_) => {}
        }
    }

    #[test]
    fn query_workspace_passes_through_for_ecp_op_with_ecp_shells_attached() {
        // Test 6: ECP operator + ECP basis attached → preflight does not
        // block. The call may fail later (planner / executor / no kernel
        // wired yet in Plan 03), but the preflight itself must not return
        // MissingEcpBasis.
        let (basis, shells) = sample_basis_with_ecp(Representation::Spheric, &[0, 0], 1);
        assert_eq!(basis.ecp_shells().len(), 1);
        let request = SessionRequest::new(
            OperatorId::INT1E_ECP_SPH,
            Representation::Spheric,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        match request.query_workspace() {
            Ok(_) => {}
            Err(FacadeError::MissingEcpBasis { .. }) => {
                panic!(
                    "ECP op with ecp_shells attached must NOT return MissingEcpBasis"
                );
            }
            Err(_) => {
                // Plan 04 wires the kernel; until then a downstream error
                // is acceptable. The point of this test is purely that the
                // ECP preflight does not fire.
            }
        }
    }

    // ------------------------------------------------------------------
    // Plan 20-07 Task 1: generic output structs IntegralTensor<F> /
    // TypedEvaluationOutput<F> with f64 defaults.
    // ------------------------------------------------------------------

    #[test]
    fn integral_tensor_default_type_param_is_f64() {
        // Unparameterized IntegralTensor resolves to IntegralTensor<f64>.
        // Compile-time test: this must type-check without a turbofish.
        let t: IntegralTensor = IntegralTensor::default();
        // owned_values must be Vec<f64>
        let _: Vec<f64> = t.owned_values;
    }

    #[test]
    fn typed_evaluation_output_default_type_param_is_f64() {
        // Unparameterized TypedEvaluationOutput resolves to TypedEvaluationOutput<f64>.
        let out: TypedEvaluationOutput = TypedEvaluationOutput::default();
        let _: Vec<f64> = out.tensor.owned_values;
    }

    #[test]
    fn integral_tensor_f32_is_constructible() {
        // TypedEvaluationOutput<f32> and IntegralTensor<f32> must be constructible.
        let t: IntegralTensor<f32> = IntegralTensor {
            extents: vec![2, 2],
            component_axis_leading: false,
            complex_interleaved: false,
            owned_values: vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        };
        assert_eq!(t.owned_values.len(), 4);
        let out: TypedEvaluationOutput<f32> = TypedEvaluationOutput {
            tensor: t,
            stats: EvaluationStats::default(),
            workspace_bytes: 0,
            chunk_count: 0,
            bytes_written: 0,
        };
        assert_eq!(out.tensor.owned_values[0], 1.0_f32);
    }
}
