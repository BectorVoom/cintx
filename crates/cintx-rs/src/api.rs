//! Safe Rust facade scaffolding for query/evaluate session flows.

use crate::error::FacadeError;
use cintx_compat::raw::enforce_safe_facade_policy_gate;
use cintx_core::{BasisSet, CintFloat, OperatorId, Representation, ShellTuple};
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

    /// Evaluate the integral using the default f64 precision.
    ///
    /// This is a thin shim that delegates to `evaluate_generic::<f64>()`. It exists
    /// so that every existing `req.evaluate()` call site compiles unchanged without
    /// a turbofish (D-03/D-12). The result is byte-identical to the pre-generic
    /// implementation.
    pub fn evaluate(self) -> Result<TypedEvaluationOutput<f64>, FacadeError> {
        self.evaluate_generic::<f64>()
    }

    /// Evaluate the integral using the precision specified by the type parameter `F`.
    ///
    /// - `evaluate_generic::<f64>()` — byte-identical to the pre-generic `evaluate()` (D-12).
    /// - `evaluate_generic::<f32>()` — opts into f32 output; sets `plan.precision = F32`;
    ///   returns `Vec<f32>` (T-20-19 type-system isolation).
    ///
    /// # Staging buffer design
    ///
    /// The facade owns a `Vec<F>` sized in `size_of::<F>()` per element. When passing to
    /// `ExecutionIo::new` (which requires `&mut [f64]` — frozen interface), the buffer is
    /// reinterpreted via `bytemuck::cast_slice_mut::<F, f64>`. This is sound because both
    /// `f32` and `f64` are `bytemuck::Pod` (proven in Plan 01, A5 spike), and the kernel
    /// dispatcher reads back the same byte pattern it wrote (it branches on `plan.precision`).
    ///
    /// OOM-safe fallible allocation (`try_reserve_exact`) and the no-partial-writes contract
    /// are preserved verbatim.
    pub fn evaluate_generic<F: CintFloat + bytemuck::Pod>(
        self,
    ) -> Result<TypedEvaluationOutput<F>, FacadeError> {
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

        // Set precision from the F type parameter (Plan 07 wiring note):
        // F::PRECISION maps to PrecisionKind::F64 for f64 and PrecisionKind::F32 for f32.
        // This must happen before any planner::evaluate call so kernel dispatchers (Plans 04/05)
        // pick the right monomorphization. Follows the f12_zeta caller-populates-after-new
        // precedent established in Plan 06.
        plan.precision = F::PRECISION;

        // Propagate f12_zeta from ExecutionOptions to operator_env_params (safe API path).
        if let Some(zeta) = self.request.options().f12_zeta {
            plan.operator_env_params.f12_zeta = Some(zeta);
        }
        // Propagate rinv_orig from ExecutionOptions to operator_env_params (safe API path, Plan 21-01).
        if let Some(origin) = self.request.options().rinv_orig {
            plan.operator_env_params.rinv_orig = Some(origin);
        }
        // Propagate common_orig from ExecutionOptions to operator_env_params (safe API path, Plan 22-01).
        if let Some(origin) = self.request.options().common_orig {
            plan.operator_env_params.common_orig = Some(origin);
        }
        // Phase 22 FND-01 (gap closure): finiteness-validate the gauge origin on the safe-API
        // path too. Mirrors the raw-path guard in cintx-compat raw.rs so the builder doc
        // contract ("NaN/inf rejected by validate_common_orig_env_params") holds on BOTH paths
        // — a `.with_common_origin([NaN, ..])` caller now gets InvalidEnvParam, not a silent
        // garbage origin threaded into the plan.
        cintx_runtime::validator::validate_common_orig_env_params(
            plan.descriptor.operator_name(),
            &plan.operator_env_params,
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

        // Allocate the output accumulator as Vec<F>.
        // Buffer is sized in elements; each element is size_of::<F>() bytes (T-20-20 mitigation).
        let staging_elements = output_layout.staging_elements;
        let staging_bytes = staging_elements
            .checked_mul(size_of::<F>())
            .ok_or(FacadeError::Memory {
                detail: "staging element byte count overflowed usize".to_owned(),
            })?;
        let mut owned_values: Vec<F> = Vec::new();
        owned_values
            .try_reserve_exact(staging_elements)
            .map_err(|_| FacadeError::Memory {
                detail: format!("failed to allocate staging buffer of {staging_bytes} bytes"),
            })?;
        owned_values.resize(staging_elements, F::zero());

        let schedule = schedule_chunks(&plan.workspace);
        let mut total_not0: i32 = 0;
        let mut total_transfer_bytes: usize = 0;
        let mut total_peak_workspace_bytes: usize = 0;

        for chunk in schedule.chunks() {
            // FND-06 / 25-02 chunk-staging contract (chunk-aware proven-sized output).
            //
            // The family kernels are MONOLITHIC whole-block writers: each launch
            // computes ALL components and ALL AO pairs of the shell tuple and scatters
            // them at ABSOLUTE output indices `[0, staging_elements)`. The executor
            // does NOT translate `chunk.work_unit_start` into a scatter offset. So
            // memory-limit chunking partitions the *workspace* (compute scratch,
            // `chunk.bytes`) — NOT the output buffer. Handing the kernel a
            // `chunk_len`-sized staging slice under-sizes the buffer (the absolute
            // `dst` overflows it); the now-removed `if dst < staging.len()` scatter
            // guards used to mask this with silent partial writes — the exact
            // anti-pattern FND-06 eliminates. Mirror of the fix in
            // `cintx-compat::eval_raw`. The OOM no-partial-write contract is preserved
            // upstream: `query_workspace` returns `MemoryLimitExceeded` (no staging
            // touched) when even one workspace chunk cannot fit the limit.
            //
            // chunk_staging is always Vec<f64> — the kernel dispatchers receive
            // &mut [f64] (frozen ExecutionIo interface) and reinterpret internally for
            // f32 precision. A Vec<f64> of `staging_elements` elements gives
            // `staging_elements * 2` f32 lanes; the f32 dispatcher bounds its writes to
            // `out_elems = staging.len()` pre-cast == `staging_elements`.
            let chunk_staging_bytes = staging_elements.checked_mul(size_of::<f64>()).ok_or(
                FacadeError::Memory {
                    detail: "chunk staging byte count overflowed usize".to_owned(),
                },
            )?;
            let mut chunk_staging: Vec<f64> = Vec::new();
            chunk_staging
                .try_reserve_exact(staging_elements)
                .map_err(|_| FacadeError::Memory {
                    detail: format!(
                        "failed to allocate chunk staging buffer of {chunk_staging_bytes} bytes"
                    ),
                })?;
            chunk_staging.resize(staging_elements, 0.0f64);

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
                // Each chunk recomputes the same full block, so `not0` is the SAME
                // full-block nonzero count every chunk — take the representative
                // value (max), not a sum, to avoid N× over-counting under multi-chunk.
                total_not0 = total_not0.max(chunk_stats.not0.max(0));
                total_transfer_bytes = total_transfer_bytes
                    .saturating_add(io.transfer_bytes())
                    .saturating_add(chunk_stats.transfer_bytes);
                total_peak_workspace_bytes =
                    total_peak_workspace_bytes.max(io.workspace().len());
            }
            allocator.release(workspace);

            // The kernel wrote the entire monolithic block into `chunk_staging`. Copy
            // the full block into the accumulator by reinterpreting the f64 buffer as
            // &[F] (f64: zero-cost identity cast; f32: the kernel wrote
            // `staging_elements` f32 values at indices 0..staging_elements in the f32
            // view). Each chunk recomputes the same full block (the kernel ignores
            // chunk boundaries for output), so the accumulator holds the complete,
            // correct output regardless of chunk_count.
            let chunk_as_f: &[F] = bytemuck::cast_slice::<f64, F>(&chunk_staging);
            let copy_len = chunk_as_f.len().min(staging_elements);
            owned_values[..copy_len].copy_from_slice(&chunk_as_f[..copy_len]);
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
            .checked_mul(size_of::<F>())
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
/// For complex outputs (`complex_interleaved == true`, e.g. Spinor), call `complex_values()`
/// to get the typed `Vec<Complex<F>>` view (D-04 / SC-2); `owned_values` remains the
/// underlying interleaved storage.
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

impl<F: CintFloat> IntegralTensor<F> {
    /// Complex view of the output for spinor/complex operators (D-04 / SC-2).
    ///
    /// Returns `Some(Vec<Complex<F>>)` when `complex_interleaved == true` (Spinor
    /// outputs): the contiguous interleaved `[re, im, re, im, ...]` `owned_values`
    /// buffer is reinterpreted element-for-element into `Complex<F>` (num_complex's
    /// `Complex<F>` is `#[repr(C)] { re, im }`, contiguous — this is a typed
    /// reinterpretation, not a data reshuffle). Returns `None` for real-valued
    /// outputs, where callers consume `owned_values: Vec<F>` directly.
    ///
    /// Migration note: callers that previously consumed the interleaved `Vec<F>` for
    /// complex outputs (manually pairing `owned_values[2i]`/`owned_values[2i+1]`) should
    /// now call `tensor.complex_values()` for the typed `Vec<Complex<F>>` view.
    pub fn complex_values(&self) -> Option<Vec<num_complex::Complex<F>>> {
        if !self.complex_interleaved {
            return None;
        }
        // owned_values.len() is even for complex_interleaved (re/im pairs).
        debug_assert_eq!(self.owned_values.len() % 2, 0);
        Some(
            self.owned_values
                .chunks_exact(2)
                .map(|pair| num_complex::Complex::new(pair[0], pair[1]))
                .collect(),
        )
    }
}

impl<F: CintFloat> TypedEvaluationOutput<F> {
    /// Convenience: complex view of the tensor (see `IntegralTensor::complex_values`).
    pub fn complex_values(&self) -> Option<Vec<num_complex::Complex<F>>> {
        self.tensor.complex_values()
    }
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

    // ------------------------------------------------------------------
    // Plan 20-07 Task 2: generic evaluate::<F: CintFloat>() and f64 shim.
    // ------------------------------------------------------------------

    #[test]
    fn cintfloat_precision_const_f64_is_f64() {
        use cintx_core::{CintFloat, PrecisionKind};
        assert_eq!(<f64 as CintFloat>::PRECISION, PrecisionKind::F64);
    }

    #[test]
    fn cintfloat_precision_const_f32_is_f32() {
        use cintx_core::{CintFloat, PrecisionKind};
        assert_eq!(<f32 as CintFloat>::PRECISION, PrecisionKind::F32);
    }

    #[test]
    fn evaluate_generic_f32_returns_vec_f32_with_nonzero_element() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0), // int1e_ovlp_cart
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let query = request.query_workspace().expect("query should succeed");
        let output: TypedEvaluationOutput<f32> = query
            .evaluate_generic::<f32>()
            .expect("evaluate::<f32> should succeed");
        // Must return Vec<f32>
        let _: Vec<f32> = output.tensor.owned_values.clone();
        // Must have at least one nonzero element
        let nonzero_count = output
            .tensor
            .owned_values
            .iter()
            .filter(|&&v| v.abs() > 1e-9_f32)
            .count();
        assert!(
            nonzero_count > 0,
            "evaluate::<f32> must produce at least one nonzero element"
        );
    }

    #[test]
    fn evaluate_unparameterized_delegates_to_f64_path() {
        // evaluate() (unparameterized) must return TypedEvaluationOutput<f64> byte-identically.
        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionRequest::new(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let q1 = request.clone().query_workspace().expect("q1");
        let out1 = q1.evaluate().expect("evaluate f64 unparameterized");
        let q2 = request.query_workspace().expect("q2");
        let out2 = q2.evaluate_generic::<f64>().expect("evaluate_generic::<f64>");
        // Must be byte-identical (same f64 values)
        assert_eq!(
            out1.tensor.owned_values, out2.tensor.owned_values,
            "evaluate() and evaluate_generic::<f64>() must be byte-identical"
        );
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

    // --- D-04 / ROADMAP SC-2 / Gap 1: Complex<F> typed view tests ---

    #[test]
    fn complex_values_returns_some_for_complex_interleaved_f64() {
        // Spinor output (complex_interleaved == true): complex_values() returns Some.
        // Buffer: [1.0, 2.0, 3.0, 4.0] -> [Complex(1,2), Complex(3,4)]
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64],
        };
        let cv = t.complex_values();
        assert!(cv.is_some(), "complex_values() must return Some when complex_interleaved == true");
        let v = cv.unwrap();
        assert_eq!(v.len(), 2, "len must be owned_values.len() / 2");
        assert_eq!(v[0], num_complex::Complex::new(1.0_f64, 2.0_f64));
        assert_eq!(v[1], num_complex::Complex::new(3.0_f64, 4.0_f64));
    }

    #[test]
    fn complex_values_returns_none_for_real_output() {
        // Real output (complex_interleaved == false): complex_values() returns None.
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: false,
            owned_values: vec![1.0_f64, 2.0_f64],
        };
        assert!(t.complex_values().is_none(), "complex_values() must return None when complex_interleaved == false");
    }

    #[test]
    fn complex_values_f32_typed() {
        // IntegralTensor<f32> complex view yields Complex<f32>.
        let t: IntegralTensor<f32> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
        };
        let cv = t.complex_values();
        assert!(cv.is_some());
        let v: Vec<num_complex::Complex<f32>> = cv.unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], num_complex::Complex::new(1.0_f32, 2.0_f32));
        assert_eq!(v[1], num_complex::Complex::new(3.0_f32, 4.0_f32));
    }

    #[test]
    fn typed_evaluation_output_complex_values_delegates_to_tensor() {
        // TypedEvaluationOutput::complex_values() delegates to tensor.complex_values().
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![1],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![5.0_f64, 6.0_f64],
        };
        let out = TypedEvaluationOutput {
            tensor: t,
            stats: EvaluationStats::default(),
            workspace_bytes: 0,
            chunk_count: 0,
            bytes_written: 0,
        };
        let cv = out.complex_values();
        assert!(cv.is_some());
        let v = cv.unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], num_complex::Complex::new(5.0_f64, 6.0_f64));
    }

    #[test]
    fn owned_values_field_unchanged_by_complex_view() {
        // The owned_values: Vec<F> field is accessible and byte-identical (PREC-04 / SemVer).
        let t: IntegralTensor<f64> = IntegralTensor {
            extents: vec![2],
            component_axis_leading: false,
            complex_interleaved: true,
            owned_values: vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64],
        };
        // Field must still be accessible as Vec<f64> directly.
        let raw: &Vec<f64> = &t.owned_values;
        assert_eq!(raw, &vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]);
    }

    /// Task 2: PREC-02 end-to-end smoke — a real spinor evaluate() exposes complex_values() == Some.
    ///
    /// This test drives `SessionRequest::evaluate()` with `Representation::Spinor` (int1e_ovlp_spinor,
    /// OperatorId 2) and asserts:
    /// 1. `output.complex_values().is_some()` — PREC-02/D-04/SC-2 literally TRUE.
    /// 2. `complex_values().unwrap().len() == output.tensor.owned_values.len() / 2` — correct pairing.
    /// 3. `owned_values` is unchanged (PREC-04 regression guard on the accessor path).
    #[test]
    fn spinor_evaluate_exposes_complex_values_some_prec02() {
        // int1e_ovlp_spinor = OperatorId 2; two s-shells (l=0) with Spinor representation.
        let (basis, shells) = sample_basis(Representation::Spinor);
        let request = SessionRequest::new(
            OperatorId::new(2), // int1e_ovlp_spinor
            Representation::Spinor,
            &basis,
            shells,
            ExecutionOptions::default(),
        );
        let query = request.query_workspace().expect("spinor query_workspace must succeed");
        let output = query.evaluate().expect("spinor evaluate must succeed");

        // PREC-02: complex_values() must return Some for Spinor output.
        assert!(
            output.complex_values().is_some(),
            "PREC-02 GAP-1: complex_values() must return Some for Spinor evaluate output (complex_interleaved == true)"
        );
        // Correct pairing: len == owned_values.len() / 2.
        let cv = output.complex_values().unwrap();
        assert_eq!(
            cv.len(),
            output.tensor.owned_values.len() / 2,
            "complex_values() length must be owned_values.len() / 2"
        );
        // PREC-04: owned_values still accessible and unchanged.
        assert!(
            !output.tensor.owned_values.is_empty(),
            "owned_values must be non-empty (PREC-04 regression guard)"
        );
        // Spinor overlap must produce at least one nonzero complex element.
        let nonzero_count = cv.iter().filter(|c| c.norm_sqr() > 1e-36).count();
        assert!(
            nonzero_count > 0,
            "spinor evaluate must produce at least one nonzero Complex<f64> element"
        );
    }

    // Phase 22 FND-01 gap closure: the safe-API path must finiteness-validate the gauge
    // origin, matching the raw-path guard in cintx-compat raw.rs. Before this fix the
    // builder doc-contract ("NaN/inf rejected") held only on the raw path; a
    // `.with_common_origin([NaN, ..])` caller silently threaded garbage into the plan.
    #[test]
    fn evaluate_rejects_non_finite_common_origin_on_safe_api_path() {
        use crate::builder::SessionBuilder;

        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionBuilder::new(OperatorId::new(0), Representation::Cart, &basis, shells)
            .with_common_origin([f64::NAN, 0.0, 0.0])
            .build();

        let query = request.query_workspace().expect("query should succeed");
        let err = query.evaluate().unwrap_err();

        assert_eq!(
            err.kind(),
            FacadeErrorKind::Validation,
            "non-finite gauge origin must surface as a Validation error on the safe path"
        );
        assert!(
            err.to_string().contains("PTR_COMMON_ORIG"),
            "error must name the PTR_COMMON_ORIG param; got: {err}"
        );
    }

    // Companion guard: a finite gauge origin on the safe-API path is accepted and
    // round-trips into the plan (no false rejection from the new validate call).
    #[test]
    fn evaluate_accepts_finite_common_origin_on_safe_api_path() {
        use crate::builder::SessionBuilder;

        let (basis, shells) = sample_basis(Representation::Cart);
        let request = SessionBuilder::new(OperatorId::new(0), Representation::Cart, &basis, shells)
            .with_common_origin([0.5, -0.3, 0.8])
            .build();

        let query = request.query_workspace().expect("query should succeed");
        query
            .evaluate()
            .expect("finite gauge origin must be accepted on the safe path");
    }
}
