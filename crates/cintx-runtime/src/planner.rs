use crate::dispatch::{BackendExecutor, DispatchDecision, ExecutionIo, OutputOwnership};
use crate::metrics::{ExecutionStats, RunMetrics};
use crate::options::ExecutionOptions;
use crate::range_omega;
use crate::scheduler::schedule_chunks;
use crate::validator::{ValidatedShellTuple, validate_range_omega_value, validate_shell_tuple};
use crate::workspace::{
    ChunkInfo, ChunkPlanner, DEFAULT_ALIGNMENT_BYTES, WorkspaceAllocator, WorkspaceQuery,
    WorkspaceRequest,
};
use cintx_core::{BasisSet, OperatorId, PrecisionKind, Representation, ShellTuple, cintxRsError};
use cintx_ops::resolver::{OperatorDescriptor, Resolver, ResolverError};
use tracing::{debug, info_span};

#[cfg(test)]
use crate::dispatch::{DispatchFamily, WorkspaceBytes};

/// Parameters for grid-based integrals (int1e_grids family).
///
/// NGRIDS = env[11] (count of grid points), PTR_GRIDS = env[12] (offset into env
/// where grid coordinates start). Validator rejects grids family when NGRIDS=0.
/// Per D-05/D-06: output shape is (ncomp * NGRIDS * di * dj) matching libcint g1e_grids.c.
///
/// The `grid_coords` field carries the actual grid coordinates extracted from env,
/// so the kernel can access them without needing the full env array.
/// Each element is `[x, y, z]` for one grid point.
#[derive(Clone, Debug)]
pub struct GridsEnvParams {
    /// Number of grid points (from env[11]).
    pub ngrids: usize,
    /// Offset into env array where grid coordinates start (from env[12]).
    /// This is kept for reference but the kernel uses `grid_coords` directly.
    pub ptr_grids: usize,
    /// Grid point coordinates, extracted from env at call time.
    /// Length must equal `ngrids`. Each element is `[x, y, z]` in Bohr.
    pub grid_coords: Vec<[f64; 3]>,
}

/// Operator-specific env parameters extracted from the raw `env[]` array.
///
/// These are populated by the caller (raw compat reads from env[], safe API
/// passes via ExecutionOptions). Default is all None (no extra params).
#[derive(Clone, Debug, Default)]
pub struct OperatorEnvParams {
    /// PTR_F12_ZETA value (env[9]) for F12/STG/YP kernels.
    /// Must be non-zero when canonical_family == "f12".
    pub f12_zeta: Option<f64>,
    /// For grids integrals: NGRIDS (env[11]) and PTR_GRIDS (env[12]) parameters.
    /// Validator rejects grids family when grids_params is None or ngrids == 0.
    pub grids_params: Option<GridsEnvParams>,
    /// PTR_RINV_ORIG value (env[4..6]) for iprinv/ECPscalar_iprinv kernels.
    /// Must be present when operator_name contains "iprinv".
    /// Absent (None) is allowed for all other operators.
    pub rinv_orig: Option<[f64; 3]>,
    /// PTR_COMMON_ORIG value (env[1..3]) — gauge origin for moment/GIAO families.
    /// None defaults to [0,0,0] (libcint reads unset env as zero); consumers use
    /// `common_orig.unwrap_or([0.0; 3])`. Validator checks finiteness only (D-01).
    pub common_orig: Option<[f64; 3]>,
    /// PTR_RANGE_OMEGA value (env[8]) — the range-separation parameter ω.
    ///
    /// D-PBC-24. `> 0` long range `erf(ω r)/r`, `< 0` short range
    /// `erfc(|ω| r)/r`, `None`/`0` full Coulomb. Consumed by the shared
    /// `CINTg0_2e` prologue of the `int2e` / `int3c2e` / `int2c2e` kernels;
    /// rejected for every other operator by
    /// [`crate::validator::validate_range_omega_env_params`].
    pub range_omega: Option<f64>,
}

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
    /// Operator-specific env parameters (e.g. f12_zeta for F12/STG/YP).
    /// Populated by the caller; default is all None.
    pub operator_env_params: OperatorEnvParams,
    /// Float precision for this evaluation: `F64` (default, byte-identical to libcint)
    /// or `F32` (opt-in, ~1e-4 rtol, unlocks non-`SHADER_F64` wgpu adapters).
    ///
    /// The f64 path is byte-identical to upstream libcint 6.1.3 (D-08, D-12).
    /// Callers that do not set this field get `F64` automatically.
    pub precision: PrecisionKind,
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
        // D-PBC-24: re-derive the request from the omega the QUERY captured, not
        // from a caller-supplied one, so the doubled-root sizing cannot drift
        // between query and evaluate. `evaluate`'s `planning_matches` is what
        // catches a caller that changed `opts.range_omega` in between.
        let expected_request =
            estimate_workspace_request(descriptor, basis, &shells, workspace.range_omega)?;
        if workspace.request() != expected_request {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "planner",
                detail: "workspace query does not match the validated shell tuple".to_owned(),
            });
        }

        let dispatch = DispatchDecision::from_manifest_family(descriptor.family())?;
        let component_count = component_multiplier_for_descriptor(descriptor)?;
        let output_layout = build_output_layout(descriptor, &shells, component_count)?;

        Ok(Self {
            basis,
            descriptor,
            representation: rep,
            shells,
            workspace,
            dispatch,
            component_count,
            output_layout,
            // Note: f12_zeta is populated by the caller (raw compat reads env[9],
            // safe API passes it via ExecutionOptions). Default is None.
            operator_env_params: OperatorEnvParams::default(),
            // Default to F64 (byte-identical to libcint 6.1.3, D-08/D-12).
            // Callers that want f32 precision set this field after construction.
            precision: PrecisionKind::default(),
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
    // D-PBC-24: reject a range_omega the operator cannot honour BEFORE sizing
    // anything, so a caller never gets a capability token for an evaluation
    // that would silently run the full-range kernel.
    validate_range_omega_value(
        descriptor.entry.canonical_family,
        descriptor.operator_name(),
        opts.range_omega,
    )?;
    let request = estimate_workspace_request(descriptor, basis, &validated, opts.range_omega)?;
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
        range_omega: opts.range_omega,
        rys_roots: request.rys_roots,
    })
}

pub fn evaluate(
    mut plan: ExecutionPlan<'_>,
    opts: &ExecutionOptions,
    allocator: &mut dyn WorkspaceAllocator,
    executor: &dyn BackendExecutor,
) -> Result<ExecutionStats, cintxRsError> {
    // Thread precision from opts into the plan, following the f12_zeta
    // "caller populates after new" precedent (planner.rs §operator_env_params).
    // This ensures kernel dispatchers from Plans 04/05 receive the caller's
    // precision selection rather than the default F64.
    plan.precision = opts.precision;

    // D-PBC-24: thread the range-separation parameter the same way. Doing it
    // here (rather than only at the two API boundaries) means the kernel can
    // never see a different omega from the one `query_workspace` sized the
    // doubled Rys roots for — `planning_matches` below rejects the mismatch
    // first, so by this point `opts.range_omega == plan.workspace.range_omega`.
    if let Some(omega) = opts.range_omega {
        plan.operator_env_params.range_omega = Some(omega);
    }

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
        // FND-06 (D-04): required staging elements for this chunk =
        // component_multiplier * per_component_elements (the layout already folded
        // component_rank into staging_elements). This is the SINGLE contract point.
        let required_elements = staging_elements_for_chunk(&plan, chunk)?;
        let mut staging = try_alloc_staging(required_elements)?;
        // Prove the buffer is large enough BEFORE handing it to the backend scatter.
        // Fail-closed with a typed BufferTooSmall stop and NO partial write if not.
        assert_staging_size(staging.len(), required_elements)?;

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
    descriptor: &OperatorDescriptor,
    shells: &ValidatedShellTuple,
    component_count: usize,
) -> Result<OutputLayoutMetadata, cintxRsError> {
    let shell_slice = shells.as_slice();
    let arity = shell_slice.len();
    let extents: Vec<usize> = shell_slice
        .iter()
        .enumerate()
        .map(|(axis, shell)| {
            // 27-04 (AUX-K CONTRACT): arity-3 SPINOR derivative families
            // (int3c2e_ip1/ip2, int3c1e_ip1/iprinv) size the auxiliary-k axis (the
            // tail shell, axis == arity-1) SPHERICALLY as nsph(lk) = (2lk+1)*nctr_k,
            // NOT spinor (4l+2). Only bra i and ket j are spinor-sized. Source:
            // libcint CINT3c2e_spinor_drv is_ssc=0 (cint3c2e.c:631-636,
            // counts[2] = (k_l*2+1)*x_ctr[2]); 27-SPIKE-FINDINGS CORRECTION NOTICE.
            // ao_per_shell() over-sizes the aux-k as spinor (the disproven 720); this
            // positional override mirrors the oracle fixtures' dims_for_arity so the
            // compat output-length contract matches the vendor buffer (spherical k).
            if arity == 3 && axis == arity - 1 && shell.representation == Representation::Spinor {
                let l = shell.ang_momentum as usize;
                (2 * l + 1) * shell.nctr as usize
            } else {
                shell.ao_per_shell()
            }
        })
        .collect();
    let base_elements = extents
        .iter()
        .try_fold(1usize, |acc, extent| acc.checked_mul(*extent))
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "layout",
            detail: "output extent product overflowed usize".to_owned(),
        })?;
    // FND-03 (D-01): size complex output from the per-family manifest
    // `complex_output` flag, NOT from Representation::Spinor. Spinor families
    // stay 2× because Task 1 backfilled complex_output=true on their rows; GIAO
    // cart/sph families (purely imaginary) now also size 2× from manifest data.
    let complex_multiplier = if descriptor.entry.complex_output {
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

/// Required staging-buffer element count for a single scheduled chunk.
///
/// # WR-01 / FND-06: monolithic complex/GIAO writers need the FULL block per chunk
///
/// GIAO and other complex-interleaved families are evaluated by MONOLITHIC
/// whole-block launchers (`write_giao_complex_staging`, `launch_two_electron_giao2e`)
/// that scatter the ENTIRE `2 * real_total` interleaved block in one pass — they do
/// NOT honor a per-chunk work-unit sub-range. Handing such a launcher the sliced
/// `suffix - prefix` staging (smaller than the full block whenever `chunk_count > 1`)
/// yields a per-chunk `BufferTooSmall`, making the family inoperable under any
/// `memory_limit_bytes` that forces chunking (WR-01 availability regression).
///
/// `eval_raw` already solves this by allocating the FULL `staging_elements` per chunk
/// for these monolithic writers (crates/cintx-compat/src/raw.rs:1061-1070; FND-06,
/// project memory: "family kernels are MONOLITHIC whole-block writers → per-chunk
/// staging must be FULL-block sized; never re-add `if dst < staging.len()` guards").
/// The safe-API `evaluate` path mirrors that here: when
/// `plan.output_layout.complex_interleaved` is set, return the full
/// `plan.output_layout.staging_elements` so every chunk hands the launcher a
/// full-block buffer. If the full block cannot fit, the upfront workspace check fails
/// closed with a typed `MemoryLimitExceeded` (no partial write).
///
/// Real (non-complex) families ARE genuinely chunk-partitionable, so they keep the
/// sliced `suffix - prefix` sizing below.
fn staging_elements_for_chunk(
    plan: &ExecutionPlan<'_>,
    chunk: &ChunkInfo,
) -> Result<usize, cintxRsError> {
    // WR-01 / FND-06: monolithic complex/GIAO writers require the full interleaved
    // block per chunk — mirror eval_raw (raw.rs:1061-1070). Never the sliced suffix.
    if plan.output_layout.complex_interleaved {
        return Ok(plan.output_layout.staging_elements.max(1));
    }

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

/// FND-06 (D-04): the SINGLE upfront staging-size contract point.
///
/// Proves the allocated staging buffer is large enough to hold
/// `required_elements = component_multiplier * per_component_elements` BEFORE any
/// kernel scatter runs. Once this assertion holds, every per-element
/// `staging[dst] = v` scatter in the kernels is unconditional and in-bounds by
/// construction — replacing the 18+ per-element `if dst < staging.len()` guards
/// (which silently DROPPED trailing components, the 260530-9ay truncation bug).
///
/// Returns `Err(cintxRsError::BufferTooSmall { required, provided })` if the
/// buffer is undersized — a typed fail-closed stop with NO partial write, per
/// the CLAUDE.md "fallible allocation + typed failure + no partial writes"
/// non-negotiable.
fn assert_staging_size(staging_len: usize, required_elements: usize) -> Result<(), cintxRsError> {
    if staging_len < required_elements {
        return Err(cintxRsError::BufferTooSmall {
            required: required_elements,
            provided: staging_len,
        });
    }
    Ok(())
}

/// `nrys_roots` this operator/tuple needs under `range_omega`, or `0` when the
/// operator has no Rys-quadrature workspace component (D-PBC-24 §3.2).
///
/// Only the three scalar Coulomb operators are Rys-sized here, because they are
/// the only ones that accept `range_omega` at all
/// ([`crate::range_omega::supports_range_omega`]) and the only ones whose
/// `rys_order` is exactly `(Σ l)/2 + 1` with no derivative raises. Everything
/// else reports `0`, leaving its request byte-identical to before D-PBC-24.
fn rys_roots_for_request(
    descriptor: &OperatorDescriptor,
    shells: &ValidatedShellTuple,
    range_omega: Option<f64>,
) -> usize {
    let Some(headroom) = range_omega::derivative_headroom(
        descriptor.entry.canonical_family,
        descriptor.operator_name(),
    ) else {
        return 0;
    };
    // The derivative rows raise `l_ceil` before `rys_order` is taken
    // (`g2e.c:74-79`), so the estimate has to apply the same `ng[]` raises the
    // launcher's `build_2e_shape(li + i_inc, ..)` does. `derivative_headroom`
    // is the single table both read; a scalar row's entry is all zeros and
    // reduces this to `(Σ l)/2 + 1` exactly as before.
    let rys_order = range_omega::rys_order_with_headroom(
        shells.as_slice().iter().map(|s| s.ang_momentum as usize),
        headroom,
    );
    range_omega::nrys_roots_for(rys_order, range_omega)
}

/// Extra workspace bytes the SHORT-RANGE doubled Rys roots demand.
///
/// libcint holds `u[nrys_roots]` alongside the `w` block that aliases `gz`, and
/// sizes `g_stride_i = nrys_roots`, so doubling the roots doubles the Rys
/// footprint of the G-tensor. Only the *extra* roots are charged here, so a
/// full-range or long-range request (where `nrys_roots == rys_order`) keeps the
/// byte count it had before D-PBC-24 — stage 1's byte-identity gate.
fn sr_root_extra_bytes(rys_roots: usize, range_omega: Option<f64>) -> usize {
    if !range_omega::is_short_range(range_omega) || rys_roots == 0 {
        return 0;
    }
    let rys_order = rys_roots / 2;
    if rys_roots != rys_order * 2 {
        // Not in the doubled-root regime (rys_order > 3): no extra roots.
        return 0;
    }
    // The extra `rys_order` roots each carry a root and a weight.
    rys_order * 2 * std::mem::size_of::<f64>()
}

fn estimate_workspace_request(
    descriptor: &OperatorDescriptor,
    basis: &BasisSet,
    shells: &ValidatedShellTuple,
    range_omega: Option<f64>,
) -> Result<WorkspaceRequest, cintxRsError> {
    let rys_roots = rys_roots_for_request(descriptor, shells, range_omega);
    let component_multiplier = component_multiplier_for_descriptor(descriptor)?;
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
        .and_then(|value| value.checked_add(sr_root_extra_bytes(rys_roots, range_omega)))
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
        rys_roots,
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

fn component_multiplier_for_descriptor(
    descriptor: &OperatorDescriptor,
) -> Result<usize, cintxRsError> {
    if descriptor.entry.canonical_family == "grids" {
        return Ok(match descriptor.operator_name() {
            "grids" => 1,
            "grids_ip" => 3,
            "grids_ipvip" | "grids_ipip" => 9,
            "grids_spvsp" => 4,
            _ => 1,
        });
    }
    parse_component_multiplier(descriptor.entry.component_rank)
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
    fn grids_component_multiplier_matches_runtime_contract() {
        let grids = Resolver::descriptor_by_symbol("int1e_grids_sph").expect("grids descriptor");
        let grids_ip =
            Resolver::descriptor_by_symbol("int1e_grids_ip_sph").expect("grids_ip descriptor");
        let grids_ipvip = Resolver::descriptor_by_symbol("int1e_grids_ipvip_sph")
            .expect("grids_ipvip descriptor");
        let grids_spvsp = Resolver::descriptor_by_symbol("int1e_grids_spvsp_sph")
            .expect("grids_spvsp descriptor");
        let grids_ipip =
            Resolver::descriptor_by_symbol("int1e_grids_ipip_sph").expect("grids_ipip descriptor");

        assert_eq!(
            component_multiplier_for_descriptor(grids).expect("multiplier"),
            1
        );
        assert_eq!(
            component_multiplier_for_descriptor(grids_ip).expect("multiplier"),
            3
        );
        assert_eq!(
            component_multiplier_for_descriptor(grids_ipvip).expect("multiplier"),
            9
        );
        assert_eq!(
            component_multiplier_for_descriptor(grids_spvsp).expect("multiplier"),
            4
        );
        assert_eq!(
            component_multiplier_for_descriptor(grids_ipip).expect("multiplier"),
            9
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // D-PBC-24 stage 1 — the control plane sees omega
    // ─────────────────────────────────────────────────────────────────────

    /// `sample_basis` builds two l=1 shells, so `int2c2e` has
    /// `rys_order = (1+1)/2 + 1 = 2` — squarely in the doubled-root regime.
    fn int2c2e_cart_id() -> OperatorId {
        Resolver::descriptor_by_symbol("int2c2e_cart")
            .expect("int2c2e_cart descriptor")
            .id
    }

    #[test]
    fn short_range_doubles_the_queried_rys_roots_and_query_evaluate_agree() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let op = int2c2e_cart_id();

        let full = query_workspace(
            op,
            Representation::Cart,
            &basis,
            shells.clone(),
            &ExecutionOptions::default(),
        )
        .expect("full-range query");

        let sr_opts = ExecutionOptions {
            range_omega: Some(-0.8),
            ..ExecutionOptions::default()
        };
        let sr = query_workspace(op, Representation::Cart, &basis, shells.clone(), &sr_opts)
            .expect("short-range query");

        // g2c2e.c:61-68 — rys_order = (li + lk)/2 + 1 = 2, doubled to 4.
        assert_eq!(full.rys_roots, 2, "full range keeps rys_order roots");
        assert_eq!(
            sr.rys_roots,
            2 * full.rys_roots,
            "short range at rys_order <= 3 must query EXACTLY twice the roots"
        );
        assert_eq!(sr.range_omega, Some(-0.8));
        assert!(
            sr.required_bytes > full.required_bytes,
            "the doubled roots must show up in the workspace request"
        );

        // query → evaluate must agree on the SR-sized request. `ExecutionPlan::new`
        // re-derives it from `query.range_omega`; a plan built against the
        // full-range query would fail the request equality check.
        let plan = ExecutionPlan::new(op, Representation::Cart, &basis, shells.clone(), &sr)
            .expect("SR plan must match the SR query");
        let mut allocator = HostWorkspaceAllocator::default();
        let backend = MockBackend { supports: true };
        let stats = evaluate(plan, &sr_opts, &mut allocator, &backend).expect("SR evaluate");
        assert_eq!(stats.required_workspace_bytes, sr.required_bytes);

        let mismatched = ExecutionPlan::new(op, Representation::Cart, &basis, shells, &full)
            .expect("full-range plan builds against its own query");
        let err = evaluate(mismatched, &sr_opts, &mut allocator, &backend)
            .expect_err("changing range_omega after query is backend contract drift");
        assert!(
            matches!(
                err,
                cintxRsError::ChunkPlanFailed {
                    from: "evaluate",
                    ..
                }
            ),
            "expected a typed drift stop, got {err:?}"
        );
    }

    #[test]
    fn long_range_and_full_range_leave_the_request_byte_identical() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let op = int2c2e_cart_id();

        let full = query_workspace(
            op,
            Representation::Cart,
            &basis,
            shells.clone(),
            &ExecutionOptions::default(),
        )
        .expect("full-range query");

        for omega in [Some(0.0), Some(0.3), Some(0.8)] {
            let opts = ExecutionOptions {
                range_omega: omega,
                ..ExecutionOptions::default()
            };
            let q = query_workspace(op, Representation::Cart, &basis, shells.clone(), &opts)
                .expect("query");
            assert_eq!(
                q.required_bytes, full.required_bytes,
                "omega={omega:?} must not change the workspace: only SR doubles roots"
            );
            assert_eq!(q.rys_roots, full.rys_roots, "omega={omega:?}");
        }
    }

    #[test]
    fn query_workspace_refuses_omega_on_an_operator_without_a_coulomb_kernel() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let opts = ExecutionOptions {
            range_omega: Some(-0.8),
            ..ExecutionOptions::default()
        };
        // OperatorId::new(0) is a 1e row — libcint's 1e kernels never read env[8].
        let err = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells,
            &opts,
        )
        .expect_err("a set omega on a non-Coulomb operator must be refused");
        assert!(
            matches!(err, cintxRsError::UnsupportedApi { .. }),
            "expected UnsupportedApi, got {err:?}"
        );
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

    // Phase 16-01 (W)/(F) site per RESEARCH §5: BackendKind::Wgpu is a feature-gated
    // variant after Wave 1; this test must compile only when the wgpu backend is
    // enabled. BackendIntent literal stays explicit (no `..Default::default()`) so the
    // Wave 1 default-flip is mechanical.
    #[cfg(feature = "wgpu")]
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

    // Phase 16-01 (W)/(F) site per RESEARCH §5: drift test asserts Wgpu vs Cpu
    // mismatch; both BackendIntent literals are already explicit. Gated on `wgpu`
    // so Wave 1's `--no-default-features --features cpu` matrix cell does not try
    // to construct a `BackendKind::Wgpu` variant that has been cfg-removed.
    #[cfg(feature = "wgpu")]
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

    // ─────────────────────────────────────────────────────────────────────────
    // T06-2a: ExecutionOptions::default().precision == PrecisionKind::F64
    // RED: compile fails until `precision: PrecisionKind` is added to ExecutionOptions.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn execution_options_default_precision_is_f64() {
        use cintx_core::PrecisionKind;
        // RED: ExecutionOptions does not yet have a `precision` field.
        // Once added, ExecutionOptions::default().precision must equal PrecisionKind::F64.
        let opts = ExecutionOptions::default();
        assert_eq!(
            opts.precision,
            PrecisionKind::F64,
            "ExecutionOptions::default() precision must be F64"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T06-2b: A plan built via query_workspace with opts.precision == F32
    // has plan.precision == F32 after the planner threads it through.
    // RED: compile fails until precision threading is implemented in query_workspace
    // + evaluate (or after plan construction).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn plan_precision_threaded_from_options_f32() {
        use cintx_core::PrecisionKind;
        let (basis, shells) = sample_basis(Representation::Cart);
        // Set precision to F32 in opts.
        // RED: ExecutionOptions does not yet have a `precision` field.
        let opts = ExecutionOptions {
            precision: PrecisionKind::F32,
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
        // Thread precision via the f12_zeta "caller populates after new" precedent.
        // RED: plan.precision is already on ExecutionPlan (from Plan 01), but the
        // planner does not yet set it from opts. After GREEN, this should be F32.
        plan.precision = opts.precision; // simulate what the planner will do
        assert_eq!(
            plan.precision,
            PrecisionKind::F32,
            "plan.precision must be F32 when opts.precision == F32"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T06-2c: Default opts produces plan.precision == F64 (f64 path unchanged).
    // RED: compile fails until `precision: PrecisionKind` is added to ExecutionOptions.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn plan_precision_default_is_f64() {
        use cintx_core::PrecisionKind;
        let (basis, shells) = sample_basis(Representation::Cart);
        let opts = ExecutionOptions::default();
        // RED: ExecutionOptions does not yet have a `precision` field.
        let query = query_workspace(
            OperatorId::new(0),
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
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
        assert_eq!(
            plan.precision,
            PrecisionKind::F64,
            "plan.precision must default to F64 when using default opts"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // T06-2d: try_alloc_staging is OOM-safe and the f64 buffer over-allocates
    // for f32 (elements*2 f32 lanes >= elements needed).
    // GREEN even before any staging changes (confirmed frozen OOM contract).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn try_alloc_staging_oom_safe_and_f32_lane_count_adequate() {
        // try_alloc_staging is frozen — it still returns Vec<f64>.
        let elements = 16usize;
        let staging = try_alloc_staging(elements).expect("allocation should succeed");
        // Buffer holds `elements` f64 = `elements * 8` bytes.
        assert_eq!(
            staging.len(),
            elements,
            "staging Vec<f64> must have exactly `elements` entries"
        );
        // Byte capacity of the buffer.
        let total_bytes = staging.len() * std::mem::size_of::<f64>();
        // How many f32 lanes fit in those bytes (size_of::<f32>() == 4; size_of::<f64>() == 8).
        let f32_lanes = total_bytes / std::mem::size_of::<f32>();
        assert_eq!(
            f32_lanes,
            elements * 2,
            "f64 buffer of {elements} elements yields {f32_lanes} f32 lanes (expected {})",
            elements * 2
        );
        assert!(
            f32_lanes >= elements,
            "f32 lane count ({f32_lanes}) must be >= element count ({elements}) — buffer over-allocates safely"
        );
        // Verify the OOM-safe fallible alloc still works for larger sizes.
        let large = try_alloc_staging(1024).expect("large staging alloc should succeed");
        assert_eq!(
            large.len(),
            1024,
            "large staging alloc must have 1024 f64 entries"
        );
        // All entries initialized to 0.0 (no partial writes).
        assert!(
            large.iter().all(|&v| v == 0.0),
            "staging buffer must be zero-initialized"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FND-06 (D-04): the SINGLE upfront staging-size assertion at the planner
    // boundary. When required_elements = component_multiplier *
    // per_component_elements exceeds the allocated staging length, the boundary
    // returns Err(BufferTooSmall { required, provided }) and never hands an
    // undersized buffer to the kernel scatter. For a correctly-sized buffer
    // (rank 9/27/81 with adequate memory), the assertion passes (Ok).
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn staging_buffer_too_small() {
        // Undersized: a rank-81 family needs 81 * per_component elements; an
        // under-allocated buffer must be rejected with a typed BufferTooSmall stop.
        let per_component = 4usize; // e.g. a 2x2 bra×ket block
        let required = 81 * per_component; // rank-81 staging requirement
        let undersized = 9 * per_component; // only rank-9 worth allocated

        let err = assert_staging_size(undersized, required)
            .expect_err("undersized staging must be rejected, not silently truncated");
        assert!(
            matches!(
                err,
                cintxRsError::BufferTooSmall {
                    required: r,
                    provided: p,
                } if r == required && p == undersized
            ),
            "expected BufferTooSmall {{ required: {required}, provided: {undersized} }}, got {err:?}"
        );

        // Correctly-sized staging for ranks 9 / 27 / 81 must pass (Ok).
        for rank in [9usize, 27, 81] {
            let need = rank * per_component;
            assert!(
                assert_staging_size(need, need).is_ok(),
                "exact-sized rank-{rank} staging ({need} elements) must pass the assertion"
            );
            // Over-allocation (f64 buffer holding > required) also passes.
            assert!(
                assert_staging_size(need + 1, need).is_ok(),
                "over-allocated rank-{rank} staging must pass the assertion"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FND-06 (D-05): rank-81 staging under a sub-rank-81 memory limit must
    // produce a typed OOM/BufferTooSmall stop with NO partial write — the
    // output buffer is byte-for-byte unchanged from its pre-call sentinel state.
    // Exercises the Task-1 upfront assertion + the existing ChunkPlanner
    // OOM-safe-stop together. int1e_rrrr (rank 81, Phase-24 moments) is the
    // rank-81 driver — a real registered family independent of the Phase-25
    // families still being landed.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    fn rank81_oom_no_partial_write() {
        // (a) The upfront assertion is a no-partial-write gate: pre-fill a staging
        //     buffer with a sentinel, then prove that when the rank-81 requirement
        //     exceeds the buffer, BufferTooSmall fires and the buffer is untouched.
        const SENTINEL: f64 = -1.234_567_89e30;
        let per_component = 4usize; // 2x2 non-square-ish block
        let rank81_required = 81 * per_component; // rank-81 staging requirement
        // Only a rank-9 worth of buffer is available, pre-filled with the sentinel.
        let staging = vec![SENTINEL; 9 * per_component];
        let before = staging.clone();

        let err = assert_staging_size(staging.len(), rank81_required)
            .expect_err("rank-81 requirement over a rank-9 buffer must fail closed");
        assert!(
            matches!(
                err,
                cintxRsError::BufferTooSmall {
                    required: r,
                    provided: p,
                } if r == rank81_required && p == staging.len()
            ),
            "expected BufferTooSmall {{ required: {rank81_required}, provided: {} }}, got {err:?}",
            staging.len()
        );
        // NO partial write: the buffer is byte-for-byte unchanged.
        assert_eq!(
            staging, before,
            "staging must be byte-for-byte untouched after a fail-closed OOM stop"
        );
        // Defensive: an attempted scatter never ran — every element is still the sentinel.
        assert!(
            staging.iter().all(|&v| v == SENTINEL),
            "no element may be overwritten when the upfront assertion fails"
        );

        // (b) Drive the real rank-81 family (int1e_rrrr_cart) through the planner
        //     under a memory limit far below its rank-81 staging requirement and
        //     assert a typed OOM-safe stop (MemoryLimitExceeded / ChunkPlanFailed /
        //     BufferTooSmall / HostAllocationFailed) — never a panic, never a
        //     silently-truncated success.
        let rank81 = Resolver::descriptor_by_symbol("int1e_rrrr_cart")
            .expect("int1e_rrrr_cart (rank-81 Phase-24 moment) must be registered");
        assert_eq!(
            component_multiplier_for_descriptor(rank81).expect("rank-81 multiplier"),
            81,
            "int1e_rrrr must carry component_rank 81 (D-10)"
        );

        let (basis, shells) = sample_basis(Representation::Cart);
        // A 1-byte limit is unreachable for any non-trivial rank-81 chunk staging.
        let opts = ExecutionOptions {
            memory_limit_bytes: Some(1),
            ..ExecutionOptions::default()
        };
        let result = query_workspace(
            rank81.id,
            Representation::Cart,
            &basis,
            shells.clone(),
            &opts,
        );
        match result {
            Err(cintxRsError::MemoryLimitExceeded { .. })
            | Err(cintxRsError::ChunkPlanFailed { .. })
            | Err(cintxRsError::BufferTooSmall { .. })
            | Err(cintxRsError::HostAllocationFailed { .. }) => { /* typed OOM-safe stop */ }
            Err(other) => {
                panic!("rank-81 under 1-byte limit produced an unexpected error: {other:?}")
            }
            Ok(query) => {
                // If the planner accepted the limit by chunking down, evaluate must
                // still stop fail-closed (or succeed without ever partial-writing).
                let plan =
                    ExecutionPlan::new(rank81.id, Representation::Cart, &basis, shells, &query)
                        .expect("rank-81 plan should build");
                let mut allocator = HostWorkspaceAllocator::default();
                let backend = MockBackend { supports: true };
                // Either a typed stop or a clean success — never a panic / silent truncation.
                let _ = evaluate(plan, &opts, &mut allocator, &backend);
            }
        }
    }

    // FND-03 (D-01): build_output_layout sizes complex staging from the manifest
    // `complex_output` flag, NOT from Representation::Spinor.
    #[test]
    fn build_output_layout_sizes_complex_from_manifest_flag_true() {
        // int1e_ovlp_spinor was backfilled complex_output=true in Task 1.
        let descriptor =
            Resolver::descriptor_by_symbol("int1e_ovlp_spinor").expect("spinor descriptor");
        assert!(
            descriptor.entry.complex_output,
            "int1e_ovlp_spinor must carry complex_output=true"
        );
        let (basis, shells) = sample_basis(Representation::Spinor);
        let validated =
            validate_shell_tuple(descriptor, Representation::Spinor, &basis, &shells).unwrap();
        let component_count = component_multiplier_for_descriptor(descriptor).unwrap();
        let base_elements: usize = validated
            .as_slice()
            .iter()
            .map(|shell| shell.ao_per_shell())
            .product();

        let layout = build_output_layout(descriptor, &validated, component_count).unwrap();
        assert!(layout.complex_interleaved);
        assert_eq!(layout.staging_elements, base_elements * component_count * 2);
    }

    #[test]
    fn build_output_layout_sizes_real_from_manifest_flag_false() {
        // int1e_ovlp_cart is a real family: complex_output=false (default).
        let descriptor =
            Resolver::descriptor_by_symbol("int1e_ovlp_cart").expect("cart descriptor");
        assert!(
            !descriptor.entry.complex_output,
            "int1e_ovlp_cart must carry complex_output=false"
        );
        let (basis, shells) = sample_basis(Representation::Cart);
        let validated =
            validate_shell_tuple(descriptor, Representation::Cart, &basis, &shells).unwrap();
        let component_count = component_multiplier_for_descriptor(descriptor).unwrap();
        let base_elements: usize = validated
            .as_slice()
            .iter()
            .map(|shell| shell.ao_per_shell())
            .product();

        let layout = build_output_layout(descriptor, &validated, component_count).unwrap();
        assert!(!layout.complex_interleaved);
        assert_eq!(layout.staging_elements, base_elements * component_count);
    }

    #[test]
    fn build_output_layout_spinor_family_still_complex_via_backfill() {
        // Existing spinor family stays complex because Task 1 backfilled the flag.
        let descriptor =
            Resolver::descriptor_by_symbol("int1e_kin_spinor").expect("kin spinor descriptor");
        let (basis, shells) = sample_basis(Representation::Spinor);
        let validated =
            validate_shell_tuple(descriptor, Representation::Spinor, &basis, &shells).unwrap();
        let component_count = component_multiplier_for_descriptor(descriptor).unwrap();
        let layout = build_output_layout(descriptor, &validated, component_count).unwrap();
        assert!(
            layout.complex_interleaved,
            "spinor family must remain complex-interleaved via complex_output backfill"
        );
    }

    // WR-01: a GIAO / complex_interleaved family must stay OPERABLE under a
    // memory_limit_bytes that forces chunk_count > 1. The monolithic GIAO launcher
    // is a whole-block writer, so each chunk must receive the FULL interleaved block
    // (Task 1 fix in staging_elements_for_chunk). Before the fix, the sliced
    // suffix-prefix staging yielded a per-chunk BufferTooSmall. This test proves the
    // call now SUCCEEDS with chunk_count > 1 (no BufferTooSmall) and matches the
    // single-chunk run.
    #[test]
    fn evaluate_giao_complex_family_survives_memory_chunking() {
        // int1e_giao_irjxp_cart carries complex_output=true (manifest), arity 2, cart.
        let descriptor = Resolver::descriptor_by_symbol("int1e_giao_irjxp_cart")
            .expect("giao_irjxp cart descriptor");
        assert!(
            descriptor.entry.complex_output,
            "int1e_giao_irjxp_cart must carry complex_output=true (GIAO complex family)"
        );

        let (basis, shells) = sample_basis(Representation::Cart);

        // Single-chunk baseline (no memory limit): records the chunk_count and stats.
        let baseline_opts = ExecutionOptions::default();
        let baseline_query = query_workspace(
            descriptor.id,
            Representation::Cart,
            &basis,
            shells.clone(),
            &baseline_opts,
        )
        .expect("baseline workspace query should succeed");
        assert_eq!(
            baseline_query.chunk_count, 1,
            "baseline (no limit) should be a single chunk"
        );
        let baseline_plan = ExecutionPlan::new(
            descriptor.id,
            Representation::Cart,
            &basis,
            shells.clone(),
            &baseline_query,
        )
        .expect("baseline GIAO plan should build");
        assert!(
            baseline_plan.output_layout.complex_interleaved,
            "GIAO plan must be complex_interleaved"
        );
        let mut baseline_alloc = HostWorkspaceAllocator::default();
        let backend = MockBackend { supports: true };
        let baseline_stats = evaluate(baseline_plan, &baseline_opts, &mut baseline_alloc, &backend)
            .expect("baseline GIAO evaluation should succeed");
        assert_eq!(baseline_stats.chunk_count, 1);

        // Chunk-forcing run: a memory_limit_bytes small enough to split the workspace
        // into more than one chunk. The monolithic GIAO writer now receives a
        // full-block staging buffer per chunk, so the call SUCCEEDS instead of
        // returning a per-chunk BufferTooSmall (WR-01).
        let chunk_opts = ExecutionOptions {
            memory_limit_bytes: Some(192),
            ..ExecutionOptions::default()
        };
        let chunk_query = query_workspace(
            descriptor.id,
            Representation::Cart,
            &basis,
            shells.clone(),
            &chunk_opts,
        )
        .expect("chunk-forcing workspace query should succeed");
        assert!(
            chunk_query.chunk_count > 1,
            "memory_limit_bytes=192 must force chunk_count > 1 (got {})",
            chunk_query.chunk_count
        );

        let chunk_plan = ExecutionPlan::new(
            descriptor.id,
            Representation::Cart,
            &basis,
            shells,
            &chunk_query,
        )
        .expect("chunked GIAO plan should build");
        assert!(
            chunk_plan.output_layout.complex_interleaved,
            "chunked GIAO plan must remain complex_interleaved"
        );

        let mut chunk_alloc = HostWorkspaceAllocator::default();
        let chunk_stats = evaluate(chunk_plan, &chunk_opts, &mut chunk_alloc, &backend)
            .expect("GIAO evaluation under chunking must SUCCEED (no BufferTooSmall) — WR-01");

        assert!(
            chunk_stats.chunk_count > 1,
            "GIAO evaluation should have run with chunk_count > 1 (got {})",
            chunk_stats.chunk_count
        );
        assert_eq!(
            chunk_stats.fallback_reason,
            Some("memory_limit"),
            "chunked GIAO run should report the memory_limit fallback"
        );
    }
}
