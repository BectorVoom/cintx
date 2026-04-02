use crate::layout::{CompatDims, ensure_cache_len};
use crate::optimizer::RawOptimizerHandle;
use cintx_core::{
    Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple, cintxRsError,
};
use cintx_cubecl::CubeClExecutor;
use cintx_ops::resolver::{HelperKind, OperatorDescriptor, Resolver, ResolverError};
use cintx_runtime::{
    BackendCapabilityToken, BackendExecutor, ExecutionIo, ExecutionOptions, ExecutionPlan,
    ExecutionStats, HostWorkspaceAllocator, WorkspaceBytes, WorkspaceQuery, evaluate,
    query_workspace,
};
use std::mem::size_of;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// RecordingExecutor — captures staging output from the backend so eval_raw()
// can retrieve real computed values instead of zero-filling the output buffer.
// ---------------------------------------------------------------------------

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

    fn owned_values(&self) -> Result<Vec<f64>, cintxRsError> {
        let staged_values = self
            .staged_values
            .lock()
            .map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "compat_recording_executor",
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
                    from: "compat_recording_executor",
                    detail: "owned output capture buffer mutex poisoned".to_owned(),
                })?;
        staged_values.extend_from_slice(io.staging_output());
        Ok(stats)
    }
}

pub const CHARGE_OF: usize = 0;
pub const PTR_COORD: usize = 1;
pub const NUC_MOD_OF: usize = 2;
pub const PTR_ZETA: usize = 3;
pub const PTR_FRAC_CHARGE: usize = 4;
pub const ATM_SLOTS: usize = 6;

pub const ATOM_OF: usize = 0;
pub const ANG_OF: usize = 1;
pub const NPRIM_OF: usize = 2;
pub const NCTR_OF: usize = 3;
pub const KAPPA_OF: usize = 4;
pub const PTR_EXP: usize = 5;
pub const PTR_COEFF: usize = 6;
pub const BAS_SLOTS: usize = 8;

pub const POINT_NUC: i32 = 1;
pub const GAUSSIAN_NUC: i32 = 2;
pub const FRAC_CHARGE_NUC: i32 = 3;
const VALIDATED_4C1E_REASON: &str = "outside Validated4C1E";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawApiId {
    Symbol(&'static str),
}

impl RawApiId {
    pub const INT1E_OVLP_CART: Self = Self::Symbol("int1e_ovlp_cart");
    pub const INT1E_OVLP_SPH: Self = Self::Symbol("int1e_ovlp_sph");
    pub const INT1E_OVLP_SPINOR: Self = Self::Symbol("int1e_ovlp_spinor");

    pub const INT1E_KIN_CART: Self = Self::Symbol("int1e_kin_cart");
    pub const INT1E_KIN_SPH: Self = Self::Symbol("int1e_kin_sph");
    pub const INT1E_KIN_SPINOR: Self = Self::Symbol("int1e_kin_spinor");

    pub const INT1E_NUC_CART: Self = Self::Symbol("int1e_nuc_cart");
    pub const INT1E_NUC_SPH: Self = Self::Symbol("int1e_nuc_sph");
    pub const INT1E_NUC_SPINOR: Self = Self::Symbol("int1e_nuc_spinor");

    pub const INT2E_CART: Self = Self::Symbol("int2e_cart");
    pub const INT2E_SPH: Self = Self::Symbol("int2e_sph");
    pub const INT2E_SPINOR: Self = Self::Symbol("int2e_spinor");

    pub const INT2C2E_CART: Self = Self::Symbol("int2c2e_cart");
    pub const INT2C2E_SPH: Self = Self::Symbol("int2c2e_sph");
    pub const INT2C2E_SPINOR: Self = Self::Symbol("int2c2e_spinor");

    pub const INT3C1E_P2_CART: Self = Self::Symbol("int3c1e_p2_cart");
    pub const INT3C1E_P2_SPH: Self = Self::Symbol("int3c1e_p2_sph");
    pub const INT3C1E_P2_SPINOR: Self = Self::Symbol("int3c1e_p2_spinor");

    pub const INT3C2E_IP1_CART: Self = Self::Symbol("int3c2e_ip1_cart");
    pub const INT3C2E_IP1_SPH: Self = Self::Symbol("int3c2e_ip1_sph");
    pub const INT3C2E_IP1_SPINOR: Self = Self::Symbol("int3c2e_ip1_spinor");

    pub const INT4C1E_CART: Self = Self::Symbol("int4c1e_cart");
    pub const INT4C1E_SPH: Self = Self::Symbol("int4c1e_sph");

    fn symbol(self) -> &'static str {
        match self {
            Self::Symbol(symbol) => symbol,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawEvalSummary {
    pub not0: i32,
    pub bytes_written: usize,
    pub workspace_bytes: usize,
}

struct ResolvedRawApi {
    descriptor: &'static OperatorDescriptor,
    representation: Representation,
}

struct PreparedRawCall {
    op: OperatorId,
    representation: Representation,
    basis: BasisSet,
    shells: ShellTuple,
    query: WorkspaceQuery,
    compat_dims: CompatDims,
    options: ExecutionOptions,
}

/// Raw atom view over libcint-style `atm` slots.
#[derive(Clone, Copy, Debug)]
pub struct RawAtmView<'a> {
    data: &'a [i32],
}

impl<'a> RawAtmView<'a> {
    pub fn new(data: &'a [i32]) -> Result<Self, cintxRsError> {
        if data.len() % ATM_SLOTS != 0 {
            return Err(cintxRsError::InvalidAtmLayout {
                slot_width: ATM_SLOTS,
                provided: data.len(),
            });
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len() / ATM_SLOTS
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<RawAtmRecord<'a>> {
        let start = index.checked_mul(ATM_SLOTS)?;
        let record = self.data.get(start..start + ATM_SLOTS)?;
        Some(RawAtmRecord { record })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RawAtmRecord<'a>> {
        self.data
            .chunks_exact(ATM_SLOTS)
            .map(|record| RawAtmRecord { record })
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        for record in self.iter() {
            record.validate(env)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawAtmRecord<'a> {
    record: &'a [i32],
}

impl<'a> RawAtmRecord<'a> {
    pub fn charge(&self) -> i32 {
        self.record[CHARGE_OF]
    }

    pub fn coord_offset(&self) -> i32 {
        self.record[PTR_COORD]
    }

    pub fn nuclear_model_raw(&self) -> i32 {
        self.record[NUC_MOD_OF]
    }

    pub fn zeta_offset(&self) -> i32 {
        self.record[PTR_ZETA]
    }

    pub fn fractional_charge_offset(&self) -> i32 {
        self.record[PTR_FRAC_CHARGE]
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        env.validate_range("PTR_COORD", self.coord_offset(), 3)?;
        match self.nuclear_model_raw() {
            POINT_NUC => {}
            GAUSSIAN_NUC => {
                env.validate_scalar("PTR_ZETA", self.zeta_offset())?;
            }
            FRAC_CHARGE_NUC => {
                env.validate_scalar("PTR_FRAC_CHARGE", self.fractional_charge_offset())?;
            }
            other => {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("unsupported nuclear model {other}"),
                });
            }
        }
        Ok(())
    }
}

/// Raw basis-shell view over libcint-style `bas` slots.
#[derive(Clone, Copy, Debug)]
pub struct RawBasView<'a> {
    data: &'a [i32],
}

impl<'a> RawBasView<'a> {
    pub fn new(data: &'a [i32]) -> Result<Self, cintxRsError> {
        if data.len() % BAS_SLOTS != 0 {
            return Err(cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: data.len(),
            });
        }
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len() / BAS_SLOTS
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<RawBasRecord<'a>> {
        let start = index.checked_mul(BAS_SLOTS)?;
        let record = self.data.get(start..start + BAS_SLOTS)?;
        Some(RawBasRecord { record })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = RawBasRecord<'a>> {
        self.data
            .chunks_exact(BAS_SLOTS)
            .map(|record| RawBasRecord { record })
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        for record in self.iter() {
            record.validate(env)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawBasRecord<'a> {
    record: &'a [i32],
}

impl<'a> RawBasRecord<'a> {
    pub fn atom_index_raw(&self) -> i32 {
        self.record[ATOM_OF]
    }

    pub fn ang_momentum_raw(&self) -> i32 {
        self.record[ANG_OF]
    }

    pub fn nprim_raw(&self) -> i32 {
        self.record[NPRIM_OF]
    }

    pub fn nctr_raw(&self) -> i32 {
        self.record[NCTR_OF]
    }

    pub fn kappa_raw(&self) -> i32 {
        self.record[KAPPA_OF]
    }

    pub fn exp_offset(&self) -> i32 {
        self.record[PTR_EXP]
    }

    pub fn coeff_offset(&self) -> i32 {
        self.record[PTR_COEFF]
    }

    pub fn validate(&self, env: &RawEnvView<'_>) -> Result<(), cintxRsError> {
        let nprim =
            usize::try_from(self.nprim_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: self.nprim_raw().unsigned_abs() as usize,
            })?;
        let nctr =
            usize::try_from(self.nctr_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: self.nctr_raw().unsigned_abs() as usize,
            })?;

        if nprim == 0 || nctr == 0 {
            return Err(cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: 0,
            });
        }

        env.validate_range("PTR_EXP", self.exp_offset(), nprim)?;
        let coeff_len = nprim
            .checked_mul(nctr)
            .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                from: "raw_bas",
                detail: "coefficient range overflowed usize".to_owned(),
            })?;
        env.validate_range("PTR_COEFF", self.coeff_offset(), coeff_len)?;
        Ok(())
    }
}

/// Raw environment view over libcint-style `env` values.
#[derive(Clone, Copy, Debug)]
pub struct RawEnvView<'a> {
    data: &'a [f64],
}

impl<'a> RawEnvView<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    pub fn validate_scalar(&self, slot: &'static str, offset: i32) -> Result<usize, cintxRsError> {
        self.validate_range(slot, offset, 1)
    }

    pub fn validate_range(
        &self,
        slot: &'static str,
        offset: i32,
        len: usize,
    ) -> Result<usize, cintxRsError> {
        let start = normalize_offset(slot, offset, self.len())?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| cintxRsError::InvalidEnvOffset {
                slot,
                offset: start,
                env_len: self.len(),
            })?;
        if end > self.len() {
            return Err(cintxRsError::InvalidEnvOffset {
                slot,
                offset: start,
                env_len: self.len(),
            });
        }
        Ok(start)
    }

    pub fn slice(
        &self,
        slot: &'static str,
        offset: i32,
        len: usize,
    ) -> Result<&'a [f64], cintxRsError> {
        let start = self.validate_range(slot, offset, len)?;
        Ok(&self.data[start..start + len])
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn query_workspace_raw(
    api: RawApiId,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<WorkspaceQuery, cintxRsError> {
    let prepared = prepare_raw_call(api, dims, shls, atm, bas, env, opt)?;
    Ok(prepared.query)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn eval_raw(
    api: RawApiId,
    out: Option<&mut [f64]>,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
    cache: Option<&mut [f64]>,
) -> Result<RawEvalSummary, cintxRsError> {
    let prepared = prepare_raw_call(api, dims, shls, atm, bas, env, opt)?;
    let required_elements = prepared.compat_dims.required_elements()?;

    if let Some(out_buffer) = out.as_ref() {
        prepared.compat_dims.ensure_output_len(out_buffer.len())?;
    } else {
        return Ok(RawEvalSummary {
            not0: 0,
            bytes_written: 0,
            workspace_bytes: prepared.query.bytes,
        });
    }

    if let Some(cache) = cache {
        ensure_cache_len(prepared.query.bytes, cache.len())?;
    }

    let plan = ExecutionPlan::new(
        prepared.op,
        prepared.representation,
        &prepared.basis,
        prepared.shells.clone(),
        &prepared.query,
    )?;

    let executor = RecordingExecutor::new(CubeClExecutor::new());
    let mut allocator = HostWorkspaceAllocator::default();
    let stats = evaluate(plan, &prepared.options, &mut allocator, &executor)?;
    let owned_values = executor.owned_values()?;

    if owned_values.len() != required_elements {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "eval_raw",
            detail: format!(
                "staging output length mismatch: expected={required_elements} got={}",
                owned_values.len()
            ),
        });
    }

    let out = out.expect("checked out.is_some()");
    let written_elements = prepared.compat_dims.write(out, &owned_values)?;
    let bytes_written = written_elements
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "compat_raw",
            detail: "written byte count overflowed usize".to_owned(),
        })?;

    Ok(RawEvalSummary {
        not0: stats.not0,
        bytes_written,
        workspace_bytes: stats.workspace_bytes,
    })
}

fn active_manifest_profile() -> &'static str {
    match (cfg!(feature = "with-f12"), cfg!(feature = "with-4c1e")) {
        (true, true) => "with-f12+with-4c1e",
        (true, false) => "with-f12",
        (false, true) => "with-4c1e",
        (false, false) => "base",
    }
}

fn unstable_source_api_enabled() -> bool {
    cfg!(feature = "unstable-source-api")
}

fn is_f12_family_symbol(symbol: &str) -> bool {
    symbol.starts_with("int2e_stg") || symbol.starts_with("int2e_yp")
}

fn f12_sph_envelope_error(symbol: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("{symbol} is outside with-f12 sph envelope"),
    }
}

fn validated_4c1e_error(reason: &str) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("{VALIDATED_4C1E_REASON} ({reason})"),
    }
}

fn validate_profile_and_source_gate(descriptor: &OperatorDescriptor) -> Result<(), cintxRsError> {
    let symbol = descriptor.operator_symbol();
    let profile = active_manifest_profile();
    if !descriptor.is_compiled_in_profile(profile) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("raw api {symbol} is not compiled in active profile {profile}"),
        });
    }
    if descriptor.is_source_only() && !unstable_source_api_enabled() {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "source-only symbol {symbol} requires feature `unstable-source-api`"
            ),
        });
    }

    Ok(())
}

fn dims_match_natural(dims: Option<&[i32]>, natural_extents: &[usize]) -> bool {
    let Some(dims) = dims else {
        return true;
    };
    if dims.len() != natural_extents.len() {
        return false;
    }
    dims.iter()
        .zip(natural_extents.iter())
        .all(|(provided, expected)| usize::try_from(*provided).ok() == Some(*expected))
}

fn validate_f12_envelope(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    dims: Option<&[i32]>,
    natural_extents: &[usize],
) -> Result<(), cintxRsError> {
    let symbol = descriptor.operator_symbol();
    if !is_f12_family_symbol(symbol) {
        return Ok(());
    }

    if !matches!(representation, Representation::Spheric) {
        return Err(f12_sph_envelope_error(symbol));
    }
    if !dims_match_natural(dims, natural_extents) {
        return Err(f12_sph_envelope_error(symbol));
    }
    Ok(())
}

fn validate_4c1e_envelope(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shells: &ShellTuple,
    dims: Option<&[i32]>,
    natural_extents: &[usize],
) -> Result<(), cintxRsError> {
    if descriptor.entry.canonical_family != "4c1e" {
        return Ok(());
    }

    if !cfg!(feature = "with-4c1e") {
        return Err(validated_4c1e_error("with-4c1e feature disabled"));
    }
    if !matches!(
        representation,
        Representation::Cart | Representation::Spheric
    ) {
        return Err(validated_4c1e_error("representation must be cart/sph"));
    }
    if !descriptor.entry.component_rank.trim().is_empty()
        && descriptor.entry.component_rank != "scalar"
    {
        return Err(validated_4c1e_error("component rank must be scalar"));
    }
    if !dims_match_natural(dims, natural_extents) {
        return Err(validated_4c1e_error("dims must be natural"));
    }
    // Validated4C1E requires max(l)<=4.
    if shells.iter().any(|shell| shell.ang_momentum > 4) {
        return Err(validated_4c1e_error("max(l)>4"));
    }
    // D-11: Validated4C1E gate requires wgpu capability — cpu-profile check removed.
    // The executor's ensure_validated_4c1e now performs a wgpu preflight check.

    Ok(())
}

/// Apply the same manifest profile/source-only/optional envelope policy gates used by
/// compat raw dispatch so safe facade callers get identical UnsupportedApi reasons.
pub fn enforce_safe_facade_policy_gate(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shells: &ShellTuple,
    natural_extents: &[usize],
) -> Result<(), cintxRsError> {
    validate_profile_and_source_gate(descriptor)?;
    validate_f12_envelope(descriptor, representation, None, natural_extents)?;
    validate_4c1e_envelope(descriptor, representation, shells, None, natural_extents)?;
    Ok(())
}

fn prepare_raw_call(
    api: RawApiId,
    dims: Option<&[i32]>,
    shls: &[i32],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    opt: Option<&RawOptimizerHandle>,
) -> Result<PreparedRawCall, cintxRsError> {
    let resolved = resolve_raw_api(api)?;
    let atm = RawAtmView::new(atm)?;
    let bas = RawBasView::new(bas)?;
    let env = RawEnvView::new(env);

    atm.validate(&env)?;
    bas.validate(&env)?;

    let (basis, shells) = build_typed_basis_and_shell_tuple(
        resolved.descriptor,
        resolved.representation,
        shls,
        &atm,
        &bas,
        &env,
    )?;

    let options = execution_options_from_opt(opt)?;
    let query = query_workspace(
        resolved.descriptor.id,
        resolved.representation,
        &basis,
        shells.clone(),
        &options,
    )?;

    let layout_plan = ExecutionPlan::new(
        resolved.descriptor.id,
        resolved.representation,
        &basis,
        shells.clone(),
        &query,
    )?;

    validate_f12_envelope(
        resolved.descriptor,
        resolved.representation,
        dims,
        &layout_plan.output_layout.extents,
    )?;
    validate_4c1e_envelope(
        resolved.descriptor,
        resolved.representation,
        &shells,
        dims,
        &layout_plan.output_layout.extents,
    )?;

    let compat_dims = CompatDims::from_override(
        &layout_plan.output_layout.extents,
        dims,
        layout_plan.component_count,
        layout_plan.output_layout.complex_interleaved,
    )?;

    Ok(PreparedRawCall {
        op: resolved.descriptor.id,
        representation: resolved.representation,
        basis,
        shells,
        query,
        compat_dims,
        options,
    })
}

fn resolve_raw_api(api: RawApiId) -> Result<ResolvedRawApi, cintxRsError> {
    let symbol = api.symbol();
    if is_f12_family_symbol(symbol) && !symbol.ends_with("_sph") {
        return Err(f12_sph_envelope_error(symbol));
    }

    let descriptor =
        Resolver::descriptor_by_symbol(symbol).map_err(|err| map_resolver_error(api, err))?;

    if !matches!(
        descriptor.entry.helper_kind,
        HelperKind::Operator | HelperKind::Legacy | HelperKind::SourceOnly
    ) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "raw api {} must resolve to operator/legacy/source manifest entries",
                symbol
            ),
        });
    }

    validate_profile_and_source_gate(descriptor)?;

    let representation = representation_from_descriptor(descriptor)?;
    Ok(ResolvedRawApi {
        descriptor,
        representation,
    })
}

fn representation_from_descriptor(
    descriptor: &OperatorDescriptor,
) -> Result<Representation, cintxRsError> {
    let rep = descriptor.entry.representation;
    match (rep.cart, rep.spheric, rep.spinor) {
        (true, false, false) => Ok(Representation::Cart),
        (false, true, false) => Ok(Representation::Spheric),
        (false, false, true) => Ok(Representation::Spinor),
        _ => Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "descriptor {} does not map to a single representation",
                descriptor.operator_symbol()
            ),
        }),
    }
}

fn execution_options_from_opt(
    opt: Option<&RawOptimizerHandle>,
) -> Result<ExecutionOptions, cintxRsError> {
    let mut options = ExecutionOptions::default();
    options.profile_label = Some(active_manifest_profile());
    if let Some(opt) = opt {
        options.memory_limit_bytes = opt.workspace_hint_bytes();
    }
    // Propagate real wgpu adapter fingerprint so planning_matches() drift check
    // compares a real adapter identity instead of default 0 == 0.
    let report = cintx_cubecl::bootstrap_wgpu_runtime(&options.backend_intent)?;
    if report.is_capable() {
        options.backend_capability_token = BackendCapabilityToken {
            adapter_name: report.snapshot.adapter_name.clone(),
            backend_api: report.snapshot.backend_api.clone(),
            capability_fingerprint: report.fingerprint,
        };
    }
    Ok(options)
}

fn build_typed_basis_and_shell_tuple(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    shls: &[i32],
    atm: &RawAtmView<'_>,
    bas: &RawBasView<'_>,
    env: &RawEnvView<'_>,
) -> Result<(BasisSet, ShellTuple), cintxRsError> {
    let mut atoms = Vec::new();
    atoms
        .try_reserve_exact(atm.len())
        .map_err(|_| cintxRsError::HostAllocationFailed {
            bytes: atm.len().saturating_mul(size_of::<Atom>()),
        })?;

    for record in atm.iter() {
        let atomic_number =
            u16::try_from(record.charge()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_atoms",
                detail: format!(
                    "atomic number is negative or too large: {}",
                    record.charge()
                ),
            })?;

        let coord = env.slice("PTR_COORD", record.coord_offset(), 3)?;
        let coord = [coord[0], coord[1], coord[2]];
        let (model, zeta, fractional_charge) = match record.nuclear_model_raw() {
            POINT_NUC => (NuclearModel::Point, None, None),
            GAUSSIAN_NUC => (
                NuclearModel::Gaussian,
                Some(env.slice("PTR_ZETA", record.zeta_offset(), 1)?[0]),
                None,
            ),
            FRAC_CHARGE_NUC => (
                NuclearModel::FiniteSpherical,
                None,
                Some(env.slice("PTR_FRAC_CHARGE", record.fractional_charge_offset(), 1)?[0]),
            ),
            other => {
                return Err(cintxRsError::UnsupportedApi {
                    requested: format!("unsupported nuclear model {other}"),
                });
            }
        };

        let atom =
            Atom::try_new(atomic_number, coord, model, zeta, fractional_charge).map_err(|err| {
                cintxRsError::ChunkPlanFailed {
                    from: "raw_atoms",
                    detail: err.to_string(),
                }
            })?;
        atoms.push(atom);
    }

    let mut shells = Vec::new();
    shells
        .try_reserve_exact(bas.len())
        .map_err(|_| cintxRsError::HostAllocationFailed {
            bytes: bas.len().saturating_mul(size_of::<Shell>()),
        })?;

    for record in bas.iter() {
        let atom_index =
            u32::try_from(record.atom_index_raw()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: format!("negative shell atom index {}", record.atom_index_raw()),
            })?;
        let ang_momentum =
            u8::try_from(record.ang_momentum_raw()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: format!("invalid angular momentum {}", record.ang_momentum_raw()),
            })?;
        let nprim =
            u16::try_from(record.nprim_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: record.nprim_raw().unsigned_abs() as usize,
            })?;
        let nctr =
            u16::try_from(record.nctr_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
                slot_width: BAS_SLOTS,
                provided: record.nctr_raw().unsigned_abs() as usize,
            })?;
        let kappa =
            i16::try_from(record.kappa_raw()).map_err(|_| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: format!("kappa does not fit i16: {}", record.kappa_raw()),
            })?;

        let exponents = Arc::<[f64]>::from(
            env.slice("PTR_EXP", record.exp_offset(), nprim as usize)?
                .to_vec()
                .into_boxed_slice(),
        );
        let coefficient_len = usize::from(nprim)
            .checked_mul(usize::from(nctr))
            .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                from: "raw_shells",
                detail: "nprim*nctr overflowed usize".to_owned(),
            })?;
        let coefficients = Arc::<[f64]>::from(
            env.slice("PTR_COEFF", record.coeff_offset(), coefficient_len)?
                .to_vec()
                .into_boxed_slice(),
        );

        let shell = Shell::try_new(
            atom_index,
            ang_momentum,
            nprim,
            nctr,
            kappa,
            representation,
            exponents,
            coefficients,
        )
        .map_err(|err| cintxRsError::ChunkPlanFailed {
            from: "raw_shells",
            detail: err.to_string(),
        })?;
        shells.push(Arc::new(shell));
    }

    let basis = BasisSet::try_new(
        Arc::<[Atom]>::from(atoms.into_boxed_slice()),
        Arc::<[Arc<Shell>]>::from(shells.into_boxed_slice()),
    )
    .map_err(|err| cintxRsError::ChunkPlanFailed {
        from: "raw_basis",
        detail: err.to_string(),
    })?;

    let expected_arity = descriptor.entry.arity as usize;
    if shls.len() != expected_arity {
        return Err(cintxRsError::InvalidShellTuple {
            expected: expected_arity,
            got: shls.len(),
        });
    }

    let mut shell_indices = Vec::new();
    shell_indices.try_reserve_exact(shls.len()).map_err(|_| {
        cintxRsError::HostAllocationFailed {
            bytes: shls.len().saturating_mul(size_of::<usize>()),
        }
    })?;
    for index in shls {
        let parsed = usize::try_from(*index).map_err(|_| cintxRsError::ChunkPlanFailed {
            from: "raw_shell_tuple",
            detail: format!("shell index must be non-negative: {index}"),
        })?;
        shell_indices.push(parsed);
    }

    let shell_tuple = basis
        .shell_tuple_for_indices(shell_indices)
        .map_err(|err| cintxRsError::ChunkPlanFailed {
            from: "raw_shell_tuple",
            detail: err.to_string(),
        })?;

    Ok((basis, shell_tuple))
}

fn map_resolver_error(api: RawApiId, err: ResolverError) -> cintxRsError {
    match err {
        ResolverError::MissingOperatorId(_) | ResolverError::MissingSymbol(_) => {
            cintxRsError::UnsupportedApi {
                requested: format!("raw api {} is missing from manifest", api.symbol()),
            }
        }
        ResolverError::MissingFamilyOperator { family, operator } => cintxRsError::UnsupportedApi {
            requested: format!("{family}/{operator}"),
        },
        ResolverError::UnsupportedRepresentation {
            family,
            operator,
            representation,
        } => cintxRsError::UnsupportedRepresentation {
            operator: format!("{family}/{operator}"),
            representation,
        },
    }
}

fn normalize_offset(
    slot: &'static str,
    offset: i32,
    env_len: usize,
) -> Result<usize, cintxRsError> {
    usize::try_from(offset).map_err(|_| cintxRsError::InvalidEnvOffset {
        slot,
        offset: env_len,
        env_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::required_f64s_for_bytes;

    struct RawFixture {
        shls_2: [i32; 2],
        shls_3: [i32; 3],
        atm: Vec<i32>,
        bas: Vec<i32>,
        env: Vec<f64>,
    }

    impl RawFixture {
        fn single_atom_three_shells() -> Self {
            // env layout:
            // 0..3 coordinates, 3 exp0, 4 coeff0, 5 exp1, 6 coeff1, 7 exp2, 8 coeff2
            let env = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6];
            let atm = vec![
                1, // charge / atomic number
                0, // PTR_COORD
                POINT_NUC, 0, // PTR_ZETA
                0, // PTR_FRAC_CHARGE
                0,
            ];
            let bas = vec![
                0, 0, 1, 1, 0, 3, 4, 0, // shell 0
                0, 1, 1, 1, 0, 5, 6, 0, // shell 1
                0, 0, 1, 1, 0, 7, 8, 0, // shell 2
            ];
            Self {
                shls_2: [0, 1],
                shls_3: [0, 1, 2],
                atm,
                bas,
                env,
            }
        }

        fn single_atom_four_shells() -> ([i32; 4], Vec<i32>, Vec<i32>, Vec<f64>) {
            let env = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
            let atm = vec![1, 0, POINT_NUC, 0, 0, 0];
            let bas = vec![
                0, 0, 1, 1, 0, 3, 4, 0, // shell 0
                0, 1, 1, 1, 0, 5, 6, 0, // shell 1
                0, 0, 1, 1, 0, 7, 8, 0, // shell 2
                0, 2, 1, 1, 0, 9, 10, 0, // shell 3
            ];
            ([0, 1, 2, 3], atm, bas, env)
        }
    }

    #[test]
    fn malformed_layouts_are_typed() {
        let err = RawAtmView::new(&[1, 2]).unwrap_err();
        assert!(matches!(err, cintxRsError::InvalidAtmLayout { .. }));

        let err = RawBasView::new(&[1, 2, 3]).unwrap_err();
        assert!(matches!(err, cintxRsError::InvalidBasLayout { .. }));
    }

    #[test]
    fn invalid_env_offsets_fail_validation() {
        let fixture = RawFixture::single_atom_three_shells();
        let mut bas = fixture.bas.clone();
        bas[PTR_EXP] = 9999;
        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &bas,
                &fixture.env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(err, cintxRsError::InvalidEnvOffset { .. }));
    }

    #[test]
    fn invalid_dims_length_is_rejected_for_each_arity() {
        // After Phase 06 Bug 2 fix, query_workspace_raw bootstraps wgpu; skip on no-GPU CI.
        let fixture = RawFixture::single_atom_three_shells();

        let err = match unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&[1]),
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Err(e) => e,
            Ok(_) => panic!("expected invalid dims error"),
        };
        match &err {
            cintxRsError::InvalidDims { expected: 2, provided: 1 } => {}
            cintxRsError::UnsupportedApi { requested } if requested.contains("wgpu-capability") => {
                return; // no GPU adapter — skip dims validation check
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let err = match unsafe {
            query_workspace_raw(
                RawApiId::INT3C1E_P2_CART,
                Some(&[1, 2]),
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Err(e) => e,
            Ok(_) => panic!("expected invalid dims error"),
        };
        match &err {
            cintxRsError::InvalidDims { expected: 3, provided: 2 } => {}
            cintxRsError::UnsupportedApi { requested } if requested.contains("wgpu-capability") => {
                // no GPU adapter
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn undersized_output_buffer_is_reported() {
        // After Phase 06 Bug 2 fix, eval_raw bootstraps wgpu; skip on no-GPU CI.
        let fixture = RawFixture::single_atom_three_shells();
        let mut out = vec![0.0; 1];
        let err = match unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        } {
            Err(e) => e,
            Ok(_) => panic!("expected buffer too small error"),
        };
        match &err {
            cintxRsError::BufferTooSmall { .. } => {}
            cintxRsError::UnsupportedApi { requested } if requested.contains("wgpu-capability") => {
                return; // no GPU adapter
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn query_workspace_raw_and_eval_raw_none_match_workspace_expectations() {
        // After Phase 06 Bug 2 fix, query_workspace_raw bootstraps wgpu for fingerprint.
        // Accept wgpu-capability error on no-GPU CI.
        let fixture = RawFixture::single_atom_three_shells();
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter — cannot verify workspace bytes match
            }
            Err(e) => panic!("unexpected query error: {e}"),
        };

        let summary = match unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        } {
            Ok(s) => s,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter — acceptable
            }
            Err(e) => panic!("unexpected eval_raw error: {e}"),
        };

        assert_eq!(summary.not0, 0);
        assert_eq!(summary.bytes_written, 0);
        assert_eq!(summary.workspace_bytes, query.bytes);
    }

    #[test]
    fn memory_limit_hint_can_chunk_successfully() {
        let fixture = RawFixture::single_atom_three_shells();
        let opt = RawOptimizerHandle::with_hints(None, Some(128));
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter
            }
            Err(e) => panic!("unexpected query error: {e}"),
        };
        assert!(query.chunk_count >= 1);

        let mut out = vec![99.0; 3];
        // D-05: eval now routes through real CubeClExecutor path.
        // Accept both GPU-success and fail-closed wgpu-capability error.
        let result = unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
                None,
            )
        };
        match result {
            Ok(summary) => {
                assert!(summary.bytes_written > 0);
            }
            Err(cintxRsError::UnsupportedApi { ref requested }) if requested.contains("wgpu-capability") => {
                // No GPU adapter — correct fail-closed behavior (D-01/D-02).
            }
            Err(other) => panic!("unexpected error from eval_raw: {other:?}"),
        }
    }

    #[test]
    fn memory_limit_failure_keeps_output_slice_unchanged() {
        let fixture = RawFixture::single_atom_three_shells();
        let opt = RawOptimizerHandle::with_hints(None, Some(1));
        let mut out = vec![7.0; 3];

        let err = match unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
                None,
            )
        } {
            Err(e) => e,
            Ok(_) => panic!("expected error from memory-limited eval_raw"),
        };

        // On no-GPU CI the bootstrap fails with UnsupportedApi before reaching the memory check.
        // On GPU CI the memory limit is enforced and MemoryLimitExceeded is returned.
        match &err {
            cintxRsError::MemoryLimitExceeded { .. } => {
                assert!(
                    out.iter().all(|value| *value == 7.0),
                    "output slice unchanged on failure (no partial write)"
                );
            }
            cintxRsError::UnsupportedApi { requested } if requested.contains("wgpu-capability") => {
                // no GPU adapter — bootstrap failed before memory limit check; output unchanged
                assert!(
                    out.iter().all(|value| *value == 7.0),
                    "output slice must be unchanged when bootstrap fails"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn cache_buffer_too_small_is_rejected_before_execution() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter
            }
            Err(e) => panic!("unexpected query error: {e}"),
        };

        let required_cache = required_f64s_for_bytes(query.bytes).expect("cache conversion");
        let mut out = vec![0.0; 3];
        let mut cache = vec![0.0; required_cache.saturating_sub(1)];
        let err = match unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                Some(&mut cache),
            )
        } {
            Err(e) => e,
            Ok(_) => panic!("expected error for undersized cache"),
        };

        // The cache check happens before execution (before wgpu bootstrap in evaluate).
        assert!(matches!(err, cintxRsError::BufferTooSmall { .. }));
    }

    #[test]
    fn three_center_contract_query_and_eval_work_for_supported_backend() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT3C1E_P2_CART,
                None,
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter
            }
            Err(e) => panic!("unexpected query error: {e}"),
        };
        assert_eq!(query.work_units, 3);

        let mut out = vec![1.0; 3];
        // D-05: eval now routes through real CubeClExecutor path.
        // Accept both GPU-success and fail-closed wgpu-capability error.
        let result = unsafe {
            eval_raw(
                RawApiId::INT3C1E_P2_CART,
                Some(&mut out),
                None,
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        };
        match result {
            Ok(summary) => {
                assert!(summary.bytes_written > 0);
            }
            Err(cintxRsError::UnsupportedApi { ref requested }) if requested.contains("wgpu-capability") => {
                // No GPU adapter — correct fail-closed behavior (D-01/D-02).
            }
            Err(other) => panic!("unexpected error from 3c eval_raw: {other:?}"),
        }
    }

    #[test]
    fn f12_cart_symbol_is_rejected_with_explicit_sph_envelope_reason() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_cart"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested } if requested.contains("with-f12 sph envelope")
        ));
    }

    #[cfg(not(feature = "with-f12"))]
    #[test]
    fn f12_sph_symbol_requires_with_f12_profile() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("active profile")
                    && requested.contains(active_manifest_profile())
        ));
    }

    #[cfg(feature = "with-f12")]
    #[test] // safe-facade policy gate
    fn safe_facade_gate_reports_with_f12_sph_envelope_for_cart_representation() {
        let descriptor = Resolver::descriptor_by_symbol("int2e_stg_sph")
            .expect("stg symbol must exist in manifest");
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let atm = RawAtmView::new(&atm).expect("atm layout");
        let bas = RawBasView::new(&bas).expect("bas layout");
        let env = RawEnvView::new(&env);
        let (_, shells) = build_typed_basis_and_shell_tuple(
            descriptor,
            Representation::Cart,
            &shls_4,
            &atm,
            &bas,
            &env,
        )
        .expect("shell tuple should build");

        let err = enforce_safe_facade_policy_gate(
            descriptor,
            Representation::Cart,
            &shells,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested } if requested.contains("with-f12 sph envelope")
        ));
    }

    #[cfg(feature = "with-f12")]
    #[test]
    fn f12_sph_symbol_is_queryable_when_feature_enabled() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter
            }
            Err(e) => panic!("with-f12 should allow sph-only f12 symbols: {e}"),
        };
        assert!(query.bytes > 0);
    }

    #[cfg(not(feature = "with-4c1e"))]
    #[test]
    fn int4c1e_requires_with_4c1e_profile() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("active profile")
                    && requested.contains(active_manifest_profile())
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn int4c1e_rejects_bug_envelope_inputs() {
        let (shls_4, atm, mut bas, env) = RawFixture::single_atom_four_shells();
        bas[ANG_OF] = 5; // max(l)>4 should fail the Validated4C1E envelope.

        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") && requested.contains("max(l)>4")
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test] // safe-facade policy gate
    fn safe_facade_gate_reports_validated_4c1e_reason_for_out_of_envelope_shells() {
        let descriptor = Resolver::descriptor_by_symbol("int4c1e_cart")
            .expect("int4c1e cart symbol must exist in manifest");
        let (shls_4, atm, mut bas, env) = RawFixture::single_atom_four_shells();
        bas[ANG_OF] = 5;

        let atm = RawAtmView::new(&atm).expect("atm layout");
        let bas = RawBasView::new(&bas).expect("bas layout");
        let env = RawEnvView::new(&env);
        let (_, shells) = build_typed_basis_and_shell_tuple(
            descriptor,
            Representation::Cart,
            &shls_4,
            &atm,
            &bas,
            &env,
        )
        .expect("shell tuple should build");

        let err = enforce_safe_facade_policy_gate(
            descriptor,
            Representation::Cart,
            &shells,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("outside Validated4C1E") && requested.contains("max(l)>4")
        ));
    }

    #[cfg(feature = "with-4c1e")]
    #[test]
    fn int4c1e_accepts_validated_inputs() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter
            }
            Err(e) => panic!("validated 4c1e envelope should be queryable: {e}"),
        };
        assert!(query.bytes > 0);
    }

    #[cfg(not(feature = "unstable-source-api"))]
    #[test] // safe-facade policy gate
    fn safe_facade_gate_rejects_source_only_symbol_without_unstable_feature() {
        let descriptor = Resolver::descriptor_by_symbol("int2e_ipip1_sph")
            .expect("source-only symbol must exist in manifest");
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let atm = RawAtmView::new(&atm).expect("atm layout");
        let bas = RawBasView::new(&bas).expect("bas layout");
        let env = RawEnvView::new(&env);
        let (_, shells) = build_typed_basis_and_shell_tuple(
            descriptor,
            Representation::Spheric,
            &shls_4,
            &atm,
            &bas,
            &env,
        )
        .expect("shell tuple should build");

        let err = enforce_safe_facade_policy_gate(
            descriptor,
            Representation::Spheric,
            &shells,
            &[1, 1, 1, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("requires feature `unstable-source-api`")
        ));
    }

    /// D-11: Validates that cpu-profile gate is gone from validate_4c1e_envelope.
    /// The CPU-profile check (`CUBECL_RUNTIME_PROFILE != "cpu"`) was removed in plan 03;
    /// this test confirms validated 4c1e envelope query succeeds without that gate.
    #[cfg(feature = "with-4c1e")]
    #[test]
    fn validate_4c1e_envelope_no_longer_references_cpu_profile_gate() {
        // This test would have failed before plan-03 because the cpu-profile gate
        // blocked validation even when shells were otherwise valid.
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        // With cpu-profile gate removed, valid 4c1e inputs succeed at the envelope check.
        // After Phase 06 Bug 2 fix, accept wgpu-capability error on no-GPU CI.
        let result = unsafe {
            query_workspace_raw(
                RawApiId::INT4C1E_CART,
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        };
        // Query must succeed or fail with wgpu-capability (not cpu-profile gate).
        match result {
            Ok(_) => {} // expected on GPU CI
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                // no GPU adapter — not a cpu-profile gate, acceptable
            }
            Err(e) => panic!(
                "validate_4c1e_envelope should not block valid inputs with cpu-profile gate: {e:?}"
            ),
        }
    }

    #[cfg(not(feature = "unstable-source-api"))]
    #[test]
    fn source_only_symbol_requires_unstable_feature() {
        let (shls_4, atm, bas, env) = RawFixture::single_atom_four_shells();
        let err = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_ipip1_sph"),
                None,
                &shls_4,
                &atm,
                &bas,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::UnsupportedApi { requested }
                if requested.contains("requires feature `unstable-source-api`")
        ));
    }

    /// D-16: Unsupported paths must assert taxonomy prefixes in error messages.
    /// Tests layered compat → runtime → cubecl interaction for unsupported behavior.
    #[test]
    fn unsupported_behavior_reports_reason_taxonomy() {
        let (shls_4, atm_data, bas_data, env) = RawFixture::single_atom_four_shells();

        // D-16: unsupported_family taxonomy via executor supports_canonical_family.
        // An unknown family string returns false from supports_canonical_family.
        assert!(
            !cintx_cubecl::kernels::supports_canonical_family("nonexistent_family"),
            "nonexistent family must not be supported (D-16 precondition)"
        );
        // And the executor returns UnsupportedApi with unsupported_family: prefix for it.
        // We verify the executor unsupported-family path via the resolve path in kernels.
        // Since resolve_family takes an ExecutionPlan, we check via executor query_workspace.
        // Use a real operator but query via executor with an unsupported representation
        // by checking the ensure_supported_family path through the compat executor call.
        // The simplest verifiable check is via the f12 cart→sph envelope text (which uses
        // explicit unsupported taxonomy from resolve_raw_api's f12 prefix check).
        let err_f12_cart = unsafe {
            query_workspace_raw(
                RawApiId::Symbol("int2e_stg_cart"),
                None,
                &shls_4,
                &atm_data,
                &bas_data,
                &env,
                None,
            )
        }
        .unwrap_err();
        assert!(
            matches!(&err_f12_cart, cintxRsError::UnsupportedApi { requested }
                if requested.contains("with-f12 sph envelope")),
            "f12 cart must report sph envelope taxonomy (D-16): {err_f12_cart:?}"
        );

        // D-16: outside Validated4C1E prefix in 4c1e envelope boundary path.
        #[cfg(feature = "with-4c1e")]
        {
            let (shls_4_l5, atm_l5, mut bas_l5, env_l5) = RawFixture::single_atom_four_shells();
            bas_l5[1] = 5; // ANG_OF = 1, max(l)>4 triggers Validated4C1E boundary
            let err_4c1e = unsafe {
                query_workspace_raw(
                    RawApiId::INT4C1E_CART,
                    None,
                    &shls_4_l5,
                    &atm_l5,
                    &bas_l5,
                    &env_l5,
                    None,
                )
            }
            .unwrap_err();
            assert!(
                matches!(&err_4c1e, cintxRsError::UnsupportedApi { requested }
                    if requested.contains("outside Validated4C1E")),
                "4c1e envelope must report 'outside Validated4C1E' taxonomy (D-16): {err_4c1e:?}"
            );
        }

        // D-16: source-only symbol unsupported path uses explicit feature text.
        #[cfg(not(feature = "unstable-source-api"))]
        {
            let err_source = unsafe {
                query_workspace_raw(
                    RawApiId::Symbol("int2e_ipip1_sph"),
                    None,
                    &shls_4,
                    &atm_data,
                    &bas_data,
                    &env,
                    None,
                )
            }
            .unwrap_err();
            assert!(
                matches!(&err_source, cintxRsError::UnsupportedApi { requested }
                    if requested.contains("unstable-source-api")),
                "source-only symbol must report unstable-source-api taxonomy (D-16): {err_source:?}"
            );
        }
    }

    /// D-13: Layered regression covering runtime + cubecl + compat interaction paths.
    /// Verifies that backend_intent contract propagates from compat options through runtime.
    #[test]
    fn backend_intent_contract_propagates_through_compat_query_path() {
        use cintx_runtime::{BackendIntent, BackendKind};

        let fixture = RawFixture::single_atom_three_shells();
        // Query workspace with explicit backend intent in options via raw path.
        // After Phase 06 Bug 2 fix, execution_options_from_opt bootstraps wgpu and populates
        // backend_capability_token; accept wgpu-capability error on no-GPU CI.
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU adapter — skip planning_matches drift check
            }
            Err(e) => panic!("unexpected query error: {e}"),
        };

        // The query result carries backend contract from bootstrapped ExecutionOptions.
        // D-08: planning_matches must compare backend_intent and backend_capability_token.
        // Verify by constructing options with a drifted backend kind.
        let mut drifted_opts = cintx_runtime::ExecutionOptions::default();
        drifted_opts.backend_intent = BackendIntent {
            backend: BackendKind::Cpu,
            selector: "test".to_owned(),
        };

        // planning_matches returns false on backend drift — D-08 layered compat + runtime coverage.
        assert!(
            !query.planning_matches(&drifted_opts),
            "drifted backend_intent must fail planning_matches across compat→runtime boundary (D-13)"
        );

        // Matching options: re-run bootstrap to get same token so planning_matches succeeds.
        let matching_opts =
            execution_options_from_opt(None).expect("bootstrap must succeed when GPU is available");
        assert!(
            query.planning_matches(&matching_opts),
            "re-bootstrapped options must pass planning_matches (D-13)"
        );
    }

    #[test]
    fn eval_raw_staging_retrieval_smoke() {
        // Smoke test: proves eval_raw routes through RecordingExecutor and does not
        // short-circuit with an allocation error. The staging buffer is written by the
        // executor (even if the kernel is a stub producing zeros); the prior bug was that
        // the staging buffer was allocated and zero-filled in eval_raw *independently* of
        // the executor — discarding whatever the executor produced.
        //
        // This test verifies the path is connected end-to-end: query succeeds, eval_raw
        // succeeds (or fails with wgpu-capability on no-GPU CI), and the output buffer
        // length matches what the workspace query promised.
        // Accepts wgpu-capability error on no-GPU CI.
        let fixture = RawFixture::single_atom_three_shells();
        let query = match unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_SPH,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        } {
            Ok(q) => q,
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                return; // no GPU — cannot verify output content
            }
            Err(e) => panic!("unexpected query error: {e}"),
        };

        let n_elements = required_f64s_for_bytes(query.bytes).expect("cache conversion");
        let mut out = vec![0.0f64; n_elements];
        match unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_SPH,
                Some(&mut out),
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
                None,
            )
        } {
            Ok(summary) => {
                // Verify the staging retrieval path is wired: bytes_written must be nonzero
                // when GPU is available, regardless of whether kernel values are nonzero yet.
                assert!(
                    summary.bytes_written > 0,
                    "eval_raw must write nonzero bytes (RecordingExecutor staging retrieval connected)"
                );
                assert_eq!(
                    out.len(),
                    n_elements,
                    "output buffer length must match queried element count"
                );
            }
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                // no GPU — acceptable fail-closed outcome
            }
            Err(e) => panic!("unexpected eval_raw error: {e}"),
        }
    }

    #[test]
    fn fingerprint_propagation_smoke() {
        // Smoke test: proves execution_options_from_opt populates a non-zero fingerprint.
        // Accepts wgpu-capability error on no-GPU CI.
        let fixture = RawFixture::single_atom_three_shells();
        let result = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_SPH,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        };
        match result {
            Ok(query) => {
                assert_ne!(
                    query.backend_capability_token.capability_fingerprint, 0,
                    "capability_fingerprint must be non-zero when GPU adapter is available"
                );
            }
            Err(cintxRsError::UnsupportedApi { ref requested })
                if requested.contains("wgpu-capability") =>
            {
                // no GPU — cannot assert fingerprint
            }
            Err(e) => panic!("unexpected query error: {e}"),
        }
    }
}
