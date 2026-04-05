use crate::layout::{CompatDims, ensure_cache_len};
use crate::optimizer::RawOptimizerHandle;
use cintx_core::{
    Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple, cintxRsError,
};
use cintx_cubecl::{CUBECL_RUNTIME_PROFILE, CubeClExecutor};
use cintx_ops::resolver::{HelperKind, OperatorDescriptor, Resolver, ResolverError};
use cintx_runtime::{
    BackendExecutor, ExecutionIo, ExecutionOptions, ExecutionPlan,
    HostWorkspaceAllocator, WorkspaceAllocator, WorkspaceQuery, schedule_chunks, query_workspace,
};
use std::mem::size_of;
use std::sync::Arc;

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

/// First usable index in the env array for user data (coordinates, exponents, coefficients).
///
/// libcint reserves env[0..PTR_ENV_START] for global parameters:
///   PTR_EXPCUTOFF = 0, PTR_COMMON_ORIG = 1..3, PTR_RINV_ORIG = 4..6,
///   PTR_RINV_ZETA = 7, PTR_RANGE_OMEGA = 8, PTR_F12_ZETA = 9, PTR_GTG_ZETA = 10,
///   PTR_GRIDS = 12..19.
///
/// User data (atom coordinates, exponents, coefficients) MUST start at env[20] or later.
/// Placing user data at env[0..19] corrupts the global parameter fields and causes
/// incorrect results for 2e+ integrals that read PTR_RANGE_OMEGA or PTR_EXPCUTOFF.
pub const PTR_ENV_START: usize = 20;

/// Index of the F12/STG/YP zeta parameter in the libcint env array.
///
/// libcint defines `PTR_F12_ZETA = 9` in `cint_bas.h`. Raw callers set `env[9] = zeta`
/// before calling any F12/STG/YP integral. This constant allows raw compat code to
/// extract the zeta value from the env array without hardcoding the magic index.
pub const PTR_F12_ZETA: usize = 9;

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

    pub const INT3C1E_CART: Self = Self::Symbol("int3c1e_cart");
    pub const INT3C1E_SPH: Self = Self::Symbol("int3c1e_sph");
    pub const INT3C1E_SPINOR: Self = Self::Symbol("int3c1e_spinor");

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
    _options: ExecutionOptions,
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

    let mut plan = ExecutionPlan::new(
        prepared.op,
        prepared.representation,
        &prepared.basis,
        prepared.shells.clone(),
        &prepared.query,
    )?;

    // Extract f12_zeta from env[PTR_F12_ZETA] for F12/STG/YP integrals (raw compat path).
    // Raw callers are expected to set env[9] = zeta before calling any F12 integral.
    // The manifest canonical_family for STG/YP operators is "f12"; operator_name is "stg"/"yp".
    // We detect F12 symbols by their full symbol name prefix (int2e_stg / int2e_yp).
    if is_f12_family_symbol(plan.descriptor.operator_symbol()) {
        let zeta = env.get(PTR_F12_ZETA).copied().unwrap_or(0.0);
        plan.operator_env_params.f12_zeta = Some(zeta);
        // Validate before dispatch so we return a typed error on bad input.
        cintx_runtime::validator::validate_f12_env_params(
            "f12",
            &plan.operator_env_params,
        )?;
    }

    let executor = CubeClExecutor::new();
    let mut allocator = HostWorkspaceAllocator::default();

    // Allocate the full staging accumulator that we own, so we can read values after execute().
    // RecordingExecutor is not needed: we construct ExecutionIo with our own staging slice and
    // read it directly after executor.execute() returns for each chunk.
    let staging_elements = plan.output_layout.staging_elements;
    let mut staging = Vec::new();
    staging
        .try_reserve_exact(staging_elements)
        .map_err(|_| cintxRsError::HostAllocationFailed {
            bytes: staging_elements.saturating_mul(size_of::<f64>()),
        })?;
    staging.resize(staging_elements, 0.0);

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

    let schedule = schedule_chunks(&plan.workspace);
    let total_units = plan.workspace.work_units.max(1);

    let mut total_not0: i32 = 0;
    let mut total_transfer_bytes: usize = 0;

    for chunk in schedule.chunks() {
        // Compute staging slice range for this chunk (mirrors staging_elements_for_chunk logic).
        let start = chunk.work_unit_start.min(total_units);
        let end = chunk
            .work_unit_start
            .saturating_add(chunk.work_unit_count)
            .min(total_units);
        let prefix =
            staging_elements.saturating_mul(start) / total_units;
        let suffix =
            staging_elements.saturating_mul(end) / total_units;
        let chunk_len = suffix.saturating_sub(prefix).max(1);

        // Allocate the chunk staging slice and workspace.
        let chunk_staging_bytes = chunk_len
            .checked_mul(size_of::<f64>())
            .ok_or(cintxRsError::HostAllocationFailed { bytes: usize::MAX })?;
        let mut chunk_staging = Vec::new();
        chunk_staging
            .try_reserve_exact(chunk_len)
            .map_err(|_| cintxRsError::HostAllocationFailed {
                bytes: chunk_staging_bytes,
            })?;
        chunk_staging.resize(chunk_len, 0.0);

        let mut workspace = allocator.try_alloc(chunk.bytes, plan.workspace.alignment)?;

        {
            let mut io =
                ExecutionIo::new(chunk, &mut chunk_staging, &mut workspace, plan.dispatch)?;
            io.ensure_output_contract()?;
            let chunk_stats = executor.execute(&plan, &mut io)?;
            total_not0 = total_not0.saturating_add(chunk_stats.not0.max(0));
            total_transfer_bytes =
                total_transfer_bytes.saturating_add(io.transfer_bytes());
        }
        allocator.release(workspace);

        // Copy chunk staging into the appropriate range of the accumulator.
        let dest_end = prefix.saturating_add(chunk_len).min(staging_elements);
        if prefix < dest_end {
            staging[prefix..dest_end]
                .copy_from_slice(&chunk_staging[..dest_end - prefix]);
        }
    }

    let out = out.expect("checked out.is_some()");
    let written_elements = prepared.compat_dims.write(out, &staging)?;
    let bytes_written = written_elements
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| cintxRsError::ChunkPlanFailed {
            from: "compat_raw",
            detail: "written byte count overflowed usize".to_owned(),
        })?;

    Ok(RawEvalSummary {
        not0: total_not0,
        bytes_written,
        workspace_bytes: plan.workspace.bytes,
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

    // Source-only symbols use the "unstable-source" profile and are gated by the
    // unstable-source-api feature. When the feature is enabled, skip the profile
    // check (the source gate below handles authorization). When the feature is
    // disabled, reject with a clear message regardless of the base profile.
    if descriptor.is_source_only() {
        if !unstable_source_api_enabled() {
            return Err(cintxRsError::UnsupportedApi {
                requested: format!(
                    "source-only symbol {symbol} requires feature `unstable-source-api`"
                ),
            });
        }
        // Feature is enabled: source gate passed, skip profile check.
        return Ok(());
    }

    // Non-source-only symbols: check the active compiled profile.
    let profile = active_manifest_profile();
    if !descriptor.is_compiled_in_profile(profile) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!("raw api {symbol} is not compiled in active profile {profile}"),
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

    // D-05: Spinor rejection FIRST — before feature gate check.
    // A Spinor 4c1e request must return UnsupportedApi with "spinor" in the message
    // regardless of whether the with-4c1e feature is enabled.
    if matches!(representation, Representation::Spinor) {
        return Err(validated_4c1e_error("spinor representation not supported for 4c1e"));
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
    if CUBECL_RUNTIME_PROFILE != "cpu" {
        return Err(validated_4c1e_error("CubeCL backend must be cpu"));
    }

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

    let options = execution_options_from_opt(opt);
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
        _options: options,
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

fn execution_options_from_opt(opt: Option<&RawOptimizerHandle>) -> ExecutionOptions {
    let mut options = ExecutionOptions::default();
    options.profile_label = Some(active_manifest_profile());
    if let Some(opt) = opt {
        options.memory_limit_bytes = opt.workspace_hint_bytes();
    }
    options
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
        let fixture = RawFixture::single_atom_three_shells();

        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&[1]),
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 2,
                provided: 1
            }
        ));

        let err = unsafe {
            query_workspace_raw(
                RawApiId::INT3C1E_P2_CART,
                Some(&[1, 2]),
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 3,
                provided: 2
            }
        ));
    }

    #[test]
    fn undersized_output_buffer_is_reported() {
        let fixture = RawFixture::single_atom_three_shells();
        let mut out = vec![0.0; 1];
        let err = unsafe {
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
        }
        .unwrap_err();
        assert!(matches!(err, cintxRsError::BufferTooSmall { .. }));
    }

    #[test]
    fn query_workspace_raw_and_eval_raw_none_match_workspace_expectations() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .expect("query should succeed");

        let summary = unsafe {
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
        }
        .expect("out == None should return requirements");

        assert_eq!(summary.not0, 0);
        assert_eq!(summary.bytes_written, 0);
        assert_eq!(summary.workspace_bytes, query.bytes);
    }

    #[test]
    fn memory_limit_hint_can_chunk_successfully() {
        let fixture = RawFixture::single_atom_three_shells();
        let opt = RawOptimizerHandle::with_hints(None, Some(128));
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                Some(&opt),
            )
        }
        .expect("query should succeed with chunking");
        assert!(query.chunk_count >= 1);

        let mut out = vec![99.0; 3];
        let summary = unsafe {
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
        }
        .expect("eval should succeed");

        assert!(summary.bytes_written > 0);
        // Kernel stubs write zeros to staging (real kernels come in Phase 9/10).
        // Verify eval_raw completed successfully and staging is populated (all zeros from stubs).
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn memory_limit_failure_keeps_output_slice_unchanged() {
        let fixture = RawFixture::single_atom_three_shells();
        let opt = RawOptimizerHandle::with_hints(None, Some(1));
        let mut out = vec![7.0; 3];

        let err = unsafe {
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
        }
        .unwrap_err();

        assert!(matches!(err, cintxRsError::MemoryLimitExceeded { .. }));
        assert!(
            out.iter().all(|value| *value == 7.0),
            "output slice unchanged on failure (no partial write)"
        );
    }

    #[test]
    fn cache_buffer_too_small_is_rejected_before_execution() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT1E_OVLP_CART,
                None,
                &fixture.shls_2,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .expect("query should succeed");

        let required_cache = required_f64s_for_bytes(query.bytes).expect("cache conversion");
        let mut out = vec![0.0; 3];
        let mut cache = vec![0.0; required_cache.saturating_sub(1)];
        let err = unsafe {
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
        }
        .unwrap_err();

        assert!(matches!(err, cintxRsError::BufferTooSmall { .. }));
    }

    #[test]
    fn three_center_contract_query_and_eval_work_for_supported_backend() {
        let fixture = RawFixture::single_atom_three_shells();
        let query = unsafe {
            query_workspace_raw(
                RawApiId::INT3C1E_P2_CART,
                None,
                &fixture.shls_3,
                &fixture.atm,
                &fixture.bas,
                &fixture.env,
                None,
            )
        }
        .expect("3c query should still resolve and plan");
        assert_eq!(query.work_units, 3);

        let mut out = vec![1.0; 3];
        let summary = unsafe {
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
        }
        .expect("3c eval should succeed when kernel support is available");
        assert!(summary.bytes_written > 0);
        // Kernel stubs write zeros to staging (real kernels come in Phase 9/10).
        assert!(out.iter().all(|value| *value == 0.0));
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
        let query = unsafe {
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
        .expect("with-f12 should allow sph-only f12 symbols");
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
        let query = unsafe {
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
        .expect("validated 4c1e envelope should be queryable");
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

    /// Verify that eval_raw() uses direct executor.execute() with an owned staging buffer,
    /// not RecordingExecutor. This is a compile-time and runtime guarantee: RecordingExecutor
    /// no longer exists in this module, and the staging path is exercised directly.
    #[test]
    fn eval_raw_reads_staging_directly() {
        let fixture = RawFixture::single_atom_three_shells();
        // Allocate enough output for a 2-shell 1e integral (int1e_ovlp_cart: 2-center, cart).
        // Shell 0 has ang=0 (1 AO), shell 1 has ang=1 (3 AOs). Output size = 1 * 3 = 3 elements.
        let mut out = vec![0.0f64; 3];
        let result = unsafe {
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
        };
        // eval_raw must succeed: the direct staging path is exercised end-to-end.
        // bytes_written > 0 confirms the staging buffer was written and output was committed.
        let summary = result.expect("eval_raw_reads_staging_directly should succeed");
        assert!(
            summary.bytes_written > 0,
            "bytes_written must be > 0 (staging path was exercised): bytes_written={}",
            summary.bytes_written
        );
    }

    /// Verify that eval_raw returns InvalidEnvParam when env[PTR_F12_ZETA] is 0.0
    /// for an F12 symbol. This tests that the raw compat path calls validate_f12_env_params.
    #[cfg(feature = "with-f12")]
    #[test]
    fn eval_raw_f12_symbol_with_zero_zeta_returns_invalid_env_param() {
        let (shls_4, atm, bas, _env) = RawFixture::single_atom_four_shells();
        // Construct env with env[PTR_F12_ZETA=9] = 0.0 (invalid zeta).
        // The error fires before execution, so we only need valid enough env for
        // descriptor lookup and plan construction. Use the four-shell env layout
        // with zeta forced to 0 at index 9.
        let mut env_full = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.0, 0.4];
        env_full[PTR_F12_ZETA] = 0.0;
        let mut out = vec![0.0f64; 16]; // 2x2x2x2 output upper bound
        let err = unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                Some(&mut out),
                None,
                &shls_4,
                &atm,
                &bas,
                &env_full,
                None,
                None,
            )
        }
        .unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "PTR_F12_ZETA"),
            "expected InvalidEnvParam(PTR_F12_ZETA) for zero zeta, got: {err:?}"
        );
    }

    /// Verify that eval_raw passes env param validation (no InvalidEnvParam) when
    /// env[PTR_F12_ZETA] is non-zero for an F12 symbol. The call may fail later at
    /// UnsupportedApi or executor level (no GPU in test), but the zeta gate must pass.
    #[cfg(feature = "with-f12")]
    #[test]
    fn eval_raw_f12_symbol_with_valid_zeta_passes_env_param_validation() {
        let (shls_4, atm, bas, mut env_full) = RawFixture::single_atom_four_shells();
        env_full[PTR_F12_ZETA] = 1.2; // valid non-zero zeta
        let mut out = vec![0.0f64; 16];
        let result = unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_stg_sph"),
                Some(&mut out),
                None,
                &shls_4,
                &atm,
                &bas,
                &env_full,
                None,
                None,
            )
        };
        // Must not be InvalidEnvParam — that would mean our validation is wrong.
        assert!(
            !matches!(result, Err(cintxRsError::InvalidEnvParam { .. })),
            "eval_raw should not return InvalidEnvParam when zeta=1.2: {result:?}"
        );
    }
}
