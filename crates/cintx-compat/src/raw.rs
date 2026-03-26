use crate::layout::{CompatDims, ensure_cache_len};
use crate::optimizer::RawOptimizerHandle;
use cintx_core::{
    Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell, ShellTuple, cintxRsError,
};
use cintx_cubecl::CubeClExecutor;
use cintx_ops::resolver::{HelperKind, OperatorDescriptor, Resolver, ResolverError};
use cintx_runtime::{
    ExecutionOptions, ExecutionPlan, HostWorkspaceAllocator, WorkspaceQuery, evaluate,
    query_workspace,
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

pub const POINT_NUC: i32 = 1;
pub const GAUSSIAN_NUC: i32 = 2;
pub const FRAC_CHARGE_NUC: i32 = 3;

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

    let executor = CubeClExecutor::new();
    let mut allocator = HostWorkspaceAllocator::default();
    let stats = evaluate(plan, &prepared.options, &mut allocator, &executor)?;

    let mut staging = Vec::new();
    staging.try_reserve_exact(required_elements).map_err(|_| {
        cintxRsError::HostAllocationFailed {
            bytes: required_elements.saturating_mul(size_of::<f64>()),
        }
    })?;
    staging.resize(required_elements, 0.0);

    let out = out.expect("checked out.is_some()");
    let written_elements = prepared.compat_dims.write(out, &staging)?;
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
    let descriptor =
        Resolver::descriptor_by_symbol(api.symbol()).map_err(|err| map_resolver_error(api, err))?;

    if !matches!(
        descriptor.entry.helper_kind,
        HelperKind::Operator | HelperKind::Legacy
    ) {
        return Err(cintxRsError::UnsupportedApi {
            requested: format!(
                "raw api {} must resolve to operator/legacy manifest entries",
                api.symbol()
            ),
        });
    }

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
    if let Some(opt) = opt {
        options.memory_limit_bytes = opt.workspace_hint_bytes();
        options.profile_label = opt.symbol_hint();
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
        assert!(out.iter().all(|value| *value == 0.0));
    }
}
