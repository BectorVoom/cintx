use core::ffi::c_void;
use core::ptr::NonNull;

use crate::contracts::{IntegralFamily, Operator, Representation};
use crate::errors::LibcintRsError;

use super::views::{
    CompatDims, RawAtmView, RawBasView, RawCacheView, RawEnvView, RawOptView, RawShellMeta,
    RawShellTuple,
};

#[derive(Debug, Clone, Copy)]
pub struct RawValidationRequest<'a> {
    pub operator: Operator,
    pub representation: Representation,
    pub shls: &'a [i32],
    pub dims: Option<&'a [i32]>,
    pub atm: &'a [i32],
    pub bas: &'a [i32],
    pub env: &'a [f64],
    pub cache: Option<&'a [f64]>,
    pub opt: Option<NonNull<c_void>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawValidationResult {
    pub shell_tuple: Vec<usize>,
    pub shell_angular_momentum: Vec<u8>,
    pub primitive_count: usize,
    pub natural_dims: Vec<usize>,
    pub dims: Vec<usize>,
    pub required_elements: usize,
    pub natm: usize,
    pub nbas: usize,
    pub env_len: usize,
    pub cache_required_len: usize,
    pub has_cache: bool,
    pub has_opt: bool,
}

pub fn validate_raw_contract(
    request: RawValidationRequest<'_>,
) -> Result<RawValidationResult, LibcintRsError> {
    let env = RawEnvView::new(request.env);
    let atm = RawAtmView::new(request.atm)?;
    atm.validate_offsets(&env)?;

    let bas = RawBasView::new(request.bas)?;
    bas.validate_offsets(atm.natm(), &env)?;

    let expected_arity = family_arity(request.operator.family());
    let shls = RawShellTuple::new(request.shls).validate(expected_arity, bas.nbas())?;

    let mut natural_dims = Vec::with_capacity(shls.len());
    let mut shell_angular_momentum = Vec::with_capacity(shls.len());
    let mut primitive_count = 0usize;
    for shell_index in &shls {
        let meta = bas.shell_meta(*shell_index)?;
        let angular_momentum =
            u8::try_from(meta.angular_momentum).map_err(|_| LibcintRsError::InvalidInput {
                field: "bas.ang_of",
                reason: format!(
                    "angular momentum {} exceeds supported u8 range",
                    meta.angular_momentum
                ),
            })?;
        shell_angular_momentum.push(angular_momentum);
        primitive_count =
            primitive_count
                .checked_add(meta.nprim)
                .ok_or_else(|| LibcintRsError::InvalidInput {
                    field: "bas.nprim_of",
                    reason: "primitive count overflows usize".to_string(),
                })?;
        natural_dims.push(shell_component_count(meta, request.representation)?);
    }

    let dims = CompatDims::new(request.dims).validate(expected_arity, &natural_dims)?;
    let required_elements = checked_product(&dims)?;

    let cache = RawCacheView::new(request.cache);
    let opt = RawOptView::new(request.opt);
    opt.validate_with_cache(&cache)?;

    let cache_required_len = dims.len().max(1);
    cache.validate_min_len(cache_required_len)?;

    Ok(RawValidationResult {
        shell_tuple: shls,
        shell_angular_momentum,
        primitive_count,
        natural_dims,
        dims,
        required_elements,
        natm: atm.natm(),
        nbas: bas.nbas(),
        env_len: env.len(),
        cache_required_len,
        has_cache: cache.has_cache(),
        has_opt: opt.has_opt(),
    })
}

fn checked_product(dims: &[usize]) -> Result<usize, LibcintRsError> {
    let mut product = 1usize;
    for dim in dims {
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "dims",
                reason: "dimension product overflows usize".to_string(),
            })?;
    }
    Ok(product)
}

fn shell_component_count(
    meta: RawShellMeta,
    representation: Representation,
) -> Result<usize, LibcintRsError> {
    let per_contracted = match representation {
        Representation::Cartesian => cartesian_len(meta.angular_momentum)?,
        Representation::Spherical => spherical_len(meta.angular_momentum)?,
        Representation::Spinor => spinor_len(meta.angular_momentum, meta.kappa)?,
    };

    per_contracted
        .checked_mul(meta.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.nctr_of",
            reason: "contracted shell component count overflows usize".to_string(),
        })
}

fn cartesian_len(angular_momentum: usize) -> Result<usize, LibcintRsError> {
    let l_plus_1 = angular_momentum
        .checked_add(1)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.ang_of",
            reason: "angular momentum overflows usize".to_string(),
        })?;
    let l_plus_2 = angular_momentum
        .checked_add(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.ang_of",
            reason: "angular momentum overflows usize".to_string(),
        })?;

    let numerator = l_plus_1
        .checked_mul(l_plus_2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.ang_of",
            reason: "cartesian component computation overflows usize".to_string(),
        })?;

    Ok(numerator / 2)
}

fn spherical_len(angular_momentum: usize) -> Result<usize, LibcintRsError> {
    angular_momentum
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.ang_of",
            reason: "spherical component computation overflows usize".to_string(),
        })
}

fn spinor_len(angular_momentum: usize, kappa: i32) -> Result<usize, LibcintRsError> {
    if kappa == 0 {
        return angular_momentum
            .checked_mul(4)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "bas.ang_of",
                reason: "spinor component computation overflows usize".to_string(),
            });
    }

    if kappa < 0 {
        return angular_momentum
            .checked_mul(2)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "bas.ang_of",
                reason: "spinor component computation overflows usize".to_string(),
            });
    }

    if angular_momentum == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.kappa_of",
            reason: "positive kappa requires angular momentum > 0 for spinor shells".to_string(),
        });
    }

    angular_momentum
        .checked_mul(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.ang_of",
            reason: "spinor component computation overflows usize".to_string(),
        })
}

fn family_arity(family: IntegralFamily) -> usize {
    match family {
        IntegralFamily::OneElectron | IntegralFamily::TwoCenterTwoElectron => 2,
        IntegralFamily::ThreeCenterOneElectron | IntegralFamily::ThreeCenterTwoElectron => 3,
        IntegralFamily::TwoElectron => 4,
    }
}
