use core::ffi::c_void;
use core::ptr::NonNull;

use crate::errors::LibcintRsError;

pub const ATM_SLOTS: usize = 6;
pub const BAS_SLOTS: usize = 8;

const ATM_PTR_COORD_SLOT: usize = 1;
const ATM_NUC_MOD_SLOT: usize = 2;
const ATM_PTR_ZETA_SLOT: usize = 3;
const GAUSSIAN_NUC_MODEL: i32 = 2;

const BAS_ATOM_OF_SLOT: usize = 0;
const BAS_ANG_OF_SLOT: usize = 1;
const BAS_NPRIM_OF_SLOT: usize = 2;
const BAS_NCTR_OF_SLOT: usize = 3;
const BAS_KAPPA_OF_SLOT: usize = 4;
const BAS_PTR_EXP_SLOT: usize = 5;
const BAS_PTR_COEFF_SLOT: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawEnvView<'a> {
    env: &'a [f64],
}

impl<'a> RawEnvView<'a> {
    pub fn new(env: &'a [f64]) -> Self {
        Self { env }
    }

    pub fn len(&self) -> usize {
        self.env.len()
    }

    pub fn checked_offset_range(
        &self,
        field: &'static str,
        offset: i32,
        width: usize,
    ) -> Result<usize, LibcintRsError> {
        if width == 0 {
            return Err(LibcintRsError::InvalidInput {
                field,
                reason: "offset width must be greater than zero".to_string(),
            });
        }

        let start = checked_non_negative(offset, field)?;
        let end = start.checked_add(width).ok_or_else(|| LibcintRsError::InvalidInput {
            field,
            reason: format!("offset range overflows usize for width {width}"),
        })?;

        if end > self.env.len() {
            return Err(LibcintRsError::InvalidInput {
                field,
                reason: format!(
                    "offset range [{start}, {end}) exceeds env length {}",
                    self.env.len()
                ),
            });
        }

        Ok(start)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawAtmView<'a> {
    atm: &'a [i32],
}

impl<'a> RawAtmView<'a> {
    pub fn new(atm: &'a [i32]) -> Result<Self, LibcintRsError> {
        if !atm.len().is_multiple_of(ATM_SLOTS) {
            return Err(LibcintRsError::InvalidInput {
                field: "atm",
                reason: format!(
                    "length {} is not divisible by ATM_SLOTS {ATM_SLOTS}",
                    atm.len()
                ),
            });
        }
        Ok(Self { atm })
    }

    pub fn natm(&self) -> usize {
        self.atm.len() / ATM_SLOTS
    }

    pub fn validate_offsets(&self, env: &RawEnvView<'_>) -> Result<(), LibcintRsError> {
        for atom_index in 0..self.natm() {
            let row = self.row(atom_index);
            env.checked_offset_range("atm.ptr_coord", row[ATM_PTR_COORD_SLOT], 3)?;

            let nuc_mod = row[ATM_NUC_MOD_SLOT];
            let ptr_zeta = row[ATM_PTR_ZETA_SLOT];
            if nuc_mod == GAUSSIAN_NUC_MODEL || ptr_zeta != 0 {
                env.checked_offset_range("atm.ptr_zeta", ptr_zeta, 1)?;
            }
        }

        Ok(())
    }

    fn row(&self, index: usize) -> &'a [i32] {
        let offset = index * ATM_SLOTS;
        &self.atm[offset..offset + ATM_SLOTS]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawBasView<'a> {
    bas: &'a [i32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawShellMeta {
    pub atom_of: usize,
    pub angular_momentum: usize,
    pub nprim: usize,
    pub nctr: usize,
    pub kappa: i32,
    pub ptr_exp: i32,
    pub ptr_coeff: i32,
}

impl<'a> RawBasView<'a> {
    pub fn new(bas: &'a [i32]) -> Result<Self, LibcintRsError> {
        if !bas.len().is_multiple_of(BAS_SLOTS) {
            return Err(LibcintRsError::InvalidInput {
                field: "bas",
                reason: format!(
                    "length {} is not divisible by BAS_SLOTS {BAS_SLOTS}",
                    bas.len()
                ),
            });
        }
        Ok(Self { bas })
    }

    pub fn nbas(&self) -> usize {
        self.bas.len() / BAS_SLOTS
    }

    pub fn shell_meta(&self, shell_index: usize) -> Result<RawShellMeta, LibcintRsError> {
        if shell_index >= self.nbas() {
            return Err(LibcintRsError::InvalidInput {
                field: "shls",
                reason: format!(
                    "shell index {shell_index} is out of bounds for {} shells",
                    self.nbas()
                ),
            });
        }

        let row = self.row(shell_index);
        Ok(RawShellMeta {
            atom_of: checked_non_negative(row[BAS_ATOM_OF_SLOT], "bas.atom_of")?,
            angular_momentum: checked_non_negative(row[BAS_ANG_OF_SLOT], "bas.ang_of")?,
            nprim: checked_positive(row[BAS_NPRIM_OF_SLOT], "bas.nprim_of")?,
            nctr: checked_positive(row[BAS_NCTR_OF_SLOT], "bas.nctr_of")?,
            kappa: row[BAS_KAPPA_OF_SLOT],
            ptr_exp: row[BAS_PTR_EXP_SLOT],
            ptr_coeff: row[BAS_PTR_COEFF_SLOT],
        })
    }

    pub fn validate_offsets(
        &self,
        natm: usize,
        env: &RawEnvView<'_>,
    ) -> Result<(), LibcintRsError> {
        for shell_index in 0..self.nbas() {
            let meta = self.shell_meta(shell_index)?;
            if meta.atom_of >= natm {
                return Err(LibcintRsError::InvalidInput {
                    field: "bas.atom_of",
                    reason: format!(
                        "shell {shell_index} references atom {}, but natm is {natm}",
                        meta.atom_of
                    ),
                });
            }

            env.checked_offset_range("bas.ptr_exp", meta.ptr_exp, meta.nprim)?;
            let coeff_width = meta
                .nprim
                .checked_mul(meta.nctr)
                .ok_or_else(|| LibcintRsError::InvalidInput {
                    field: "bas.ptr_coeff",
                    reason: "nprim*nctr overflows usize".to_string(),
                })?;
            env.checked_offset_range("bas.ptr_coeff", meta.ptr_coeff, coeff_width)?;
        }

        Ok(())
    }

    fn row(&self, index: usize) -> &'a [i32] {
        let offset = index * BAS_SLOTS;
        &self.bas[offset..offset + BAS_SLOTS]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawShellTuple<'a> {
    shls: &'a [i32],
}

impl<'a> RawShellTuple<'a> {
    pub fn new(shls: &'a [i32]) -> Self {
        Self { shls }
    }

    pub fn validate(&self, expected_arity: usize, nbas: usize) -> Result<Vec<usize>, LibcintRsError> {
        if self.shls.len() != expected_arity {
            return Err(LibcintRsError::InvalidLayout {
                item: "shls_arity",
                expected: expected_arity,
                got: self.shls.len(),
            });
        }

        let mut validated = Vec::with_capacity(self.shls.len());
        for shell_index in self.shls {
            let shell_index = checked_non_negative(*shell_index, "shls")?;
            if shell_index >= nbas {
                return Err(LibcintRsError::InvalidInput {
                    field: "shls",
                    reason: format!("index {shell_index} is out of bounds for {nbas} shells"),
                });
            }
            validated.push(shell_index);
        }
        Ok(validated)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompatDims<'a> {
    dims: Option<&'a [i32]>,
}

impl<'a> CompatDims<'a> {
    pub fn new(dims: Option<&'a [i32]>) -> Self {
        Self { dims }
    }

    pub fn validate(
        &self,
        expected_arity: usize,
        natural_dims: &[usize],
    ) -> Result<Vec<usize>, LibcintRsError> {
        let Some(dims) = self.dims else {
            return Ok(natural_dims.to_vec());
        };

        if dims.len() != expected_arity {
            return Err(LibcintRsError::InvalidLayout {
                item: "dims_arity",
                expected: expected_arity,
                got: dims.len(),
            });
        }

        let mut provided = Vec::with_capacity(dims.len());
        for dim in dims {
            provided.push(checked_positive(*dim, "dims")?);
        }

        if provided != natural_dims {
            return Err(LibcintRsError::DimsBufferMismatch {
                expected: natural_dims.to_vec(),
                provided,
            });
        }

        Ok(provided)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawCacheView<'a> {
    cache: Option<&'a [f64]>,
}

impl<'a> RawCacheView<'a> {
    pub fn new(cache: Option<&'a [f64]>) -> Self {
        Self { cache }
    }

    pub fn has_cache(&self) -> bool {
        self.cache.is_some()
    }

    pub fn validate_min_len(&self, min_len: usize) -> Result<(), LibcintRsError> {
        if let Some(cache) = self.cache
            && cache.len() < min_len
        {
            return Err(LibcintRsError::InvalidLayout {
                item: "cache_length",
                expected: min_len,
                got: cache.len(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawOptView {
    opt: Option<NonNull<c_void>>,
}

impl RawOptView {
    pub fn new(opt: Option<NonNull<c_void>>) -> Self {
        Self { opt }
    }

    pub fn has_opt(&self) -> bool {
        self.opt.is_some()
    }

    pub fn validate_with_cache(&self, cache: &RawCacheView<'_>) -> Result<(), LibcintRsError> {
        if self.has_opt() && !cache.has_cache() {
            return Err(LibcintRsError::InvalidInput {
                field: "cache",
                reason: "cache must be provided when opt is supplied".to_string(),
            });
        }
        Ok(())
    }
}

fn checked_non_negative(value: i32, field: &'static str) -> Result<usize, LibcintRsError> {
    if value < 0 {
        return Err(LibcintRsError::InvalidInput {
            field,
            reason: format!("value {value} must be greater than or equal to zero"),
        });
    }

    usize::try_from(value).map_err(|_| LibcintRsError::InvalidInput {
        field,
        reason: format!("value {value} cannot be represented as usize"),
    })
}

fn checked_positive(value: i32, field: &'static str) -> Result<usize, LibcintRsError> {
    if value <= 0 {
        return Err(LibcintRsError::InvalidInput {
            field,
            reason: format!("value {value} must be greater than zero"),
        });
    }

    usize::try_from(value).map_err(|_| LibcintRsError::InvalidInput {
        field,
        reason: format!("value {value} cannot be represented as usize"),
    })
}
