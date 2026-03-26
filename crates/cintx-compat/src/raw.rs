use cintx_core::cintxRsError;

pub const CHARGE_OF: usize = 0;
pub const PTR_COORD: usize = 1;
pub const NUC_MOD_OF: usize = 2;
pub const PTR_ZETA: usize = 3;
pub const PTR_FRAC_CHARGE: usize = 4;
// Upstream slot widths: ATM_SLOTS = 6, BAS_SLOTS = 8.
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
        let nprim = usize::try_from(self.nprim_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
            slot_width: BAS_SLOTS,
            provided: self.nprim_raw().unsigned_abs() as usize,
        })?;
        let nctr = usize::try_from(self.nctr_raw()).map_err(|_| cintxRsError::InvalidBasLayout {
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
        let coeff_len =
            nprim
                .checked_mul(nctr)
                .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                    from: "raw_bas",
                    detail: "coefficient range overflowed usize".to_owned(),
                })?;
        env.validate_range("PTR_COEFF", self.coeff_offset(), coeff_len)?;
        Ok(())
    }
}

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

    pub fn slice(&self, slot: &'static str, offset: i32, len: usize) -> Result<&'a [f64], cintxRsError> {
        let start = self.validate_range(slot, offset, len)?;
        Ok(&self.data[start..start + len])
    }
}

fn normalize_offset(slot: &'static str, offset: i32, env_len: usize) -> Result<usize, cintxRsError> {
    usize::try_from(offset).map_err(|_| cintxRsError::InvalidEnvOffset {
        slot,
        offset: env_len,
        env_len,
    })
}
