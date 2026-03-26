use cintx_core::cintxRsError;
use smallvec::SmallVec;
use std::mem::size_of;

const MAX_COMPAT_ARITY: usize = 4;

/// Compat-owned output contract. The component axis is manifest-derived and
/// never provided by raw `dims`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatDims {
    extents: SmallVec<[usize; MAX_COMPAT_ARITY]>,
    component_count: usize,
    complex_interleaved: bool,
}

impl CompatDims {
    pub fn natural(
        extents: &[usize],
        component_count: usize,
        complex_interleaved: bool,
    ) -> Result<Self, cintxRsError> {
        Self::new(extents, component_count, complex_interleaved)
    }

    pub fn from_override(
        expected_extents: &[usize],
        dims: Option<&[i32]>,
        component_count: usize,
        complex_interleaved: bool,
    ) -> Result<Self, cintxRsError> {
        let natural = Self::new(expected_extents, component_count, complex_interleaved)?;
        let Some(dims) = dims else {
            return Ok(natural);
        };

        if dims.len() != natural.arity() {
            return Err(cintxRsError::InvalidDims {
                expected: natural.arity(),
                provided: dims.len(),
            });
        }

        let mut parsed = SmallVec::<[usize; MAX_COMPAT_ARITY]>::with_capacity(dims.len());
        for dim in dims {
            let parsed_dim = usize::try_from(*dim).map_err(|_| cintxRsError::InvalidDims {
                expected: natural.arity(),
                provided: dims.len(),
            })?;
            parsed.push(parsed_dim);
        }

        if parsed.as_slice() != natural.extents() {
            return Err(cintxRsError::InvalidDims {
                expected: natural.arity(),
                provided: dims.len(),
            });
        }

        Ok(Self {
            extents: parsed,
            component_count,
            complex_interleaved,
        })
    }

    fn new(
        extents: &[usize],
        component_count: usize,
        complex_interleaved: bool,
    ) -> Result<Self, cintxRsError> {
        if extents.len() > MAX_COMPAT_ARITY {
            return Err(cintxRsError::InvalidDims {
                expected: MAX_COMPAT_ARITY,
                provided: extents.len(),
            });
        }

        let mut owned = SmallVec::<[usize; MAX_COMPAT_ARITY]>::with_capacity(extents.len());
        owned.extend_from_slice(extents);

        Ok(Self {
            extents: owned,
            component_count,
            complex_interleaved,
        })
    }

    pub fn arity(&self) -> usize {
        self.extents.len()
    }

    pub fn extents(&self) -> &[usize] {
        &self.extents
    }

    pub fn component_count(&self) -> usize {
        self.component_count
    }

    pub fn complex_interleaved(&self) -> bool {
        self.complex_interleaved
    }

    pub fn required_elements(&self) -> Result<usize, cintxRsError> {
        required_elems_from_dims(
            self.arity(),
            self.component_count,
            self.extents(),
            self.complex_interleaved,
        )
    }

    pub fn ensure_output_len(&self, provided: usize) -> Result<usize, cintxRsError> {
        let required = self.required_elements()?;
        if provided < required {
            return Err(cintxRsError::BufferTooSmall { required, provided });
        }
        Ok(required)
    }

    /// Compat owns the final caller-visible write. Backends only fill staging.
    pub fn write(&self, out: &mut [f64], staging: &[f64]) -> Result<usize, cintxRsError> {
        let required = self.ensure_output_len(out.len())?;
        if staging.len() < required {
            return Err(cintxRsError::BufferTooSmall {
                required,
                provided: staging.len(),
            });
        }
        out[..required].copy_from_slice(&staging[..required]);
        Ok(required)
    }
}

pub fn required_elems_from_dims(
    arity: usize,
    component_count: usize,
    dims: &[usize],
    complex_interleaved: bool,
) -> Result<usize, cintxRsError> {
    if dims.len() != arity {
        return Err(cintxRsError::InvalidDims {
            expected: arity,
            provided: dims.len(),
        });
    }

    let base = dims
        .iter()
        .try_fold(component_count.max(1), |acc, extent| {
            acc.checked_mul(*extent).ok_or(cintxRsError::ChunkPlanFailed {
                from: "compat_layout",
                detail: "required element count overflowed usize".to_owned(),
            })
        })?;

    if complex_interleaved {
        base.checked_mul(2).ok_or(cintxRsError::ChunkPlanFailed {
            from: "compat_layout",
            detail: "complex element count overflowed usize".to_owned(),
        })
    } else {
        Ok(base)
    }
}

pub fn required_f64s_for_bytes(bytes: usize) -> Result<usize, cintxRsError> {
    bytes.checked_add(size_of::<f64>() - 1)
        .map(|rounded| rounded / size_of::<f64>())
        .ok_or(cintxRsError::ChunkPlanFailed {
            from: "compat_layout",
            detail: "workspace byte requirement overflowed usize".to_owned(),
        })
}

pub fn ensure_cache_len(required_bytes: usize, provided: usize) -> Result<usize, cintxRsError> {
    let required = required_f64s_for_bytes(required_bytes)?;
    if provided < required {
        return Err(cintxRsError::BufferTooSmall { required, provided });
    }
    Ok(required)
}
