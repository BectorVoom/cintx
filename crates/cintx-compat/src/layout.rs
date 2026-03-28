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
            acc.checked_mul(*extent)
                .ok_or(cintxRsError::ChunkPlanFailed {
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
    bytes
        .checked_add(size_of::<f64>() - 1)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_elements_reject_invalid_dims_arity() {
        let err = required_elems_from_dims(2, 1, &[3], false).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 2,
                provided: 1
            }
        ));
    }

    #[test]
    fn from_override_requires_exact_arity_and_extents() {
        let natural = CompatDims::from_override(&[2, 3], None, 1, false).unwrap();
        assert_eq!(natural.extents(), &[2, 3]);

        let err = CompatDims::from_override(&[2, 3], Some(&[2]), 1, false).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 2,
                provided: 1
            }
        ));

        let err = CompatDims::from_override(&[2, 3], Some(&[2, 3, 4]), 1, false).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 2,
                provided: 3
            }
        ));

        let err = CompatDims::from_override(&[2, 3], Some(&[2, 4]), 1, false).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 2,
                provided: 2
            }
        ));
    }

    #[test]
    fn ensure_output_len_and_write_enforce_buffer_requirements() {
        let dims = CompatDims::natural(&[2, 3], 1, false).unwrap();
        let required = dims.required_elements().unwrap();
        assert_eq!(required, 6);

        let err = dims.ensure_output_len(5).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::BufferTooSmall {
                required: 6,
                provided: 5
            }
        ));

        let mut out = vec![9.0; 8];
        let staging = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let written = dims.write(&mut out, &staging).unwrap();
        assert_eq!(written, 6);
        assert_eq!(&out[..6], &staging[..6]);
        assert_eq!(out[6], 9.0);
        assert_eq!(out[7], 9.0);
    }

    #[test]
    fn interleaved_complex_multiplier_doubles_element_count() {
        let dims = CompatDims::natural(&[2, 3], 1, true).unwrap();
        let required = dims.required_elements().unwrap();
        assert_eq!(required, 12);
    }

    #[test]
    fn cache_guard_uses_f64_ceil_conversion() {
        let required = required_f64s_for_bytes(17).unwrap();
        assert_eq!(required, 3);

        let err = ensure_cache_len(17, 2).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::BufferTooSmall {
                required: 3,
                provided: 2
            }
        ));
        assert_eq!(ensure_cache_len(17, 3).unwrap(), 3);
    }
}
