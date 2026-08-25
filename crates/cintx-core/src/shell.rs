use smallvec::SmallVec;
use std::sync::Arc;

use crate::error::{CoreError, CoreResult};
use crate::operator::Representation;

pub(crate) const SHELL_TUPLE_CAPACITY: usize = 4;

/// Highest angular momentum a **spherical** shell may carry.
///
/// This is the ceiling of the Cartesian-to-spherical coefficient table, which
/// is libcint's own: its `g_c2s` array (`cart2sph.c`) has entries for
/// `l = 0..=15` and nothing above, so beyond it there is no upstream reference
/// to be compatible with. `cintx-cubecl`'s generated table carries exactly that
/// range, and `transform::c2s::tests::spheric_l_max_matches_the_table_ceiling`
/// pins the two constants together.
///
/// Cartesian shells are unaffected — they need no transform — so this is
/// validated per shell against its own `representation`, not globally.
///
/// The check exists because the alternative was silent: the transform used to
/// return `0.0` above `l = 4`, so an out-of-range spherical shell came back
/// entirely zeroed with an `Ok` status.
pub const SPHERIC_L_MAX: u8 = 15;

/// Error when a shell tuple would exceed libcint's arity limits.
#[derive(Debug, thiserror::Error)]
#[error("shell tuple arity cannot exceed {0}")]
pub struct ShellTupleArityError(usize);

/// A primitive shell containing angular momentum, contraction, and coefficient data.
#[derive(Clone, Debug, PartialEq)]
pub struct Shell {
    pub atom_index: u32,
    pub ang_momentum: u8,
    pub nprim: u16,
    pub nctr: u16,
    pub kappa: i16,
    pub representation: Representation,
    pub exponents: Arc<[f64]>,
    pub coefficients: Arc<[f64]>,
}

impl Shell {
    /// Try to build a shell while validating primitives and contraction data.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        atom_index: u32,
        ang_momentum: u8,
        nprim: u16,
        nctr: u16,
        kappa: i16,
        representation: Representation,
        exponents: Arc<[f64]>,
        coefficients: Arc<[f64]>,
    ) -> CoreResult<Self> {
        let nprim_usize = nprim as usize;
        let nctr_usize = nctr as usize;

        if nprim_usize == 0 || nctr_usize == 0 {
            return Err(CoreError::InvalidShellCounts {
                nprim: nprim_usize,
                nctr: nctr_usize,
            });
        }

        if representation == Representation::Spheric && ang_momentum > SPHERIC_L_MAX {
            return Err(CoreError::SphericAngularMomentumTooHigh {
                requested: ang_momentum,
                max: SPHERIC_L_MAX,
            });
        }

        if exponents.len() != nprim_usize {
            return Err(CoreError::ShellPrimitiveMismatch {
                field: "exponents",
                expected: nprim_usize,
                actual: exponents.len(),
            });
        }

        let expected_coeffs =
            nprim_usize
                .checked_mul(nctr_usize)
                .ok_or(CoreError::InvalidShellCounts {
                    nprim: nprim_usize,
                    nctr: nctr_usize,
                })?;
        if coefficients.len() != expected_coeffs {
            return Err(CoreError::ShellPrimitiveMismatch {
                field: "coefficients",
                expected: expected_coeffs,
                actual: coefficients.len(),
            });
        }

        if !exponents.iter().all(|value| value.is_finite())
            || !coefficients.iter().all(|value| value.is_finite())
        {
            return Err(CoreError::InvalidNuclearDetail);
        }

        Ok(Self {
            atom_index,
            ang_momentum,
            nprim,
            nctr,
            kappa,
            representation,
            exponents,
            coefficients,
        })
    }

    pub fn ao_per_shell(&self) -> usize {
        let base = match self.representation {
            Representation::Cart => {
                let l = self.ang_momentum as usize;
                (l + 1) * (l + 2) / 2
            }
            Representation::Spheric => {
                let l = self.ang_momentum as usize;
                2 * l + 1
            }
            Representation::Spinor => spinor_len(self.ang_momentum as usize, self.kappa),
        };
        base * self.nctr as usize
    }
}

fn spinor_len(l: usize, kappa: i16) -> usize {
    match kappa {
        0 => 4 * l + 2,
        neg if neg < 0 => 2 * l + 2,
        _ => 2 * l,
    }
}

/// Arity-safe collection of shells matching libcint's tuple inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellTuple {
    shells: SmallVec<[Arc<Shell>; SHELL_TUPLE_CAPACITY]>,
}

impl ShellTuple {
    pub fn try_from_iter<I>(iter: I) -> Result<Self, ShellTupleArityError>
    where
        I: IntoIterator<Item = Arc<Shell>>,
    {
        let mut shells = SmallVec::new();
        for shell in iter {
            if shells.len() >= SHELL_TUPLE_CAPACITY {
                return Err(ShellTupleArityError(SHELL_TUPLE_CAPACITY));
            }
            shells.push(shell);
        }
        Ok(Self { shells })
    }

    pub fn len(&self) -> usize {
        self.shells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shells.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Arc<Shell>> {
        self.shells.iter()
    }

    pub fn as_slice(&self) -> &[Arc<Shell>] {
        &self.shells
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CoreError;
    use crate::operator::Representation;
    use std::sync::Arc;

    fn arc_from_slice(values: &[f64]) -> Arc<[f64]> {
        Arc::from(values.to_owned().into_boxed_slice())
    }

    #[test]
    fn mismatched_exponent_count_is_rejected() {
        let err = Shell::try_new(
            0,
            0,
            2,
            1,
            0,
            Representation::Cart,
            arc_from_slice(&[1.0]),
            arc_from_slice(&[1.0, 2.0]),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CoreError::ShellPrimitiveMismatch {
                field: "exponents",
                ..
            }
        ));
    }

    #[test]
    fn mismatched_coefficient_count_is_rejected() {
        let err = Shell::try_new(
            0,
            0,
            1,
            2,
            0,
            Representation::Cart,
            arc_from_slice(&[1.0]),
            arc_from_slice(&[1.0]),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            CoreError::ShellPrimitiveMismatch {
                field: "coefficients",
                ..
            }
        ));
    }
}
