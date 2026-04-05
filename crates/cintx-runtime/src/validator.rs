use cintx_core::{BasisSet, Representation, Shell, ShellTuple, cintxRsError};
use cintx_ops::resolver::OperatorDescriptor;
use std::sync::Arc;

use crate::planner::OperatorEnvParams;

#[derive(Clone, Debug)]
pub struct ValidatedShellTuple {
    shells: Vec<Arc<Shell>>,
    total_ao: usize,
    output_elements: usize,
    representation: Representation,
}

impl ValidatedShellTuple {
    pub fn len(&self) -> usize {
        self.shells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shells.is_empty()
    }

    pub fn total_ao(&self) -> usize {
        self.total_ao
    }

    pub fn output_elements(&self) -> usize {
        self.output_elements
    }

    pub fn work_units(&self) -> usize {
        self.output_elements.max(1)
    }

    pub fn representation(&self) -> Representation {
        self.representation
    }

    pub fn as_slice(&self) -> &[Arc<Shell>] {
        &self.shells
    }
}

pub fn validate_dims(expected: usize, provided: usize) -> Result<(), cintxRsError> {
    if expected == provided {
        return Ok(());
    }

    Err(cintxRsError::InvalidDims { expected, provided })
}

pub fn validate_shell_tuple(
    descriptor: &OperatorDescriptor,
    representation: Representation,
    basis: &BasisSet,
    shells: &ShellTuple,
) -> Result<ValidatedShellTuple, cintxRsError> {
    let expected = descriptor.entry.arity as usize;
    let got = shells.len();
    if expected != got {
        return Err(cintxRsError::InvalidShellTuple { expected, got });
    }

    if !descriptor.entry.supports_representation(representation) {
        return Err(cintxRsError::UnsupportedRepresentation {
            operator: format!("{}/{}", descriptor.family(), descriptor.operator_name()),
            representation,
        });
    }

    let atom_count = basis.atoms().len();
    let mut total_ao = 0usize;
    let mut output_elements = 1usize;
    let mut validated = Vec::with_capacity(got);

    for shell in shells.as_slice() {
        let shell = shell.clone();
        if shell.atom_index as usize >= atom_count {
            return Err(cintxRsError::InvalidShellAtomIndex {
                index: shell.atom_index as usize,
                atom_count,
            });
        }

        if shell.representation != representation {
            return Err(cintxRsError::UnsupportedRepresentation {
                operator: descriptor.operator_name().to_owned(),
                representation,
            });
        }

        let ao_per_shell = shell.ao_per_shell();
        total_ao =
            total_ao
                .checked_add(ao_per_shell)
                .ok_or_else(|| cintxRsError::ChunkPlanFailed {
                    from: "validator",
                    detail: "shell AO count overflowed usize".to_owned(),
                })?;
        output_elements = output_elements.checked_mul(ao_per_shell).ok_or_else(|| {
            cintxRsError::ChunkPlanFailed {
                from: "validator",
                detail: "output element count overflowed usize".to_owned(),
            }
        })?;
        validated.push(shell);
    }

    Ok(ValidatedShellTuple {
        shells: validated,
        total_ao,
        output_elements,
        representation,
    })
}

/// Validates that F12/STG/YP operator env params are correct.
///
/// Returns `InvalidEnvParam` if `f12_zeta` is `None` or `0.0` for an f12-family plan.
/// This is called before kernel launch to reject invalid configurations early (D-01, D-02, F12-05).
pub fn validate_f12_env_params(
    canonical_family: &str,
    params: &OperatorEnvParams,
) -> Result<(), cintxRsError> {
    if canonical_family == "f12" {
        match params.f12_zeta {
            None => {
                return Err(cintxRsError::InvalidEnvParam {
                    param: "PTR_F12_ZETA",
                    reason: "env[9] (PTR_F12_ZETA) must be non-zero for F12/STG/YP integrals"
                        .to_owned(),
                });
            }
            Some(z) if z == 0.0_f64 => {
                return Err(cintxRsError::InvalidEnvParam {
                    param: "PTR_F12_ZETA",
                    reason: "env[9] (PTR_F12_ZETA) must be non-zero for F12/STG/YP integrals"
                        .to_owned(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell};
    use cintx_ops::resolver::Resolver;
    use std::sync::Arc;

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
    fn invalid_dims_are_typed() {
        let err = validate_dims(4, 3).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidDims {
                expected: 4,
                provided: 3
            }
        ));
    }

    #[test]
    fn shell_tuple_arity_mismatch_is_typed() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let descriptor = Resolver::descriptor(OperatorId::new(9)).unwrap();

        let err =
            validate_shell_tuple(descriptor, Representation::Cart, &basis, &shells).unwrap_err();
        assert!(matches!(
            err,
            cintxRsError::InvalidShellTuple {
                expected: 4,
                got: 2
            }
        ));
    }

    #[test]
    fn valid_tuple_preserves_ao_counts() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let descriptor = Resolver::descriptor(OperatorId::new(0)).unwrap();

        let validated = validate_shell_tuple(descriptor, Representation::Cart, &basis, &shells)
            .expect("tuple should validate");

        assert_eq!(validated.len(), 2);
        assert_eq!(validated.total_ao(), 12);
        assert_eq!(validated.output_elements(), 36);
        assert_eq!(validated.representation(), Representation::Cart);
    }

    #[test]
    fn shell_atom_index_mismatch_is_typed() {
        let (basis, shells) = sample_basis(Representation::Cart);
        let descriptor = Resolver::descriptor(OperatorId::new(0)).unwrap();
        let invalid_shell = Arc::new(
            Shell::try_new(
                1,
                1,
                1,
                2,
                0,
                Representation::Cart,
                arc_f64(&[1.0]),
                arc_f64(&[1.0, 0.5]),
            )
            .unwrap(),
        );
        let invalid_tuple =
            ShellTuple::try_from_iter([invalid_shell, shells.as_slice()[1].clone()]).unwrap();

        let err = validate_shell_tuple(descriptor, Representation::Cart, &basis, &invalid_tuple)
            .unwrap_err();

        assert!(matches!(
            err,
            cintxRsError::InvalidShellAtomIndex {
                index: 1,
                atom_count: 1,
            }
        ));
    }

    #[test]
    fn f12_env_params_zeta_zero_is_rejected() {
        let params = OperatorEnvParams { f12_zeta: Some(0.0_f64) };
        let err = validate_f12_env_params("f12", &params).unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param: "PTR_F12_ZETA", .. }),
            "expected InvalidEnvParam(PTR_F12_ZETA), got {err:?}"
        );
    }

    #[test]
    fn f12_env_params_zeta_none_is_rejected() {
        let params = OperatorEnvParams { f12_zeta: None };
        let err = validate_f12_env_params("f12", &params).unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param: "PTR_F12_ZETA", .. }),
            "expected InvalidEnvParam(PTR_F12_ZETA), got {err:?}"
        );
    }

    #[test]
    fn f12_env_params_valid_zeta_passes() {
        let params = OperatorEnvParams { f12_zeta: Some(1.2_f64) };
        validate_f12_env_params("f12", &params).expect("valid zeta should pass");
    }

    #[test]
    fn f12_env_params_non_f12_family_skips_check() {
        // Non-f12 families should not be gated even with no f12_zeta.
        let params = OperatorEnvParams::default();
        validate_f12_env_params("2e", &params).expect("non-f12 family should not be checked");
        validate_f12_env_params("1e", &params).expect("non-f12 family should not be checked");
    }
}
