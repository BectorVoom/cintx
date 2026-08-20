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

/// Validates that grids operator env params are correct.
///
/// Returns `InvalidEnvParam` if `grids_params` is `None` or `ngrids == 0` for a grids-family plan.
/// Called before kernel launch to reject invalid configurations early (D-05).
pub fn validate_grids_env_params(
    canonical_family: &str,
    params: &OperatorEnvParams,
) -> Result<(), cintxRsError> {
    if canonical_family != "grids" {
        return Ok(());
    }
    match &params.grids_params {
        None => Err(cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            reason: "grids family requires GridsEnvParams with NGRIDS > 0 and valid PTR_GRIDS"
                .to_owned(),
        }),
        Some(gp) if gp.ngrids == 0 => Err(cintxRsError::InvalidEnvParam {
            param: "NGRIDS",
            reason: "NGRIDS must be > 0 for grids integrals".to_owned(),
        }),
        _ => Ok(()),
    }
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

/// Validates that iprinv-family operator env params include a rinv origin.
///
/// Returns `InvalidEnvParam` if `rinv_orig` is `None` for an operator whose name
/// contains `"iprinv"`. Called before kernel launch so we surface a typed error
/// before kernel entry — no garbage-origin evaluation, no UB (T-21-01-01/02).
///
/// The predicate uses `.contains("iprinv")` (not `==`) so it covers both
/// `"iprinv"` (int1e_iprinv) and `"ecp_iprinv"` (ECPscalar_iprinv) variants.
/// Non-iprinv operators (overlap, kinetic, nuclear-attraction, etc.) are never gated.
pub fn validate_rinv_orig_env_params(
    operator_name: &str,
    params: &OperatorEnvParams,
) -> Result<(), cintxRsError> {
    if operator_name.contains("iprinv") {
        match params.rinv_orig {
            None => {
                return Err(cintxRsError::InvalidEnvParam {
                    param: "PTR_RINV_ORIG",
                    reason: "env[4..6] (PTR_RINV_ORIG) must be set for iprinv operators".to_owned(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validates the common (gauge) origin env params.
///
/// D-01 (diverges from `validate_rinv_orig_env_params`): `None` is VALID — an unset
/// gauge origin defaults to `[0,0,0]` (libcint reads unset env as zero), so this is a
/// FINITENESS check, not a presence check. Only a `Some([..])` containing a non-finite
/// component (NaN/inf) is rejected.
///
/// D-02: operator-AGNOSTIC — no operator-name predicate. No dispatchable consumer exists
/// in this phase (moments/GIAO add their own in Phases 24/26), so a name-list would be dead.
pub fn validate_common_orig_env_params(
    _operator_name: &str,
    params: &OperatorEnvParams,
) -> Result<(), cintxRsError> {
    if let Some(origin) = params.common_orig {
        if origin.iter().any(|v| !v.is_finite()) {
            return Err(cintxRsError::InvalidEnvParam {
                param: "PTR_COMMON_ORIG",
                reason: "env[1..3] (PTR_COMMON_ORIG) gauge origin must be finite".to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::GridsEnvParams;
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
        let params = OperatorEnvParams {
            f12_zeta: Some(0.0_f64),
            ..OperatorEnvParams::default()
        };
        let err = validate_f12_env_params("f12", &params).unwrap_err();
        assert!(
            matches!(
                err,
                cintxRsError::InvalidEnvParam {
                    param: "PTR_F12_ZETA",
                    ..
                }
            ),
            "expected InvalidEnvParam(PTR_F12_ZETA), got {err:?}"
        );
    }

    #[test]
    fn f12_env_params_zeta_none_is_rejected() {
        let params = OperatorEnvParams {
            f12_zeta: None,
            ..OperatorEnvParams::default()
        };
        let err = validate_f12_env_params("f12", &params).unwrap_err();
        assert!(
            matches!(
                err,
                cintxRsError::InvalidEnvParam {
                    param: "PTR_F12_ZETA",
                    ..
                }
            ),
            "expected InvalidEnvParam(PTR_F12_ZETA), got {err:?}"
        );
    }

    #[test]
    fn f12_env_params_valid_zeta_passes() {
        let params = OperatorEnvParams {
            f12_zeta: Some(1.2_f64),
            ..OperatorEnvParams::default()
        };
        validate_f12_env_params("f12", &params).expect("valid zeta should pass");
    }

    #[test]
    fn f12_env_params_non_f12_family_skips_check() {
        // Non-f12 families should not be gated even with no f12_zeta.
        let params = OperatorEnvParams::default();
        validate_f12_env_params("2e", &params).expect("non-f12 family should not be checked");
        validate_f12_env_params("1e", &params).expect("non-f12 family should not be checked");
    }

    #[test]
    fn validate_grids_env_params_none_is_rejected() {
        let params = OperatorEnvParams::default(); // grids_params: None
        let err = validate_grids_env_params("grids", &params).unwrap_err();
        assert!(
            matches!(
                err,
                cintxRsError::InvalidEnvParam {
                    param: "NGRIDS",
                    ..
                }
            ),
            "expected InvalidEnvParam(NGRIDS), got {err:?}"
        );
    }

    #[test]
    fn validate_grids_env_params_ngrids_zero_is_rejected() {
        let params = OperatorEnvParams {
            grids_params: Some(GridsEnvParams {
                ngrids: 0,
                ptr_grids: 20,
                grid_coords: vec![],
            }),
            ..OperatorEnvParams::default()
        };
        let err = validate_grids_env_params("grids", &params).unwrap_err();
        assert!(
            matches!(
                err,
                cintxRsError::InvalidEnvParam {
                    param: "NGRIDS",
                    ..
                }
            ),
            "expected InvalidEnvParam(NGRIDS), got {err:?}"
        );
    }

    #[test]
    fn validate_grids_env_params_valid_passes() {
        let params = OperatorEnvParams {
            grids_params: Some(GridsEnvParams {
                ngrids: 5,
                ptr_grids: 20,
                grid_coords: vec![[0.0, 0.0, 0.0]; 5],
            }),
            ..OperatorEnvParams::default()
        };
        validate_grids_env_params("grids", &params).expect("valid grids params should pass");
    }

    #[test]
    fn validate_grids_env_params_non_grids_family_skips_check() {
        // Non-grids families should not be gated even with no grids_params.
        let params = OperatorEnvParams::default();
        validate_grids_env_params("1e", &params).expect("non-grids family should not be checked");
        validate_grids_env_params("origi", &params)
            .expect("non-grids family should not be checked");
    }

    #[test]
    fn rinv_orig_default_is_none() {
        let params = OperatorEnvParams::default();
        assert!(params.rinv_orig.is_none(), "rinv_orig must default to None");
    }

    #[test]
    fn validate_rinv_orig_rejects_none_for_iprinv() {
        let params = OperatorEnvParams::default(); // rinv_orig: None
        let err = validate_rinv_orig_env_params("iprinv", &params).unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "PTR_RINV_ORIG"),
            "expected InvalidEnvParam(PTR_RINV_ORIG) for iprinv with None origin, got {err:?}"
        );
    }

    #[test]
    fn validate_rinv_orig_rejects_none_for_ecp_iprinv() {
        let params = OperatorEnvParams::default(); // rinv_orig: None
        let err = validate_rinv_orig_env_params("ecp_iprinv", &params).unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "PTR_RINV_ORIG"),
            "expected InvalidEnvParam(PTR_RINV_ORIG) for ecp_iprinv with None origin, got {err:?}"
        );
    }

    #[test]
    fn validate_rinv_orig_accepts_non_iprinv() {
        let params = OperatorEnvParams::default();
        validate_rinv_orig_env_params("overlap", &params)
            .expect("non-iprinv operator must not be gated by rinv-origin check");
    }

    #[test]
    fn validate_rinv_orig_accepts_some() {
        let params = OperatorEnvParams {
            rinv_orig: Some([0.0, 0.0, 1.4]),
            ..OperatorEnvParams::default()
        };
        validate_rinv_orig_env_params("iprinv", &params)
            .expect("iprinv with rinv_orig=Some(...) must pass");
    }

    #[test]
    fn common_orig_default_is_none() {
        assert!(OperatorEnvParams::default().common_orig.is_none());
    }

    #[test]
    fn validate_common_orig_accepts_none() {
        let params = OperatorEnvParams::default();
        validate_common_orig_env_params("int1e_ovlp", &params)
            .expect("common_orig=None must pass (defaults to [0,0,0])");
    }

    #[test]
    fn validate_common_orig_accepts_some_finite() {
        let params = OperatorEnvParams {
            common_orig: Some([0.5, -1.2, 0.0]),
            ..OperatorEnvParams::default()
        };
        validate_common_orig_env_params("", &params).expect("finite gauge origin must pass");
    }

    #[test]
    fn validate_common_orig_rejects_non_finite() {
        for bad in [
            [f64::NAN, 0.0, 0.0],
            [0.0, f64::INFINITY, 0.0],
            [0.0, 0.0, f64::NEG_INFINITY],
        ] {
            let params = OperatorEnvParams {
                common_orig: Some(bad),
                ..OperatorEnvParams::default()
            };
            let err = validate_common_orig_env_params("int1e_r", &params)
                .expect_err("non-finite gauge origin must be rejected");
            match err {
                cintxRsError::InvalidEnvParam { param, .. } => assert_eq!(param, "PTR_COMMON_ORIG"),
                other => panic!("expected InvalidEnvParam, got {other:?}"),
            }
        }
    }
}
