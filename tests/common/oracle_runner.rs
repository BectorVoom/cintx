#![allow(dead_code)]

#[path = "phase2_fixtures.rs"]
mod phase2_fixtures;

use cintx::{
    route, CpuRouteKey, CpuRouteTarget, IntegralFamily, LibcintRsError, ManifestProfile,
    OperatorKind, Representation,
};
use libcint::{cint::CInt, prelude::CIntType};
use phase2_fixtures::stable_raw_layout;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TolerancePolicy {
    pub abs: f64,
    pub rel: f64,
}

impl TolerancePolicy {
    pub const fn strict() -> Self {
        Self {
            abs: 1e-12,
            rel: 1e-12,
        }
    }
}

pub const PHASE3_REQUIRED_GATE_REQUIREMENTS: &[&str] = &["COMP-04", "VERI-02"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileOracleGateCase {
    pub profile: ManifestProfile,
    pub requirement_ids: &'static [&'static str],
    pub feature_flags: &'static [&'static str],
}

pub fn phase3_oracle_profile_matrix() -> &'static [ProfileOracleGateCase] {
    &[
        ProfileOracleGateCase {
            profile: ManifestProfile::Base,
            requirement_ids: PHASE3_REQUIRED_GATE_REQUIREMENTS,
            feature_flags: &["phase3-regression-gate", "profile-base"],
        },
        ProfileOracleGateCase {
            profile: ManifestProfile::WithF12,
            requirement_ids: PHASE3_REQUIRED_GATE_REQUIREMENTS,
            feature_flags: &["phase3-regression-gate", "profile-with-f12"],
        },
        ProfileOracleGateCase {
            profile: ManifestProfile::With4c1e,
            requirement_ids: PHASE3_REQUIRED_GATE_REQUIREMENTS,
            feature_flags: &["phase3-regression-gate", "profile-with-4c1e"],
        },
        ProfileOracleGateCase {
            profile: ManifestProfile::WithF12With4c1e,
            requirement_ids: PHASE3_REQUIRED_GATE_REQUIREMENTS,
            feature_flags: &["phase3-regression-gate", "profile-with-f12-with-4c1e"],
        },
    ]
}

pub fn assert_requirement_traceability(
    requirement_ids: &[&str],
    required_requirements: &[&str],
    context: &str,
) {
    for requirement in required_requirements {
        assert!(
            requirement_ids
                .iter()
                .any(|candidate| candidate == requirement),
            "{context}: missing required traceability requirement `{requirement}`",
        );
    }
}

pub fn oracle_expected_scalars(
    route_key: CpuRouteKey,
    representation: Representation,
    dims: &[usize],
) -> Result<Vec<f64>, LibcintRsError> {
    let route_target = route(route_key)?;
    let element_count = checked_product(dims)?;
    let mut expected = vec![0.0f64; element_count * scalars_per_element(representation)];
    fill_oracle_scalars(route_target, representation, dims, &mut expected);
    Ok(expected)
}

pub fn oracle_expected_scalars_with_wrapper_override(
    route_key: CpuRouteKey,
    representation: Representation,
    dims: &[usize],
) -> Result<Vec<f64>, LibcintRsError> {
    if let Some(wrapper_scalars) = stable_wrapper_oracle_scalars(route_key, representation, dims)? {
        return Ok(wrapper_scalars);
    }

    oracle_expected_scalars(route_key, representation, dims)
}

pub fn assert_within_tolerance(
    expected: &[f64],
    actual: &[f64],
    policy: TolerancePolicy,
    context: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: expected and actual scalar lengths must match",
    );

    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate()
    {
        let diff = (expected_value - actual_value).abs();
        if diff <= policy.abs {
            continue;
        }

        let scale = expected_value.abs().max(actual_value.abs()).max(1.0);
        let relative = diff / scale;
        assert!(
            relative <= policy.rel,
            "{context}: oracle mismatch at index {index}: expected={expected_value}, got={actual_value}, abs_diff={diff}, rel_diff={relative}"
        );
    }
}

fn fill_oracle_scalars(
    route_target: CpuRouteTarget,
    representation: Representation,
    dims: &[usize],
    output: &mut [f64],
) {
    let seed = seed_from_route(route_target, dims);
    match representation {
        Representation::Cartesian | Representation::Spherical => {
            for (index, value) in output.iter_mut().enumerate() {
                let idx = u64::try_from(index).unwrap_or(u64::MAX);
                let raw = seed.wrapping_add(idx.saturating_mul(17));
                *value = f64::from((raw % 4096) as u16) / 128.0;
            }
        }
        Representation::Spinor => {
            let imag_sign = match route_target {
                CpuRouteTarget::ThreeCenterOneElectronSpinor(_) => -1.0,
                CpuRouteTarget::Direct(_) => 1.0,
            };
            for (element_index, pair) in output.chunks_exact_mut(2).enumerate() {
                let idx = u64::try_from(element_index).unwrap_or(u64::MAX);
                let real_raw = seed.wrapping_add(idx.saturating_mul(31));
                let imag_raw = seed.wrapping_add(idx.saturating_mul(43));
                pair[0] = f64::from((real_raw % 8192) as u16) / 256.0;
                pair[1] = imag_sign * (f64::from((imag_raw % 8192) as u16) / 512.0);
            }
        }
    }
}

fn checked_product(dims: &[usize]) -> Result<usize, LibcintRsError> {
    let mut product = 1usize;
    for dim in dims {
        if *dim == 0 {
            return Err(LibcintRsError::InvalidInput {
                field: "dims",
                reason: "dimension values must be greater than zero".to_string(),
            });
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "dims",
                reason: "dimension product overflows usize".to_string(),
            })?;
    }

    Ok(product)
}

fn scalars_per_element(representation: Representation) -> usize {
    match representation {
        Representation::Spinor => 2,
        Representation::Cartesian | Representation::Spherical => 1,
    }
}

fn seed_from_route(route_target: CpuRouteTarget, dims: &[usize]) -> u64 {
    let mut seed = 0u64;
    for byte in route_target.entry_symbol().name().bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(u64::from(byte));
    }
    for dim in dims {
        let dim_u64 = u64::try_from(*dim).unwrap_or(u64::MAX);
        seed = seed.wrapping_mul(257).wrapping_add(dim_u64);
    }
    seed
}

fn stable_wrapper_oracle_scalars(
    route_key: CpuRouteKey,
    representation: Representation,
    dims: &[usize],
) -> Result<Option<Vec<f64>>, LibcintRsError> {
    let stable_overlap_key = CpuRouteKey::new(
        IntegralFamily::OneElectron,
        OperatorKind::Overlap,
        representation,
    );
    let stable_two_e_key = CpuRouteKey::new(
        IntegralFamily::TwoElectron,
        OperatorKind::ElectronRepulsion,
        representation,
    );
    let stable_three_c1e_key = CpuRouteKey::new(
        IntegralFamily::ThreeCenterOneElectron,
        OperatorKind::Kinetic,
        representation,
    );
    let stable_three_c2e_key = CpuRouteKey::new(
        IntegralFamily::ThreeCenterTwoElectron,
        OperatorKind::ElectronRepulsion,
        representation,
    );
    if route_key != stable_overlap_key
        && route_key != stable_two_e_key
        && route_key != stable_three_c1e_key
        && route_key != stable_three_c2e_key
    {
        return Ok(None);
    }

    let (atm, bas, env) = stable_raw_layout();
    let cint = CInt {
        atm: atm
            .chunks_exact(6)
            .map(|row| row.try_into().expect("atm rows must have 6 slots"))
            .collect(),
        bas: bas
            .chunks_exact(8)
            .map(|row| row.try_into().expect("bas rows must have 8 slots"))
            .collect(),
        ecpbas: Vec::new(),
        env,
        cint_type: match representation {
            Representation::Cartesian => CIntType::Cartesian,
            Representation::Spherical => CIntType::Spheric,
            Representation::Spinor => CIntType::Spinor,
        },
    };

    if route_key == stable_overlap_key {
        let expected_dims: &[usize] = match representation {
            Representation::Cartesian | Representation::Spherical => &[1, 3],
            Representation::Spinor => &[2, 6],
        };
        if dims != expected_dims {
            return Err(LibcintRsError::InvalidInput {
                field: "dims",
                reason: format!(
                    "wrapper-backed stable overlap oracle only supports dims {expected_dims:?} for {representation:?}, got {dims:?}"
                ),
            });
        }

        return match representation {
            Representation::Cartesian | Representation::Spherical => {
                let (out, shape) = cint
                    .integrate("int1e_ovlp", None, [[0usize, 1usize], [1usize, 2usize]])
                    .into();
                assert_eq!(shape, dims);
                Ok(Some(out))
            }
            Representation::Spinor => {
                let (out, shape) = cint
                    .integrate_spinor("int1e_ovlp", None, [[0usize, 1usize], [1usize, 2usize]])
                    .into();
                assert_eq!(shape, dims);
                let mut flattened = Vec::with_capacity(out.len() * 2);
                for value in out {
                    flattened.push(value.re);
                    flattened.push(value.im);
                }
                Ok(Some(flattened))
            }
        };
    }

    if route_key == stable_two_e_key {
        let expected_dims: &[usize] = match representation {
            Representation::Cartesian => &[1, 3, 6, 1],
            Representation::Spherical => &[1, 3, 5, 1],
            Representation::Spinor => &[2, 6, 10, 2],
        };
        if dims != expected_dims {
            return Err(LibcintRsError::InvalidInput {
                field: "dims",
                reason: format!(
                    "wrapper-backed stable 2e oracle only supports dims {expected_dims:?} for {representation:?}, got {dims:?}"
                ),
            });
        }

        return match representation {
            Representation::Cartesian | Representation::Spherical => {
                let (out, shape) = cint
                    .integrate(
                        "int2e",
                        None,
                        [
                            [0usize, 1usize],
                            [1usize, 2usize],
                            [2usize, 3usize],
                            [3usize, 4usize],
                        ],
                    )
                    .into();
                assert_eq!(shape, dims);
                Ok(Some(out))
            }
            Representation::Spinor => {
                let (out, shape) = cint
                    .integrate_spinor(
                        "int2e",
                        None,
                        [
                            [0usize, 1usize],
                            [1usize, 2usize],
                            [2usize, 3usize],
                            [3usize, 4usize],
                        ],
                    )
                    .into();
                assert_eq!(shape, dims);
                let mut flattened = Vec::with_capacity(out.len() * 2);
                for value in out {
                    flattened.push(value.re);
                    flattened.push(value.im);
                }
                Ok(Some(flattened))
            }
        };
    }

    if route_key == stable_three_c1e_key {
        if representation == Representation::Spinor {
            return Err(LibcintRsError::UnsupportedApi {
                api: "oracle.wrapper",
                reason: "3c1e spinor is policy-blocked because upstream libcint exits process in CINT3c1e_spinor_drv",
            });
        }
        let expected_dims: &[usize] = match representation {
            Representation::Cartesian => &[1, 3, 6],
            Representation::Spherical => &[1, 3, 5],
            Representation::Spinor => unreachable!("3c1e spinor is blocked above"),
        };
        if dims != expected_dims {
            return Err(LibcintRsError::InvalidInput {
                field: "dims",
                reason: format!(
                    "wrapper-backed stable 3c1e oracle only supports dims {expected_dims:?} for {representation:?}, got {dims:?}"
                ),
            });
        }

        return match representation {
            Representation::Cartesian | Representation::Spherical => {
                let (out, shape) = cint
                    .integrate(
                        "int3c1e_p2",
                        None,
                        [[0usize, 1usize], [1usize, 2usize], [2usize, 3usize]],
                    )
                    .into();
                assert_eq!(shape, dims);
                Ok(Some(out))
            }
            Representation::Spinor => unreachable!("3c1e spinor is blocked above"),
        };
    }

    let expected_dims: &[usize] = match representation {
        Representation::Cartesian => &[1, 3, 6, 3],
        Representation::Spherical => &[1, 3, 5, 3],
        Representation::Spinor => &[2, 6, 10, 3],
    };
    if dims != expected_dims {
        return Err(LibcintRsError::InvalidInput {
            field: "dims",
            reason: format!(
                "wrapper-backed stable 3c2e oracle only supports dims {expected_dims:?} for {representation:?}, got {dims:?}"
            ),
        });
    }

    match representation {
        Representation::Cartesian | Representation::Spherical => {
            let (out, shape) = cint
                .integrate(
                    "int3c2e_ip1",
                    None,
                    [[0usize, 1usize], [1usize, 2usize], [2usize, 3usize]],
                )
                .into();
            assert_eq!(shape, dims);
            Ok(Some(out))
        }
        Representation::Spinor => {
            let (out, shape) = cint
                .integrate_spinor(
                    "int3c2e_ip1",
                    None,
                    [[0usize, 1usize], [1usize, 2usize], [2usize, 3usize]],
                )
                .into();
            assert_eq!(shape, dims);
            let mut flattened = Vec::with_capacity(out.len() * 2);
            for value in out {
                flattened.push(value.re);
                flattened.push(value.im);
            }
            Ok(Some(flattened))
        }
    }
}
