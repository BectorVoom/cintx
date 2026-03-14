use cintx::{CpuRouteKey, CpuRouteTarget, LibcintRsError, Representation, route};

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
