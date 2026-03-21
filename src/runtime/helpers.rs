use crate::contracts::Representation;
use crate::errors::LibcintRsError;

use super::backend::cpu::CpuRouteTarget;
use super::raw::{RawBasView, RawEnvView};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellAoLayout {
    pub counts: Vec<usize>,
    pub offsets: Vec<usize>,
    pub total_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShellNormalizationMetadata {
    pub shell_index: usize,
    pub angular_momentum: usize,
    pub kappa: i32,
    pub nprim: usize,
    pub nctr: usize,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub primitive_norms: Vec<f64>,
    pub normalized_coefficients: Vec<f64>,
}

pub fn shell_ao_layout(
    bas: &[i32],
    representation: Representation,
) -> Result<ShellAoLayout, LibcintRsError> {
    let bas_view = RawBasView::new(bas)?;
    let nbas = bas_view.nbas();
    let mut counts = Vec::with_capacity(nbas);
    for shell_index in 0..nbas {
        let shell = bas_view.shell_meta(shell_index)?;
        counts.push(contracted_shell_count(
            shell.angular_momentum,
            shell.nctr,
            representation,
            shell.kappa,
        )?);
    }

    let mut offsets = Vec::with_capacity(nbas);
    let mut running = 0usize;
    for count in &counts {
        offsets.push(running);
        running = running
            .checked_add(*count)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "bas.nctr_of",
                reason: "shell AO offset computation overflows usize".to_string(),
            })?;
    }

    Ok(ShellAoLayout {
        counts,
        offsets,
        total_count: running,
    })
}

pub fn shell_ao_counts(
    bas: &[i32],
    representation: Representation,
) -> Result<Vec<usize>, LibcintRsError> {
    Ok(shell_ao_layout(bas, representation)?.counts)
}

pub fn shell_offsets(
    bas: &[i32],
    representation: Representation,
) -> Result<Vec<usize>, LibcintRsError> {
    Ok(shell_ao_layout(bas, representation)?.offsets)
}

pub fn total_ao_count(
    bas: &[i32],
    representation: Representation,
) -> Result<usize, LibcintRsError> {
    Ok(shell_ao_layout(bas, representation)?.total_count)
}

pub fn contracted_shell_count(
    angular_momentum: usize,
    nctr: usize,
    representation: Representation,
    kappa: i32,
) -> Result<usize, LibcintRsError> {
    if nctr == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.nctr_of",
            reason: "value 0 must be greater than zero".to_string(),
        });
    }

    let components = component_count(angular_momentum, representation, kappa)?;
    components
        .checked_mul(nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "bas.nctr_of",
            reason: "contracted shell component count overflows usize".to_string(),
        })
}

pub fn cartesian_component_count(angular_momentum: usize) -> Result<usize, LibcintRsError> {
    let l_plus_1 = angular_momentum
        .checked_add(1)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;
    let l_plus_2 = angular_momentum
        .checked_add(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;

    l_plus_1
        .checked_mul(l_plus_2)
        .map(|value| value / 2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "cartesian component computation overflows usize".to_string(),
        })
}

pub fn spherical_component_count(angular_momentum: usize) -> Result<usize, LibcintRsError> {
    angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "spherical component computation overflows usize".to_string(),
        })
}

pub fn spinor_component_count(
    angular_momentum: usize,
    kappa: i32,
) -> Result<usize, LibcintRsError> {
    if kappa == 0 {
        return angular_momentum
            .checked_mul(4)
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "shell.angular_momentum",
                reason: "spinor component computation overflows usize".to_string(),
            });
    }

    if kappa < 0 {
        return angular_momentum
            .checked_mul(2)
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "shell.angular_momentum",
                reason: "spinor component computation overflows usize".to_string(),
            });
    }

    if angular_momentum == 0 {
        return Err(LibcintRsError::InvalidInput {
            field: "shell.kappa",
            reason: "positive kappa requires angular momentum > 0".to_string(),
        });
    }

    angular_momentum
        .checked_mul(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "spinor component computation overflows usize".to_string(),
        })
}

pub fn shell_normalization_metadata(
    shell_index: usize,
    bas: &[i32],
    env: &[f64],
) -> Result<ShellNormalizationMetadata, LibcintRsError> {
    let bas_view = RawBasView::new(bas)?;
    let env_view = RawEnvView::new(env);
    let shell = bas_view.shell_meta(shell_index)?;

    let exp_start = env_view.checked_offset_range("bas.ptr_exp", shell.ptr_exp, shell.nprim)?;
    let coeff_width =
        shell
            .nprim
            .checked_mul(shell.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "bas.ptr_coeff",
                reason: "nprim*nctr overflows usize".to_string(),
            })?;
    let coeff_start =
        env_view.checked_offset_range("bas.ptr_coeff", shell.ptr_coeff, coeff_width)?;

    let exponents = env[exp_start..exp_start + shell.nprim].to_vec();
    let coefficients = env[coeff_start..coeff_start + coeff_width].to_vec();

    let mut primitive_norms = Vec::with_capacity(exponents.len());
    for exponent in &exponents {
        primitive_norms.push(gto_norm(shell.angular_momentum, *exponent)?);
    }

    let mut normalized_coefficients = Vec::with_capacity(coefficients.len());
    for (index, coefficient) in coefficients.iter().enumerate() {
        let primitive_index = index % shell.nprim;
        normalized_coefficients.push(coefficient * primitive_norms[primitive_index]);
    }

    Ok(ShellNormalizationMetadata {
        shell_index,
        angular_momentum: shell.angular_momentum,
        kappa: shell.kappa,
        nprim: shell.nprim,
        nctr: shell.nctr,
        exponents,
        coefficients,
        primitive_norms,
        normalized_coefficients,
    })
}

pub fn gto_norm(angular_momentum: usize, exponent: f64) -> Result<f64, LibcintRsError> {
    if !exponent.is_finite() || exponent <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "exponent",
            reason: "must be a finite value greater than zero".to_string(),
        });
    }

    let l_plus_1 = angular_momentum
        .checked_add(1)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;
    let two_l_plus_2 = angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;
    let two_l_plus_3 = angular_momentum
        .checked_mul(2)
        .and_then(|value| value.checked_add(3))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum overflows usize".to_string(),
        })?;
    let two_l_plus_3_i32 =
        i32::try_from(two_l_plus_3).map_err(|_| LibcintRsError::InvalidInput {
            field: "shell.angular_momentum",
            reason: "angular momentum exceeds supported normalization range".to_string(),
        })?;

    let numerator = 2f64.powi(two_l_plus_3_i32)
        * factorial_f64(l_plus_1)?
        * (2.0 * exponent).powf((angular_momentum as f64) + 1.5);
    let denominator = factorial_f64(two_l_plus_2)? * std::f64::consts::PI.sqrt();

    let ratio = numerator / denominator;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "exponent",
            reason: "normalization computation produced non-finite value".to_string(),
        });
    }

    Ok(ratio.sqrt())
}

pub fn deterministic_transform_scalars(
    route_target: CpuRouteTarget,
    representation: Representation,
    dims: &[usize],
) -> Result<Vec<f64>, LibcintRsError> {
    let element_count = checked_product(dims)?;
    let scalars_per_element = match representation {
        Representation::Spinor => 2,
        Representation::Cartesian | Representation::Spherical => 1,
    };
    let scalar_count = element_count
        .checked_mul(scalars_per_element)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "dims",
            reason: "transform scalar count overflows usize".to_string(),
        })?;

    let seed = seed_from_route(route_target, dims);
    let mut output = vec![0.0f64; scalar_count];
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

    Ok(output)
}

fn component_count(
    angular_momentum: usize,
    representation: Representation,
    kappa: i32,
) -> Result<usize, LibcintRsError> {
    match representation {
        Representation::Cartesian => cartesian_component_count(angular_momentum),
        Representation::Spherical => spherical_component_count(angular_momentum),
        Representation::Spinor => spinor_component_count(angular_momentum, kappa),
    }
}

fn factorial_f64(value: usize) -> Result<f64, LibcintRsError> {
    let mut acc = 1.0f64;
    for item in 2..=value {
        acc *= item as f64;
        if !acc.is_finite() {
            return Err(LibcintRsError::InvalidInput {
                field: "shell.angular_momentum",
                reason: "factorial computation overflowed f64".to_string(),
            });
        }
    }
    Ok(acc)
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
