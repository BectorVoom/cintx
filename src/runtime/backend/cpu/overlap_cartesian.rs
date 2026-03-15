use crate::contracts::BasisSet;
use crate::errors::LibcintRsError;
use crate::runtime::raw::{RawAtmView, RawBasView, RawEnvView};

#[derive(Debug, Clone)]
struct CartesianShellData {
    center: [f64; 3],
    angular_momentum: usize,
    kappa: i32,
    nctr: usize,
    exponents: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct NuclearCenter {
    center: [f64; 3],
    charge: f64,
}

pub(crate) fn fill_safe_one_e_overlap_cartesian(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    fill_overlap_shell_pair(&shell_i, &shell_j, output)
}

pub(crate) fn fill_safe_one_e_kinetic_cartesian(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    fill_kinetic_shell_pair(&shell_i, &shell_j, output)
}

pub(crate) fn fill_safe_one_e_nuclear_cartesian(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    let nuclei = safe_nuclear_centers(basis)?;
    fill_nuclear_shell_pair(&shell_i, &shell_j, nuclei.as_slice(), output)
}

pub(crate) fn fill_safe_one_e_kinetic_spherical(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    fill_kinetic_shell_pair_spherical(&shell_i, &shell_j, output)
}

pub(crate) fn fill_safe_one_e_nuclear_spherical(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    let nuclei = safe_nuclear_centers(basis)?;
    fill_nuclear_shell_pair_spherical(&shell_i, &shell_j, nuclei.as_slice(), output)
}

pub(crate) fn fill_safe_one_e_overlap_spherical(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    fill_overlap_shell_pair_spherical(&shell_i, &shell_j, output)
}

pub(crate) fn fill_safe_one_e_overlap_spinor(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    fill_overlap_shell_pair_spinor(&shell_i, &shell_j, output)
}

pub(crate) fn fill_safe_one_e_kinetic_spinor(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    fill_kinetic_shell_pair_spinor(&shell_i, &shell_j, output)
}

pub(crate) fn fill_safe_one_e_nuclear_spinor(
    basis: &BasisSet,
    shell_tuple: &[usize],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shell_tuple_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = safe_shell_data(basis, shell_tuple[0])?;
    let shell_j = safe_shell_data(basis, shell_tuple[1])?;
    let nuclei = safe_nuclear_centers(basis)?;
    fill_nuclear_shell_pair_spinor(&shell_i, &shell_j, nuclei.as_slice(), output)
}

pub(crate) fn fill_raw_one_e_overlap_cartesian(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    fill_overlap_shell_pair(&shell_i, &shell_j, output)
}

pub(crate) fn fill_raw_one_e_kinetic_cartesian(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    fill_kinetic_shell_pair(&shell_i, &shell_j, output)
}

pub(crate) fn fill_raw_one_e_nuclear_cartesian(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    let nuclei = raw_nuclear_centers(atm, env)?;
    fill_nuclear_shell_pair(&shell_i, &shell_j, nuclei.as_slice(), output)
}

pub(crate) fn fill_raw_one_e_overlap_spherical(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    fill_overlap_shell_pair_spherical(&shell_i, &shell_j, output)
}

pub(crate) fn fill_raw_one_e_kinetic_spherical(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    fill_kinetic_shell_pair_spherical(&shell_i, &shell_j, output)
}

pub(crate) fn fill_raw_one_e_nuclear_spherical(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    let nuclei = raw_nuclear_centers(atm, env)?;
    fill_nuclear_shell_pair_spherical(&shell_i, &shell_j, nuclei.as_slice(), output)
}

pub(crate) fn fill_raw_one_e_overlap_spinor(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    fill_overlap_shell_pair_spinor_flattened(&shell_i, &shell_j, output)
}

pub(crate) fn fill_raw_one_e_kinetic_spinor(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    fill_kinetic_shell_pair_spinor_flattened(&shell_i, &shell_j, output)
}

pub(crate) fn fill_raw_one_e_nuclear_spinor(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if shell_tuple.len() != 2 {
        return Err(LibcintRsError::InvalidLayout {
            item: "shls_arity",
            expected: 2,
            got: shell_tuple.len(),
        });
    }

    let shell_i = raw_shell_data(atm, bas, env, shell_tuple[0])?;
    let shell_j = raw_shell_data(atm, bas, env, shell_tuple[1])?;
    let nuclei = raw_nuclear_centers(atm, env)?;
    fill_nuclear_shell_pair_spinor_flattened(&shell_i, &shell_j, nuclei.as_slice(), output)
}

fn safe_shell_data(
    basis: &BasisSet,
    shell_index: usize,
) -> Result<CartesianShellData, LibcintRsError> {
    let shell = basis
        .shells()
        .get(shell_index)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell_tuple",
            reason: format!(
                "index {shell_index} is out of bounds for {} shells",
                basis.shells().len()
            ),
        })?;
    let center = basis
        .atoms()
        .get(shell.center_index())
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "shell.center_index",
            reason: format!(
                "index {} is out of bounds for {} atoms",
                shell.center_index(),
                basis.atoms().len()
            ),
        })?
        .coordinates();

    Ok(CartesianShellData {
        center,
        angular_momentum: usize::from(shell.angular_momentum()),
        kappa: 0,
        nctr: 1,
        exponents: shell
            .primitives()
            .iter()
            .map(|primitive| primitive.exponent())
            .collect(),
        coefficients: shell
            .primitives()
            .iter()
            .map(|primitive| primitive.coefficient())
            .collect(),
    })
}

fn raw_shell_data(
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_index: usize,
) -> Result<CartesianShellData, LibcintRsError> {
    let atm_view = RawAtmView::new(atm)?;
    let bas_view = RawBasView::new(bas)?;
    let env_view = RawEnvView::new(env);
    let meta = bas_view.shell_meta(shell_index)?;

    let coeff_len =
        meta.nprim
            .checked_mul(meta.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "bas.ptr_coeff",
                reason: "nprim*nctr overflows usize".to_string(),
            })?;

    Ok(CartesianShellData {
        center: atm_view.atom_coordinates(meta.atom_of, &env_view)?,
        angular_momentum: meta.angular_momentum,
        kappa: meta.kappa,
        nctr: meta.nctr,
        exponents: env_view
            .checked_slice("bas.ptr_exp", meta.ptr_exp, meta.nprim)?
            .to_vec(),
        coefficients: env_view
            .checked_slice("bas.ptr_coeff", meta.ptr_coeff, coeff_len)?
            .to_vec(),
    })
}

fn safe_nuclear_centers(basis: &BasisSet) -> Result<Vec<NuclearCenter>, LibcintRsError> {
    let mut nuclei = Vec::with_capacity(basis.atoms().len());
    for atom in basis.atoms() {
        let charge = f64::from(atom.atomic_number());
        if charge <= 0.0 || !charge.is_finite() {
            return Err(LibcintRsError::InvalidInput {
                field: "atomic_number",
                reason: "nuclear charge must be finite and greater than zero".to_string(),
            });
        }
        nuclei.push(NuclearCenter {
            center: atom.coordinates(),
            charge,
        });
    }
    Ok(nuclei)
}

fn raw_nuclear_centers(atm: &[i32], env: &[f64]) -> Result<Vec<NuclearCenter>, LibcintRsError> {
    const ATM_SLOTS: usize = 6;
    const ATM_CHARGE_SLOT: usize = 0;
    const ATM_PTR_COORD_SLOT: usize = 1;
    const ATM_NUC_MOD_SLOT: usize = 2;
    const ATM_PTR_ZETA_SLOT: usize = 3;
    const GAUSSIAN_NUC_MODEL: i32 = 2;

    let atm_view = RawAtmView::new(atm)?;
    let env_view = RawEnvView::new(env);
    let natm = atm_view.natm();
    let mut nuclei = Vec::with_capacity(natm);
    for atom_index in 0..natm {
        let row_start = atom_index * ATM_SLOTS;
        let row = &atm[row_start..row_start + ATM_SLOTS];
        if row[ATM_NUC_MOD_SLOT] == GAUSSIAN_NUC_MODEL || row[ATM_PTR_ZETA_SLOT] != 0 {
            return Err(LibcintRsError::UnsupportedApi {
                api: "cpu.one_e.nuclear.cartesian",
                reason: "Gaussian-distributed nuclei are not supported in Rust-native 1e nuclear implementation",
            });
        }

        let charge = f64::from(row[ATM_CHARGE_SLOT].abs());
        if charge <= 0.0 || !charge.is_finite() {
            return Err(LibcintRsError::InvalidInput {
                field: "atm.charge_of",
                reason: format!("atom {atom_index} has invalid nuclear charge {}", row[0]),
            });
        }

        let coord_offset =
            env_view.checked_offset_range("atm.ptr_coord", row[ATM_PTR_COORD_SLOT], 3)?;
        nuclei.push(NuclearCenter {
            center: [
                env[coord_offset],
                env[coord_offset + 1],
                env[coord_offset + 2],
            ],
            charge,
        });
    }
    Ok(nuclei)
}

fn fill_overlap_shell_pair(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let components_i = cartesian_component_powers(shell_i.angular_momentum);
    let components_j = cartesian_component_powers(shell_j.angular_momentum);
    let di = components_i
        .len()
        .checked_mul(shell_i.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "i-shell output dimension overflows usize".to_string(),
        })?;
    let dj = components_j
        .len()
        .checked_mul(shell_j.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "j-shell output dimension overflows usize".to_string(),
        })?;
    let expected_len = di
        .checked_mul(dj)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "output length overflows usize".to_string(),
        })?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let nprim_i = shell_i.exponents.len();
    let nprim_j = shell_j.exponents.len();
    let common_factor =
        common_fac_sp(shell_i.angular_momentum) * common_fac_sp(shell_j.angular_momentum);
    for jc in 0..shell_j.nctr {
        for ic in 0..shell_i.nctr {
            for (j_component_index, j_powers) in components_j.iter().enumerate() {
                for (i_component_index, i_powers) in components_i.iter().enumerate() {
                    let mut value = 0.0f64;
                    for jp in 0..nprim_j {
                        let coefficient_j = shell_j.coefficients[jc * nprim_j + jp];
                        if coefficient_j == 0.0 {
                            continue;
                        }
                        for ip in 0..nprim_i {
                            let coefficient_i = shell_i.coefficients[ic * nprim_i + ip];
                            if coefficient_i == 0.0 {
                                continue;
                            }
                            value += common_factor
                                * coefficient_i
                                * coefficient_j
                                * primitive_overlap_cartesian(
                                    shell_i.exponents[ip],
                                    shell_j.exponents[jp],
                                    shell_i.center,
                                    shell_j.center,
                                    *i_powers,
                                    *j_powers,
                                )?;
                        }
                    }

                    let i_index = ic * components_i.len() + i_component_index;
                    let j_index = jc * components_j.len() + j_component_index;
                    output[i_index + di * j_index] = value;
                }
            }
        }
    }

    Ok(())
}

fn fill_kinetic_shell_pair(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let components_i = cartesian_component_powers(shell_i.angular_momentum);
    let components_j = cartesian_component_powers(shell_j.angular_momentum);
    let di = components_i
        .len()
        .checked_mul(shell_i.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "i-shell output dimension overflows usize".to_string(),
        })?;
    let dj = components_j
        .len()
        .checked_mul(shell_j.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "j-shell output dimension overflows usize".to_string(),
        })?;
    let expected_len = di
        .checked_mul(dj)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "output length overflows usize".to_string(),
        })?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let nprim_i = shell_i.exponents.len();
    let nprim_j = shell_j.exponents.len();
    let common_factor =
        common_fac_sp(shell_i.angular_momentum) * common_fac_sp(shell_j.angular_momentum);
    for jc in 0..shell_j.nctr {
        for ic in 0..shell_i.nctr {
            for (j_component_index, j_powers) in components_j.iter().enumerate() {
                for (i_component_index, i_powers) in components_i.iter().enumerate() {
                    let mut value = 0.0f64;
                    for jp in 0..nprim_j {
                        let coefficient_j = shell_j.coefficients[jc * nprim_j + jp];
                        if coefficient_j == 0.0 {
                            continue;
                        }
                        for ip in 0..nprim_i {
                            let coefficient_i = shell_i.coefficients[ic * nprim_i + ip];
                            if coefficient_i == 0.0 {
                                continue;
                            }
                            value += common_factor
                                * coefficient_i
                                * coefficient_j
                                * primitive_kinetic_cartesian(
                                    shell_i.exponents[ip],
                                    shell_j.exponents[jp],
                                    shell_i.center,
                                    shell_j.center,
                                    *i_powers,
                                    *j_powers,
                                )?;
                        }
                    }

                    let i_index = ic * components_i.len() + i_component_index;
                    let j_index = jc * components_j.len() + j_component_index;
                    output[i_index + di * j_index] = value;
                }
            }
        }
    }

    Ok(())
}

fn fill_nuclear_shell_pair(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    nuclei: &[NuclearCenter],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let components_i = cartesian_component_powers(shell_i.angular_momentum);
    let components_j = cartesian_component_powers(shell_j.angular_momentum);
    let di = components_i
        .len()
        .checked_mul(shell_i.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "i-shell output dimension overflows usize".to_string(),
        })?;
    let dj = components_j
        .len()
        .checked_mul(shell_j.nctr)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "j-shell output dimension overflows usize".to_string(),
        })?;
    let expected_len = di
        .checked_mul(dj)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "output length overflows usize".to_string(),
        })?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let nprim_i = shell_i.exponents.len();
    let nprim_j = shell_j.exponents.len();
    let common_factor =
        common_fac_sp(shell_i.angular_momentum) * common_fac_sp(shell_j.angular_momentum);
    for jc in 0..shell_j.nctr {
        for ic in 0..shell_i.nctr {
            for (j_component_index, j_powers) in components_j.iter().enumerate() {
                for (i_component_index, i_powers) in components_i.iter().enumerate() {
                    let mut value = 0.0f64;
                    for jp in 0..nprim_j {
                        let coefficient_j = shell_j.coefficients[jc * nprim_j + jp];
                        if coefficient_j == 0.0 {
                            continue;
                        }
                        for ip in 0..nprim_i {
                            let coefficient_i = shell_i.coefficients[ic * nprim_i + ip];
                            if coefficient_i == 0.0 {
                                continue;
                            }
                            value += common_factor
                                * coefficient_i
                                * coefficient_j
                                * primitive_nuclear_cartesian(
                                    shell_i.exponents[ip],
                                    shell_j.exponents[jp],
                                    shell_i.center,
                                    shell_j.center,
                                    *i_powers,
                                    *j_powers,
                                    nuclei,
                                )?;
                        }
                    }

                    let i_index = ic * components_i.len() + i_component_index;
                    let j_index = jc * components_j.len() + j_component_index;
                    output[i_index + di * j_index] = value;
                }
            }
        }
    }

    Ok(())
}

const SPH_L0: [f64; 1] = [1.0];
const SPH_L1: [f64; 9] = [
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
    0.0, 0.0, 1.0,
];
const SPH_L2: [f64; 30] = [
    0.0,
    1.092_548_430_592_079_1,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.092_548_430_592_079_1,
    0.0,
    -0.315_391_565_252_52,
    0.0,
    0.0,
    -0.315_391_565_252_52,
    0.0,
    0.630_783_130_505_04,
    0.0,
    0.0,
    1.092_548_430_592_079_1,
    0.0,
    0.0,
    0.0,
    0.546_274_215_296_039_5,
    0.0,
    0.0,
    -0.546_274_215_296_039_5,
    0.0,
    0.0,
];

const SPINOR_LT_L0_R: [f64; 4] = [0.0, 1.0, 1.0, 0.0];
const SPINOR_LT_L0_I: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

const SPINOR_LT_L1_R: [f64; 36] = [
    -0.577_350_269_189_625_7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.577_350_269_189_625_7,
    0.0,
    0.0,
    -0.577_350_269_189_625_7,
    -0.577_350_269_189_625_7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.707_106_781_186_547_6,
    0.0,
    0.0,
    0.408_248_290_463_863,
    0.0,
    0.0,
    0.0,
    0.0,
    0.816_496_580_927_726,
    0.0,
    0.0,
    0.816_496_580_927_726,
    -0.408_248_290_463_863,
    0.0,
    0.0,
    -0.707_106_781_186_547_6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

const SPINOR_LT_L1_I: [f64; 36] = [
    0.0,
    0.577_350_269_189_625_7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.577_350_269_189_625_7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.707_106_781_186_547_6,
    0.0,
    0.0,
    -0.408_248_290_463_863,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.408_248_290_463_863,
    0.0,
    0.0,
    -0.707_106_781_186_547_6,
    0.0,
    0.0,
    0.0,
    0.0,
];

const SPINOR_LT_L2_R: [f64; 120] = [
    -0.345_494_149_471_335_5,
    0.0,
    0.0,
    0.345_494_149_471_335_5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.345_494_149_471_335_5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.598_413_420_602_149,
    0.0,
    0.0,
    0.0,
    -0.199_471_140_200_716_35,
    0.0,
    0.0,
    -0.199_471_140_200_716_35,
    0.0,
    0.398_942_280_401_432_7,
    0.199_471_140_200_716_35,
    0.0,
    0.0,
    0.199_471_140_200_716_35,
    0.0,
    -0.398_942_280_401_432_7,
    0.0,
    0.0,
    -0.598_413_420_602_149,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.345_494_149_471_335_5,
    0.0,
    0.0,
    0.0,
    0.345_494_149_471_335_5,
    0.0,
    0.0,
    -0.345_494_149_471_335_5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.386_274_202_023_189_57,
    0.0,
    0.0,
    -0.386_274_202_023_189_57,
    0.0,
    0.0,
    0.172_747_074_735_667_75,
    0.0,
    0.0,
    -0.172_747_074_735_667_75,
    0.0,
    0.0,
    0.0,
    0.0,
    0.690_988_298_942_671,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.488_602_511_902_919_9,
    0.0,
    0.0,
    0.0,
    -0.244_301_255_951_459_96,
    0.0,
    0.0,
    -0.244_301_255_951_459_96,
    0.0,
    0.488_602_511_902_919_9,
    -0.244_301_255_951_459_96,
    0.0,
    0.0,
    -0.244_301_255_951_459_96,
    0.0,
    0.488_602_511_902_919_9,
    0.0,
    0.0,
    -0.488_602_511_902_919_9,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.690_988_298_942_671,
    0.0,
    0.0,
    0.0,
    0.172_747_074_735_667_75,
    0.0,
    0.0,
    -0.172_747_074_735_667_75,
    0.0,
    0.0,
    0.386_274_202_023_189_57,
    0.0,
    0.0,
    -0.386_274_202_023_189_57,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

const SPINOR_LT_L2_I: [f64; 120] = [
    0.00000000000000000,
    0.690988298942670998,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.345494149471335499,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.598413420602148971,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.598413420602148971,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.345494149471335499,
    0.00000000000000000,
    0.00000000000000000,
    0.690988298942670998,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.772548404046379145,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.345494149471335499,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.690988298942670998,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.488602511902919923,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.488602511902919923,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    -0.690988298942670998,
    0.00000000000000000,
    0.00000000000000000,
    0.345494149471335499,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.772548404046379145,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
    0.00000000000000000,
];

#[derive(Clone, Copy)]
struct SpinorTransform {
    nd: usize,
    nf: usize,
    coeff_r: &'static [f64],
    coeff_i: &'static [f64],
}

fn spherical_transform(l: usize) -> Result<(usize, usize, &'static [f64]), LibcintRsError> {
    match l {
        0 => Ok((1, 1, &SPH_L0)),
        1 => Ok((3, 3, &SPH_L1)),
        2 => Ok((5, 6, &SPH_L2)),
        _ => Err(LibcintRsError::UnsupportedApi {
            api: "cpu.one_e.transform.spherical",
            reason: "spherical transform currently supports angular momentum <= 2",
        }),
    }
}

fn spinor_transform(l: usize, kappa: i32) -> Result<SpinorTransform, LibcintRsError> {
    if kappa != 0 {
        return Err(LibcintRsError::UnsupportedApi {
            api: "cpu.one_e.transform.spinor",
            reason: "spinor transform currently supports kappa=0 shells only",
        });
    }

    match l {
        0 => Ok(SpinorTransform {
            nd: 2,
            nf: 1,
            coeff_r: &SPINOR_LT_L0_R,
            coeff_i: &SPINOR_LT_L0_I,
        }),
        1 => Ok(SpinorTransform {
            nd: 6,
            nf: 3,
            coeff_r: &SPINOR_LT_L1_R,
            coeff_i: &SPINOR_LT_L1_I,
        }),
        2 => Ok(SpinorTransform {
            nd: 10,
            nf: 6,
            coeff_r: &SPINOR_LT_L2_R,
            coeff_i: &SPINOR_LT_L2_I,
        }),
        _ => Err(LibcintRsError::UnsupportedApi {
            api: "cpu.one_e.transform.spinor",
            reason: "spinor transform currently supports angular momentum <= 2",
        }),
    }
}

fn fill_overlap_shell_pair_spherical(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian overlap output length overflows usize".to_string(),
        })?;
    let mut cart = vec![0.0f64; cart_len];
    fill_overlap_shell_pair(shell_i, shell_j, cart.as_mut_slice())?;
    transform_cartesian_block_to_spherical(
        shell_i,
        shell_j,
        cart.as_slice(),
        output,
        "cpu.one_e.overlap.spherical",
    )
}

fn fill_kinetic_shell_pair_spherical(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian kinetic output length overflows usize".to_string(),
        })?;
    let mut cart = vec![0.0f64; cart_len];
    fill_kinetic_shell_pair(shell_i, shell_j, cart.as_mut_slice())?;
    transform_cartesian_block_to_spherical(
        shell_i,
        shell_j,
        cart.as_slice(),
        output,
        "cpu.one_e.kinetic.spherical",
    )
}

fn fill_nuclear_shell_pair_spherical(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    nuclei: &[NuclearCenter],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian nuclear-attraction output length overflows usize".to_string(),
        })?;
    let mut cart = vec![0.0f64; cart_len];
    fill_nuclear_shell_pair(shell_i, shell_j, nuclei, cart.as_mut_slice())?;
    transform_cartesian_block_to_spherical(
        shell_i,
        shell_j,
        cart.as_slice(),
        output,
        "cpu.one_e.nuclear.spherical",
    )
}

fn transform_cartesian_block_to_spherical(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    cart: &[f64],
    output: &mut [f64],
    api_label: &'static str,
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian transform source length overflows usize".to_string(),
        })?;
    if cart.len() != cart_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: cart_len,
            got: cart.len(),
        });
    }

    let (sph_comp_i, sph_cart_i, sph_i) = spherical_transform(shell_i.angular_momentum)?;
    let (sph_comp_j, sph_cart_j, sph_j) = spherical_transform(shell_j.angular_momentum)?;
    if sph_cart_i != cart_comp_i || sph_cart_j != cart_comp_j {
        return Err(LibcintRsError::BackendFailure {
            backend: api_label,
            detail: "spherical transform/cartesian component mismatch".to_string(),
        });
    }

    let di_sph =
        sph_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "spherical i-shell dimension overflows usize".to_string(),
            })?;
    let dj_sph =
        sph_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "spherical j-shell dimension overflows usize".to_string(),
            })?;
    let expected_len = di_sph
        .checked_mul(dj_sph)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "spherical transform output length overflows usize".to_string(),
        })?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    for jc in 0..shell_j.nctr {
        for ic in 0..shell_i.nctr {
            for sj in 0..sph_comp_j {
                for si in 0..sph_comp_i {
                    let mut value = 0.0f64;
                    for cj in 0..cart_comp_j {
                        let tj = sph_j[sj * cart_comp_j + cj];
                        if tj == 0.0 {
                            continue;
                        }
                        for ci in 0..cart_comp_i {
                            let ti = sph_i[si * cart_comp_i + ci];
                            if ti == 0.0 {
                                continue;
                            }

                            let cart_i = (ic * cart_comp_i) + ci;
                            let cart_j = (jc * cart_comp_j) + cj;
                            value += ti * tj * cart[cart_i + di_cart * cart_j];
                        }
                    }

                    let out_i = (ic * sph_comp_i) + si;
                    let out_j = (jc * sph_comp_j) + sj;
                    output[out_i + di_sph * out_j] = value;
                }
            }
        }
    }

    Ok(())
}

fn fill_overlap_shell_pair_spinor(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian overlap output length overflows usize".to_string(),
        })?;
    let mut cart = vec![0.0f64; cart_len];
    fill_overlap_shell_pair(shell_i, shell_j, cart.as_mut_slice())?;
    transform_cartesian_block_to_spinor(
        shell_i,
        shell_j,
        cart.as_slice(),
        output,
        "cpu.one_e.overlap.spinor",
    )
}

fn fill_kinetic_shell_pair_spinor(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian kinetic output length overflows usize".to_string(),
        })?;
    let mut cart = vec![0.0f64; cart_len];
    fill_kinetic_shell_pair(shell_i, shell_j, cart.as_mut_slice())?;
    transform_cartesian_block_to_spinor(
        shell_i,
        shell_j,
        cart.as_slice(),
        output,
        "cpu.one_e.kinetic.spinor",
    )
}

fn fill_nuclear_shell_pair_spinor(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    nuclei: &[NuclearCenter],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian nuclear-attraction output length overflows usize".to_string(),
        })?;
    let mut cart = vec![0.0f64; cart_len];
    fill_nuclear_shell_pair(shell_i, shell_j, nuclei, cart.as_mut_slice())?;
    transform_cartesian_block_to_spinor(
        shell_i,
        shell_j,
        cart.as_slice(),
        output,
        "cpu.one_e.nuclear.spinor",
    )
}

fn transform_cartesian_block_to_spinor(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    cart: &[f64],
    output: &mut [[f64; 2]],
    api_label: &'static str,
) -> Result<(), LibcintRsError> {
    let cart_comp_i = cartesian_component_powers(shell_i.angular_momentum).len();
    let cart_comp_j = cartesian_component_powers(shell_j.angular_momentum).len();
    let di_cart =
        cart_comp_i
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian i-shell dimension overflows usize".to_string(),
            })?;
    let dj_cart =
        cart_comp_j
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "cartesian j-shell dimension overflows usize".to_string(),
            })?;
    let cart_len = di_cart
        .checked_mul(dj_cart)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "cartesian transform source length overflows usize".to_string(),
        })?;
    if cart.len() != cart_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: cart_len,
            got: cart.len(),
        });
    }

    let spin_i = spinor_transform(shell_i.angular_momentum, shell_i.kappa)?;
    let spin_j = spinor_transform(shell_j.angular_momentum, shell_j.kappa)?;
    if spin_i.nf != cart_comp_i || spin_j.nf != cart_comp_j {
        return Err(LibcintRsError::BackendFailure {
            backend: api_label,
            detail: "spinor transform/cartesian component mismatch".to_string(),
        });
    }

    let di_spin =
        spin_i
            .nd
            .checked_mul(shell_i.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "spinor i-shell dimension overflows usize".to_string(),
            })?;
    let dj_spin =
        spin_j
            .nd
            .checked_mul(shell_j.nctr)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "spinor j-shell dimension overflows usize".to_string(),
            })?;
    let expected_len =
        di_spin
            .checked_mul(dj_spin)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "spinor transform output length overflows usize".to_string(),
            })?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let ket_cols = spin_j
        .nf
        .checked_mul(2)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "output",
            reason: "spinor ket column count overflows usize".to_string(),
        })?;
    let interm_len =
        spin_i
            .nd
            .checked_mul(ket_cols)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "output",
                reason: "spinor intermediate length overflows usize".to_string(),
            })?;
    let mut interm_r = vec![0.0f64; interm_len];
    let mut interm_i = vec![0.0f64; interm_len];

    for jc in 0..shell_j.nctr {
        for ic in 0..shell_i.nctr {
            for q in 0..spin_j.nf {
                for si in 0..spin_i.nd {
                    let coeff_row = si * spin_i.nf * 2;
                    let mut alpha_r = 0.0f64;
                    let mut alpha_i = 0.0f64;
                    let mut beta_r = 0.0f64;
                    let mut beta_i = 0.0f64;

                    for pi in 0..spin_i.nf {
                        let cart_i = (ic * spin_i.nf) + pi;
                        let cart_j = (jc * spin_j.nf) + q;
                        let value = cart[cart_i + di_cart * cart_j];

                        let ca_r = spin_i.coeff_r[coeff_row + pi];
                        let ca_i = spin_i.coeff_i[coeff_row + pi];
                        let cb_r = spin_i.coeff_r[coeff_row + spin_i.nf + pi];
                        let cb_i = spin_i.coeff_i[coeff_row + spin_i.nf + pi];

                        alpha_r += ca_r * value;
                        alpha_i -= ca_i * value;
                        beta_r += cb_r * value;
                        beta_i -= cb_i * value;
                    }

                    interm_r[si + spin_i.nd * q] = alpha_r;
                    interm_i[si + spin_i.nd * q] = alpha_i;
                    interm_r[si + spin_i.nd * (q + spin_j.nf)] = beta_r;
                    interm_i[si + spin_i.nd * (q + spin_j.nf)] = beta_i;
                }
            }

            for sj in 0..spin_j.nd {
                let coeff_row = sj * ket_cols;
                for si in 0..spin_i.nd {
                    let mut real = 0.0f64;
                    let mut imag = 0.0f64;

                    for n in 0..ket_cols {
                        let c_r = spin_j.coeff_r[coeff_row + n];
                        let c_i = spin_j.coeff_i[coeff_row + n];
                        let g_r = interm_r[si + spin_i.nd * n];
                        let g_i = interm_i[si + spin_i.nd * n];
                        real += (c_r * g_r) - (c_i * g_i);
                        imag += (c_i * g_r) + (c_r * g_i);
                    }

                    let out_i = (ic * spin_i.nd) + si;
                    let out_j = (jc * spin_j.nd) + sj;
                    output[out_i + di_spin * out_j] = [real, imag];
                }
            }
        }
    }

    Ok(())
}

fn fill_overlap_shell_pair_spinor_flattened(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if !output.len().is_multiple_of(2) {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: output.len().saturating_add(1),
            got: output.len(),
        });
    }

    let element_count = output.len() / 2;
    let mut complex = vec![[0.0f64; 2]; element_count];
    fill_overlap_shell_pair_spinor(shell_i, shell_j, complex.as_mut_slice())?;
    for (slot, value) in output.chunks_exact_mut(2).zip(complex.into_iter()) {
        slot[0] = value[0];
        slot[1] = value[1];
    }
    Ok(())
}

fn fill_kinetic_shell_pair_spinor_flattened(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if !output.len().is_multiple_of(2) {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: output.len().saturating_add(1),
            got: output.len(),
        });
    }

    let element_count = output.len() / 2;
    let mut complex = vec![[0.0f64; 2]; element_count];
    fill_kinetic_shell_pair_spinor(shell_i, shell_j, complex.as_mut_slice())?;
    for (slot, value) in output.chunks_exact_mut(2).zip(complex.into_iter()) {
        slot[0] = value[0];
        slot[1] = value[1];
    }
    Ok(())
}

fn fill_nuclear_shell_pair_spinor_flattened(
    shell_i: &CartesianShellData,
    shell_j: &CartesianShellData,
    nuclei: &[NuclearCenter],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    if !output.len().is_multiple_of(2) {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: output.len().saturating_add(1),
            got: output.len(),
        });
    }

    let element_count = output.len() / 2;
    let mut complex = vec![[0.0f64; 2]; element_count];
    fill_nuclear_shell_pair_spinor(shell_i, shell_j, nuclei, complex.as_mut_slice())?;
    for (slot, value) in output.chunks_exact_mut(2).zip(complex.into_iter()) {
        slot[0] = value[0];
        slot[1] = value[1];
    }
    Ok(())
}

fn common_fac_sp(angular_momentum: usize) -> f64 {
    match angular_momentum {
        0 => 0.282_094_791_773_878_14,
        1 => 0.488_602_511_902_919_9,
        _ => 1.0,
    }
}

fn primitive_nuclear_cartesian(
    alpha: f64,
    beta: f64,
    center_i: [f64; 3],
    center_j: [f64; 3],
    i_powers: [usize; 3],
    j_powers: [usize; 3],
    nuclei: &[NuclearCenter],
) -> Result<f64, LibcintRsError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason: "primitive exponent must be finite and greater than zero".to_string(),
        });
    }
    if !beta.is_finite() || beta <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason: "primitive exponent must be finite and greater than zero".to_string(),
        });
    }

    let mut value = 0.0f64;
    for nucleus in nuclei {
        value += primitive_nuclear_single_center(
            alpha, beta, center_i, center_j, i_powers, j_powers, nucleus,
        )?;
    }
    Ok(value)
}

fn primitive_nuclear_single_center(
    alpha: f64,
    beta: f64,
    center_i: [f64; 3],
    center_j: [f64; 3],
    i_powers: [usize; 3],
    j_powers: [usize; 3],
    nucleus: &NuclearCenter,
) -> Result<f64, LibcintRsError> {
    let integral = adaptive_simpson_result(
        &|u| {
            let clamped_u = u.clamp(0.0, 1.0 - 1e-14);
            let denom = 1.0 - clamped_u;
            let s = clamped_u / denom;
            let s2 = s * s;
            let jacobian = 1.0 / (denom * denom);

            let x = primitive_three_gaussian_axis(
                alpha,
                beta,
                s2,
                center_i[0],
                center_j[0],
                nucleus.center[0],
                i_powers[0],
                j_powers[0],
            );
            let y = primitive_three_gaussian_axis(
                alpha,
                beta,
                s2,
                center_i[1],
                center_j[1],
                nucleus.center[1],
                i_powers[1],
                j_powers[1],
            );
            let z = primitive_three_gaussian_axis(
                alpha,
                beta,
                s2,
                center_i[2],
                center_j[2],
                nucleus.center[2],
                i_powers[2],
                j_powers[2],
            );

            Ok((2.0 / std::f64::consts::PI.sqrt()) * x * y * z * jacobian)
        },
        0.0,
        1.0 - 1e-12,
        1e-12,
        14,
    )?;
    Ok((-nucleus.charge) * integral)
}

fn cartesian_axis_expansion_coeffs(
    power_i: usize,
    power_j: usize,
    shift_i: f64,
    shift_j: f64,
) -> Vec<f64> {
    let mut coeffs = vec![0.0f64; power_i + power_j + 1];
    for i in 0..=power_i {
        let coeff_i = binomial(power_i, i) * shift_i.powi((power_i - i) as i32);
        for j in 0..=power_j {
            let coeff_j = binomial(power_j, j) * shift_j.powi((power_j - j) as i32);
            coeffs[i + j] += coeff_i * coeff_j;
        }
    }
    coeffs
}

fn primitive_three_gaussian_axis(
    alpha: f64,
    beta: f64,
    gamma: f64,
    center_i: f64,
    center_j: f64,
    center_k: f64,
    i_power: usize,
    j_power: usize,
) -> f64 {
    let p = alpha + beta;
    let p_center = (alpha * center_i + beta * center_j) / p;
    let q = p + gamma;
    let q_center = (p * p_center + gamma * center_k) / q;

    let mu_ab = alpha * beta / p;
    let mu_pk = p * gamma / q;
    let prefactor =
        (-mu_ab * (center_i - center_j).powi(2) - mu_pk * (p_center - center_k).powi(2)).exp();

    let coeffs =
        cartesian_axis_expansion_coeffs(i_power, j_power, q_center - center_i, q_center - center_j);
    let mut moment_sum = 0.0f64;
    for (order, coefficient) in coeffs.into_iter().enumerate() {
        if coefficient == 0.0 {
            continue;
        }
        moment_sum += coefficient * gaussian_moment(order, q);
    }
    prefactor * moment_sum
}

fn adaptive_simpson_result<F>(
    f: &F,
    a: f64,
    b: f64,
    tolerance: f64,
    max_depth: usize,
) -> Result<f64, LibcintRsError>
where
    F: Fn(f64) -> Result<f64, LibcintRsError>,
{
    let fa = f(a)?;
    let fb = f(b)?;
    let c = 0.5 * (a + b);
    let fc = f(c)?;
    let whole = simpson_estimate(a, b, fa, fb, fc);
    adaptive_simpson_recursive(f, a, b, fa, fb, fc, whole, tolerance, max_depth)
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_recursive<F>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fc: f64,
    whole: f64,
    tolerance: f64,
    depth: usize,
) -> Result<f64, LibcintRsError>
where
    F: Fn(f64) -> Result<f64, LibcintRsError>,
{
    let c = 0.5 * (a + b);
    let left_mid = 0.5 * (a + c);
    let right_mid = 0.5 * (c + b);
    let f_left_mid = f(left_mid)?;
    let f_right_mid = f(right_mid)?;

    let left = simpson_estimate(a, c, fa, fc, f_left_mid);
    let right = simpson_estimate(c, b, fc, fb, f_right_mid);
    let delta = left + right - whole;
    if depth == 0 || delta.abs() <= 15.0 * tolerance {
        return Ok(left + right + delta / 15.0);
    }

    let left_integral = adaptive_simpson_recursive(
        f,
        a,
        c,
        fa,
        fc,
        f_left_mid,
        left,
        0.5 * tolerance,
        depth - 1,
    )?;
    let right_integral = adaptive_simpson_recursive(
        f,
        c,
        b,
        fc,
        fb,
        f_right_mid,
        right,
        0.5 * tolerance,
        depth - 1,
    )?;
    Ok(left_integral + right_integral)
}

fn simpson_estimate(a: f64, b: f64, fa: f64, fb: f64, fm: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

fn primitive_kinetic_cartesian(
    alpha: f64,
    beta: f64,
    center_i: [f64; 3],
    center_j: [f64; 3],
    i_powers: [usize; 3],
    j_powers: [usize; 3],
) -> Result<f64, LibcintRsError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason: "primitive exponent must be finite and greater than zero".to_string(),
        });
    }
    if !beta.is_finite() || beta <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason: "primitive exponent must be finite and greater than zero".to_string(),
        });
    }

    let overlap = primitive_overlap_cartesian(alpha, beta, center_i, center_j, i_powers, j_powers)?;
    let total_j_angular = j_powers.iter().copied().sum::<usize>();
    let mut value = beta * ((2 * total_j_angular) + 3) as f64 * overlap;

    for axis in 0..3 {
        let mut raised = j_powers;
        raised[axis] = raised[axis]
            .checked_add(2)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "bas.ang_of",
                reason: "angular momentum overflow while evaluating kinetic integral".to_string(),
            })?;
        value -= 2.0
            * beta
            * beta
            * primitive_overlap_cartesian(alpha, beta, center_i, center_j, i_powers, raised)?;

        let axis_power = j_powers[axis];
        if axis_power >= 2 {
            let mut lowered = j_powers;
            lowered[axis] -= 2;
            value -= 0.5
                * (axis_power * (axis_power - 1)) as f64
                * primitive_overlap_cartesian(alpha, beta, center_i, center_j, i_powers, lowered)?;
        }
    }

    Ok(value)
}

fn primitive_overlap_cartesian(
    alpha: f64,
    beta: f64,
    center_i: [f64; 3],
    center_j: [f64; 3],
    i_powers: [usize; 3],
    j_powers: [usize; 3],
) -> Result<f64, LibcintRsError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason: "primitive exponent must be finite and greater than zero".to_string(),
        });
    }
    if !beta.is_finite() || beta <= 0.0 {
        return Err(LibcintRsError::InvalidInput {
            field: "bas.ptr_exp",
            reason: "primitive exponent must be finite and greater than zero".to_string(),
        });
    }

    let mut value = 1.0f64;
    for axis in 0..3 {
        value *= primitive_overlap_1d(
            alpha,
            beta,
            center_i[axis],
            center_j[axis],
            i_powers[axis],
            j_powers[axis],
        );
    }
    Ok(value)
}

fn primitive_overlap_1d(
    alpha: f64,
    beta: f64,
    center_i: f64,
    center_j: f64,
    i_power: usize,
    j_power: usize,
) -> f64 {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let product_center = (alpha * center_i + beta * center_j) / p;
    let center_delta = center_i - center_j;
    let prefactor = (-mu * center_delta * center_delta).exp();
    let shifted_i = product_center - center_i;
    let shifted_j = product_center - center_j;

    let mut sum = 0.0f64;
    for ki in 0..=i_power {
        let coeff_i = binomial(i_power, ki) * shifted_i.powi((i_power.saturating_sub(ki)) as i32);
        for kj in 0..=j_power {
            let coeff_j =
                binomial(j_power, kj) * shifted_j.powi((j_power.saturating_sub(kj)) as i32);
            sum += coeff_i * coeff_j * gaussian_moment(ki + kj, p);
        }
    }

    prefactor * sum
}

fn gaussian_moment(power: usize, exponent_sum: f64) -> f64 {
    if power % 2 == 1 {
        return 0.0;
    }

    let half_power = power / 2;
    let mut moment = (std::f64::consts::PI / exponent_sum).sqrt();
    for order in 1..=half_power {
        moment *= (2 * order - 1) as f64 / (2.0 * exponent_sum);
    }
    moment
}

fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k);
    let mut value = 1.0f64;
    for step in 0..k {
        value *= (n - step) as f64;
        value /= (step + 1) as f64;
    }
    value
}

fn cartesian_component_powers(angular_momentum: usize) -> Vec<[usize; 3]> {
    let component_count = (angular_momentum + 1) * (angular_momentum + 2) / 2;
    let mut powers = Vec::with_capacity(component_count);
    for lx in (0..=angular_momentum).rev() {
        for ly in (0..=angular_momentum - lx).rev() {
            let lz = angular_momentum - lx - ly;
            powers.push([lx, ly, lz]);
        }
    }
    powers
}
