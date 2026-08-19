//! 3-way Parity Test: SIMD-Kernel vs. CubeCL-Kernel vs. libcint
//!
//! Validates result compatibility across all three execution engines:
//! 1. `simd-kernel`: Portable SIMD kernel using `wide` + `rmath` (`cintx-simd`)
//! 2. `cubecl-kernel`: CubeCL compute kernel backend (`cintx-compat` / `cintx-cubecl`)
//! 3. `libcint`: Upstream vendored libcint 6.1.3 reference oracle (`cintx-oracle::vendor_ffi`)

#![cfg(feature = "cpu")]
#![cfg(has_vendor_libcint)]

use approx::assert_relative_eq;
use cintx_compat::raw::{
    ANG_OF, ATOM_OF, ATM_SLOTS, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_oracle::vendor_ffi;
use cintx_simd::{
    AtomCoord, Center2c2eInput, Center3c1eInput, Center3c2eInput, OneElectronInput,
    SimdCenter2c2eKernel, SimdCenter3c1eKernel, SimdCenter3c2eKernel,
    SimdOneElectronKernel, SimdTwoElectronKernel, TwoElectronInput,
};

/// Build H2O STO-3G libcint-style atm/bas/env.
fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];

    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let o1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_exp);
    let o1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_1s_coeff);

    let o2s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_exp);
    let o2s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2s_coeff);

    let o2p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_exp);
    let o2p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&o_2p_coeff);

    let h1s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_exp);
    let h1s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_1s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];

    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 5 * BAS_SLOTS];

    // Shell 0: O-1s (l=0)
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    // Shell 1: O-2s (l=0)
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    // Shell 2: O-2p (l=1)
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    // Shell 3: H1-1s (l=0)
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    // Shell 4: H2-1s (l=0)
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

fn ncart_l(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn get_shell_data<'a>(
    s: usize,
    atm: &[i32],
    bas: &[i32],
    env: &'a [f64],
) -> (u8, [f64; 3], &'a [f64], &'a [f64]) {
    let atom_idx = bas[s * BAS_SLOTS + ATOM_OF] as usize;
    let l = bas[s * BAS_SLOTS + ANG_OF] as u8;
    let nprim = bas[s * BAS_SLOTS + NPRIM_OF] as usize;
    let ptr_exp = bas[s * BAS_SLOTS + PTR_EXP] as usize;
    let ptr_coeff = bas[s * BAS_SLOTS + PTR_COEFF] as usize;
    let ptr_coord = atm[atom_idx * ATM_SLOTS + PTR_COORD] as usize;

    let coord = [env[ptr_coord], env[ptr_coord + 1], env[ptr_coord + 2]];
    let exps = &env[ptr_exp..ptr_exp + nprim];
    let coeffs = &env[ptr_coeff..ptr_coeff + nprim];

    (l, coord, exps, coeffs)
}

fn get_atoms(atm: &[i32], env: &[f64]) -> Vec<AtomCoord> {
    let natm = atm.len() / ATM_SLOTS;
    let mut atoms = Vec::with_capacity(natm);
    for a in 0..natm {
        let charge = atm[a * ATM_SLOTS + CHARGE_OF] as f64;
        let ptr_coord = atm[a * ATM_SLOTS + PTR_COORD] as usize;
        let coord = [env[ptr_coord], env[ptr_coord + 1], env[ptr_coord + 2]];
        atoms.push(AtomCoord { coord, charge });
    }
    atoms
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-Electron 3-way Parity (Overlap, Kinetic, Nuclear Attraction)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_3way_int1e_ovlp_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let nbas = 5;

    for si in 0..nbas {
        for sj in 0..nbas {
            let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
            let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
            let len = ncart_l(li as i32) * ncart_l(lj as i32);

            // 1. SIMD Kernel
            let mut out_simd = vec![0.0; len];
            let input = OneElectronInput {
                li,
                lj,
                ri,
                rj,
                exps_i,
                exps_j,
                coeff_i,
                coeff_j,
                atoms: &[],
            };
            SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut out_simd);

            // 2. CubeCL Kernel (via eval_raw)
            let mut out_cubecl = vec![0.0; len];
            let shls = [si as i32, sj as i32];
            unsafe {
                eval_raw(
                    RawApiId::INT1E_OVLP_CART,
                    Some(&mut out_cubecl),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap();
            }

            // 3. libcint Reference (via vendor_ffi)
            let mut out_libcint = vec![0.0; len];
            vendor_ffi::vendor_int1e_ovlp_cart(
                &mut out_libcint,
                &shls,
                &atm,
                3,
                &bas,
                5,
                &env,
            );

            // Assert 3-way parity
            for k in 0..len {
                assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-12);
                assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-12);
                assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn test_3way_int1e_kin_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let nbas = 5;

    for si in 0..nbas {
        for sj in 0..nbas {
            let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
            let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
            let len = ncart_l(li as i32) * ncart_l(lj as i32);

            // 1. SIMD Kernel
            let mut out_simd = vec![0.0; len];
            let input = OneElectronInput {
                li,
                lj,
                ri,
                rj,
                exps_i,
                exps_j,
                coeff_i,
                coeff_j,
                atoms: &[],
            };
            SimdOneElectronKernel::eval_kin::<f64>(&input, &mut out_simd);

            // 2. CubeCL Kernel (via eval_raw)
            let mut out_cubecl = vec![0.0; len];
            let shls = [si as i32, sj as i32];
            unsafe {
                eval_raw(
                    RawApiId::INT1E_KIN_CART,
                    Some(&mut out_cubecl),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap();
            }

            // 3. libcint Reference (via vendor_ffi)
            let mut out_libcint = vec![0.0; len];
            vendor_ffi::vendor_int1e_kin_cart(
                &mut out_libcint,
                &shls,
                &atm,
                3,
                &bas,
                5,
                &env,
            );

            for k in 0..len {
                assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-12);
                assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-12);
                assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn test_3way_int1e_nuc_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let atoms = get_atoms(&atm, &env);
    let nbas = 5;

    for si in 0..nbas {
        for sj in 0..nbas {
            let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
            let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
            let len = ncart_l(li as i32) * ncart_l(lj as i32);

            // 1. SIMD Kernel
            let mut out_simd = vec![0.0; len];
            let input = OneElectronInput {
                li,
                lj,
                ri,
                rj,
                exps_i,
                exps_j,
                coeff_i,
                coeff_j,
                atoms: &atoms,
            };
            SimdOneElectronKernel::eval_nuc::<f64>(&input, &mut out_simd);

            // 2. CubeCL Kernel (via eval_raw)
            let mut out_cubecl = vec![0.0; len];
            let shls = [si as i32, sj as i32];
            unsafe {
                eval_raw(
                    RawApiId::INT1E_NUC_CART,
                    Some(&mut out_cubecl),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap();
            }

            // 3. libcint Reference (via vendor_ffi)
            let mut out_libcint = vec![0.0; len];
            vendor_ffi::vendor_int1e_nuc_cart(
                &mut out_libcint,
                &shls,
                &atm,
                3,
                &bas,
                5,
                &env,
            );

            for k in 0..len {
                assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-10);
                assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-10);
                assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-10);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-Center 2-Electron 3-way Parity ((i|k))
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_3way_int2c2e_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let nbas = 5;

    for si in 0..nbas {
        for sk in 0..nbas {
            let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
            let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
            let len = ncart_l(li as i32) * ncart_l(lk as i32);

            // 1. SIMD Kernel
            let mut out_simd = vec![0.0; len];
            let input = Center2c2eInput {
                li,
                lk,
                ri,
                rk,
                exps_i,
                exps_k,
                coeff_i,
                coeff_k,
            };
            SimdCenter2c2eKernel::eval::<f64>(&input, &mut out_simd);

            // 2. CubeCL Kernel (via eval_raw)
            let mut out_cubecl = vec![0.0; len];
            let shls = [si as i32, sk as i32];
            unsafe {
                eval_raw(
                    RawApiId::INT2C2E_CART,
                    Some(&mut out_cubecl),
                    None,
                    &shls,
                    &atm,
                    &bas,
                    &env,
                    None,
                    None,
                )
                .unwrap();
            }

            // 3. libcint Reference (via vendor_ffi)
            let mut out_libcint = vec![0.0; len];
            vendor_ffi::vendor_int2c2e_cart(
                &mut out_libcint,
                &shls,
                &atm,
                3,
                &bas,
                5,
                &env,
            );

            for k in 0..len {
                assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-10);
                assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-10);
                assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-10);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-Center 2-Electron 3-way Parity ((ij|k))
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_3way_int3c2e_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let nbas = 4; // Sample representative subset of shells

    for si in 0..nbas {
        for sj in 0..nbas {
            for sk in 0..nbas {
                let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
                let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
                let len = ncart_l(li as i32) * ncart_l(lj as i32) * ncart_l(lk as i32);

                // 1. SIMD Kernel
                let mut out_simd = vec![0.0; len];
                let input = Center3c2eInput {
                    li,
                    lj,
                    lk,
                    ri,
                    rj,
                    rk,
                    exps_i,
                    exps_j,
                    exps_k,
                    coeff_i,
                    coeff_j,
                    coeff_k,
                };
                SimdCenter3c2eKernel::eval::<f64>(&input, &mut out_simd);

                // 2. CubeCL Kernel (via eval_raw)
                let mut out_cubecl = vec![0.0; len];
                let shls = [si as i32, sj as i32, sk as i32];
                unsafe {
                    eval_raw(
                        RawApiId::INT3C2E_CART,
                        Some(&mut out_cubecl),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap();
                }

                // 3. libcint Reference (via vendor_ffi)
                let mut out_libcint = vec![0.0; len];
                vendor_ffi::vendor_int3c2e_cart(
                    &mut out_libcint,
                    &shls,
                    &atm,
                    3,
                    &bas,
                    5,
                    &env,
                );

                for k in 0..len {
                    assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-9);
                    assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-9);
                    assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-9);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-Center 1-Electron 3-way Parity ((i|O_k|j))
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_3way_int3c1e_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let nbas = 4;

    for si in 0..nbas {
        for sj in 0..nbas {
            for sk in 0..nbas {
                let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
                let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
                let len = ncart_l(li as i32) * ncart_l(lj as i32) * ncart_l(lk as i32);

                // 1. SIMD Kernel
                let mut out_simd = vec![0.0; len];
                let input = Center3c1eInput {
                    li,
                    lj,
                    lk,
                    ri,
                    rj,
                    rk,
                    exps_i,
                    exps_j,
                    exps_k,
                    coeff_i,
                    coeff_j,
                    coeff_k,
                };
                SimdCenter3c1eKernel::eval::<f64>(&input, &mut out_simd);

                // 2. CubeCL Kernel (via eval_raw)
                let mut out_cubecl = vec![0.0; len];
                let shls = [si as i32, sj as i32, sk as i32];
                unsafe {
                    eval_raw(
                        RawApiId::INT3C1E_CART,
                        Some(&mut out_cubecl),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap();
                }

                // 3. libcint Reference (via vendor_ffi)
                let mut out_libcint = vec![0.0; len];
                vendor_ffi::vendor_int3c1e_cart(
                    &mut out_libcint,
                    &shls,
                    &atm,
                    3,
                    &bas,
                    5,
                    &env,
                );

                for k in 0..len {
                    assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-10);
                    assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-10);
                    assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-10);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4-Center 2-Electron 3-way Parity ((ij|kl))
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_3way_int2e_parity() {
    let (atm, bas, env) = build_h2o_sto3g();
    let quartets = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 2, 0, 0], // includes p-shell
        [1, 2, 3, 4],
        [2, 2, 0, 0],
    ];

    for &shls in &quartets {
        let (li, ri, exps_i, coeff_i) = get_shell_data(shls[0], &atm, &bas, &env);
        let (lj, rj, exps_j, coeff_j) = get_shell_data(shls[1], &atm, &bas, &env);
        let (lk, rk, exps_k, coeff_k) = get_shell_data(shls[2], &atm, &bas, &env);
        let (ll, rl, exps_l, coeff_l) = get_shell_data(shls[3], &atm, &bas, &env);
        let len = ncart_l(li as i32) * ncart_l(lj as i32) * ncart_l(lk as i32) * ncart_l(ll as i32);

        // 1. SIMD Kernel
        let mut out_simd = vec![0.0; len];
        let input = TwoElectronInput {
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            exps_i,
            exps_j,
            exps_k,
            exps_l,
            coeff_i,
            coeff_j,
            coeff_k,
            coeff_l,
        };
        SimdTwoElectronKernel::eval::<f64>(&input, &mut out_simd);

        // 2. CubeCL Kernel (via eval_raw)
        let mut out_cubecl = vec![0.0; len];
        let shls_i32 = [shls[0] as i32, shls[1] as i32, shls[2] as i32, shls[3] as i32];
        unsafe {
            eval_raw(
                RawApiId::INT2E_CART,
                Some(&mut out_cubecl),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        // 3. libcint Reference (via vendor_ffi)
        let mut out_libcint = vec![0.0; len];
        vendor_ffi::vendor_int2e_cart(
            &mut out_libcint,
            &shls_i32,
            &atm,
            3,
            &bas,
            5,
            &env,
        );

        for k in 0..len {
            assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-10);
            assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-10);
            assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-10);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// High Angular Momentum Shells (l >= 4: g-shells l=4, h-shells l=5)
// ─────────────────────────────────────────────────────────────────────────────

fn build_high_l_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let r1 = [0.0_f64, 0.0, 0.0];
    let r2 = [0.6_f64, 0.9, -0.4];
    let r3 = [-0.5_f64, 0.3, 0.7];

    let s_exp = [3.4252509_f64, 0.6239137];
    let s_coeff = [0.15432897_f64, 0.53532814];

    let p_exp = [5.0331513_f64, 1.1695961];
    let p_coeff = [0.15591627_f64, 0.60768372];

    let d_exp = [2.5_f64, 0.75];
    let d_coeff = [0.45_f64, 0.55];

    let f_exp = [1.8_f64, 0.55];
    let f_coeff = [0.50_f64, 0.50];

    let g_exp = [1.35_f64, 0.40];
    let g_coeff = [0.60_f64, 0.40];

    let h_exp = [1.60_f64, 0.45];
    let h_coeff = [0.55_f64, 0.45];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let r1_ptr = env.len() as i32;
    env.extend_from_slice(&r1);
    let r2_ptr = env.len() as i32;
    env.extend_from_slice(&r2);
    let r3_ptr = env.len() as i32;
    env.extend_from_slice(&r3);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let f_exp_ptr = env.len() as i32;
    env.extend_from_slice(&f_exp);
    let f_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&f_coeff);

    let g_exp_ptr = env.len() as i32;
    env.extend_from_slice(&g_exp);
    let g_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&g_coeff);

    let h_exp_ptr = env.len() as i32;
    env.extend_from_slice(&h_exp);
    let h_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&h_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[0 * ATM_SLOTS + CHARGE_OF] = 6;
    atm[0 * ATM_SLOTS + PTR_COORD] = r1_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = r2_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = r3_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 6 * BAS_SLOTS];

    // Shell 0: s (l=0)
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 2;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = s_coeff_ptr;

    // Shell 1: p (l=1)
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 1;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 2;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = p_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = p_coeff_ptr;

    // Shell 2: d (l=2)
    bas[2 * BAS_SLOTS + ATOM_OF] = 1;
    bas[2 * BAS_SLOTS + ANG_OF] = 2;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 2;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;

    // Shell 3: f (l=3)
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 3;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 2;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = f_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = f_coeff_ptr;

    // Shell 4: g (l=4)
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 4;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 2;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + PTR_EXP] = g_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = g_coeff_ptr;

    // Shell 5: h (l=5)
    bas[5 * BAS_SLOTS + ATOM_OF] = 2;
    bas[5 * BAS_SLOTS + ANG_OF] = 5;
    bas[5 * BAS_SLOTS + NPRIM_OF] = 2;
    bas[5 * BAS_SLOTS + NCTR_OF] = 1;
    bas[5 * BAS_SLOTS + PTR_EXP] = h_exp_ptr;
    bas[5 * BAS_SLOTS + PTR_COEFF] = h_coeff_ptr;

    (atm, bas, env)
}

#[test]
fn test_3way_high_l_1e_ovlp_kin_nuc_parity() {
    let (atm, bas, env) = build_high_l_fixture();
    let natm = 3;
    let nbas = 6;

    let test_pairs = [
        (4, 0), // g-s
        (4, 1), // g-p
        (4, 2), // g-d
        (4, 4), // g-g
        (5, 0), // h-s
        (5, 1), // h-p
        (5, 4), // h-g
        (5, 5), // h-h
    ];

    let atoms = get_atoms(&atm, &env);

    for &(si, sj) in &test_pairs {
        let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
        let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
        let len = ncart_l(li as i32) * ncart_l(lj as i32);
        let shls_i32 = [si as i32, sj as i32];

        // 1. Overlap
        let mut out_simd = vec![0.0; len];
        let input = OneElectronInput {
            li,
            lj,
            ri,
            rj,
            exps_i,
            exps_j,
            coeff_i,
            coeff_j,
            atoms: &atoms,
        };
        SimdOneElectronKernel::eval_ovlp::<f64>(&input, &mut out_simd);

        let mut out_cubecl = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::INT1E_OVLP_CART,
                Some(&mut out_cubecl),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        let mut out_libcint = vec![0.0; len];
        vendor_ffi::vendor_int1e_ovlp_cart(&mut out_libcint, &shls_i32, &atm, natm, &bas, nbas, &env);

        for k in 0..len {
            assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-10);
            assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-10);
            assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-10);
        }

        // 2. Kinetic
        let mut out_simd_kin = vec![0.0; len];
        SimdOneElectronKernel::eval_kin::<f64>(&input, &mut out_simd_kin);

        let mut out_cubecl_kin = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::INT1E_KIN_CART,
                Some(&mut out_cubecl_kin),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        let mut out_libcint_kin = vec![0.0; len];
        vendor_ffi::vendor_int1e_kin_cart(&mut out_libcint_kin, &shls_i32, &atm, natm, &bas, nbas, &env);

        for k in 0..len {
            assert_relative_eq!(out_simd_kin[k], out_libcint_kin[k], epsilon = 1e-10);
            assert_relative_eq!(out_cubecl_kin[k], out_libcint_kin[k], epsilon = 1e-10);
            assert_relative_eq!(out_simd_kin[k], out_cubecl_kin[k], epsilon = 1e-10);
        }

        // 3. Nuclear
        let mut out_simd_nuc = vec![0.0; len];
        SimdOneElectronKernel::eval_nuc::<f64>(&input, &mut out_simd_nuc);

        let mut out_cubecl_nuc = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::INT1E_NUC_CART,
                Some(&mut out_cubecl_nuc),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        let mut out_libcint_nuc = vec![0.0; len];
        vendor_ffi::vendor_int1e_nuc_cart(&mut out_libcint_nuc, &shls_i32, &atm, natm, &bas, nbas, &env);

        for k in 0..len {
            assert_relative_eq!(out_simd_nuc[k], out_libcint_nuc[k], epsilon = 1e-9);
            assert_relative_eq!(out_cubecl_nuc[k], out_libcint_nuc[k], epsilon = 1e-9);
            assert_relative_eq!(out_simd_nuc[k], out_cubecl_nuc[k], epsilon = 1e-9);
        }
    }
}

#[test]
fn test_3way_high_l_2c2e_parity() {
    let (atm, bas, env) = build_high_l_fixture();
    let natm = 3;
    let nbas = 6;

    let test_pairs = [
        (4, 0), // g-s
        (4, 2), // g-d
        (4, 4), // g-g
        (5, 0), // h-s
        (5, 3), // h-f
        (5, 5), // h-h
    ];

    for &(si, sk) in &test_pairs {
        let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
        let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
        let len = ncart_l(li as i32) * ncart_l(lk as i32);
        let shls_i32 = [si as i32, sk as i32];

        let mut out_simd = vec![0.0; len];
        let input = Center2c2eInput {
            li,
            lk,
            ri,
            rk,
            exps_i,
            exps_k,
            coeff_i,
            coeff_k,
        };
        SimdCenter2c2eKernel::eval::<f64>(&input, &mut out_simd);

        let mut out_cubecl = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::INT2C2E_CART,
                Some(&mut out_cubecl),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        let mut out_libcint = vec![0.0; len];
        vendor_ffi::vendor_int2c2e_cart(&mut out_libcint, &shls_i32, &atm, natm, &bas, nbas, &env);

        for k in 0..len {
            assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-9);
            assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-9);
            assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-9);
        }
    }
}

#[test]
fn test_3way_high_l_3c1e_and_3c2e_parity() {
    let (atm, bas, env) = build_high_l_fixture();
    let natm = 3;
    let nbas = 6;

    let test_triples = [
        (4, 0, 1), // g-s-p
        (4, 2, 0), // g-d-s
        (4, 4, 0), // g-g-s
        (5, 0, 1), // h-s-p
    ];

    for &(si, sj, sk) in &test_triples {
        let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
        let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
        let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
        let len = ncart_l(li as i32) * ncart_l(lj as i32) * ncart_l(lk as i32);
        let shls_i32 = [si as i32, sj as i32, sk as i32];

        // 3c1e
        let mut out_simd_3c1e = vec![0.0; len];
        let input_3c1e = Center3c1eInput {
            li,
            lj,
            lk,
            ri,
            rj,
            rk,
            exps_i,
            exps_j,
            exps_k,
            coeff_i,
            coeff_j,
            coeff_k,
        };
        SimdCenter3c1eKernel::eval::<f64>(&input_3c1e, &mut out_simd_3c1e);

        let mut out_cubecl_3c1e = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::INT3C1E_CART,
                Some(&mut out_cubecl_3c1e),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        let mut out_libcint_3c1e = vec![0.0; len];
        vendor_ffi::vendor_int3c1e_cart(&mut out_libcint_3c1e, &shls_i32, &atm, natm, &bas, nbas, &env);

        for k in 0..len {
            assert_relative_eq!(out_simd_3c1e[k], out_libcint_3c1e[k], epsilon = 1e-9);
            assert_relative_eq!(out_cubecl_3c1e[k], out_libcint_3c1e[k], epsilon = 1e-9);
            assert_relative_eq!(out_simd_3c1e[k], out_cubecl_3c1e[k], epsilon = 1e-9);
        }

        // 3c2e
        let mut out_simd_3c2e = vec![0.0; len];
        let input_3c2e = Center3c2eInput {
            li,
            lj,
            lk,
            ri,
            rj,
            rk,
            exps_i,
            exps_j,
            exps_k,
            coeff_i,
            coeff_j,
            coeff_k,
        };
        SimdCenter3c2eKernel::eval::<f64>(&input_3c2e, &mut out_simd_3c2e);

        let mut out_cubecl_3c2e = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::INT3C2E_CART,
                Some(&mut out_cubecl_3c2e),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        let mut out_libcint_3c2e = vec![0.0; len];
        vendor_ffi::vendor_int3c2e_cart(&mut out_libcint_3c2e, &shls_i32, &atm, natm, &bas, nbas, &env);

        for k in 0..len {
            assert_relative_eq!(out_simd_3c2e[k], out_libcint_3c2e[k], epsilon = 1e-9);
            assert_relative_eq!(out_cubecl_3c2e[k], out_libcint_3c2e[k], epsilon = 1e-9);
            assert_relative_eq!(out_simd_3c2e[k], out_cubecl_3c2e[k], epsilon = 1e-9);
        }
    }
}

#[test]
fn test_3way_high_l_2e_quartets_parity() {
    let (atm, bas, env) = build_high_l_fixture();
    let natm = 3;
    let nbas = 6;

    let quartets = [
        [4, 0, 0, 0], // g-s-s-s
        [4, 1, 0, 0], // g-p-s-s
        [4, 2, 0, 0], // g-d-s-s (nroots = 4)
        [4, 4, 0, 0], // g-g-s-s (nroots = 5)
        [5, 0, 0, 0], // h-s-s-s
        [5, 1, 0, 0], // h-p-s-s (nroots = 4)
        [5, 5, 0, 0], // h-h-s-s (nroots = 6)
        [4, 2, 2, 0], // g-d-d-s (nroots = 5)
    ];

    for &shls in &quartets {
        let (li, ri, exps_i, coeff_i) = get_shell_data(shls[0], &atm, &bas, &env);
        let (lj, rj, exps_j, coeff_j) = get_shell_data(shls[1], &atm, &bas, &env);
        let (lk, rk, exps_k, coeff_k) = get_shell_data(shls[2], &atm, &bas, &env);
        let (ll, rl, exps_l, coeff_l) = get_shell_data(shls[3], &atm, &bas, &env);
        let len = ncart_l(li as i32) * ncart_l(lj as i32) * ncart_l(lk as i32) * ncart_l(ll as i32);

        // 1. SIMD Kernel
        let mut out_simd = vec![0.0; len];
        let input = TwoElectronInput {
            li,
            lj,
            lk,
            ll,
            ri,
            rj,
            rk,
            rl,
            exps_i,
            exps_j,
            exps_k,
            exps_l,
            coeff_i,
            coeff_j,
            coeff_k,
            coeff_l,
        };
        SimdTwoElectronKernel::eval::<f64>(&input, &mut out_simd);

        // 2. CubeCL Kernel (via eval_raw)
        let mut out_cubecl = vec![0.0; len];
        let shls_i32 = [shls[0] as i32, shls[1] as i32, shls[2] as i32, shls[3] as i32];
        unsafe {
            eval_raw(
                RawApiId::INT2E_CART,
                Some(&mut out_cubecl),
                None,
                &shls_i32,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap();
        }

        // 3. libcint Reference (via vendor_ffi)
        let mut out_libcint = vec![0.0; len];
        vendor_ffi::vendor_int2e_cart(
            &mut out_libcint,
            &shls_i32,
            &atm,
            natm,
            &bas,
            nbas,
            &env,
        );

        for k in 0..len {
            assert_relative_eq!(out_simd[k], out_libcint[k], epsilon = 1e-9);
            assert_relative_eq!(out_cubecl[k], out_libcint[k], epsilon = 1e-9);
            assert_relative_eq!(out_simd[k], out_cubecl[k], epsilon = 1e-9);
        }
    }
}

