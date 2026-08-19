//! Performance Benchmark: SIMD-Kernel vs. CubeCL-Kernel vs. libcint
//!
//! Measures evaluation throughput and latency across all three engines in release mode:
//! 1. `simd-kernel` (wide + rmath f64x4 SIMD vectorization)
//! 2. `cubecl-kernel` (CubeCL CPU compute engine via cintx-compat raw API)
//! 3. `libcint` (Upstream C library 6.1.3 compiled with -O3)

#![cfg(feature = "cpu")]
#![cfg(has_vendor_libcint)]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_oracle::vendor_ffi;
use cintx_simd::{
    AtomCoord, Center2c2eInput, Center3c1eInput, Center3c2eInput, OneElectronInput,
    SimdCenter2c2eKernel, SimdCenter3c1eKernel, SimdCenter3c2eKernel, SimdOneElectronKernel,
    SimdTwoElectronKernel, TwoElectronInput,
};
use std::hint::black_box;
use std::time::Instant;
use wide::f64x4;

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

    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;

    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;

    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

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

struct BenchResult {
    family: &'static str,
    iterations: usize,
    simd_duration_ns: f64,
    cubecl_duration_ns: f64,
    libcint_duration_ns: f64,
}

impl BenchResult {
    fn print_summary(&self) {
        let simd_per_op_us = self.simd_duration_ns / (self.iterations as f64) / 1000.0;
        let cubecl_per_op_us = self.cubecl_duration_ns / (self.iterations as f64) / 1000.0;
        let libcint_per_op_us = self.libcint_duration_ns / (self.iterations as f64) / 1000.0;

        let simd_thru = (self.iterations as f64) / (self.simd_duration_ns * 1e-9);
        let cubecl_thru = (self.iterations as f64) / (self.cubecl_duration_ns * 1e-9);
        let libcint_thru = (self.iterations as f64) / (self.libcint_duration_ns * 1e-9);

        let speedup_vs_libcint = libcint_per_op_us / simd_per_op_us;
        let speedup_vs_cubecl = cubecl_per_op_us / simd_per_op_us;

        println!(
            "| {:<16} | {:>9.3} µs ({:>9.1} k/s) | {:>9.3} µs ({:>9.1} k/s) | {:>9.3} µs ({:>9.1} k/s) | {:>7.2}x | {:>7.2}x |",
            self.family,
            simd_per_op_us,
            simd_thru / 1000.0,
            cubecl_per_op_us,
            cubecl_thru / 1000.0,
            libcint_per_op_us,
            libcint_thru / 1000.0,
            speedup_vs_libcint,
            speedup_vs_cubecl,
        );
    }
}

#[test]
fn test_benchmark_speed_all_families() {
    let (atm, bas, env) = build_h2o_sto3g();
    let atoms = get_atoms(&atm, &env);
    let nbas = 5;

    println!("\n========================================================================================================================");
    println!("                                   3-WAY PERFORMANCE BENCHMARK (RELEASE MODE)                                           ");
    println!("========================================================================================================================");
    println!("| Integral Family  | SIMD (wide f64x4)           | CubeCL CPU Backend          | libcint (C Reference)       | vs libcint| vs CubeCL |");
    println!("|------------------|-----------------------------|-----------------------------|-----------------------------|-----------|-----------|");

    // 1. Overlap (int1e_ovlp_cart)
    {
        let repeats = 2000;
        let total_evals = repeats * nbas * nbas;
        let mut out = [0.0; 9];

        // Warmup
        for _ in 0..100 {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                    let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
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
                    SimdOneElectronKernel::eval_ovlp::<f64x4>(&input, &mut out);
                }
            }
        }

        // Benchmark SIMD
        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                    let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
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
                    SimdOneElectronKernel::eval_ovlp::<f64x4>(&input, &mut out);
                    black_box(&out);
                }
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        // Benchmark CubeCL
        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let shls = [si as i32, sj as i32];
                    unsafe {
                        eval_raw(
                            RawApiId::INT1E_OVLP_CART,
                            Some(&mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])]),
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
                    black_box(&out);
                }
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        // Benchmark libcint
        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let shls = [si as i32, sj as i32];
                    vendor_ffi::vendor_int1e_ovlp_cart(
                        &mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])],
                        &shls,
                        &atm,
                        3,
                        &bas,
                        5,
                        &env,
                    );
                    black_box(&out);
                }
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "1e_ovlp",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    // 2. Kinetic (int1e_kin_cart)
    {
        let repeats = 2000;
        let total_evals = repeats * nbas * nbas;
        let mut out = [0.0; 9];

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                    let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
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
                    SimdOneElectronKernel::eval_kin::<f64x4>(&input, &mut out);
                    black_box(&out);
                }
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let shls = [si as i32, sj as i32];
                    unsafe {
                        eval_raw(
                            RawApiId::INT1E_KIN_CART,
                            Some(&mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])]),
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
                    black_box(&out);
                }
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let shls = [si as i32, sj as i32];
                    vendor_ffi::vendor_int1e_kin_cart(
                        &mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])],
                        &shls,
                        &atm,
                        3,
                        &bas,
                        5,
                        &env,
                    );
                    black_box(&out);
                }
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "1e_kin",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    // 3. Nuclear (int1e_nuc_cart)
    {
        let repeats = 1000;
        let total_evals = repeats * nbas * nbas;
        let mut out = [0.0; 9];

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                    let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
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
                    SimdOneElectronKernel::eval_nuc::<f64x4>(&input, &mut out);
                    black_box(&out);
                }
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let shls = [si as i32, sj as i32];
                    unsafe {
                        eval_raw(
                            RawApiId::INT1E_NUC_CART,
                            Some(&mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])]),
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
                    black_box(&out);
                }
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    let shls = [si as i32, sj as i32];
                    vendor_ffi::vendor_int1e_nuc_cart(
                        &mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])],
                        &shls,
                        &atm,
                        3,
                        &bas,
                        5,
                        &env,
                    );
                    black_box(&out);
                }
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "1e_nuc",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    // 4. 2c2e (int2c2e_cart)
    {
        let repeats = 2000;
        let total_evals = repeats * nbas * nbas;
        let mut out = [0.0; 9];

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sk in 0..nbas {
                    let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                    let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
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
                    SimdCenter2c2eKernel::eval::<f64x4>(&input, &mut out);
                    black_box(&out);
                }
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sk in 0..nbas {
                    let shls = [si as i32, sk as i32];
                    unsafe {
                        eval_raw(
                            RawApiId::INT2C2E_CART,
                            Some(&mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sk * BAS_SLOTS + ANG_OF])]),
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
                    black_box(&out);
                }
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sk in 0..nbas {
                    let shls = [si as i32, sk as i32];
                    vendor_ffi::vendor_int2c2e_cart(
                        &mut out[..ncart_l(bas[si * BAS_SLOTS + ANG_OF]) * ncart_l(bas[sk * BAS_SLOTS + ANG_OF])],
                        &shls,
                        &atm,
                        3,
                        &bas,
                        5,
                        &env,
                    );
                    black_box(&out);
                }
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "2c2e",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    // 5. 3c1e (int3c1e_cart)
    {
        let repeats = 500;
        let total_evals = repeats * nbas * nbas * nbas;
        let mut out = [0.0; 27];

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    for sk in 0..nbas {
                        let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                        let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
                        let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
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
                        SimdCenter3c1eKernel::eval::<f64x4>(&input, &mut out);
                        black_box(&out);
                    }
                }
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    for sk in 0..nbas {
                        let shls = [si as i32, sj as i32, sk as i32];
                        let len = ncart_l(bas[si * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sk * BAS_SLOTS + ANG_OF]);
                        unsafe {
                            eval_raw(
                                RawApiId::INT3C1E_CART,
                                Some(&mut out[..len]),
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
                        black_box(&out);
                    }
                }
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    for sk in 0..nbas {
                        let shls = [si as i32, sj as i32, sk as i32];
                        let len = ncart_l(bas[si * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sk * BAS_SLOTS + ANG_OF]);
                        vendor_ffi::vendor_int3c1e_cart(
                            &mut out[..len],
                            &shls,
                            &atm,
                            3,
                            &bas,
                            5,
                            &env,
                        );
                        black_box(&out);
                    }
                }
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "3c1e",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    // 6. 3c2e (int3c2e_cart)
    {
        let repeats = 500;
        let total_evals = repeats * nbas * nbas * nbas;
        let mut out = [0.0; 27];

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    for sk in 0..nbas {
                        let (li, ri, exps_i, coeff_i) = get_shell_data(si, &atm, &bas, &env);
                        let (lj, rj, exps_j, coeff_j) = get_shell_data(sj, &atm, &bas, &env);
                        let (lk, rk, exps_k, coeff_k) = get_shell_data(sk, &atm, &bas, &env);
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
                        SimdCenter3c2eKernel::eval::<f64x4>(&input, &mut out);
                        black_box(&out);
                    }
                }
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    for sk in 0..nbas {
                        let shls = [si as i32, sj as i32, sk as i32];
                        let len = ncart_l(bas[si * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sk * BAS_SLOTS + ANG_OF]);
                        unsafe {
                            eval_raw(
                                RawApiId::INT3C2E_CART,
                                Some(&mut out[..len]),
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
                        black_box(&out);
                    }
                }
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for si in 0..nbas {
                for sj in 0..nbas {
                    for sk in 0..nbas {
                        let shls = [si as i32, sj as i32, sk as i32];
                        let len = ncart_l(bas[si * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sj * BAS_SLOTS + ANG_OF])
                            * ncart_l(bas[sk * BAS_SLOTS + ANG_OF]);
                        vendor_ffi::vendor_int3c2e_cart(
                            &mut out[..len],
                            &shls,
                            &atm,
                            3,
                            &bas,
                            5,
                            &env,
                        );
                        black_box(&out);
                    }
                }
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "3c2e",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    // 7. 2e (int2e_cart)
    {
        let repeats = 100;
        let quartets = [
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 2, 0, 0],
            [1, 2, 3, 4],
            [2, 2, 0, 0],
            [2, 2, 2, 2],
        ];
        let total_evals = repeats * quartets.len();
        let mut out = [0.0; 81];

        let t0 = Instant::now();
        for _ in 0..repeats {
            for &shls in &quartets {
                let (li, ri, exps_i, coeff_i) = get_shell_data(shls[0], &atm, &bas, &env);
                let (lj, rj, exps_j, coeff_j) = get_shell_data(shls[1], &atm, &bas, &env);
                let (lk, rk, exps_k, coeff_k) = get_shell_data(shls[2], &atm, &bas, &env);
                let (ll, rl, exps_l, coeff_l) = get_shell_data(shls[3], &atm, &bas, &env);
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
                SimdTwoElectronKernel::eval::<f64x4>(&input, &mut out);
                black_box(&out);
            }
        }
        let simd_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for &shls in &quartets {
                let shls_i32 = [shls[0] as i32, shls[1] as i32, shls[2] as i32, shls[3] as i32];
                let len = ncart_l(bas[shls[0] * BAS_SLOTS + ANG_OF])
                    * ncart_l(bas[shls[1] * BAS_SLOTS + ANG_OF])
                    * ncart_l(bas[shls[2] * BAS_SLOTS + ANG_OF])
                    * ncart_l(bas[shls[3] * BAS_SLOTS + ANG_OF]);
                unsafe {
                    eval_raw(
                        RawApiId::INT2E_CART,
                        Some(&mut out[..len]),
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
                black_box(&out);
            }
        }
        let cubecl_dur = t0.elapsed().as_nanos() as f64;

        let t0 = Instant::now();
        for _ in 0..repeats {
            for &shls in &quartets {
                let shls_i32 = [shls[0] as i32, shls[1] as i32, shls[2] as i32, shls[3] as i32];
                let len = ncart_l(bas[shls[0] * BAS_SLOTS + ANG_OF])
                    * ncart_l(bas[shls[1] * BAS_SLOTS + ANG_OF])
                    * ncart_l(bas[shls[2] * BAS_SLOTS + ANG_OF])
                    * ncart_l(bas[shls[3] * BAS_SLOTS + ANG_OF]);
                vendor_ffi::vendor_int2e_cart(
                    &mut out[..len],
                    &shls_i32,
                    &atm,
                    3,
                    &bas,
                    5,
                    &env,
                );
                black_box(&out);
            }
        }
        let libcint_dur = t0.elapsed().as_nanos() as f64;

        let res = BenchResult {
            family: "2e (ERIs)",
            iterations: total_evals,
            simd_duration_ns: simd_dur,
            cubecl_duration_ns: cubecl_dur,
            libcint_duration_ns: libcint_dur,
        };
        res.print_summary();
    }

    println!("========================================================================================================================\n");
}
