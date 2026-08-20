//! Oracle parity test for 3c2e spherical integrals (int3c2e_sph): H2O STO-3G.
//!
//! Validates the end-to-end compute pipeline for `int3c2e_sph` by comparing:
//! - cintx values via `eval_raw` (dispatches through `launch_center_3c2e`)
//! - reference values from vendored libcint 6.1.3 FFI (when enabled)
//!
//! Tolerance: atol 1e-9 for 3c2e per phase research D-06.

// Module gate widened to allow `--features rocm` (without cpu) for the
// Phase 16-04 ROCm oracle suite (D-15). Cpu tests remain unconditional under
// the (cpu OR rocm) gate; rocm tests inside are individually gated
// `#[cfg(feature = "rocm")]` + `#[ignore]` + env-gated.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

/// Build H2O STO-3G libcint-style atm/bas/env arrays.
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

    // 2e-family kernels read libcint global env slots (e.g. PTR_RANGE_OMEGA),
    // so user payload must start at PTR_ENV_START to avoid corrupting them.
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

const N_SHELLS: usize = 5;

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "output length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        if diff > atol {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, diff={diff:.3e}, atol={atol:.1e}"
            );
        }
    }
    mismatches
}

#[test]
fn test_center_3c2e_sph_h2o_sto3g_nonzero() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C2E_IP1_SPH;

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                // Phase 21-06: int3c2e_ip1 is a 3-component derivative (3 * ni*nj*nk).
                let n_elem = 3 * ni * nj * nk;
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];
                let mut out1 = vec![0.0_f64; n_elem];
                let mut out2 = vec![0.0_f64; n_elem];

                unsafe {
                    eval_raw(
                        api_id,
                        Some(&mut out1),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!("eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}")
                    });
                    eval_raw(
                        api_id,
                        Some(&mut out2),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "eval_raw second call failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"
                        )
                    });
                }

                mismatch_count += count_mismatches(&out1, &out2, 1e-15);
                if out1.iter().any(|&v| v.abs() > 1e-18) {
                    any_nonzero = true;
                }
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "int3c2e_sph idempotency failed: {mismatch_count} mismatches"
    );
    assert!(
        any_nonzero,
        "int3c2e_sph output is all zeros - 3c2e kernel stub not replaced"
    );
}

// Phase 21-06 (GRAD-08 / Risk R1): int3c2e_ip1 now ships the REAL ∇_A derivative.
// FLIPPED: the vendor reference is vendor_int3c2e_ip1_sph (the derivative), NOT
// plain vendor_int3c2e_sph; the buffer is 3-component (3 * ni*nj*nk); the tolerance
// is the Phase 15 unified atol=1e-12. The element-for-element comparison validates
// the component-leading [3, nk, nj, ni] F-order (same convention as int2e_ip1).
#[test]
#[cfg(has_vendor_libcint)]
fn test_center_3c2e_sph_h2o_sto3g_vendor_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C2E_IP1_SPH;
    let atol = 1e-12_f64;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                // 3-component derivative output (component-leading): 3 * ni*nj*nk.
                let n_elem = 3 * ni * nj * nk;
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

                let mut vendor_out = vec![0.0_f64; n_elem];
                let mut cintx_out = vec![0.0_f64; n_elem];

                // REAL derivative reference (R1 flip from plain vendor_int3c2e_sph).
                vendor_ffi::vendor_int3c2e_ip1_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                unsafe {
                    eval_raw(
                        api_id,
                        Some(&mut cintx_out),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!("eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}")
                    });
                }

                if vendor_out.iter().any(|&v| v.abs() > 1e-18)
                    || cintx_out.iter().any(|&v| v.abs() > 1e-18)
                {
                    any_nonzero = true;
                }

                mismatch_count += count_mismatches(&vendor_out, &cintx_out, atol);
            }
        }
    }

    assert!(any_nonzero, "int3c2e_ip1_sph outputs are all zeros");
    assert_eq!(
        mismatch_count, 0,
        "int3c2e_ip1_sph vendor parity failed: {mismatch_count} elements exceed atol=1e-12 \
         vs the REAL vendor_int3c2e_ip1_sph derivative"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROCm oracle parity test (Phase 16-04 / D-15)
//
// Idempotency check at atol=1e-12 / rtol=1e-10 across all 5^3 shell triples
// for `int3c2e_ip1_sph` under the rocm backend (mirroring the cpu sibling
// test's `RawApiId::INT3C2E_IP1_SPH` choice — Phase 10 STATE.md decision).
// Gated `#[cfg(feature = "rocm")] + #[ignore] + CINTX_ROCM_ORACLE=1` env-gate.
// ─────────────────────────────────────────────────────────────────────────────

/// int3c2e_ip1_sph H2O STO-3G ROCm oracle parity (atol=1e-12 / rtol=1e-10).
#[cfg(feature = "rocm")]
#[test]
#[ignore]
fn test_int3c2e_sph_h2o_sto3g_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C2E_IP1_SPH;
    let atol = 1e-12_f64;
    let rtol = 1e-10_f64;

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut triple_count = 0usize;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                triple_count += 1;
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                // Phase 21-06: int3c2e_ip1 is a 3-component derivative (3 * ni*nj*nk).
                let n_elem = 3 * ni * nj * nk;
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];
                let mut out1 = vec![0.0_f64; n_elem];
                let mut out2 = vec![0.0_f64; n_elem];

                unsafe {
                    eval_raw(
                        api_id,
                        Some(&mut out1),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| {
                        panic!("rocm eval_raw failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}")
                    });
                    eval_raw(
                        api_id,
                        Some(&mut out2),
                        None,
                        &shls,
                        &atm,
                        &bas,
                        &env,
                        None,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("rocm eval_raw second call failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"));
                }

                // Inline abs+rel tolerance check (count_mismatches in this file is abs-only).
                for (idx, (&r, &o)) in out1.iter().zip(out2.iter()).enumerate() {
                    let diff = (o - r).abs();
                    let threshold = atol + rtol * r.abs();
                    if diff > threshold {
                        mismatch_count += 1;
                        eprintln!(
                            "  rocm MISMATCH shells ({i_sh},{j_sh},{k_sh}) idx {idx}: \
                             ref={r:.15e}, obs={o:.15e}, diff={diff:.3e}, threshold={threshold:.3e}"
                        );
                    }
                }
            }
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "rocm oracle parity failed: {mismatch_count} mismatches in int3c2e_ip1_sph across {triple_count} triples"
    );
    println!(
        "  PASS: rocm int3c2e_ip1_sph mismatch_count=0 across {triple_count} triples at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// RANDOM ROCm oracle idempotency (Phase 260529-exs).
//
// Exercises the int3c2e_ip1_sph #[cube(launch)] device kernel on the rocm backend
// across 64 random 3-shell systems (random li,lj,lk ∈ {0,1,2}, nprim ∈ {1..3},
// random exps/coeffs/coords). The on-device run is the orchestrator's post-merge
// job; here it compiles and is collected under `--list`.
//
// Gated `#[cfg(feature = "rocm")]` + `#[ignore]` + `CINTX_ROCM_ORACLE=1`,
// identically to the 3c1e/2c2e siblings. Trigger via `xtask rocm-oracle`.
// ─────────────────────────────────────────────────────────────────────────────

/// Tiny deterministic LCG (Numerical Recipes constants) — keeps the random suite
/// reproducible without an external rng crate.
#[cfg(feature = "rocm")]
struct Lcg(u64);

#[cfg(feature = "rocm")]
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform f64 in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + frac * (hi - lo)
    }
    /// Uniform integer in [lo, hi] inclusive.
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i32
    }
}

/// Build a random 3-shell int3c2e_ip1 system. Returns `(atm, bas, env, li, lj, lk)`.
///
/// Angular momenta are drawn from {0,1,2}, then redrawn until the int3c2e_ip1
/// ELEVATED Rys-root count `(li+1 + lj + 0 + lk)/2 + 1` is `<= 5` (the device
/// `rys_root1..5` ceiling — the `li→li+1` raise is the gradient headroom).
/// Primitive counts are {1,2,3}; the three shells sit on three distinct atoms at
/// random, distinct coordinates. Uses the libcint env-pointer layout.
#[cfg(feature = "rocm")]
fn build_random_3shell_3c2e(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>, i32, i32, i32) {
    // Redraw l-triples whose elevated nroots exceeds the rys_root1..5 ceiling.
    let (li, lj, lk) = loop {
        let li = rng.range_i32(0, 2);
        let lj = rng.range_i32(0, 2);
        let lk = rng.range_i32(0, 2);
        let elevated_nroots = ((li + 1) + lj + 0 + lk) / 2 + 1;
        if elevated_nroots <= 5 {
            break (li, lj, lk);
        }
    };
    let nprim_i = rng.range_i32(1, 3) as usize;
    let nprim_j = rng.range_i32(1, 3) as usize;
    let nprim_k = rng.range_i32(1, 3) as usize;

    let coord_i = [
        rng.uniform(-1.5, 1.5),
        rng.uniform(-1.5, 1.5),
        rng.uniform(-1.5, 1.5),
    ];
    // Offset shells j and k so the three centers never coincide.
    let coord_j = [
        coord_i[0] + rng.uniform(0.8, 2.5),
        coord_i[1] + rng.uniform(-1.5, 1.5),
        coord_i[2] + rng.uniform(-1.5, 1.5),
    ];
    let coord_k = [
        coord_i[0] + rng.uniform(-2.5, -0.8),
        coord_i[1] + rng.uniform(-1.5, 1.5),
        coord_i[2] + rng.uniform(0.8, 2.5),
    ];

    let exps_i: Vec<f64> = (0..nprim_i).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_i: Vec<f64> = (0..nprim_i).map(|_| rng.uniform(0.15, 1.0)).collect();
    let exps_j: Vec<f64> = (0..nprim_j).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_j: Vec<f64> = (0..nprim_j).map(|_| rng.uniform(0.15, 1.0)).collect();
    let exps_k: Vec<f64> = (0..nprim_k).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_k: Vec<f64> = (0..nprim_k).map(|_| rng.uniform(0.15, 1.0)).collect();

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let coord_i_ptr = env.len() as i32;
    env.extend_from_slice(&coord_i);
    let coord_j_ptr = env.len() as i32;
    env.extend_from_slice(&coord_j);
    let coord_k_ptr = env.len() as i32;
    env.extend_from_slice(&coord_k);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let exp_i_ptr = env.len() as i32;
    env.extend_from_slice(&exps_i);
    let coeff_i_ptr = env.len() as i32;
    env.extend_from_slice(&coeff_i);
    let exp_j_ptr = env.len() as i32;
    env.extend_from_slice(&exps_j);
    let coeff_j_ptr = env.len() as i32;
    env.extend_from_slice(&coeff_j);
    let exp_k_ptr = env.len() as i32;
    env.extend_from_slice(&exps_k);
    let coeff_k_ptr = env.len() as i32;
    env.extend_from_slice(&coeff_k);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[CHARGE_OF] = 1;
    atm[PTR_COORD] = coord_i_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = coord_j_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = coord_k_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = li;
    bas[NPRIM_OF] = nprim_i as i32;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = exp_i_ptr;
    bas[PTR_COEFF] = coeff_i_ptr;
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = lj;
    bas[BAS_SLOTS + NPRIM_OF] = nprim_j as i32;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = exp_j_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = coeff_j_ptr;
    bas[2 * BAS_SLOTS + ATOM_OF] = 2;
    bas[2 * BAS_SLOTS + ANG_OF] = lk;
    bas[2 * BAS_SLOTS + NPRIM_OF] = nprim_k as i32;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = exp_k_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = coeff_k_ptr;

    (atm, bas, env, li, lj, lk)
}

/// int3c2e_ip1_sph RANDOM idempotency on the ROCm backend (atol=1e-12 / rtol=1e-10).
#[cfg(feature = "rocm")]
#[test]
#[ignore]
fn test_int3c2e_ip1_sph_random_rocm_idempotency() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let atol = 1e-12_f64;
    let rtol = 1e-10_f64;
    let n_cases = 64usize;
    let mut rng = Lcg::new(0x5ec0_3c2e_1234_5678);

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for case in 0..n_cases {
        let (atm, bas, env, li, lj, lk) = build_random_3shell_3c2e(&mut rng);
        let ni = nsph_for_l(li);
        let nj = nsph_for_l(lj);
        let nk = nsph_for_l(lk);
        // int3c2e_ip1 is a 3-component (component-leading) derivative.
        let n_elem = 3 * ni * nj * nk;
        let mut out1 = vec![0.0_f64; n_elem];
        let mut out2 = vec![0.0_f64; n_elem];
        let shls = [0_i32, 1, 2];

        unsafe {
            eval_raw(
                RawApiId::INT3C2E_IP1_SPH,
                Some(&mut out1),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("rocm random eval_raw failed for case {case}: {e:?}"));
            eval_raw(
                RawApiId::INT3C2E_IP1_SPH,
                Some(&mut out2),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap_or_else(|e| {
                panic!("rocm random eval_raw second call failed for case {case}: {e:?}")
            });
        }

        assert_eq!(
            out1.len(),
            out2.len(),
            "case {case}: output length mismatch (li={li}, lj={lj}, lk={lk})"
        );
        for (idx, (&r, &o)) in out1.iter().zip(out2.iter()).enumerate() {
            let diff = (o - r).abs();
            let threshold = atol + rtol * r.abs();
            if diff > threshold {
                mismatch_count += 1;
                eprintln!(
                    "  rocm RANDOM MISMATCH case {case} (li={li},lj={lj},lk={lk}) idx {idx}: \
                     ref={r:.15e}, obs={o:.15e}, diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
        if out1.iter().any(|&v| v.abs() > 1e-18) {
            any_nonzero = true;
        }
    }

    assert_eq!(
        mismatch_count, 0,
        "rocm random oracle idempotency failed: {mismatch_count} mismatches across {n_cases} cases"
    );
    assert!(
        any_nonzero,
        "rocm random int3c2e_ip1_sph output is all zeros across {n_cases} cases — device kernel not running"
    );
    println!(
        "  PASS: rocm random int3c2e_ip1_sph idempotency mismatch_count=0 across {n_cases} cases \
         at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}
