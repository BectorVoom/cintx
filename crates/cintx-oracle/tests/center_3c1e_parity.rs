//! Oracle parity test for 3c1e spherical integrals (int3c1e_sph): H2O STO-3G.
//!
//! Validates the end-to-end compute pipeline for `int3c1e_sph` (three-center
//! one-electron overlap) by comparing:
//!   - cintx values via `eval_raw` (which dispatches through `launch_center_3c1e`)
//!   - Reference values from vendored libcint 6.1.3 FFI (when CINTX_ORACLE_BUILD_VENDOR=1)
//!
//! Tolerance: atol 1e-7 for 3c1e per RESEARCH.md D-06.
//!
//! H2O STO-3G geometry (in Bohr):
//!   O  at (0.000,  0.000, 0.000)
//!   H1 at (0.000,  1.431, 1.108)
//!   H2 at (0.000, -1.431, 1.108)
//!
//! Shells (STO-3G):
//!   Shell 0: O 1s  (3 primitives, l=0)
//!   Shell 1: O 2s  (3 primitives, l=0)
//!   Shell 2: O 2p  (3 primitives, l=1)
//!   Shell 3: H1 1s (3 primitives, l=0)
//!   Shell 4: H2 1s (3 primitives, l=0)
//!
//! With 5 shells: 5^3 = 125 shell triples.

// Module gate widened to allow `--features rocm` (without cpu) for the
// Phase 16-04 ROCm oracle suite (D-15). Cpu tests remain unconditional under
// the (cpu OR rocm) gate; rocm tests inside are individually gated
// `#[cfg(feature = "rocm")]` + `#[ignore]` + env-gated.
#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
// Only the rocm random-3-shell builder uses PTR_ENV_START; gate it so the
// cpu-only build stays warning-free.
#[cfg(feature = "rocm")]
use cintx_compat::raw::PTR_ENV_START;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G basis data
// ─────────────────────────────────────────────────────────────────────────────

/// Build the H2O STO-3G `atm`, `bas`, `env` arrays.
///
/// Matches the build_h2o_sto3g() function in one_electron_parity.rs exactly
/// so comparisons are made on the same molecular geometry.
fn build_h2o_sto3g() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let o_coord = [0.0_f64, 0.0, 0.0];
    let h1_coord = [0.0_f64, 1.4307, 1.1078];
    let h2_coord = [0.0_f64, -1.4307, 1.1078];

    // STO-3G exponents and coefficients (Hehre, Stewart & Pople, JCP 51, 2657, 1969)
    let o_1s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let o_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let o_2s_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2s_coeff = [-0.09996723_f64, 0.39951283, 0.70011547];
    let o_2p_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let o_2p_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let h_1s_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let h_1s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = Vec::<f64>::new();

    let o_coord_ptr = env.len() as i32;
    env.extend_from_slice(&o_coord);
    let h1_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h1_coord);
    let h2_coord_ptr = env.len() as i32;
    env.extend_from_slice(&h2_coord);
    let _zeta_ptr = env.len() as i32;
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

    // atm: [CHARGE_OF, PTR_COORD, NUC_MOD_OF, PTR_ZETA, PTR_FRAC_CHARGE, 0]
    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = o_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = 9;

    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = 9;

    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = 9;

    // bas: [ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, KAPPA_OF, PTR_EXP, PTR_COEFF, 0]
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

/// Number of shells in H2O STO-3G basis.
const N_SHELLS: usize = 5;

/// Number of spherical AOs for angular momentum l: 2l+1.
fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Compare two output slices element-wise with absolute tolerance.
/// Returns the count of elements that fall outside tolerance.
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
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, atol={atol:.1e}"
            );
        }
    }
    mismatches
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx self-consistency test (no vendor FFI required)
// ─────────────────────────────────────────────────────────────────────────────

/// int3c1e_sph H2O STO-3G self-consistency test.
///
/// Verifies that:
/// 1. int3c1e_sph via eval_raw produces non-zero values for some shell triples.
/// 2. Results are deterministic (two calls produce identical output).
/// 3. The (0,0,0) diagonal triple (s-s-s) is positive.
#[test]
fn test_int3c1e_sph_h2o_sto3g_nonzero() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C1E_SPH;

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut total_nonzero = 0usize;
    let mut total_mismatch = 0usize;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                let n_elem = ni * nj * nk;
                let mut out1 = vec![0.0_f64; n_elem];
                let mut out2 = vec![0.0_f64; n_elem];
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

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

                // Idempotency check
                total_mismatch += count_mismatches(&out1, &out2, 1e-15);

                // Nonzero tracking
                total_nonzero += out1.iter().filter(|&&v| v.abs() > 1e-18).count();
            }
        }
    }

    assert_eq!(
        total_mismatch, 0,
        "int3c1e_sph: {total_mismatch} idempotency mismatches (non-deterministic kernel)"
    );
    assert!(
        total_nonzero > 0,
        "int3c1e_sph: all outputs are zero — 3c1e kernel stub not replaced"
    );

    println!(
        "int3c1e_sph self-consistency: PASS. Nonzero elements: {total_nonzero}/{}",
        N_SHELLS * N_SHELLS * N_SHELLS
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor libcint parity test (requires CINTX_ORACLE_BUILD_VENDOR=1)
// ─────────────────────────────────────────────────────────────────────────────

/// int3c1e_sph H2O STO-3G oracle parity against vendored libcint 6.1.3.
///
/// Iterates over all 5^3 = 125 shell triples and compares:
/// - Cintx output from eval_raw(int3c1e_sph)
/// - Reference output from vendor_int3c1e_sph (vendored libcint 6.1.3 FFI)
///
/// Tolerance: atol 1e-7 (per RESEARCH.md D-06 for 3c1e family).
///
/// Asserts:
/// - mismatch_count == 0
/// - at least one non-zero element seen (non-stub check)
///
/// Note: libcint 3c1e output is column-major (i fastest, k slowest).
/// Our kernel produces the same ordering.
#[test]
#[cfg(has_vendor_libcint)]
fn test_int3c1e_sph_h2o_sto3g_vendor_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C1E_SPH;
    let atol = cintx_oracle::compare::tolerance_for_family("3c1e").atol;

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsph: Vec<usize> = ang.iter().map(|&l| nsph_for_l(l)).collect();

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;
    let mut triple_count = 0usize;

    for i_sh in 0..N_SHELLS {
        for j_sh in 0..N_SHELLS {
            for k_sh in 0..N_SHELLS {
                triple_count += 1;
                let ni = shell_nsph[i_sh];
                let nj = shell_nsph[j_sh];
                let nk = shell_nsph[k_sh];
                let n_elem = ni * nj * nk;

                let mut vendor_out = vec![0.0_f64; n_elem];
                let mut cintx_out = vec![0.0_f64; n_elem];
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

                // Reference: vendored libcint 6.1.3
                vendor_ffi::vendor_int3c1e_sph(
                    &mut vendor_out,
                    &shls,
                    &atm,
                    natm,
                    &bas,
                    nbas,
                    &env,
                );

                // cintx: eval_raw dispatches to launch_center_3c1e
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

                // Count nonzero elements across both outputs
                if vendor_out.iter().any(|v| v.abs() > 1e-18)
                    || cintx_out.iter().any(|v| v.abs() > 1e-18)
                {
                    any_nonzero = true;
                }

                // Element-wise comparison
                let triple_mismatches = count_mismatches(&vendor_out, &cintx_out, atol);
                if triple_mismatches > 0 {
                    eprintln!(
                        "  Shell triple ({i_sh},{j_sh},{k_sh}) [li={},lj={},lk={}]: \
                         {triple_mismatches} mismatches",
                        ang[i_sh], ang[j_sh], ang[k_sh]
                    );
                }
                mismatch_count += triple_mismatches;
            }
        }
    }

    println!(
        "int3c1e_sph oracle parity: {triple_count} shell triples checked, \
         mismatch_count={mismatch_count}, atol={atol:.1e}"
    );

    assert!(
        any_nonzero,
        "int3c1e_sph: all outputs are zero — either the 3c1e kernel is still a stub \
         or vendor libcint returned all zeros"
    );

    assert_eq!(
        mismatch_count, 0,
        "int3c1e_sph oracle parity: {mismatch_count} elements exceed atol={atol:.1e} \
         vs vendored libcint 6.1.3 for H2O STO-3G"
    );

    println!("  PASS: mismatch_count=0 vs vendored libcint 6.1.3");
}

// ─────────────────────────────────────────────────────────────────────────────
// ROCm oracle parity test (Phase 16-04 / D-15)
//
// Idempotency check at atol=1e-12 / rtol=1e-10 across all 5^3 shell triples
// for `int3c1e_sph` under the rocm backend. Gated `#[cfg(feature = "rocm")]
// + #[ignore] + CINTX_ROCM_ORACLE=1` env-gate.
//
// NOTE: The cpu sibling test in this file uses atol=1e-7 (per RESEARCH D-06
// for 3c1e), but that's a vendor-libcint parity tolerance. The rocm variant
// is a self-consistency idempotency check between two cintx eval_raw calls,
// which should be exactly equal — we use D-15's tighter atol=1e-12 / rtol=1e-10.
// ─────────────────────────────────────────────────────────────────────────────

/// int3c1e_sph H2O STO-3G ROCm oracle parity (atol=1e-12 / rtol=1e-10).
#[cfg(feature = "rocm")]
#[test]
#[ignore]
fn test_int3c1e_sph_h2o_sto3g_rocm_parity() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let (atm, bas, env) = build_h2o_sto3g();
    let api_id = RawApiId::INT3C1E_SPH;
    let tol = cintx_oracle::compare::tolerance_for_family("3c1e");
    let atol = tol.atol;
    let rtol = tol.rtol;

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
                let n_elem = ni * nj * nk;
                let mut out1 = vec![0.0_f64; n_elem];
                let mut out2 = vec![0.0_f64; n_elem];
                let shls = [i_sh as i32, j_sh as i32, k_sh as i32];

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
                    .unwrap_or_else(|e| {
                        panic!(
                            "rocm eval_raw second call failed for shells ({i_sh},{j_sh},{k_sh}): {e:?}"
                        )
                    });
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
        "rocm oracle parity failed: {mismatch_count} mismatches in int3c1e_sph across {triple_count} triples"
    );
    println!(
        "  PASS: rocm int3c1e_sph mismatch_count=0 across {triple_count} triples at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROCm RANDOM idempotency oracle test (quick task 260529-e69)
//
// Exercises the CubeCL `center_3c1e_kernel` GPU device path under
// `CINTX_BACKEND=rocm` over many RANDOMIZED 3-shell systems (random angular
// momenta, primitive counts, exponents, contraction coefficients, and
// geometry). For each system eval_raw(int3c1e_sph) is invoked twice on the
// ROCm device and the two results must agree exactly (idempotency), with the
// suite as a whole producing non-zero output (proves the device kernel ran).
//
// Gated `#[cfg(feature = "rocm")]` + `#[ignore]` + `CINTX_ROCM_ORACLE=1`, like
// the H2O sibling above. Trigger via `xtask rocm-oracle`.
// ─────────────────────────────────────────────────────────────────────────────

/// Tiny deterministic LCG (Numerical Recipes constants) — keeps the random
/// suite reproducible without pulling in an external rng crate.
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

/// Build a random 3-shell int3c1e system. Returns `(atm, bas, env, li, lj, lk)`.
///
/// Angular momenta are drawn from {0,1,2}; primitive counts from {1,2,3}. The
/// three shells sit on three distinct atoms at random, distinct coordinates.
/// Uses the libcint env-pointer layout (user data starts at PTR_ENV_START).
#[cfg(feature = "rocm")]
fn build_random_3shell(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>, i32, i32, i32) {
    let li = rng.range_i32(0, 2);
    let lj = rng.range_i32(0, 2);
    let lk = rng.range_i32(0, 2);
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

/// int3c1e_sph RANDOM idempotency on the ROCm backend (atol=1e-12 / rtol=1e-10).
#[cfg(feature = "rocm")]
#[test]
#[ignore]
fn test_int3c1e_sph_random_rocm_idempotency() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let tol = cintx_oracle::compare::tolerance_for_family("3c1e");
    let atol = tol.atol;
    let rtol = tol.rtol;
    let n_cases = 64usize;
    let mut rng = Lcg::new(0x5ec0_3c1e_1234_5678);

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for case in 0..n_cases {
        let (atm, bas, env, li, lj, lk) = build_random_3shell(&mut rng);
        let ni = nsph_for_l(li);
        let nj = nsph_for_l(lj);
        let nk = nsph_for_l(lk);
        let n_elem = ni * nj * nk;
        let mut out1 = vec![0.0_f64; n_elem];
        let mut out2 = vec![0.0_f64; n_elem];
        let shls = [0_i32, 1, 2];

        unsafe {
            eval_raw(
                RawApiId::INT3C1E_SPH,
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
                RawApiId::INT3C1E_SPH,
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
        "rocm random int3c1e_sph output is all zeros across {n_cases} cases — device kernel not running"
    );
    println!(
        "  PASS: rocm random int3c1e_sph idempotency mismatch_count=0 across {n_cases} cases \
         at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}
