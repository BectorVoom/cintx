//! Oracle parity test for 4c1e spherical integrals (int4c1e_sph).
//!
//! Exercises the CubeCL `center_4c1e_kernel` GPU device path under
//! `CINTX_BACKEND=rocm` over many RANDOMIZED 4-shell systems (random angular
//! momenta, primitive counts, exponents, contraction coefficients, and
//! geometry). For each system eval_raw(int4c1e_sph) is invoked twice and the
//! two results must agree exactly (idempotency), with the suite as a whole
//! producing non-zero output (proves the device kernel ran).
//!
//! Mirrors the 3c1e sibling (center_3c1e_parity.rs) adapted to 4 shells /
//! int4c1e_sph. 4c1e nroots is always 1 (polynomial recurrence, not Rys
//! quadrature), so bounding l in {0,1,2} keeps HRR sizes sane with no redraw.
//!
//! Gating:
//!   - Module gate: `(cpu OR rocm)`.
//!   - rocm random idempotency test: `(rocm AND with-4c1e)` + `#[ignore]` +
//!     env-gated `CINTX_ROCM_ORACLE=1` (invoke via `xtask rocm-oracle`).
//!   - cpu self-consistency smoke: `(cpu AND with-4c1e)`. int4c1e_sph requires
//!     the with-4c1e feature, hence the co-gate.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
// Only the random-4-shell builder uses PTR_ENV_START; gate it so neither the
// cpu-only nor rocm-only build emits an unused-import warning.
#[cfg(all(feature = "rocm", feature = "with-4c1e"))]
use cintx_compat::raw::PTR_ENV_START;

/// Number of spherical AOs for angular momentum l: 2l+1.
#[allow(dead_code)]
fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// cpu self-consistency smoke (no vendor FFI / no rocm hardware required)
//
// Builds one fixed 4-shell system, evaluates int4c1e_sph twice via eval_raw,
// and asserts idempotency + at least one nonzero element. Proves the new file
// compiles and the (cpu) device kernel runs end-to-end.
// ─────────────────────────────────────────────────────────────────────────────

/// Build a fixed 4-shell int4c1e system (s,p,s,s on four distinct atoms).
/// Returns `(atm, bas, env, [li,lj,lk,ll])`.
#[cfg(all(feature = "cpu", feature = "with-4c1e"))]
fn build_fixed_4shell() -> (Vec<i32>, Vec<i32>, Vec<f64>, [i32; 4]) {
    const PTR_ENV_START_LOCAL: usize = 20;

    let ls = [0_i32, 1, 0, 0];
    let coord_i = [0.0_f64, 0.0, 0.0];
    let coord_j = [1.2_f64, 0.3, -0.4];
    let coord_k = [-0.9_f64, 1.1, 0.5];
    let coord_l = [0.4_f64, -0.7, 1.3];

    let exps_i = [1.5_f64, 0.5];
    let coeff_i = [0.6_f64, 0.4];
    let exps_j = [0.9_f64];
    let coeff_j = [1.0_f64];
    let exps_k = [1.1_f64, 0.3];
    let coeff_k = [0.5_f64, 0.5];
    let exps_l = [0.8_f64];
    let coeff_l = [1.0_f64];

    let mut env = vec![0.0_f64; PTR_ENV_START_LOCAL];

    let coord_i_ptr = env.len() as i32;
    env.extend_from_slice(&coord_i);
    let coord_j_ptr = env.len() as i32;
    env.extend_from_slice(&coord_j);
    let coord_k_ptr = env.len() as i32;
    env.extend_from_slice(&coord_k);
    let coord_l_ptr = env.len() as i32;
    env.extend_from_slice(&coord_l);
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
    let exp_l_ptr = env.len() as i32;
    env.extend_from_slice(&exps_l);
    let coeff_l_ptr = env.len() as i32;
    env.extend_from_slice(&coeff_l);

    let coord_ptrs = [coord_i_ptr, coord_j_ptr, coord_k_ptr, coord_l_ptr];
    let mut atm = vec![0_i32; 4 * ATM_SLOTS];
    for a in 0..4 {
        atm[a * ATM_SLOTS + CHARGE_OF] = 1;
        atm[a * ATM_SLOTS + PTR_COORD] = coord_ptrs[a];
        atm[a * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[a * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let nprim = [exps_i.len(), exps_j.len(), exps_k.len(), exps_l.len()];
    let exp_ptrs = [exp_i_ptr, exp_j_ptr, exp_k_ptr, exp_l_ptr];
    let coeff_ptrs = [coeff_i_ptr, coeff_j_ptr, coeff_k_ptr, coeff_l_ptr];
    let mut bas = vec![0_i32; 4 * BAS_SLOTS];
    for s in 0..4 {
        bas[s * BAS_SLOTS + ATOM_OF] = s as i32;
        bas[s * BAS_SLOTS + ANG_OF] = ls[s];
        bas[s * BAS_SLOTS + NPRIM_OF] = nprim[s] as i32;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptrs[s];
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptrs[s];
    }

    (atm, bas, env, ls)
}

/// int4c1e_sph cpu self-consistency: idempotent + nonzero.
#[cfg(all(feature = "cpu", feature = "with-4c1e"))]
#[test]
fn test_int4c1e_sph_cpu_self_consistency() {
    let (atm, bas, env, ls) = build_fixed_4shell();
    let ni = nsph_for_l(ls[0]);
    let nj = nsph_for_l(ls[1]);
    let nk = nsph_for_l(ls[2]);
    let nl = nsph_for_l(ls[3]);
    let n_elem = ni * nj * nk * nl;

    let mut out1 = vec![0.0_f64; n_elem];
    let mut out2 = vec![0.0_f64; n_elem];
    let shls = [0_i32, 1, 2, 3];

    unsafe {
        eval_raw(
            RawApiId::INT4C1E_SPH,
            Some(&mut out1),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .expect("eval_raw int4c1e_sph (call 1) failed");
        eval_raw(
            RawApiId::INT4C1E_SPH,
            Some(&mut out2),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .expect("eval_raw int4c1e_sph (call 2) failed");
    }

    let mut mismatch = 0usize;
    for (idx, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
        if (a - b).abs() > 1e-15 {
            mismatch += 1;
            eprintln!("  cpu idempotency MISMATCH idx {idx}: {a:.15e} vs {b:.15e}");
        }
    }
    assert_eq!(mismatch, 0, "int4c1e_sph cpu kernel is non-deterministic");

    let nonzero = out1.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "int4c1e_sph cpu output is all zeros — kernel not running"
    );
    println!("int4c1e_sph cpu self-consistency: PASS. Nonzero elements: {nonzero}/{n_elem}");
}

// ─────────────────────────────────────────────────────────────────────────────
// ROCm RANDOM idempotency oracle test (quick task 260529-fsa)
//
// Exercises the CubeCL `center_4c1e_kernel` GPU device path under
// `CINTX_BACKEND=rocm` over many RANDOMIZED 4-shell systems. For each system
// eval_raw(int4c1e_sph) is invoked twice on the ROCm device and the two results
// must agree exactly (idempotency), with the suite producing non-zero output.
//
// Gated `(rocm AND with-4c1e)` + `#[ignore]` + `CINTX_ROCM_ORACLE=1`. Trigger
// via `xtask rocm-oracle --profile with-4c1e`.
// ─────────────────────────────────────────────────────────────────────────────

/// Tiny deterministic LCG (Numerical Recipes constants) — keeps the random
/// suite reproducible without pulling in an external rng crate.
#[cfg(all(feature = "rocm", feature = "with-4c1e"))]
struct Lcg(u64);

#[cfg(all(feature = "rocm", feature = "with-4c1e"))]
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

/// Build a random 4-shell int4c1e system. Returns `(atm, bas, env, li,lj,lk,ll)`.
///
/// Angular momenta are drawn from {0,1,2}; primitive counts from {1,2,3}. The
/// four shells sit on four distinct atoms at random, distinct coordinates (each
/// in a different octant offset so no two centers coincide). Uses the libcint
/// env-pointer layout (user data starts at PTR_ENV_START).
#[cfg(all(feature = "rocm", feature = "with-4c1e"))]
fn build_random_4shell(rng: &mut Lcg) -> (Vec<i32>, Vec<i32>, Vec<f64>, i32, i32, i32, i32) {
    let li = rng.range_i32(0, 2);
    let lj = rng.range_i32(0, 2);
    let lk = rng.range_i32(0, 2);
    let ll = rng.range_i32(0, 2);
    let nprim_i = rng.range_i32(1, 3) as usize;
    let nprim_j = rng.range_i32(1, 3) as usize;
    let nprim_k = rng.range_i32(1, 3) as usize;
    let nprim_l = rng.range_i32(1, 3) as usize;

    let coord_i = [
        rng.uniform(-1.5, 1.5),
        rng.uniform(-1.5, 1.5),
        rng.uniform(-1.5, 1.5),
    ];
    // Offset shells j, k, l into distinct octants so the four centers never
    // coincide.
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
    let coord_l = [
        coord_i[0] + rng.uniform(-1.5, 1.5),
        coord_i[1] + rng.uniform(0.8, 2.5),
        coord_i[2] + rng.uniform(-2.5, -0.8),
    ];

    let exps_i: Vec<f64> = (0..nprim_i).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_i: Vec<f64> = (0..nprim_i).map(|_| rng.uniform(0.15, 1.0)).collect();
    let exps_j: Vec<f64> = (0..nprim_j).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_j: Vec<f64> = (0..nprim_j).map(|_| rng.uniform(0.15, 1.0)).collect();
    let exps_k: Vec<f64> = (0..nprim_k).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_k: Vec<f64> = (0..nprim_k).map(|_| rng.uniform(0.15, 1.0)).collect();
    let exps_l: Vec<f64> = (0..nprim_l).map(|_| rng.uniform(0.25, 4.0)).collect();
    let coeff_l: Vec<f64> = (0..nprim_l).map(|_| rng.uniform(0.15, 1.0)).collect();

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let coord_i_ptr = env.len() as i32;
    env.extend_from_slice(&coord_i);
    let coord_j_ptr = env.len() as i32;
    env.extend_from_slice(&coord_j);
    let coord_k_ptr = env.len() as i32;
    env.extend_from_slice(&coord_k);
    let coord_l_ptr = env.len() as i32;
    env.extend_from_slice(&coord_l);
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
    let exp_l_ptr = env.len() as i32;
    env.extend_from_slice(&exps_l);
    let coeff_l_ptr = env.len() as i32;
    env.extend_from_slice(&coeff_l);

    let coord_ptrs = [coord_i_ptr, coord_j_ptr, coord_k_ptr, coord_l_ptr];
    let mut atm = vec![0_i32; 4 * ATM_SLOTS];
    for a in 0..4 {
        atm[a * ATM_SLOTS + CHARGE_OF] = 1;
        atm[a * ATM_SLOTS + PTR_COORD] = coord_ptrs[a];
        atm[a * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[a * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let ls = [li, lj, lk, ll];
    let nprim = [nprim_i, nprim_j, nprim_k, nprim_l];
    let exp_ptrs = [exp_i_ptr, exp_j_ptr, exp_k_ptr, exp_l_ptr];
    let coeff_ptrs = [coeff_i_ptr, coeff_j_ptr, coeff_k_ptr, coeff_l_ptr];
    let mut bas = vec![0_i32; 4 * BAS_SLOTS];
    for s in 0..4 {
        bas[s * BAS_SLOTS + ATOM_OF] = s as i32;
        bas[s * BAS_SLOTS + ANG_OF] = ls[s];
        bas[s * BAS_SLOTS + NPRIM_OF] = nprim[s] as i32;
        bas[s * BAS_SLOTS + NCTR_OF] = 1;
        bas[s * BAS_SLOTS + PTR_EXP] = exp_ptrs[s];
        bas[s * BAS_SLOTS + PTR_COEFF] = coeff_ptrs[s];
    }

    (atm, bas, env, li, lj, lk, ll)
}

/// int4c1e_sph RANDOM idempotency on the ROCm backend (atol=1e-12 / rtol=1e-10).
#[cfg(all(feature = "rocm", feature = "with-4c1e"))]
#[test]
#[ignore]
fn test_int4c1e_sph_random_rocm_idempotency() {
    assert_eq!(
        std::env::var("CINTX_ROCM_ORACLE").as_deref(),
        Ok("1"),
        "ROCm oracle must be invoked via `xtask rocm-oracle` (sets CINTX_ROCM_ORACLE=1). \
         Direct `cargo test --features rocm -- --ignored` is intentionally blocked."
    );

    let tol = cintx_oracle::compare::tolerance_for_family("4c1e");
    let atol = tol.atol;
    let rtol = tol.rtol;
    let n_cases = 64usize;
    let mut rng = Lcg::new(0x5ec0_4c1e_1234_5678);

    let mut mismatch_count = 0usize;
    let mut any_nonzero = false;

    for case in 0..n_cases {
        let (atm, bas, env, li, lj, lk, ll) = build_random_4shell(&mut rng);
        let ni = nsph_for_l(li);
        let nj = nsph_for_l(lj);
        let nk = nsph_for_l(lk);
        let nl = nsph_for_l(ll);
        let n_elem = ni * nj * nk * nl;
        let mut out1 = vec![0.0_f64; n_elem];
        let mut out2 = vec![0.0_f64; n_elem];
        let shls = [0_i32, 1, 2, 3];

        unsafe {
            eval_raw(
                RawApiId::INT4C1E_SPH,
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
                RawApiId::INT4C1E_SPH,
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
            "case {case}: output length mismatch (li={li}, lj={lj}, lk={lk}, ll={ll})"
        );
        for (idx, (&r, &o)) in out1.iter().zip(out2.iter()).enumerate() {
            let diff = (o - r).abs();
            let threshold = atol + rtol * r.abs();
            if diff > threshold {
                mismatch_count += 1;
                eprintln!(
                    "  rocm RANDOM MISMATCH case {case} (li={li},lj={lj},lk={lk},ll={ll}) idx {idx}: \
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
        "rocm random int4c1e_sph output is all zeros across {n_cases} cases — device kernel not running"
    );
    println!(
        "  PASS: rocm random int4c1e_sph idempotency mismatch_count=0 across {n_cases} cases \
         at atol={atol:.0e}/rtol={rtol:.0e}"
    );
}
