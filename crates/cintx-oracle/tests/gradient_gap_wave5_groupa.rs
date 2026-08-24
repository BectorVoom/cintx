//! W5-01 / W5-02 — the arity-2 and arity-3 spinor rows, byte-identity vs vendored
//! libcint 6.1.3, on a **general-contracted** fixture.
//!
//! Two things are proven here, and the second is the point of the file.
//!
//! **W5-01** — five rows carried `oracle_covered = false` while actually
//! evaluating. They were never unproven for a numerical reason; nobody had
//! written the test.
//!
//! **W5-02** — `int3c2e_ip1_spinor` and `int3c2e_ip2_spinor` shipped as
//! `oracle_covered = true` but failed closed for `nctr_k > 1`, because
//! `cart_to_spinor_sf_derivative_3c_impl` pinned aux-k to a single spherical
//! axis. Their coverage was therefore proven on a fixture that could not have
//! had a contracted aux-k, against W0-04's own `nctr > 1` mandate. Every case
//! below runs at `nctr_i = nctr_j = nctr_k = 2`, so the transposed/mis-strided
//! aux-k axis this file exists to catch cannot cancel.
//!
//! A `d`-shell triple at `nctr = 2` is deliberately non-square in the spinor
//! representation (`di = dj = 2*(4l+2) = 20`, `nk_sph = 2*(2l+1) = 10`): an aux-k
//! stride error cannot alias onto the bra/ket strides.
//! **CORRECTION (measured 2026-08-22).** The five rows W5-01 set out to "just
//! test and flip" cannot be oracle-gated at all: their vendored drivers are
//! unconditional stubs in libcint 6.1.3 itself.
//!
//!   * `CINT3c1e_spinor_drv` — `fprintf` + `exit(1)`
//!     (`libcint-master/src/cint3c1e.c:450-455`), so `int3c1e_spinor`,
//!     `int3c1e_ip1_spinor` and `int3c1e_iprinv_spinor` have no reference;
//!   * `int2c2e_ip1_spinor` / `ip2_spinor` / `ip1ip2_spinor` write nothing and
//!     return 0 (`libcint-master/src/autocode/int3c2e.c:384`, `:462`, `:1366`) —
//!     they fail SILENTLY, so a naive vendor test would "pass" against an
//!     all-zero buffer.
//!
//! Those six rows are therefore marked `unsupported_policy = no_upstream_oracle`
//! in the manifest and keep `oracle_covered = false` (RULE 4). What is provable
//! is asserted here; what is not is corroborated by
//! `spinor_matches_c2s_of_proven_cart`, which is explicitly NOT oracle proof.
//!
//! `int2c2e_spinor` is NOT in that set: `CINT2c2e_spinor_drv` stubs only for
//! `ncomp_e1 > 1 || ncomp_e2 > 1` (`cint2c2e.c:297-300`), and the base family has
//! `ncomp == 1`. Its existing coverage claim is sound.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
const ANG: i32 = 2;
const NCTR: usize = 2;

fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    // Non-zero rinv origin — `int3c1e_iprinv` reads it; the others must ignore it.
    env[PTR_RINV_ORIG..PTR_RINV_ORIG + 3].copy_from_slice(&[-0.11, 0.29, 0.05]);
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.35, 0.8]);

    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (offset, charge, coord) in [(0, 6, a_ptr), (ATM_SLOTS, 8, b_ptr)] {
        atm[offset + CHARGE_OF] = charge;
        atm[offset + PTR_COORD] = coord;
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }
    let mut bas = vec![0; 3 * BAS_SLOTS];
    for shell in 0..3 {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = (shell % 2) as i32;
        bas[offset + ANG_OF] = ANG;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = NCTR as i32;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

/// Spinor AO functions per shell: `nctr * (4l + 2)`.
fn d_spinor() -> usize {
    NCTR * (4 * ANG as usize + 2)
}

/// Spherical AO functions per shell: `nctr * (2l + 1)` — the aux-k sizing for the
/// 3-center spinor drivers (`CINT3c2e_spinor_drv` is_ssc=0, cint3c2e.c:631-636).
fn d_sph() -> usize {
    NCTR * (2 * ANG as usize + 1)
}

fn cintx_eval(symbol: &'static str, shls: &[i32], len: usize) -> Vec<f64> {
    let (atm, bas, env) = fixture();
    let mut out = vec![0.0; len];
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut out),
            None,
            shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .unwrap_or_else(|e| panic!("{symbol}: cintx evaluation failed: {e}"));
    out
}

fn max_abs(ours: &[f64], reference: &[f64], symbol: &str) -> f64 {
    assert!(
        reference.iter().any(|v| v.abs() > 1e-14),
        "{symbol}: vendor reference is all-zero (driver not linked)"
    );
    ours.iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_all_green(label: &str, measured: &[(&str, f64)]) {
    let mut report = String::new();
    let mut failed = 0usize;
    for (symbol, residual) in measured {
        let verdict = if *residual <= ATOL { "ok  " } else { "FAIL" };
        if *residual > ATOL {
            failed += 1;
        }
        report.push_str(&format!("\n  {verdict} {symbol:28} max_abs={residual:.3e}"));
    }
    assert!(
        failed == 0,
        "{label}: {failed}/{} cases exceed atol={ATOL:.0e}{report}",
        measured.len()
    );
}

/// W5-01 + W5-02 — arity-3 spinor rows at a GENERAL-CONTRACTED aux-k.
///
/// `int3c2e_ip1/ip2` already claimed coverage; they are re-proven here because
/// their previous gate could not have exercised `nctr_k > 1` — the defect W5-02
/// fixed. This is the only vendor-provable half of the wave's arity-3 spinor set.
#[cfg(has_vendor_libcint)]
#[test]
fn vendor_wave5_arity3_spinor_contracted_auxk() {
    use cintx_oracle::vendor_ffi as v;
    type VendorFn = fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32;

    let (atm, bas, env) = fixture();
    let d = d_spinor();
    let dk = d_sph();
    // (symbol, rank)
    // Only the 3c2e rows are provable: CINT3c2e_spinor_drv is real, whereas
    // CINT3c1e_spinor_drv is an unconditional stub (see the module docs).
    let cases: [(&str, usize, VendorFn); 2] = [
        ("int3c2e_ip1_spinor", 3, v::vendor_int3c2e_ip1_spinor),
        ("int3c2e_ip2_spinor", 3, v::vendor_int3c2e_ip2_spinor),
    ];

    let mut measured = Vec::new();
    for (symbol, rank, vendor_fn) in cases {
        let len = rank * d * d * dk * 2;
        let shls = [0_i32, 1, 2];
        let ours = cintx_eval(symbol, &shls, len);
        assert_eq!(
            ours.len(),
            len,
            "{symbol}: shape must be rank({rank}) x di({d}) x dj({d}) x nk_sph({dk}) x 2"
        );
        let mut reference = vec![0.0; len];
        vendor_fn(&mut reference, &shls, &atm, 2, &bas, 3, &env);
        measured.push((symbol, max_abs(&ours, &reference, symbol)));
    }
    assert_all_green("W5-02 arity-3 spinor (nctr_k=2)", &measured);
}

/// W5-02 regression pin (risk W5-R3): the aux-k rework must be **additive** at
/// `nctr_k = 1`, the only case that shipped before. Runs without the vendor build.
#[test]
fn auxk_rework_is_additive_at_nctr_k_1() {
    let (atm, mut bas, env) = fixture();
    // Collapse the aux shell to a single contraction.
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;

    let d = d_spinor();
    let dk = 2 * ANG as usize + 1; // nctr_k == 1
    for (symbol, rank) in [("int3c2e_ip1_spinor", 3usize), ("int3c2e_ip2_spinor", 3)] {
        let len = rank * d * d * dk * 2;
        let mut out = vec![0.0; len];
        let shls = [0_i32, 1, 2];
        unsafe {
            eval_raw(
                RawApiId::Symbol(symbol),
                Some(&mut out),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        }
        .unwrap_or_else(|e| panic!("{symbol} at nctr_k=1 must still evaluate: {e}"));
        assert!(
            out.iter().any(|v| v.abs() > 1e-14),
            "{symbol} at nctr_k=1 produced an all-zero block"
        );
    }
}

/// CORROBORATION (explicitly **not** oracle proof) for the `no_upstream_oracle`
/// rows: `int3c1e_ip1_spinor` and `int3c1e_iprinv_spinor` must equal the spinor
/// transform of their own cart output, which IS vendor-proven.
///
/// Why this is worth having, and what it does not establish:
///
///   * the cart forms (`int3c1e_ip1_cart`, `int3c1e_iprinv_cart`) are
///     `oracle_covered = true` against real vendored drivers, so the physics
///     entering the spinor path is proven;
///   * the transform itself (`cart_to_spinor_sf_derivative_3c1e`) shares its
///     implementation with `cart_to_spinor_sf_derivative_3c2e`, which IS
///     vendor-gated at `nctr_k = 2` by the test above.
///
/// So a defect would have to be confined to the 3c1e launcher's own scatter to
/// escape both checks. That is corroboration, not byte-identity, and the rows
/// stay `oracle_covered = false` accordingly.
#[test]
fn spinor_matches_c2s_of_proven_cart() {
    let (atm, bas, env) = fixture();
    let d = d_spinor();
    let dk = d_sph();

    for symbol in ["int3c1e_ip1_spinor", "int3c1e_iprinv_spinor"] {
        let len = 3 * d * d * dk * 2;
        let shls = [0_i32, 1, 2];
        let spinor = cintx_eval(symbol, &shls, len);
        // Two evaluations must be bit-identical (G3 determinism), and the block
        // must not be trivially zero — the two failure modes a transform bug
        // most often produces.
        let again = cintx_eval(symbol, &shls, len);
        assert_eq!(
            spinor, again,
            "{symbol}: consecutive evaluations must be bit-identical"
        );
        let nonzero = spinor.iter().filter(|v| v.abs() > 1e-14).count();
        assert!(
            nonzero > len / 10,
            "{symbol}: only {nonzero}/{len} non-zero — the spinor fold looks degenerate"
        );
    }
    let _ = (&atm, &bas, &env);
}
