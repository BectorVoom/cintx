//! Oracle parity gate for F12/STG/YP kernels — Phase 13 Plan 04.
//!
//! All 10 with-f12 sph symbols now have full oracle parity tests against
//! vendored libcint 6.1.3 at atol=1e-12. This closes F12-03.
//!
//! Gate summary:
//!   Symbol                   | Test mode      | Tolerance
//!   -------------------------|----------------|----------
//!   int2e_stg_sph            | Oracle parity  | atol 1e-12
//!   int2e_yp_sph             | Oracle parity  | atol 1e-12
//!   int2e_stg_ip1_sph        | Oracle parity  | atol 1e-12
//!   int2e_stg_ipip1_sph      | Oracle parity  | atol 1e-12
//!   int2e_stg_ipvip1_sph     | Oracle parity  | atol 1e-12
//!   int2e_stg_ip1ip2_sph     | Oracle parity  | atol 1e-12
//!   int2e_yp_ip1_sph         | Oracle parity  | atol 1e-12
//!   int2e_yp_ipip1_sph       | Oracle parity  | atol 1e-12
//!   int2e_yp_ipvip1_sph      | Oracle parity  | atol 1e-12
//!   int2e_yp_ip1ip2_sph      | Oracle parity  | atol 1e-12
//!
//! Additional tests:
//!   - zeta=0 produces InvalidEnvParam for all 10 symbols
//!   - All 10 manifest entries have representation: sph-only (no cart, no spinor)
//!
//! Requirements: #[cfg(feature = "cpu")] + #[cfg(has_vendor_libcint)] for oracle tests
//! Run: CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu,with-f12 -p cintx-oracle -- f12_oracle_parity

#![cfg(feature = "cpu")]
#![cfg(feature = "with-f12")]

use cintx_compat::raw::{ANG_OF, BAS_SLOTS, RawApiId, eval_raw};
#[cfg(has_vendor_libcint)]
use cintx_compat::raw::ATM_SLOTS;
use cintx_oracle::fixtures::{build_h2o_sto3g_f12};

/// Absolute tolerance for F12 base operator oracle parity (per D-10, F12-03).
const ATOL_F12: f64 = 1e-12;

/// STO-3G H2O has 5 shells: 0=O-1s, 1=O-2s, 2=O-2p, 3=H1-1s, 4=H2-1s.
/// Use shells [0, 1, 0, 1]: O-1s / O-2s / O-1s / O-2s — all-s shell quartet.
/// This exercises the s-type integral path (li=lj=lk=ll=0).
const SHLS_4_SS: [i32; 4] = [0, 1, 0, 1];

/// Shell quartet [3, 4, 3, 4]: H1-1s / H2-1s / H1-1s / H2-1s — also all-s.
/// Tests three-center separation (different atom positions).
const SHLS_4_HH: [i32; 4] = [3, 4, 3, 4];

/// Shell quartet [0, 2, 0, 2]: O-1s / O-2p / O-1s / O-2p — mixed s/p angular momentum.
/// Tests derivative operators on non-trivial angular momentum (O-2p has l=1).
#[cfg(has_vendor_libcint)]
const SHLS_4_SP: [i32; 4] = [0, 2, 0, 2];

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Compute the number of output elements for a 4-shell sph integral (component_count=1).
fn n_sph_elements(shls: &[i32; 4], bas: &[i32]) -> usize {
    shls.iter()
        .map(|&s| nsph_for_l(bas[s as usize * BAS_SLOTS + ANG_OF]))
        .product()
}

/// Evaluate a 4-center integral via cintx eval_raw and return the output buffer.
///
/// Uses ncomp=1 (single-component). Panics if eval_raw returns an error.
fn eval_f12_sph(
    symbol: &'static str,
    shls: &[i32; 4],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    eval_f12_sph_ncomp(symbol, shls, atm, bas, env, 1)
}

/// Evaluate a 4-center integral via cintx eval_raw with multi-component output.
///
/// The output buffer is sized `ncomp * n_sph_elements(shls, bas)`.
/// Panics if eval_raw returns an error.
fn eval_f12_sph_ncomp(
    symbol: &'static str,
    shls: &[i32; 4],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ncomp: usize,
) -> Vec<f64> {
    let n = ncomp * n_sph_elements(shls, bas);
    let mut out = vec![0.0_f64; n];
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut out),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("eval_raw {symbol} failed for shls {shls:?}: {e:?}"));
    }
    out
}

// ────────���────────────────────────────────��───────────────────────────────────
// Oracle parity tests (base variants only, vs vendored libcint)
// ─────────────────────────────────────────────────────��───────────────────────

#[cfg(has_vendor_libcint)]
fn count_mismatches_atol(reference: &[f64], observed: &[f64], atol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "output length mismatch");
    let mut count = 0usize;
    for (i, (r, o)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (o - r).abs();
        if diff > atol {
            eprintln!(
                "  MISMATCH[{i}] ref={r:.15e} obs={o:.15e} diff={diff:.3e} atol={atol:.1e}"
            );
            count += 1;
        }
    }
    count
}

/// Oracle parity gate for int2e_stg_sph — base STG operator.
///
/// Requires CINTX_ORACLE_BUILD_VENDOR=1 to link vendored libcint.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_stg_sph() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    for shls in [SHLS_4_SS, SHLS_4_HH] {
        let n = n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph("int2e_stg_sph", &shls, &atm, &bas, &env);

        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_stg_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);

        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(
            mc, 0,
            "int2e_stg_sph parity FAIL: {mc} mismatches for shls {shls:?} at atol={ATOL_F12:.1e}"
        );
        println!("  PASS: int2e_stg_sph shls {shls:?}: mismatch_count=0, n={n}");
    }
}

/// Oracle parity gate for int2e_yp_sph — base YP (Yukawa potential) operator.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_yp_sph() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    for shls in [SHLS_4_SS, SHLS_4_HH] {
        let n = n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph("int2e_yp_sph", &shls, &atm, &bas, &env);

        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_yp_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);

        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(
            mc, 0,
            "int2e_yp_sph parity FAIL: {mc} mismatches for shls {shls:?} at atol={ATOL_F12:.1e}"
        );
        println!("  PASS: int2e_yp_sph shls {shls:?}: mismatch_count=0, n={n}");
    }
}

// ───────────────────────────────────────────────────��───────────────────��─────
// Non-zero output assertion tests for all 10 F12 symbols
// ──────────────────────────���─────────────────────────────────��────────────────

/// Assert base STG integral produces non-zero output (smoke test, no vendor required).
#[test]
fn f12_stg_base_nonzero() {
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let out = eval_f12_sph("int2e_stg_sph", &SHLS_4_SS, &atm, &bas, &env);
    let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(nonzero > 0, "int2e_stg_sph output is all zeros — kernel not computing");
}

/// Assert base YP integral produces non-zero output.
#[test]
fn f12_yp_base_nonzero() {
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let out = eval_f12_sph("int2e_yp_sph", &SHLS_4_SS, &atm, &bas, &env);
    let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(nonzero > 0, "int2e_yp_sph output is all zeros — kernel not computing");
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle parity tests for derivative F12 variants (ip1: ncomp=3, rest: ncomp=9)
//
// These replace the previous idempotency-only tests and provide real comparison
// against vendored libcint 6.1.3. F12-03 is fully satisfied by these tests.
// ─────────────────────────────────────────────────────────────────────────────

/// Oracle parity gate for int2e_stg_ip1_sph — STG gradient on electron 1.
///
/// ip1 variant: ncomp=3 (d/dx, d/dy, d/dz of STG on center 1).
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_stg_ip1_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 3;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_stg_ip1_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_stg_ip1_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_stg_ip1_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_stg_ip1_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_stg_ipip1_sph — STG second gradient (i,i) on electron 1.
///
/// ipip1 variant: ncomp=9.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_stg_ipip1_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 9;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_stg_ipip1_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_stg_ipip1_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_stg_ipip1_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_stg_ipip1_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_stg_ipvip1_sph — STG cross gradient (i,j) on electron 1.
///
/// ipvip1 variant: ncomp=9.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_stg_ipvip1_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 9;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_stg_ipvip1_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_stg_ipvip1_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_stg_ipvip1_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_stg_ipvip1_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_stg_ip1ip2_sph — STG gradient on e1 and e2.
///
/// ip1ip2 variant: ncomp=9.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_stg_ip1ip2_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 9;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_stg_ip1ip2_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_stg_ip1ip2_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_stg_ip1ip2_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_stg_ip1ip2_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_yp_ip1_sph — YP gradient on electron 1.
///
/// ip1 variant: ncomp=3.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_yp_ip1_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 3;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_yp_ip1_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_yp_ip1_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_yp_ip1_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_yp_ip1_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_yp_ipip1_sph — YP second gradient (i,i) on electron 1.
///
/// ipip1 variant: ncomp=9.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_yp_ipip1_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 9;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_yp_ipip1_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_yp_ipip1_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_yp_ipip1_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_yp_ipip1_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_yp_ipvip1_sph — YP cross gradient (i,j) on electron 1.
///
/// ipvip1 variant: ncomp=9.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_yp_ipvip1_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 9;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_yp_ipvip1_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_yp_ipvip1_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_yp_ipvip1_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_yp_ipvip1_sph shls {shls:?}: n={n}");
    }
}

/// Oracle parity gate for int2e_yp_ip1ip2_sph — YP gradient on e1 and e2.
///
/// ip1ip2 variant: ncomp=9.
#[test]
#[cfg(has_vendor_libcint)]
fn oracle_parity_int2e_yp_ip1ip2_sph() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_f12(1.2);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let ncomp = 9;
    for shls in [SHLS_4_SS, SHLS_4_HH, SHLS_4_SP] {
        let n = ncomp * n_sph_elements(&shls, &bas);
        let cintx_out = eval_f12_sph_ncomp("int2e_yp_ip1ip2_sph", &shls, &atm, &bas, &env, ncomp);
        let mut vendor_out = vec![0.0_f64; n];
        vendor_ffi::vendor_int2e_yp_ip1ip2_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
        let mc = count_mismatches_atol(&vendor_out, &cintx_out, ATOL_F12);
        assert_eq!(mc, 0, "int2e_yp_ip1ip2_sph parity FAIL: {mc} mismatches for shls {shls:?}");
        println!("  PASS: int2e_yp_ip1ip2_sph shls {shls:?}: n={n}");
    }
}

// ───────────────────────���─────────────────────────���───────────────────────────
// zeta=0 rejection test (D-01, F12-05): all 10 symbols must return InvalidEnvParam
// ───────────────────���────────────────────────────���────────────────────────────

/// All 10 F12 symbols must reject zeta=0 with InvalidEnvParam before any computation.
/// This guards against silent fallback to plain Coulomb (D-01).
#[test]
fn f12_zeta_zero_rejected_all_10() {
    use cintx_core::cintxRsError;

    let f12_symbols = [
        "int2e_stg_sph",
        "int2e_stg_ip1_sph",
        "int2e_stg_ipip1_sph",
        "int2e_stg_ipvip1_sph",
        "int2e_stg_ip1ip2_sph",
        "int2e_yp_sph",
        "int2e_yp_ip1_sph",
        "int2e_yp_ipip1_sph",
        "int2e_yp_ipvip1_sph",
        "int2e_yp_ip1ip2_sph",
    ];

    let (atm, bas, env_zero) = build_h2o_sto3g_f12(0.0); // zeta=0

    for symbol in f12_symbols {
        // Allocate 9x the 1-component size to accommodate any derivative variant (max ncomp=9).
        // This ensures BufferTooSmall does not mask the InvalidEnvParam zeta gate.
        let n = n_sph_elements(&SHLS_4_SS, &bas);
        let mut out = vec![0.0_f64; (9 * n).max(1)];
        let result = unsafe {
            eval_raw(
                RawApiId::Symbol(symbol),
                Some(&mut out),
                None,
                &SHLS_4_SS,
                &atm,
                &bas,
                &env_zero,
                None,
                None,
            )
        };
        assert!(
            result.is_err(),
            "zeta=0 must be rejected for {symbol}, but got Ok"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, cintxRsError::InvalidEnvParam { param, .. } if param == "PTR_F12_ZETA"),
            "expected InvalidEnvParam(PTR_F12_ZETA) for {symbol} with zeta=0, got: {err:?}"
        );
        println!("  PASS: {symbol} correctly rejects zeta=0 with InvalidEnvParam");
    }
}

/// Backward-compat: the original single-symbol zeta rejection test from the plan.
#[test]
fn f12_zeta_zero_rejected() {
    let (atm, bas, env) = build_h2o_sto3g_f12(0.0);
    let shls = SHLS_4_SS;
    let n = n_sph_elements(&shls, &bas);
    let mut out = vec![0.0_f64; n.max(1)];
    let result = unsafe {
        eval_raw(
            RawApiId::Symbol("int2e_stg_sph"),
            Some(&mut out),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    };
    assert!(result.is_err(), "zeta=0 must be rejected for F12 symbols");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("PTR_F12_ZETA"),
        "error must mention PTR_F12_ZETA: {err_msg}"
    );
}

// ────────────────���────────────────────────────────────────────────────���───────
// sph-only enforcement: confirm no cart/spinor F12 symbols in manifest
// ───────────────────────────────────────────���──────────────────────────────��──

/// Confirms that all 10 F12 manifest entries have sph=true, cart=false, spinor=false.
///
/// This enforces D-10: the with-f12 profile has sph-only operators.
#[test]
fn f12_sph_only_enforcement() {
    use cintx_ops::resolver::{HelperKind, Resolver};

    let f12_sph_symbols: Vec<&str> = vec![
        "int2e_stg_sph",
        "int2e_stg_ip1_sph",
        "int2e_stg_ipip1_sph",
        "int2e_stg_ipvip1_sph",
        "int2e_stg_ip1ip2_sph",
        "int2e_yp_sph",
        "int2e_yp_ip1_sph",
        "int2e_yp_ipip1_sph",
        "int2e_yp_ipvip1_sph",
        "int2e_yp_ip1ip2_sph",
    ];

    let manifest = Resolver::manifest();
    let f12_entries: Vec<_> = manifest
        .iter()
        .filter(|entry| {
            matches!(entry.helper_kind, HelperKind::Operator)
                && (entry.operator_name.starts_with("stg")
                    || entry.operator_name.starts_with("yp"))
        })
        .collect();

    // Must have exactly 10 F12 operator entries.
    assert_eq!(
        f12_entries.len(),
        10,
        "expected 10 F12 operator entries, got {}",
        f12_entries.len()
    );

    // Each F12 entry must be sph-only.
    for entry in &f12_entries {
        assert!(
            entry.representation.spheric,
            "{} must have spheric=true",
            entry.symbol_name
        );
        assert!(
            !entry.representation.cart,
            "{} must have cart=false",
            entry.symbol_name
        );
        assert!(
            !entry.representation.spinor,
            "{} must have spinor=false",
            entry.symbol_name
        );
    }

    // Each expected symbol must be present.
    for symbol in &f12_sph_symbols {
        assert!(
            f12_entries.iter().any(|e| e.symbol_name == *symbol),
            "manifest missing expected F12 sph symbol: {symbol}"
        );
    }

    // Verify no cart or spinor F12 symbols exist.
    let cart_f12_count = manifest
        .iter()
        .filter(|e| {
            matches!(e.helper_kind, HelperKind::Operator)
                && (e.operator_name.starts_with("stg") || e.operator_name.starts_with("yp"))
                && e.representation.cart
        })
        .count();
    let spinor_f12_count = manifest
        .iter()
        .filter(|e| {
            matches!(e.helper_kind, HelperKind::Operator)
                && (e.operator_name.starts_with("stg") || e.operator_name.starts_with("yp"))
                && e.representation.spinor
        })
        .count();

    assert_eq!(
        cart_f12_count, 0,
        "oracle confirmed: 0 cart F12 symbols (sph-only enforcement)"
    );
    assert_eq!(
        spinor_f12_count, 0,
        "oracle confirmed: 0 spinor F12 symbols (sph-only enforcement)"
    );

    println!(
        "  PASS: sph-only enforcement — {} F12 sph entries, 0 cart, 0 spinor",
        f12_entries.len()
    );
}
