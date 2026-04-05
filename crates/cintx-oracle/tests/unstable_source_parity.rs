//! Oracle parity tests for unstable-source API families (Phase 14).
//!
//! Per D-09: single file with per-family modules, all gated behind
//! `#[cfg(feature = "unstable-source-api")]`.
//! Per D-10: reuse H2O/STO-3G fixture molecule for all unstable families.
//!
//! Gate summary:
//!   Family  | Symbols | Status
//!   --------|---------|-------
//!   origi   | 4       | Phase 14 Plan 02
//!   origk   | 6       | Phase 14 Plan 02
//!   ssc     | 1       | Phase 14 Plan 02
//!   grids   | 5       | Wave 2 (Phase 14 Plan 03)
//!   breit   | 2       | Wave 2 (Phase 14 Plan 03)
//!
//! Requirements: #[cfg(feature = "cpu")] + #[cfg(feature = "unstable-source-api")]
//! Run: CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu,unstable-source-api -p cintx-oracle -- unstable_source_parity

#![cfg(feature = "cpu")]
#![cfg(feature = "unstable-source-api")]

use cintx_compat::raw::{ANG_OF, ATM_SLOTS, BAS_SLOTS, RawApiId, eval_raw};
use cintx_oracle::fixtures::build_h2o_sto3g;

/// Absolute tolerance for unstable-source oracle parity (matching Phase 13 convention).
const ATOL: f64 = 1e-12;

/// STO-3G H2O shells: 0=O-1s, 1=O-2s, 2=O-2p, 3=H1-1s, 4=H2-1s.
const SHLS_2_01: [i32; 2] = [0, 1]; // O-1s / O-2s (s-s pair)
const SHLS_2_02: [i32; 2] = [0, 2]; // O-1s / O-2p (s-p pair)

/// 3-shell triples for 3c1e/3c2e (matching Phase 10 convention).
const SHLS_3_340: [i32; 3] = [3, 4, 0]; // H1-1s / H2-1s / O-1s
const SHLS_3_012: [i32; 3] = [0, 1, 2]; // O-1s / O-2s / O-2p

fn nsph_for_l(l: i32) -> usize {
    (2 * l + 1) as usize
}

fn ncart_for_l(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

/// Count mismatches above absolute tolerance.
#[cfg(has_vendor_libcint)]
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64) -> usize {
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

/// Evaluate a 1e (2-shell) integral via cintx eval_raw.
fn eval_1e_sph(symbol: &'static str, shls: &[i32; 2], atm: &[i32], bas: &[i32], env: &[f64], ncomp: usize) -> Vec<f64> {
    let ni = nsph_for_l(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nsph_for_l(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let n = ncomp * ni * nj;
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

/// Evaluate a 3-shell integral via cintx eval_raw.
fn eval_3c_sph(symbol: &'static str, shls: &[i32; 3], atm: &[i32], bas: &[i32], env: &[f64], ncomp: usize, k_cartesian: bool) -> Vec<f64> {
    let ni = nsph_for_l(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nsph_for_l(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = if k_cartesian {
        ncart_for_l(bas[shls[2] as usize * BAS_SLOTS + ANG_OF])
    } else {
        nsph_for_l(bas[shls[2] as usize * BAS_SLOTS + ANG_OF])
    };
    let n = ncomp * ni * nj * nk;
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

/// origi family parity tests.
/// 4 symbols: int1e_r2_origi_sph, int1e_r4_origi_sph,
///            int1e_r2_origi_ip2_sph, int1e_r4_origi_ip2_sph.
mod origi_parity {
    use super::*;

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int1e_r2_origi_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for shls in [SHLS_2_01, SHLS_2_02] {
            let cintx_out = eval_1e_sph("int1e_r2_origi_sph", &shls, &atm, &bas, &env, 1);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int1e_r2_origi_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int1e_r2_origi_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int1e_r4_origi_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for shls in [SHLS_2_01, SHLS_2_02] {
            let cintx_out = eval_1e_sph("int1e_r4_origi_sph", &shls, &atm, &bas, &env, 1);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int1e_r4_origi_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int1e_r4_origi_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int1e_r2_origi_ip2_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let ncomp = 3;

        for shls in [SHLS_2_01, SHLS_2_02] {
            let cintx_out = eval_1e_sph("int1e_r2_origi_ip2_sph", &shls, &atm, &bas, &env, ncomp);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int1e_r2_origi_ip2_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int1e_r2_origi_ip2_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int1e_r4_origi_ip2_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let ncomp = 3;

        for shls in [SHLS_2_01, SHLS_2_02] {
            let cintx_out = eval_1e_sph("int1e_r4_origi_ip2_sph", &shls, &atm, &bas, &env, ncomp);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int1e_r4_origi_ip2_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int1e_r4_origi_ip2_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }
}

/// origk family parity tests.
/// 6 symbols: int3c1e_r2/r4/r6_origk_sph, int3c1e_ip1_r2/r4/r6_origk_sph.
mod origk_parity {
    use super::*;

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c1e_r2_origk_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for shls in [SHLS_3_340, SHLS_3_012] {
            let cintx_out = eval_3c_sph("int3c1e_r2_origk_sph", &shls, &atm, &bas, &env, 1, false);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c1e_r2_origk_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c1e_r2_origk_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c1e_r4_origk_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for shls in [SHLS_3_340, SHLS_3_012] {
            let cintx_out = eval_3c_sph("int3c1e_r4_origk_sph", &shls, &atm, &bas, &env, 1, false);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c1e_r4_origk_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c1e_r4_origk_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c1e_r6_origk_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for shls in [SHLS_3_340, SHLS_3_012] {
            let cintx_out = eval_3c_sph("int3c1e_r6_origk_sph", &shls, &atm, &bas, &env, 1, false);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c1e_r6_origk_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c1e_r6_origk_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c1e_ip1_r2_origk_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let ncomp = 3;

        for shls in [SHLS_3_340, SHLS_3_012] {
            let cintx_out = eval_3c_sph("int3c1e_ip1_r2_origk_sph", &shls, &atm, &bas, &env, ncomp, false);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c1e_ip1_r2_origk_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c1e_ip1_r2_origk_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c1e_ip1_r4_origk_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let ncomp = 3;

        for shls in [SHLS_3_340, SHLS_3_012] {
            let cintx_out = eval_3c_sph("int3c1e_ip1_r4_origk_sph", &shls, &atm, &bas, &env, ncomp, false);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c1e_ip1_r4_origk_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c1e_ip1_r4_origk_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c1e_ip1_r6_origk_sph_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;
        let ncomp = 3;

        for shls in [SHLS_3_340, SHLS_3_012] {
            let cintx_out = eval_3c_sph("int3c1e_ip1_r6_origk_sph", &shls, &atm, &bas, &env, ncomp, false);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c1e_ip1_r6_origk_sph(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c1e_ip1_r6_origk_sph parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }
}

/// ssc family parity tests.
/// 1 symbol: int3c2e_sph_ssc.
mod ssc_parity {
    use super::*;

    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int3c2e_sph_ssc_oracle_parity() {
        use cintx_oracle::vendor_ffi;
        let (atm, bas, env) = build_h2o_sto3g();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for shls in [SHLS_3_340, SHLS_3_012] {
            // SSC: k stays Cartesian
            let cintx_out = eval_3c_sph("int3c2e_sph_ssc", &shls, &atm, &bas, &env, 1, true);
            let mut vendor_out = vec![0.0_f64; cintx_out.len()];
            vendor_ffi::vendor_int3c2e_sph_ssc(&mut vendor_out, &shls, &atm, natm, &bas, nbas, &env);
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(mc, 0, "int3c2e_sph_ssc parity FAIL: {mc} mismatches for shls {shls:?} at epsilon={ATOL:.1e}");
        }
    }
}

/// grids family parity tests.
/// 5 symbols. Implementation pending in Phase 14 Plan 03.
mod grids_parity {}

/// breit family parity tests.
/// 2 spinor-only symbols. Implementation pending in Phase 14 Plan 03.
mod breit_parity {}
