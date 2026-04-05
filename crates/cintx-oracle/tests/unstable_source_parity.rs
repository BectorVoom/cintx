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

/// grids family parity tests.
/// 5 symbols: int1e_grids_sph, int1e_grids_ip_sph, int1e_grids_ipvip_sph,
///            int1e_grids_spvsp_sph, int1e_grids_ipip_sph.
/// Uses H2O/STO-3G fixture with grid point coordinates in env.
/// Implementation added in Phase 14 Plan 02.
mod grids_parity {
    use cintx_compat::raw::{
        ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
        NGRIDS, POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_GRIDS, PTR_ZETA,
        RawApiId, eval_raw,
    };

    /// Absolute tolerance for grids oracle parity comparisons.
    const ATOL: f64 = 1e-11;

    /// Build H2O STO-3G libcint env with `ngrids` grid points appended.
    ///
    /// Grid layout:
    ///   env[0..PTR_ENV_START]  — reserved libcint global params (zeros)
    ///   env[11] = ngrids       — NGRIDS slot
    ///   env[12] = ptr_grids    — PTR_GRIDS slot points to first grid coord
    ///   env[PTR_ENV_START..PTR_ENV_START+12] — atom coords (3 atoms × 3)
    ///   env[...] — exponents, coefficients
    ///   env[ptr_grids..] — grid point coordinates (ngrids × 3)
    fn build_h2o_sto3g_grids(ngrids: usize) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

        // env[0..PTR_ENV_START] = zeros (libcint global param slots)
        let mut env = vec![0.0_f64; PTR_ENV_START];

        // Atom coordinates starting at PTR_ENV_START
        let o_coord_ptr = env.len() as i32;
        env.extend_from_slice(&o_coord);
        let h1_coord_ptr = env.len() as i32;
        env.extend_from_slice(&h1_coord);
        let h2_coord_ptr = env.len() as i32;
        env.extend_from_slice(&h2_coord);

        // Exponents and coefficients
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

        // Grid coordinates start here
        let ptr_grids_val = env.len() as i32;

        // Append grid point coordinates: use a 3×3 grid around the O center
        // and near the H1/H2 positions to exercise range
        let grid_points: Vec<[f64; 3]> = if ngrids == 3 {
            vec![
                [0.0, 0.0, 0.0],      // at O
                [0.0, 1.4307, 1.1078], // at H1
                [0.5, -0.5, 0.5],      // off-center
            ]
        } else {
            // For ngrids != 3: evenly space along x axis
            (0..ngrids)
                .map(|g| {
                    let t = g as f64 / (ngrids.max(2) - 1) as f64;
                    [-1.0 + 2.0 * t, 0.0, 0.0]
                })
                .collect()
        };
        assert_eq!(grid_points.len(), ngrids);

        for coord in &grid_points {
            env.extend_from_slice(coord);
        }

        // Fill in NGRIDS and PTR_GRIDS global env slots
        env[NGRIDS] = ngrids as f64;
        env[PTR_GRIDS] = ptr_grids_val as f64;

        // Build atm (3 atoms: O, H1, H2)
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

        // Build bas (5 shells: O-1s, O-2s, O-2p, H1-1s, H2-1s)
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

    /// Number of spherical harmonics for angular momentum l: 2l+1.
    fn nsph(l: i32) -> usize {
        (2 * l + 1) as usize
    }

    /// Count element-wise mismatches above atol.
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

    /// Evaluate a grids integral via eval_raw.
    ///
    /// `shls_4` is `[i, j, grid_start, grid_end]`.
    /// `ncomp` is the number of operator components.
    /// Returns a buffer of length `ncomp * ngrids * nsph_i * nsph_j`.
    fn eval_grids(
        symbol: &'static str,
        shls_4: &[i32; 4],
        ncomp: usize,
        ngrids: usize,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
    ) -> Vec<f64> {
        let si = shls_4[0] as usize;
        let sj = shls_4[1] as usize;
        let ni = nsph(bas[si * BAS_SLOTS + ANG_OF]);
        let nj = nsph(bas[sj * BAS_SLOTS + ANG_OF]);
        let n = ncomp * ngrids * ni * nj;
        let mut out = vec![0.0_f64; n];
        unsafe {
            eval_raw(
                RawApiId::Symbol(symbol),
                Some(&mut out),
                None,
                shls_4.as_slice(),
                atm,
                bas,
                env,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("eval_raw {symbol} failed for shls {shls_4:?}: {e:?}"));
        }
        out
    }

    // ────────────────────────────────────────────────────────────────────────
    // Non-vendor smoke tests: verify non-zero output for each grids symbol.
    // These run without CINTX_ORACLE_BUILD_VENDOR and catch stub regressions.
    // ────────────────────────────────────────────────────────────────────────

    /// int1e_grids_sph: base grids operator produces non-zero output.
    #[test]
    fn test_int1e_grids_sph_nonzero() {
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        // Use O-1s / O-2s shell pair (s/s: ni=1, nj=1)
        let shls_4: [i32; 4] = [0, 1, 0, ngrids as i32];
        let out = eval_grids("int1e_grids_sph", &shls_4, 1, ngrids, &atm, &bas, &env);
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
        assert!(nonzero > 0, "int1e_grids_sph output is all zeros — kernel not computing");
        println!("  PASS: int1e_grids_sph nonzero={nonzero}/{}", out.len());
    }

    /// int1e_grids_ip_sph: ip variant (ncomp=3) produces non-zero output.
    #[test]
    fn test_int1e_grids_ip_sph_nonzero() {
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let shls_4: [i32; 4] = [0, 1, 0, ngrids as i32];
        let out = eval_grids("int1e_grids_ip_sph", &shls_4, 3, ngrids, &atm, &bas, &env);
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
        assert!(nonzero > 0, "int1e_grids_ip_sph output is all zeros — kernel not computing");
        println!("  PASS: int1e_grids_ip_sph ncomp=3 nonzero={nonzero}/{}", out.len());
    }

    /// int1e_grids_ipvip_sph: ipvip variant (ncomp=9) produces non-zero output.
    #[test]
    fn test_int1e_grids_ipvip_sph_nonzero() {
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let shls_4: [i32; 4] = [0, 1, 0, ngrids as i32];
        let out = eval_grids("int1e_grids_ipvip_sph", &shls_4, 9, ngrids, &atm, &bas, &env);
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
        assert!(nonzero > 0, "int1e_grids_ipvip_sph output is all zeros — kernel not computing");
        println!("  PASS: int1e_grids_ipvip_sph ncomp=9 nonzero={nonzero}/{}", out.len());
    }

    /// int1e_grids_spvsp_sph: spvsp variant (ncomp=4) produces non-zero output.
    #[test]
    fn test_int1e_grids_spvsp_sph_nonzero() {
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let shls_4: [i32; 4] = [0, 1, 0, ngrids as i32];
        let out = eval_grids("int1e_grids_spvsp_sph", &shls_4, 4, ngrids, &atm, &bas, &env);
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
        assert!(nonzero > 0, "int1e_grids_spvsp_sph output is all zeros — kernel not computing");
        println!("  PASS: int1e_grids_spvsp_sph ncomp=4 nonzero={nonzero}/{}", out.len());
    }

    /// int1e_grids_ipip_sph: ipip variant (ncomp=9) produces non-zero output.
    #[test]
    fn test_int1e_grids_ipip_sph_nonzero() {
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let shls_4: [i32; 4] = [0, 1, 0, ngrids as i32];
        let out = eval_grids("int1e_grids_ipip_sph", &shls_4, 9, ngrids, &atm, &bas, &env);
        let nonzero = out.iter().filter(|&&v| v.abs() > 1e-18).count();
        assert!(nonzero > 0, "int1e_grids_ipip_sph output is all zeros — kernel not computing");
        println!("  PASS: int1e_grids_ipip_sph ncomp=9 nonzero={nonzero}/{}", out.len());
    }

    // ────────────────────────────────────────────────────────────────────────
    // Vendor oracle parity tests: compare against libcint 6.1.3.
    // Require CINTX_ORACLE_BUILD_VENDOR=1.
    // ────────────────────────────────────────────────────────────────────────

    /// Oracle parity gate for int1e_grids_sph.
    ///
    /// Tests shell pairs: (O-1s, O-2s), (O-1s, O-2p), (H1-1s, H2-1s).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn oracle_parity_int1e_grids_sph() {
        use cintx_oracle::vendor_ffi;
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for (si, sj) in [(0, 1), (0, 2), (3, 4)] {
            let shls_4: [i32; 4] = [si, sj, 0, ngrids as i32];
            let cintx_out = eval_grids("int1e_grids_sph", &shls_4, 1, ngrids, &atm, &bas, &env);
            let ni = nsph(bas[si as usize * BAS_SLOTS + ANG_OF]);
            let nj = nsph(bas[sj as usize * BAS_SLOTS + ANG_OF]);
            let n = 1 * ngrids * ni * nj;
            let mut vendor_out = vec![0.0_f64; n];
            vendor_ffi::vendor_int1e_grids_sph(
                &mut vendor_out, &shls_4, &atm, natm, &bas, nbas, &env,
            );
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(
                mc, 0,
                "int1e_grids_sph parity FAIL: {mc} mismatches for shls {shls_4:?} at atol={ATOL:.1e}"
            );
            println!("  PASS: int1e_grids_sph shls {shls_4:?}: mismatch_count=0, n={n}");
        }
    }

    /// Oracle parity gate for int1e_grids_ip_sph (ncomp=3).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn oracle_parity_int1e_grids_ip_sph() {
        use cintx_oracle::vendor_ffi;
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for (si, sj) in [(0i32, 1i32), (0, 2), (3, 4)] {
            let shls_4: [i32; 4] = [si, sj, 0, ngrids as i32];
            let ncomp = 3;
            let cintx_out = eval_grids("int1e_grids_ip_sph", &shls_4, ncomp, ngrids, &atm, &bas, &env);
            let ni = nsph(bas[si as usize * BAS_SLOTS + ANG_OF]);
            let nj = nsph(bas[sj as usize * BAS_SLOTS + ANG_OF]);
            let n = ncomp * ngrids * ni * nj;
            let mut vendor_out = vec![0.0_f64; n];
            vendor_ffi::vendor_int1e_grids_ip_sph(
                &mut vendor_out, &shls_4, &atm, natm, &bas, nbas, &env,
            );
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(
                mc, 0,
                "int1e_grids_ip_sph parity FAIL: {mc} mismatches for shls {shls_4:?} at atol={ATOL:.1e}"
            );
            println!("  PASS: int1e_grids_ip_sph shls {shls_4:?}: mismatch_count=0, n={n}");
        }
    }

    /// Oracle parity gate for int1e_grids_ipvip_sph (ncomp=9).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn oracle_parity_int1e_grids_ipvip_sph() {
        use cintx_oracle::vendor_ffi;
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for (si, sj) in [(0i32, 1i32), (0, 2), (3, 4)] {
            let shls_4: [i32; 4] = [si, sj, 0, ngrids as i32];
            let ncomp = 9;
            let cintx_out = eval_grids("int1e_grids_ipvip_sph", &shls_4, ncomp, ngrids, &atm, &bas, &env);
            let ni = nsph(bas[si as usize * BAS_SLOTS + ANG_OF]);
            let nj = nsph(bas[sj as usize * BAS_SLOTS + ANG_OF]);
            let n = ncomp * ngrids * ni * nj;
            let mut vendor_out = vec![0.0_f64; n];
            vendor_ffi::vendor_int1e_grids_ipvip_sph(
                &mut vendor_out, &shls_4, &atm, natm, &bas, nbas, &env,
            );
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(
                mc, 0,
                "int1e_grids_ipvip_sph parity FAIL: {mc} mismatches for shls {shls_4:?} at atol={ATOL:.1e}"
            );
            println!("  PASS: int1e_grids_ipvip_sph shls {shls_4:?}: mismatch_count=0, n={n}");
        }
    }

    /// Oracle parity gate for int1e_grids_spvsp_sph (ncomp=4).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn oracle_parity_int1e_grids_spvsp_sph() {
        use cintx_oracle::vendor_ffi;
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for (si, sj) in [(0i32, 1i32), (0, 2), (3, 4)] {
            let shls_4: [i32; 4] = [si, sj, 0, ngrids as i32];
            let ncomp = 4;
            let cintx_out = eval_grids("int1e_grids_spvsp_sph", &shls_4, ncomp, ngrids, &atm, &bas, &env);
            let ni = nsph(bas[si as usize * BAS_SLOTS + ANG_OF]);
            let nj = nsph(bas[sj as usize * BAS_SLOTS + ANG_OF]);
            let n = ncomp * ngrids * ni * nj;
            let mut vendor_out = vec![0.0_f64; n];
            vendor_ffi::vendor_int1e_grids_spvsp_sph(
                &mut vendor_out, &shls_4, &atm, natm, &bas, nbas, &env,
            );
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(
                mc, 0,
                "int1e_grids_spvsp_sph parity FAIL: {mc} mismatches for shls {shls_4:?} at atol={ATOL:.1e}"
            );
            println!("  PASS: int1e_grids_spvsp_sph shls {shls_4:?}: mismatch_count=0, n={n}");
        }
    }

    /// Oracle parity gate for int1e_grids_ipip_sph (ncomp=9).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn oracle_parity_int1e_grids_ipip_sph() {
        use cintx_oracle::vendor_ffi;
        let ngrids = 3;
        let (atm, bas, env) = build_h2o_sto3g_grids(ngrids);
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        for (si, sj) in [(0i32, 1i32), (0, 2), (3, 4)] {
            let shls_4: [i32; 4] = [si, sj, 0, ngrids as i32];
            let ncomp = 9;
            let cintx_out = eval_grids("int1e_grids_ipip_sph", &shls_4, ncomp, ngrids, &atm, &bas, &env);
            let ni = nsph(bas[si as usize * BAS_SLOTS + ANG_OF]);
            let nj = nsph(bas[sj as usize * BAS_SLOTS + ANG_OF]);
            let n = ncomp * ngrids * ni * nj;
            let mut vendor_out = vec![0.0_f64; n];
            vendor_ffi::vendor_int1e_grids_ipip_sph(
                &mut vendor_out, &shls_4, &atm, natm, &bas, nbas, &env,
            );
            let mc = count_mismatches(&vendor_out, &cintx_out, ATOL);
            assert_eq!(
                mc, 0,
                "int1e_grids_ipip_sph parity FAIL: {mc} mismatches for shls {shls_4:?} at atol={ATOL:.1e}"
            );
            println!("  PASS: int1e_grids_ipip_sph shls {shls_4:?}: mismatch_count=0, n={n}");
        }
    }
}

/// breit family parity tests.
/// 2 spinor-only symbols: int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor.
/// Implementation added in Phase 14 Plan 04.
mod breit_parity {
    use cintx_compat::raw::{
        ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
        POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
    };

    /// Count of mismatches between reference and observed f64 slices at given atol.
    fn count_mismatches_atol(reference: &[f64], observed: &[f64], atol: f64) -> usize {
        assert_eq!(
            reference.len(),
            observed.len(),
            "output length mismatch: {} vs {}",
            reference.len(),
            observed.len()
        );
        let mut mismatches = 0usize;
        for (idx, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
            let diff = (obs_val - ref_val).abs();
            if diff > atol {
                mismatches += 1;
                if mismatches <= 16 {
                    eprintln!(
                        "  MISMATCH[{idx}] ref={ref_val:.15e} obs={obs_val:.15e} diff={diff:.3e}"
                    );
                }
            }
        }
        mismatches
    }

    /// Build H2O/STO-3G fixture (same as two_electron_parity.rs — PTR_ENV_START-aligned).
    ///
    /// Returns (atm, bas, env) with 5 shells:
    ///   Shell 0: O 1s  (l=0, 3 primitives, kappa=0)
    ///   Shell 1: O 2s  (l=0, 3 primitives, kappa=0)
    ///   Shell 2: O 2p  (l=1, 3 primitives, kappa=0)
    ///   Shell 3: H1 1s (l=0, 3 primitives, kappa=0)
    ///   Shell 4: H2 1s (l=0, 3 primitives, kappa=0)
    fn build_h2o_sto3g_breit() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    /// Compute spinor output size (in f64) for a shell quartet with kappa=0 shells.
    ///
    /// For kappa=0, CINTcgto_spinor = 2*(l+1) per contracted shell.
    /// The output buffer for a complex spinor integral has 2*ns_i*ns_j*ns_k*ns_l f64 values
    /// (factor 2 for complex interleave: re/im pairs).
    fn spinor_n_elem_kappa0(bas: &[i32], shls: &[i32; 4]) -> usize {
        let mut ns = 1usize;
        for &sh in shls {
            let l = bas[sh as usize * BAS_SLOTS + ANG_OF] as usize;
            let ns_sh = 2 * (l + 1); // CINTcgto_spinor for kappa=0
            ns *= ns_sh;
        }
        2 * ns // factor 2 for complex interleave
    }

    /// Oracle parity test for int2e_breit_r1p2_spinor against vendored libcint 6.1.3.
    ///
    /// Uses H2O/STO-3G fixture, shell quartet [0, 1, 0, 1] (O-1s/O-2s pair).
    /// Tolerance: atol=1e-12 applied to each f64 element (real and imaginary parts).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int2e_breit_r1p2_spinor_oracle_parity() {
        let (atm, bas, env) = build_h2o_sto3g_breit();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        let shls: [i32; 4] = [0, 1, 0, 1];
        let n_elem = spinor_n_elem_kappa0(&bas, &shls);
        let atol = 1.0e-12_f64;
        let epsilon = atol;

        // Reference: vendored libcint
        let mut vendor_out = vec![0.0_f64; n_elem];
        use cintx_oracle::vendor_ffi;
        vendor_ffi::vendor_int2e_breit_r1p2_spinor(
            &mut vendor_out,
            &shls,
            &atm,
            natm,
            &bas,
            nbas,
            &env,
        );

        // cintx: eval_raw with Breit spinor symbol
        let mut cintx_out = vec![0.0_f64; n_elem];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_breit_r1p2_spinor"),
                Some(&mut cintx_out),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("eval_raw int2e_breit_r1p2_spinor failed: {e:?}"));
        }

        let mismatches = count_mismatches_atol(&vendor_out, &cintx_out, epsilon);

        // Verify at least one non-zero element (kernel is not a stub)
        let any_nonzero = cintx_out.iter().any(|v| v.abs() > 1e-18)
            || vendor_out.iter().any(|v| v.abs() > 1e-18);

        println!(
            "int2e_breit_r1p2_spinor: vendor parity PASS, n_elem={n_elem}, mismatches={mismatches}, any_nonzero={any_nonzero} (atol={atol:.1e})"
        );

        assert_eq!(
            mismatches, 0,
            "int2e_breit_r1p2_spinor: {mismatches} parity mismatch(es) against vendored libcint at atol={atol:.1e}"
        );
        assert!(
            any_nonzero,
            "int2e_breit_r1p2_spinor: all outputs are zero — kernel appears stubbed"
        );
    }

    /// Oracle parity test for int2e_breit_r2p2_spinor against vendored libcint 6.1.3.
    ///
    /// Uses H2O/STO-3G fixture, shell quartet [0, 1, 0, 1] (O-1s/O-2s pair).
    /// Tolerance: atol=1e-12 applied to each f64 element (real and imaginary parts).
    #[test]
    #[cfg(has_vendor_libcint)]
    fn test_int2e_breit_r2p2_spinor_oracle_parity() {
        let (atm, bas, env) = build_h2o_sto3g_breit();
        let natm = (atm.len() / ATM_SLOTS) as i32;
        let nbas = (bas.len() / BAS_SLOTS) as i32;

        let shls: [i32; 4] = [0, 1, 0, 1];
        let n_elem = spinor_n_elem_kappa0(&bas, &shls);
        let atol = 1.0e-12_f64;
        let epsilon = atol;

        // Reference: vendored libcint
        let mut vendor_out = vec![0.0_f64; n_elem];
        use cintx_oracle::vendor_ffi;
        vendor_ffi::vendor_int2e_breit_r2p2_spinor(
            &mut vendor_out,
            &shls,
            &atm,
            natm,
            &bas,
            nbas,
            &env,
        );

        // cintx: eval_raw with Breit spinor symbol
        let mut cintx_out = vec![0.0_f64; n_elem];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_breit_r2p2_spinor"),
                Some(&mut cintx_out),
                None,
                &shls,
                &atm,
                &bas,
                &env,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("eval_raw int2e_breit_r2p2_spinor failed: {e:?}"));
        }

        let mismatches = count_mismatches_atol(&vendor_out, &cintx_out, epsilon);

        // Verify at least one non-zero element (kernel is not a stub)
        let any_nonzero = cintx_out.iter().any(|v| v.abs() > 1e-18)
            || vendor_out.iter().any(|v| v.abs() > 1e-18);

        println!(
            "int2e_breit_r2p2_spinor: vendor parity PASS, n_elem={n_elem}, mismatches={mismatches}, any_nonzero={any_nonzero} (atol={atol:.1e})"
        );

        assert_eq!(
            mismatches, 0,
            "int2e_breit_r2p2_spinor: {mismatches} parity mismatch(es) against vendored libcint at atol={atol:.1e}"
        );
        assert!(
            any_nonzero,
            "int2e_breit_r2p2_spinor: all outputs are zero — kernel appears stubbed"
        );
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

