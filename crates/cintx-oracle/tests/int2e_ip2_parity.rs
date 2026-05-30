//! Oracle parity test for the Phase 23 rank-3 ket-side 2e gradient family
//! `int2e_ip2` (∇ on the 2nd-electron bra-center k), cart + sph: H2O STO-3G.
//!
//! DRV1-01. Validates the 3-component (∇_k) output vs vendored libcint 6.1.3 at
//! atol=1e-12. Requires the `cpu` feature (cubecl cpu backend); vendor parity
//! additionally requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint`
//! cfg). Without BOTH gates the vendor parity test is not compiled and parity
//! silently skips — run:
//!
//!   CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!       --test int2e_ip2_parity -- --test-threads=1
//!
//! Layout / D-14 discipline: each evaluated quartet pins the element count to
//! `3 * ni*nj*nk*nl` (catches a too-low component_rank truncation) AND asserts
//! `any_nonzero` (catches a zero-fill / short-buffer stub). The quartet is
//! deliberately NON-SQUARE (p,s,p,s) so a transposed component layout cannot pass.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;
const NCOMP: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G fixture (verbatim from one_electron_grad_both_parity.rs)
// Shells: 0=O-1s(s) 1=O-2s(s) 2=O-2p(p) 3=H1-1s(s) 4=H2-1s(s)
// ─────────────────────────────────────────────────────────────────────────────

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

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = o_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = 9;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = 9;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = 9;

    let mut bas = vec![0_i32; 5 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 0;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = o1s_exp_ptr;
    bas[PTR_COEFF] = o1s_coeff_ptr;

    bas[BAS_SLOTS + ATOM_OF] = 0;
    bas[BAS_SLOTS + ANG_OF] = 0;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;

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

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// A small set of explicitly NON-SQUARE quartets covering the s/p l-range within
// the int2e_ip2 nroots ceiling (d-max). The p-shell is slot 2 (O-2p); pairing it
// against s shells produces rectangular (ni != nk) blocks that a transposed
// component layout cannot pass (D-05 discipline).
const QUARTETS: &[[i32; 4]] = &[
    [2, 0, 0, 1], // p s s p — non-square, ket-bra k = s
    [0, 0, 2, 0], // s s p s — ∇ on the p ket-bra center
    [2, 3, 0, 4], // p s s s — mixed-center non-square
    [0, 2, 2, 0], // s p p s — non-square in two slots
];

// ─────────────────────────────────────────────────────────────────────────────
// 3-component gradient collector (cintx via eval_raw)
// ─────────────────────────────────────────────────────────────────────────────

fn collect_3c_quartet(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 4],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
    let nl = nf(bas[shls[3] as usize * BAS_SLOTS + ANG_OF]);
    let n_elem = NCOMP * ni * nj * nk * nl;
    let mut out = vec![0.0_f64; n_elem];

    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for quartet {shls:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_3c_quartet<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shls: &[i32; 4],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
    let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
    let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
    let nl = nf(bas[shls[3] as usize * BAS_SLOTS + ANG_OF]);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; NCOMP * ni * nj * nk * nl];
    vendor_fn(&mut out, shls, atm, natm, bas, nbas, env);
    out
}

#[allow(dead_code)]
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "length mismatch");
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: matrix is all-zero (zero-fill regression)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism test (always under cpu) — also pins the 3*ni*nj*nk*nl element count
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_int2e_ip2_determinism_and_shape() {
    let (atm, bas, env) = build_h2o_sto3g();
    for (api, nf, rep) in [
        (RawApiId::INT2E_IP2_SPH, nsph as fn(i32) -> usize, "sph"),
        (RawApiId::INT2E_IP2_CART, ncart as fn(i32) -> usize, "cart"),
    ] {
        for shls in QUARTETS {
            let ni = nf(bas[shls[0] as usize * BAS_SLOTS + ANG_OF]);
            let nj = nf(bas[shls[1] as usize * BAS_SLOTS + ANG_OF]);
            let nk = nf(bas[shls[2] as usize * BAS_SLOTS + ANG_OF]);
            let nl = nf(bas[shls[3] as usize * BAS_SLOTS + ANG_OF]);
            let m1 = collect_3c_quartet(api, &atm, &bas, &env, shls, nf);
            let m2 = collect_3c_quartet(api, &atm, &bas, &env, shls, nf);
            // D-14: pin the element count to component_rank=3 × AO product.
            assert_eq!(
                m1.len(),
                NCOMP * ni * nj * nk * nl,
                "int2e_ip2_{rep} {shls:?} size must be 3*ni*nj*nk*nl"
            );
            for (a, b) in m1.iter().zip(m2.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "int2e_ip2_{rep} {shls:?} not bit-identical");
            }
            assert_any_nonzero(&m1, &format!("int2e_ip2_{rep} {shls:?}"));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity test (requires has_vendor_libcint + cpu)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int2e_ip2_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g();

    for shls in QUARTETS {
        // sph
        let vendor_s =
            collect_vendor_3c_quartet(vendor_ffi::vendor_int2e_ip2_sph, &atm, &bas, &env, shls, nsph);
        let cintx_s = collect_3c_quartet(RawApiId::INT2E_IP2_SPH, &atm, &bas, &env, shls, nsph);
        assert_any_nonzero(&cintx_s, &format!("int2e_ip2_sph {shls:?} cintx"));
        assert_any_nonzero(&vendor_s, &format!("int2e_ip2_sph {shls:?} vendor"));
        let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
        assert_eq!(mm, 0, "int2e_ip2_sph {shls:?}: {mm} mismatches vs vendored libcint at atol={ATOL}");

        // cart
        let vendor_c =
            collect_vendor_3c_quartet(vendor_ffi::vendor_int2e_ip2_cart, &atm, &bas, &env, shls, ncart);
        let cintx_c = collect_3c_quartet(RawApiId::INT2E_IP2_CART, &atm, &bas, &env, shls, ncart);
        assert_any_nonzero(&cintx_c, &format!("int2e_ip2_cart {shls:?} cintx"));
        assert_any_nonzero(&vendor_c, &format!("int2e_ip2_cart {shls:?} vendor"));
        let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
        assert_eq!(mm, 0, "int2e_ip2_cart {shls:?}: {mm} mismatches vs vendored libcint at atol={ATOL}");
    }
}
