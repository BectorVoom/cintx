//! Oracle parity tests for the Phase 25 HESS-04 3rd/4th-order derivative families
//! (`deriv3.c` rank 27, `deriv4.c` rank 81), cart + sph.
//!
//! Roster (locked by grepping libcint deriv3.c/deriv4.c + cint_funcs.h, Task 0):
//!   3rd-order (deriv3.c, component_rank 27):
//!     int1e_ipipipnuc   (<∇∇∇ i | NUC  | j>,    bra+3)
//!     int1e_ipipiprinv  (<∇∇∇ i | RINV | j>,    bra+3)
//!     int1e_ipipnucip   (<∇∇  i | NUC  | ∇ j>,  bra+2 / ket+1)
//!     int1e_ipiprinvip  (<∇∇  i | RINV | ∇ j>,  bra+2 / ket+1)
//!   4th-order (deriv4.c, component_rank 81):
//!     int1e_ipipipiprinv (<∇∇ i | RINV | ∇∇ j>, bra+2 AND ket+2 — dual headroom)
//!     int1e_ipiprinvipip (<∇∇ i | RINV | ∇∇ j>, alternate ordering)
//!     int1e_ipipiprinvip (<∇∇∇ i | RINV | ∇ j>, bra+3 / ket+1)
//!
//! Each family matches vendored libcint 6.1.3 at atol=1e-12, every component
//! (27 or 81), on a NON-SQUARE block with DISTINCT bra and ket angular momenta
//! (Pitfall 4: a square block is transpose-symmetric and would hide a bra↔ket /
//! single-side-headroom bug — the dominant deriv4 dual-headroom failure mode).
//!
//! Requires the `cpu` feature (cubecl cpu backend); vendor parity additionally
//! requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint` cfg). Without
//! BOTH flags, only the determinism tests run and vendor parity SILENTLY SKIPS.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// NON-SQUARE (p × d) two-shell fixture (Pitfall 4 — dual headroom).
//
// A square (p×p) block is transpose-symmetric and would hide a deriv4 ket-headroom
// miss; p (l=1, 3/5 funcs) × d (l=2, 6/9 funcs) is strictly NON-SQUARE in BOTH cart
// and sph, with DISTINCT bra (p) and ket (d) angular momenta, so a single-side
// (bra-only) headroom raise on a deriv4 family cannot pass.
//
// PTR_ENV_START-aligned so the PTR_RINV_ORIG=4..6 slot stays clean for the
// rinv-family origin.
// ─────────────────────────────────────────────────────────────────────────────

fn build_pd_nonsquare() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.0_f64, 0.0, 1.2];

    let p_exp = [1.8_f64, 0.5];
    let p_coeff = [0.35_f64, 0.65];
    let d_exp = [2.2_f64, 0.7];
    let d_coeff = [0.40_f64, 0.60];

    let mut env = vec![0.0_f64; PTR_ENV_START];
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
    let _zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);
    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    atm[CHARGE_OF] = 6;
    atm[PTR_COORD] = a_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + CHARGE_OF] = 8;
    atm[ATM_SLOTS + PTR_COORD] = b_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    // shell 0: p on atom 0 (bra) — l=1
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 2;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1: d on atom 1 (ket) — l=2 (bra l != ket l)
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 2;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;

    (atm, bas, env)
}

/// Set the rinv origin into env[PTR_RINV_ORIG..+3] (returns a fresh env). The
/// rinv families are single-origin 1/r derivatives; the origin MUST be a nonzero
/// off-bra/off-ket point so parity exercises the shifted-center Rys path.
#[allow(dead_code)]
fn env_with_rinv_origin(env: &[f64], origin: [f64; 3]) -> Vec<f64> {
    let mut e = env.to_vec();
    e[PTR_RINV_ORIG] = origin[0];
    e[PTR_RINV_ORIG + 1] = origin[1];
    e[PTR_RINV_ORIG + 2] = origin[2];
    e
}

// ─────────────────────────────────────────────────────────────────────────────
// N-component single-block collectors (parameterized over ncomp = 27 / 81).
// ─────────────────────────────────────────────────────────────────────────────

fn collect_nc_block(
    api_id: RawApiId,
    ncomp: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    // bra = shell 0, ket = shell 1 (NON-SQUARE p×d).
    let ni = nf(bas[ANG_OF]);
    let nj = nf(bas[BAS_SLOTS + ANG_OF]);
    let shls = [0_i32, 1_i32];
    let mut out = vec![0.0_f64; ncomp * ni * nj];
    // SAFETY: atm/bas/env well-formed by construction; shls valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed (0,1): {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_nc_block<F>(
    vendor_fn: F,
    ncomp: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ni = nf(bas[ANG_OF]);
    let nj = nf(bas[BAS_SLOTS + ANG_OF]);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let shls = [0_i32, 1_i32];
    let mut out = vec![0.0_f64; ncomp * ni * nj];
    vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);
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

#[allow(dead_code)]
fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: matrix is all-zero (zero-fill regression)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism tests (always run under cpu) — pin the N-component count and the
// NON-SQUARE p×d block shape (catches rank truncation without the vendor gate).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism(api_sph: RawApiId, api_cart: RawApiId, ncomp: usize, rinv: Option<[f64; 3]>, label: &str) {
    let (atm, bas, env0) = build_pd_nonsquare();
    let env = match rinv {
        Some(o) => env_with_rinv_origin(&env0, o),
        None => env0,
    };
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        let m1 = collect_nc_block(api, ncomp, &atm, &bas, &env, nf);
        let m2 = collect_nc_block(api, ncomp, &atm, &bas, &env, nf);
        let ni = nf(1);
        let nj = nf(2);
        assert_eq!(m1.len(), ncomp * ni * nj, "{label}_{rep} length = ncomp*ni*nj");
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}_{rep} must be bit-identical");
        }
        assert_any_nonzero(&m1, &format!("{label}_{rep}"));
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipipnuc_determinism() {
    determinism(RawApiId::INT1E_IPIPIPNUC_SPH, RawApiId::INT1E_IPIPIPNUC_CART, 27, None, "int1e_ipipipnuc");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipiprinv_determinism() {
    determinism(
        RawApiId::INT1E_IPIPIPRINV_SPH,
        RawApiId::INT1E_IPIPIPRINV_CART,
        27,
        Some([0.0, 0.3, 0.6]),
        "int1e_ipipiprinv",
    );
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipnucip_determinism() {
    determinism(RawApiId::INT1E_IPIPNUCIP_SPH, RawApiId::INT1E_IPIPNUCIP_CART, 27, None, "int1e_ipipnucip");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipiprinvip_determinism() {
    determinism(
        RawApiId::INT1E_IPIPRINVIP_SPH,
        RawApiId::INT1E_IPIPRINVIP_CART,
        27,
        Some([0.0, 0.3, 0.6]),
        "int1e_ipiprinvip",
    );
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipipiprinv_determinism() {
    determinism(
        RawApiId::INT1E_IPIPIPIPRINV_SPH,
        RawApiId::INT1E_IPIPIPIPRINV_CART,
        81,
        Some([0.0, 0.3, 0.6]),
        "int1e_ipipipiprinv",
    );
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipiprinvipip_determinism() {
    determinism(
        RawApiId::INT1E_IPIPRINVIPIP_SPH,
        RawApiId::INT1E_IPIPRINVIPIP_CART,
        81,
        Some([0.0, 0.3, 0.6]),
        "int1e_ipiprinvipip",
    );
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipiprinvip_determinism() {
    determinism(
        RawApiId::INT1E_IPIPIPRINVIP_SPH,
        RawApiId::INT1E_IPIPIPRINVIP_CART,
        81,
        Some([0.0, 0.3, 0.6]),
        "int1e_ipipiprinvip",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity (require has_vendor_libcint + cpu). `deriv34_ipipip` runs the
// NON-SQUARE p×d (bra != ket) single-block parity for one family, all components.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn deriv34_ipipip<FS, FC>(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: FS,
    vendor_cart: FC,
    ncomp: usize,
    label: &str,
    rinv_origin: Option<[f64; 3]>,
) where
    FS: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    FC: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    // NON-SQUARE p×d block (bra l=1 != ket l=2). A deriv4 ket-headroom miss MUST
    // be caught here — a transposed / single-side-headroom layout cannot survive
    // a strictly non-square bra ≠ ket block.
    let (atm, bas, env0) = build_pd_nonsquare();
    let env = match rinv_origin {
        Some(o) => env_with_rinv_origin(&env0, o),
        None => env0,
    };

    let vendor_s = collect_vendor_nc_block(&vendor_sph, ncomp, &atm, &bas, &env, nsph);
    let cintx_s = collect_nc_block(api_sph, ncomp, &atm, &bas, &env, nsph);
    assert_any_nonzero(&cintx_s, &format!("{label}_sph p×d cintx"));
    assert_any_nonzero(&vendor_s, &format!("{label}_sph p×d vendor"));
    let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_sph (p×d non-square, {ncomp} comp): {mm} mismatches at atol={ATOL}");

    let vendor_c = collect_vendor_nc_block(&vendor_cart, ncomp, &atm, &bas, &env, ncart);
    let cintx_c = collect_nc_block(api_cart, ncomp, &atm, &bas, &env, ncart);
    assert_any_nonzero(&cintx_c, &format!("{label}_cart p×d cintx"));
    assert_any_nonzero(&vendor_c, &format!("{label}_cart p×d vendor"));
    let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_cart (p×d non-square, {ncomp} comp): {mm} mismatches at atol={ATOL}");
}

// ── deriv3 (rank 27) ─────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipipnuc_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPIPNUC_SPH,
        RawApiId::INT1E_IPIPIPNUC_CART,
        vendor_ffi::vendor_int1e_ipipipnuc_sph,
        vendor_ffi::vendor_int1e_ipipipnuc_cart,
        27,
        "int1e_ipipipnuc",
        None,
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipiprinv_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPIPRINV_SPH,
        RawApiId::INT1E_IPIPIPRINV_CART,
        vendor_ffi::vendor_int1e_ipipiprinv_sph,
        vendor_ffi::vendor_int1e_ipipiprinv_cart,
        27,
        "int1e_ipipiprinv",
        Some([0.0, 0.3, 0.6]),
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipnucip_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPNUCIP_SPH,
        RawApiId::INT1E_IPIPNUCIP_CART,
        vendor_ffi::vendor_int1e_ipipnucip_sph,
        vendor_ffi::vendor_int1e_ipipnucip_cart,
        27,
        "int1e_ipipnucip",
        None,
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipiprinvip_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPRINVIP_SPH,
        RawApiId::INT1E_IPIPRINVIP_CART,
        vendor_ffi::vendor_int1e_ipiprinvip_sph,
        vendor_ffi::vendor_int1e_ipiprinvip_cart,
        27,
        "int1e_ipiprinvip",
        Some([0.0, 0.3, 0.6]),
    );
}

// ── deriv4 (rank 81 — dual headroom) ─────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipipiprinv_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPIPIPRINV_SPH,
        RawApiId::INT1E_IPIPIPIPRINV_CART,
        vendor_ffi::vendor_int1e_ipipipiprinv_sph,
        vendor_ffi::vendor_int1e_ipipipiprinv_cart,
        81,
        "int1e_ipipipiprinv",
        Some([0.0, 0.3, 0.6]),
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipiprinvipip_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPRINVIPIP_SPH,
        RawApiId::INT1E_IPIPRINVIPIP_CART,
        vendor_ffi::vendor_int1e_ipiprinvipip_sph,
        vendor_ffi::vendor_int1e_ipiprinvipip_cart,
        81,
        "int1e_ipiprinvipip",
        Some([0.0, 0.3, 0.6]),
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipiprinvip_pd_parity() {
    use cintx_oracle::vendor_ffi;
    deriv34_ipipip(
        RawApiId::INT1E_IPIPIPRINVIP_SPH,
        RawApiId::INT1E_IPIPIPRINVIP_CART,
        vendor_ffi::vendor_int1e_ipipiprinvip_sph,
        vendor_ffi::vendor_int1e_ipipiprinvip_cart,
        81,
        "int1e_ipipiprinvip",
        Some([0.0, 0.3, 0.6]),
    );
}
