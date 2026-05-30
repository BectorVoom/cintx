//! Oracle parity tests for int3c1e GENERAL-CONTRACTION (nctr>1) support — WR-03.
//!
//! Covers the scalar `int3c1e` and the two derivative launchers `int3c1e_ip1`
//! (∇ on bra i of the 3-center OVERLAP, no Rys) and `int3c1e_iprinv` (∇ on bra i
//! of the 3-center rinv-COULOMB, Rys-driven) on a GENERALLY-CONTRACTED, NON-SQUARE
//! shell triple so neither a dropped contraction column nor a transposed layout
//! can pass.
//!
//! Fixture: bra i = p-shell with nctr_i = 2 (two distinct coefficient columns over
//! the SAME 3 primitives), j = d-shell (nctr=1), k = s-shell (nctr=1). The block is
//! genuinely non-square (di != dj != dk) AND general-contracted in i.
//!
//! ── Confirmed libcint nctr>1 output block ordering (the heart of WR-03) ──
//! From `CINT3c1e_drv` + `c2s_{cart,sph}_3c2e1` (cint3c1e.c / cart2sph.c):
//!   * The output is a SINGLE dense 3-D array of shape
//!       [ni_full, nj_full, nk_full] = [di*nctr_i, dj*nctr_j, dk*nctr_k]
//!     in i-fastest / k-slowest (column-major i,j,k) order, where di/dj/dk are the
//!     per-shell angular block lengths (ncart or nsph).
//!   * Contraction is the MAJOR (outer / slow) index WITHIN each axis:
//!       i_global = ci*di + i_idx ,  j_global = cj*dj + j_idx ,  k_global = ck*dk + k_idx
//!     i.e. `pijk = ofk*ck + ofj*cj + di*ci` with ofj = ni_full*dj, ofk = ni_full*nj_full*dk,
//!     then `dcopy_iklj` scatters the per-(ci,cj,ck) block with strides (1, ni_full, ni_full*nj_full).
//!   * This is exactly the 1e contraction-MAJOR convention (AO = ctr*(2l+1)+m).
//!     It is NOT a stack of independent (ci*nctr_j+cj)*block_len blocks — the
//!     contraction index interleaves with the angular index per axis.
//! For gradients (ip1 / iprinv) the 3 components are the OUTERMOST (component-leading)
//! dimension: out[comp * (ni_full*nj_full*nk_full) + <interleaved nctr/angular index>].
//!
//! Double gate: vendor parity assertions require `--features cpu` (cubecl cpu
//! backend) AND `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint` cfg).
//! Without BOTH the parity tests compile out and the banner shows `running 0 tests`.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

// ─────────────────────────────────────────────────────────────────────────────
// General-contraction fixture.
//
//   shell 0: bra i  = p-shell (l=1), 3 primitives, nctr_i = 2  (gc, two columns)
//   shell 1: ket j  = d-shell (l=2), 3 primitives, nctr_j = 1
//   shell 2: aux k  = s-shell (l=0), 3 primitives, nctr_k = 1
//
// Distinct, displaced atom centers so the gradient is non-trivial. A non-zero
// rinv origin (env[PTR_RINV_ORIG..+3]) drives the iprinv path.
// ─────────────────────────────────────────────────────────────────────────────

fn build_genctr_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let i_coord = [0.0_f64, 0.0, 0.0];
    let j_coord = [0.0_f64, 1.3, 0.7];
    let k_coord = [0.9_f64, -0.4, 0.2];

    // p-shell (bra i) — 3 primitives, two general-contraction columns.
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    // coefficients[ip * nctr_i + ci]: row-major (libcint/cintx canonical).
    //   column 0 = (0.70, 0.30, 0.15) , column 1 = (0.20, 0.55, 0.80)
    let p_coeff = [0.70_f64, 0.20, 0.30, 0.55, 0.15, 0.80];

    // d-shell (ket j) — 3 primitives, single contraction.
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    // s-shell (aux k) — 3 primitives, single contraction.
    let s_exp = [130.7093200_f64, 23.8088610, 6.4436083];
    let s_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    let mut env = Vec::<f64>::new();
    // Reserve the libcint global-parameter region; rinv origin lives in env[4..6].
    env.resize(20, 0.0);
    env[PTR_RINV_ORIG] = 0.30;
    env[PTR_RINV_ORIG + 1] = -0.45;
    env[PTR_RINV_ORIG + 2] = 0.60;

    let i_coord_ptr = env.len() as i32;
    env.extend_from_slice(&i_coord);
    let j_coord_ptr = env.len() as i32;
    env.extend_from_slice(&j_coord);
    let k_coord_ptr = env.len() as i32;
    env.extend_from_slice(&k_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p_exp);
    let p_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p_coeff);

    let d_exp_ptr = env.len() as i32;
    env.extend_from_slice(&d_exp);
    let d_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&d_coeff);

    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for (n, &ptr) in [i_coord_ptr, j_coord_ptr, k_coord_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    // shell 0: p, nctr=2 (general contraction)
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 2;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1: d, nctr=1
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 3;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;
    // shell 2: s, nctr=1
    bas[2 * BAS_SLOTS + ATOM_OF] = 2;
    bas[2 * BAS_SLOTS + ANG_OF] = 0;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + PTR_EXP] = s_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = s_coeff_ptr;

    (atm, bas, env)
}

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}
fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

fn shell_ang(bas: &[i32], s: usize) -> i32 {
    bas[s * BAS_SLOTS + ANG_OF]
}
fn shell_nctr(bas: &[i32], s: usize) -> usize {
    bas[s * BAS_SLOTS + NCTR_OF] as usize
}

// The single shell triple under test: (i=p nctr2, j=d, k=s). Non-square + gc.
const TRIPLE: (usize, usize, usize) = (0, 1, 2);

/// Total element count = ncomp * (di*nctr_i) * (dj*nctr_j) * (dk*nctr_k).
fn elem_count(bas: &[i32], triple: (usize, usize, usize), nf: impl Fn(i32) -> usize, ncomp: usize) -> usize {
    let (si, sj, sk) = triple;
    let ni = nf(shell_ang(bas, si)) * shell_nctr(bas, si);
    let nj = nf(shell_ang(bas, sj)) * shell_nctr(bas, sj);
    let nk = nf(shell_ang(bas, sk)) * shell_nctr(bas, sk);
    ncomp * ni * nj * nk
}

fn collect_cintx(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    triple: (usize, usize, usize),
    nf: impl Fn(i32) -> usize,
    ncomp: usize,
) -> Vec<f64> {
    let n_elem = elem_count(bas, triple, &nf, ncomp);
    let mut out = vec![0.0_f64; n_elem];
    let (si, sj, sk) = triple;
    let shls = [si as i32, sj as i32, sk as i32];
    // SAFETY: atm/bas/env well-formed by construction; shls valid; out sized exactly.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for {api_id:?} triple {triple:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    triple: (usize, usize, usize),
    nf: impl Fn(i32) -> usize,
    ncomp: usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let n_elem = elem_count(bas, triple, &nf, ncomp);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; n_elem];
    let (si, sj, sk) = triple;
    let shls = [si as i32, sj as i32, sk as i32];
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
fn assert_any_nonzero(buf: &[f64], label: &str) {
    let any_nonzero = buf.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: buffer is all-zero (zero-fill regression)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism (always under cpu): the nctr>1 element count is correct and the
// buffer is bit-reproducible and non-zero.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism(api_sph: RawApiId, api_cart: RawApiId, ncomp: usize, label: &str) {
    let (atm, bas, env) = build_genctr_fixture();
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        let m1 = collect_cintx(api, &atm, &bas, &env, TRIPLE, nf, ncomp);
        let m2 = collect_cintx(api, &atm, &bas, &env, TRIPLE, nf, ncomp);
        let expect = elem_count(&bas, TRIPLE, nf, ncomp);
        assert_eq!(m1.len(), expect, "{label}_{rep}: element count = ncomp*(di*nctr_i)*(dj*nctr_j)*(dk*nctr_k)");
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}_{rep} must be bit-identical");
        }
        assert_any_nonzero(&m1, &format!("{label}_{rep}"));
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_genctr_determinism() {
    determinism(RawApiId::INT3C1E_SPH, RawApiId::INT3C1E_CART, 1, "int3c1e");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_ip1_genctr_determinism() {
    determinism(RawApiId::INT3C1E_IP1_SPH, RawApiId::INT3C1E_IP1_CART, 3, "int3c1e_ip1");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_iprinv_genctr_determinism() {
    determinism(RawApiId::INT3C1E_IPRINV_SPH, RawApiId::INT3C1E_IPRINV_CART, 3, "int3c1e_iprinv");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor byte-identity (require has_vendor_libcint + cpu).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
fn vendor_parity<FS, FC>(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: FS,
    vendor_cart: FC,
    ncomp: usize,
    label: &str,
) where
    FS: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    FC: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let (atm, bas, env) = build_genctr_fixture();

    let vendor_s = collect_vendor(&vendor_sph, &atm, &bas, &env, TRIPLE, nsph, ncomp);
    let cintx_s = collect_cintx(api_sph, &atm, &bas, &env, TRIPLE, nsph, ncomp);
    assert_any_nonzero(&cintx_s, &format!("{label}_sph cintx"));
    assert_any_nonzero(&vendor_s, &format!("{label}_sph vendor"));
    let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_sph (nctr>1): {mm} mismatches vs vendored libcint at atol={ATOL}");

    let vendor_c = collect_vendor(&vendor_cart, &atm, &bas, &env, TRIPLE, ncart, ncomp);
    let cintx_c = collect_cintx(api_cart, &atm, &bas, &env, TRIPLE, ncart, ncomp);
    assert_any_nonzero(&cintx_c, &format!("{label}_cart cintx"));
    assert_any_nonzero(&vendor_c, &format!("{label}_cart vendor"));
    let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_cart (nctr>1): {mm} mismatches vs vendored libcint at atol={ATOL}");
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_genctr_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT3C1E_SPH,
        RawApiId::INT3C1E_CART,
        vendor_ffi::vendor_int3c1e_sph,
        vendor_ffi::vendor_int3c1e_cart,
        1,
        "int3c1e",
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_ip1_genctr_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT3C1E_IP1_SPH,
        RawApiId::INT3C1E_IP1_CART,
        vendor_ffi::vendor_int3c1e_ip1_sph,
        vendor_ffi::vendor_int3c1e_ip1_cart,
        3,
        "int3c1e_ip1",
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_iprinv_genctr_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT3C1E_IPRINV_SPH,
        RawApiId::INT3C1E_IPRINV_CART,
        vendor_ffi::vendor_int3c1e_iprinv_sph,
        vendor_ffi::vendor_int3c1e_iprinv_cart,
        3,
        "int3c1e_iprinv",
    );
}
