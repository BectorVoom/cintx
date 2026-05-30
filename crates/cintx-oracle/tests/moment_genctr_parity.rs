//! Phase 24 GENERAL-CONTRACTION (nctr>1) vendor parity for a moment family — WR-03.
//!
//! Every other Phase 24 moment parity test runs on the H2O/STO-3G corpus, where ALL
//! shells are `nctr == 1`. The project's hard-won lesson (the raw `eval_raw`
//! column-major↔row-major coefficient transpose bug, latent because *all prior
//! fixtures were nctr==1*) is that a new family's `nctr>1` path is exercised by NO
//! test until one is deliberately added. This file closes that gap for the Cluster-A
//! moment kernels: it byte-checks the contraction-blocked staging stride
//! (`ii = ci*nci + ic`, `jj = cj*ncj + jc`, `ni_full = nctr_i*nci`) that the moment
//! path emits, against vendored libcint, on a generally-contracted NON-SQUARE block.
//!
//! Representative family: `int1e_rr` (rank 9) — high enough rank to exercise the
//! component-leading + contraction-blocked interleave, low enough that the rank-9
//! buffer stays small.
//!
//! ── Confirmed libcint nctr>1 1e block ordering (the heart of WR-03) ──
//! From `CINT1e_drv` + `c2s_{cart,sph}_1e` (cint1e.c / cart2sph.c): the per-component
//! output is a dense 2-D array of shape `[ni_full, nj_full] = [di*nctr_i, dj*nctr_j]`
//! in i-fastest (column-major i,j) order, contraction the MAJOR index WITHIN each
//! axis: `i_global = ci*di + i_idx`, `j_global = cj*dj + j_idx`. The `rr` rank-9
//! components are the OUTERMOST (component-leading) dimension:
//!   out[comp * (ni_full*nj_full) + (j_global*ni_full + i_global)].
//! This exactly matches the cintx moment staging loops.
//!
//! Double gate: vendor parity assertions require `--features cpu` (cubecl cpu
//! backend) AND `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint` cfg). Without
//! BOTH the parity test compiles out and the banner shows `running 0 tests`.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COMMON_ORIG, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

// ─────────────────────────────────────────────────────────────────────────────
// General-contraction, non-square, two-center fixture.
//
//   shell 0: bra i = p-shell (l=1), 3 primitives, nctr_i = 2  (gc, two columns)
//   shell 1: ket j = d-shell (l=2), 3 primitives, nctr_j = 1
//
// Distinct, displaced atom centers so the block is genuinely cross-center. A
// NON-ZERO gauge origin (env[PTR_COMMON_ORIG..+3]) drives the `rr` position tensor
// (D-02: base moment families read the gauge origin via G1E_RCJ).
// ─────────────────────────────────────────────────────────────────────────────

fn build_moment_genctr_fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let i_coord = [0.0_f64, 0.0, 0.0];
    let j_coord = [0.0_f64, 1.3, 0.7];

    // p-shell (bra i) — 3 primitives, two general-contraction columns.
    let p_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    // The libcint env coefficient block is COLUMN-MAJOR: env[ci*nprim + ip] (see
    // CINTprim_to_ctr_0 in g1e.c). cintx transposes it to row-major internally.
    //   column 0 = (0.70, 0.30, 0.15) , column 1 = (0.20, 0.55, 0.80)
    // → env layout [c0_p0, c0_p1, c0_p2, c1_p0, c1_p1, c1_p2].
    let p_coeff = [0.70_f64, 0.30, 0.15, 0.20, 0.55, 0.80];

    // d-shell (ket j) — 3 primitives, single contraction.
    let d_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let d_coeff = [0.15591627_f64, 0.60768372, 0.39195739];

    let mut env = Vec::<f64>::new();
    // Reserve the libcint global-parameter region; gauge origin lives in env[1..4].
    env.resize(20, 0.0);
    env[PTR_COMMON_ORIG] = 0.30;
    env[PTR_COMMON_ORIG + 1] = -0.45;
    env[PTR_COMMON_ORIG + 2] = 0.60;

    let i_coord_ptr = env.len() as i32;
    env.extend_from_slice(&i_coord);
    let j_coord_ptr = env.len() as i32;
    env.extend_from_slice(&j_coord);
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

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (n, &ptr) in [i_coord_ptr, j_coord_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
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

// The single shell pair under test: (i = p nctr2, j = d). Non-square + gc.
const PAIR: (usize, usize) = (0, 1);

/// Total element count = rank * (di*nctr_i) * (dj*nctr_j).
fn elem_count(bas: &[i32], pair: (usize, usize), nf: impl Fn(i32) -> usize, rank: usize) -> usize {
    let (si, sj) = pair;
    let ni = nf(shell_ang(bas, si)) * shell_nctr(bas, si);
    let nj = nf(shell_ang(bas, sj)) * shell_nctr(bas, sj);
    rank * ni * nj
}

fn collect_cintx(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    pair: (usize, usize),
    nf: impl Fn(i32) -> usize,
    rank: usize,
) -> Vec<f64> {
    let n_elem = elem_count(bas, pair, &nf, rank);
    let mut out = vec![0.0_f64; n_elem];
    let (si, sj) = pair;
    let shls = [si as i32, sj as i32];
    // SAFETY: atm/bas/env well-formed by construction; shls valid; out sized exactly.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for {api_id:?} pair {pair:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    pair: (usize, usize),
    nf: impl Fn(i32) -> usize,
    rank: usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let n_elem = elem_count(bas, pair, &nf, rank);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; n_elem];
    let (si, sj) = pair;
    let shls = [si as i32, sj as i32];
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

/// Per-component stuck-at-zero gate (WR-04 applied to the nctr>1 block): every
/// vendor-populated component must be populated by cintx too.
#[allow(dead_code)]
fn assert_components_match_vendor_support(vendor: &[f64], observed: &[f64], rank: usize, label: &str) {
    assert_eq!(vendor.len(), observed.len(), "{label}: length mismatch");
    assert_eq!(vendor.len() % rank, 0, "{label}: not divisible by rank {rank}");
    let comp_len = vendor.len() / rank;
    for comp in 0..rank {
        let lo = comp * comp_len;
        let hi = lo + comp_len;
        if vendor[lo..hi].iter().any(|v| v.abs() > 1e-14) {
            assert!(
                observed[lo..hi].iter().any(|v| v.abs() > 1e-14),
                "{label}: component {comp}/{rank} all-zero in cintx but non-zero in vendor"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism (always under cpu): the nctr>1 element count is correct and the
// buffer is bit-reproducible and non-zero.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism(api_sph: RawApiId, api_cart: RawApiId, rank: usize, label: &str) {
    let (atm, bas, env) = build_moment_genctr_fixture();
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        let m1 = collect_cintx(api, &atm, &bas, &env, PAIR, nf, rank);
        let m2 = collect_cintx(api, &atm, &bas, &env, PAIR, nf, rank);
        let expect = elem_count(&bas, PAIR, nf, rank);
        assert_eq!(
            m1.len(),
            expect,
            "{label}_{rep}: element count = rank*(di*nctr_i)*(dj*nctr_j)"
        );
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}_{rep} must be bit-identical");
        }
        assert_any_nonzero(&m1, &format!("{label}_{rep}"));
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_rr_genctr_determinism() {
    determinism(RawApiId::INT1E_RR_SPH, RawApiId::INT1E_RR_CART, 9, "int1e_rr");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor byte-identity (require has_vendor_libcint + cpu). This is the WR-03 gate:
// the contraction-blocked staging stride for a Phase-24 moment family is byte-checked
// against vendored libcint on an nctr>1, non-square, cross-center block.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
fn vendor_parity<FS, FC>(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: FS,
    vendor_cart: FC,
    rank: usize,
    label: &str,
) where
    FS: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    FC: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let (atm, bas, env) = build_moment_genctr_fixture();

    let vendor_s = collect_vendor(&vendor_sph, &atm, &bas, &env, PAIR, nsph, rank);
    let cintx_s = collect_cintx(api_sph, &atm, &bas, &env, PAIR, nsph, rank);
    assert_any_nonzero(&cintx_s, &format!("{label}_sph cintx"));
    assert_any_nonzero(&vendor_s, &format!("{label}_sph vendor"));
    assert_components_match_vendor_support(&vendor_s, &cintx_s, rank, &format!("{label}_sph"));
    let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_sph (nctr>1): {mm} mismatches vs vendored libcint at atol={ATOL}");

    let vendor_c = collect_vendor(&vendor_cart, &atm, &bas, &env, PAIR, ncart, rank);
    let cintx_c = collect_cintx(api_cart, &atm, &bas, &env, PAIR, ncart, rank);
    assert_any_nonzero(&cintx_c, &format!("{label}_cart cintx"));
    assert_any_nonzero(&vendor_c, &format!("{label}_cart vendor"));
    assert_components_match_vendor_support(&vendor_c, &cintx_c, rank, &format!("{label}_cart"));
    let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_cart (nctr>1): {mm} mismatches vs vendored libcint at atol={ATOL}");
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_rr_genctr_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT1E_RR_SPH,
        RawApiId::INT1E_RR_CART,
        vendor_ffi::vendor_int1e_rr_sph,
        vendor_ffi::vendor_int1e_rr_cart,
        9,
        "int1e_rr",
    );
}
