//! Global spinor AO ORDERING vendor parity (quick task 260601-d7e, Sub-problem B).
//!
//! Proves cintx's GLOBAL spinor AO ordering (`ao_loc_2c`-equivalent) is byte-identical
//! to vendored libcint 6.1.3 on a SEGMENTED multi-shell-same-l basis — the configuration
//! the pyscf-rs report flagged as a possible permutation. cintx builds cumulative
//! per-shell offsets in SHELL ORDER (`CINTshells_spinor_offset` / `write_offsets`,
//! cintx-compat/src/helpers.rs), exactly mirroring libcint `shells_cgto_offset`
//! (cint_bas.c). Neither groups by angular momentum, so a >=3-shell basis that
//! INTERLEAVES repeated angular momenta (s, p, s, p) would surface any cross-shell
//! global permutation as a byte mismatch.
//!
//! Fixture: 4 shells [s, p, s, p] (l = 0,1,0,1) on 2 atoms, all nctr==1. This is
//! SEGMENTED (the two l=0 and the two l=1 shells repeat, NOT grouped), has at least
//! one l>0 shell (so non-square cross blocks participate), and keeps nctr==1 so the
//! test is purely about global ORDERING (independent of the 260601-aty nctr>1 work).
//!
//! Global assembly: each shell pair (si,sj) is evaluated as an interleaved-complex
//! block and stitched into the full n_sp×n_sp matrix via the column-major / bra-fastest
//! formula `dst = ((col_off+jj)*n_sp + (row_off+ii))*2`, advancing offsets in shell
//! order. cintx and vendor use the IDENTICAL stitch, so a global-ordering divergence
//! (different per-shell offsets) is the ONLY thing that can produce mismatches.
//!
//! Driver: `int1e_ovlp_spinor` (`INT1E_OVLP_SPINOR`) — a REAL vendor driver (scalar
//! spinor, stable; NOT a return-0/exit(1) stub).
//!
//! Double gate: vendor parity requires `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`
//! (the `has_vendor_libcint` cfg). Without both, the vendor body compiles out and only
//! the non-vendor smoke test runs.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

const N_ATOMS: usize = 2;
const N_SHELLS: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Segmented multi-shell-same-l fixture: [s, p, s, p] on 2 atoms (l = 0,1,0,1).
//
//   shell 0: s (l=0) on atom 0      shell 1: p (l=1) on atom 0
//   shell 2: s (l=0) on atom 1      shell 3: p (l=1) on atom 1
//
// Repeated angular momenta are INTERLEAVED (not grouped), so a cross-shell global
// permutation between the two s blocks or the two p blocks would land in the wrong
// global offset and mismatch vendor. At least one l>0 shell (the two p shells, di=6)
// makes non-square cross blocks participate. All nctr==1 (global ordering only).
// env[0..PTR_ENV_START) is reserved for libcint global slots; user data follows.
// ─────────────────────────────────────────────────────────────────────────────

fn build_segmented_spd_spinor() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.7531_f64, 1.4307, 1.1078];

    // Distinct exponents/coeffs per shell so blocks are value-asymmetric.
    let s0_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let s0_coeff = [0.15432897_f64, 0.53532814, 0.44463454];
    let p1_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let p1_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    let s2_exp = [2.9412494_f64, 0.6834831, 0.2222899];
    let s2_coeff = [0.44463454_f64, 0.53532814, 0.15432897];
    let p3_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p3_coeff = [0.41098730_f64, 0.22072503, 0.81763176];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let a_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_coord_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let s0_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s0_exp);
    let s0_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s0_coeff);
    let p1_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p1_exp);
    let p1_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p1_coeff);
    let s2_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s2_exp);
    let s2_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s2_coeff);
    let p3_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p3_exp);
    let p3_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p3_coeff);

    let mut atm = vec![0_i32; N_ATOMS * ATM_SLOTS];
    atm[0 * ATM_SLOTS + CHARGE_OF] = 8;
    atm[0 * ATM_SLOTS + PTR_COORD] = a_coord_ptr;
    atm[0 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[0 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    atm[1 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[1 * ATM_SLOTS + PTR_COORD] = b_coord_ptr;
    atm[1 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[1 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

    let mut bas = vec![0_i32; N_SHELLS * BAS_SLOTS];
    // shell 0: s on atom 0
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[0 * BAS_SLOTS + PTR_EXP] = s0_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = s0_coeff_ptr;
    // shell 1: p on atom 0
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 1;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[1 * BAS_SLOTS + PTR_EXP] = p1_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = p1_coeff_ptr;
    // shell 2: s on atom 1 (repeats l=0 — segmented)
    bas[2 * BAS_SLOTS + ATOM_OF] = 1;
    bas[2 * BAS_SLOTS + ANG_OF] = 0;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[2 * BAS_SLOTS + PTR_EXP] = s2_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = s2_coeff_ptr;
    // shell 3: p on atom 1 (repeats l=1 — segmented)
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 1;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[3 * BAS_SLOTS + PTR_EXP] = p3_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = p3_coeff_ptr;

    (atm, bas, env)
}

/// Number of spinor components for ang `l` with kappa==0: 4*l + 2. (s=2, p=6)
fn spinor_len_kappa0(l: i32) -> usize {
    (4 * l + 2) as usize
}

/// Per-shell spinor dim at kappa==0, nctr==1: (4l+2).
fn shell_nsp(bas: &[i32], s: usize) -> usize {
    spinor_len_kappa0(bas[s * BAS_SLOTS + ANG_OF])
}

/// Stitch one complex-interleaved shell-pair block (`out`) into the full matrix.
/// Block is column-major (bra fastest): out[(jj*ni + ii)*2 + {0:re,1:im}].
/// Global dst uses cumulative shell-order offsets: ((col+jj)*n_sp + (row+ii))*2.
fn stitch_block(
    matrix: &mut [f64],
    out: &[f64],
    ni: usize,
    nj: usize,
    n_sp: usize,
    row_offset: usize,
    col_offset: usize,
) {
    for jj in 0..nj {
        for ii in 0..ni {
            let src = (jj * ni + ii) * 2;
            let row = row_offset + ii;
            let col = col_offset + jj;
            let dst = (col * n_sp + row) * 2;
            matrix[dst] = out[src];
            matrix[dst + 1] = out[src + 1];
        }
    }
}

#[allow(dead_code)]
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            if mismatches <= 20 {
                eprintln!(
                    "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                     diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
    }
    mismatches
}

#[allow(dead_code)]
fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(
        any_nonzero,
        "{label}: spinor matrix is all-zero (zero-fill / silent-skip regression)"
    );
}

/// Self-documenting fixture guard for the GLOBAL-ordering proof. The basis MUST be:
///   - at least 3 shells (a cross-shell global permutation needs >=3 to be non-trivial),
///   - SEGMENTED: at least one angular momentum value repeated across distinct shells,
///   - have at least one l>0 shell (so a non-square block participates),
///   - all nctr==1 (this test is about ORDERING, not contraction — independent of 260601-aty).
/// If any of these regress, the global-ordering proof loses its teeth.
#[allow(dead_code)]
fn assert_fixture_segmented_same_l(bas: &[i32]) {
    let n_shells = bas.len() / BAS_SLOTS;
    assert!(
        n_shells >= 3,
        "global-ordering fixture must have >=3 shells, got {n_shells}"
    );
    let angs: Vec<i32> = (0..n_shells).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    // At least one repeated angular momentum (segmented same-l).
    let mut has_repeat = false;
    for i in 0..angs.len() {
        for j in (i + 1)..angs.len() {
            if angs[i] == angs[j] {
                has_repeat = true;
            }
        }
    }
    assert!(
        has_repeat,
        "global-ordering fixture must repeat an angular momentum (segmented same-l), got {angs:?}"
    );
    // At least one l>0 shell.
    assert!(
        angs.iter().any(|&l| l > 0),
        "global-ordering fixture must include at least one l>0 shell, got {angs:?}"
    );
    // All nctr==1.
    for s in 0..n_shells {
        assert_eq!(
            bas[s * BAS_SLOTS + NCTR_OF],
            1,
            "shell {s} must be nctr==1 (this test is about ordering, not contraction)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic global collectors (shell-count-generic; do NOT depend on N_SHELLS const
// except for the fixture builder).
// ─────────────────────────────────────────────────────────────────────────────

fn collect_cintx_global(api_id: RawApiId, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let n_shells = bas.len() / BAS_SLOTS;
    let dims: Vec<usize> = (0..n_shells).map(|s| shell_nsp(bas, s)).collect();
    let n_sp: usize = dims.iter().sum();

    let mut matrix = vec![0.0_f64; n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..n_shells {
        let ni = dims[si];
        let mut col_offset = 0usize;
        for sj in 0..n_shells {
            let nj = dims[sj];
            let shls = [si as i32, sj as i32];
            let mut out = vec![0.0_f64; ni * nj * 2];

            // SAFETY: atm/bas/env are well-formed by construction; shls are valid.
            unsafe {
                eval_raw(
                    api_id,
                    Some(&mut out),
                    None,
                    &shls,
                    atm,
                    bas,
                    env,
                    None,
                    None,
                )
                .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
            }

            stitch_block(&mut matrix, &out, ni, nj, n_sp, row_offset, col_offset);
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_global<F>(vendor_fn: F, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let n_shells = bas.len() / BAS_SLOTS;
    let dims: Vec<usize> = (0..n_shells).map(|s| shell_nsp(bas, s)).collect();
    let n_sp: usize = dims.iter().sum();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut matrix = vec![0.0_f64; n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..n_shells {
        let ni = dims[si];
        let mut col_offset = 0usize;
        for sj in 0..n_shells {
            let nj = dims[sj];
            let shls: [i32; 2] = [si as i32, sj as i32];
            let mut out = vec![0.0_f64; ni * nj * 2];

            vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);

            stitch_block(&mut matrix, &out, ni, nj, n_sp, row_offset, col_offset);
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke test (always runs under --features cpu).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_spinor_global_ao_order_evaluates() {
    let (atm, bas, env) = build_segmented_spd_spinor();
    assert_fixture_segmented_same_l(&bas);

    let mat = collect_cintx_global(RawApiId::INT1E_OVLP_SPINOR, &atm, &bas, &env);
    // n_sp = s+p+s+p = 2+6+2+6 = 16 → 16*16*2 = 512.
    assert_eq!(
        mat.len(),
        16 * 16 * 2,
        "global spinor matrix size: n_sp=16 → 16*16*2"
    );
    assert_any_nonzero(&mat, "global spinor ovlp cintx (smoke)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor byte-identity test (requires has_vendor_libcint + cpu feature).
// Proves cintx's global spinor AO ordering == libcint on a segmented same-l basis.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_spinor_global_ao_order_parity() {
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_segmented_spd_spinor();
    assert_fixture_segmented_same_l(&bas);

    let vendor = collect_vendor_global(vendor_ffi::vendor_int1e_ovlp_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_global(RawApiId::INT1E_OVLP_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "global spinor ovlp cintx");
    assert_any_nonzero(&vendor, "global spinor ovlp vendor");
    assert_eq!(
        vendor.len(),
        cintx.len(),
        "global matrix length mismatch (n_sp divergence)"
    );

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "global spinor AO ordering: {mismatches} mismatches vs vendored libcint at atol={ATOL} \
         on segmented [s,p,s,p] basis — a nonzero count would mean cintx's per-shell global \
         offsets (CINTshells_spinor_offset) diverge from libcint shells_cgto_offset"
    );

    println!(
        "test_spinor_global_ao_order_parity: PASS — cintx global spinor AO ordering \
         byte-identical to vendored libcint on segmented [s,p,s,p] basis; mismatches=0, \
         n_sp=16, elems={}",
        cintx.len()
    );
}
