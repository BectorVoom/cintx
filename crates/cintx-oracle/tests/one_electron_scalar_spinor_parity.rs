//! Oracle parity tests for the three SCALAR spinor int1e operators:
//!   int1e_ovlp_spinor, int1e_kin_spinor, int1e_nuc_spinor.
//!
//! These tests prove the cart→spinor block-orientation fix for the SCALAR spinor 1e
//! path (quick task 260529-kke), the exact mirror of the GRADIENT-path fix (260529-jtd).
//!
//! The scalar spinor arm (one_electron.rs ~line 2906) feeds the device scalar kernel's
//! ket-major / bra-fastest Cartesian block (`block[cj*nci + ci]`) into
//! `cart_to_spinor_sf_2d`, which reads bra-major / ket-fastest (`cart[bra*ncj + ket]`,
//! c2spinor.rs apply_bra_block). The orientation only matters when BOTH nci>1 AND
//! ncj>1 for two DISTINCT shells — every H2O/STO-3G cross block has an s side
//! (nci==1 or ncj==1) and is transpose-invariant, so the bug stays hidden there.
//!
//! This fixture deliberately uses TWO DISTINCT p shells on DIFFERENT centers with
//! DIFFERENT exponents/coefficients so the (0,1) / (1,0) p⁺×p⁺ cross block is a
//! genuine asymmetric nci=ncj=3 block — the only configuration that surfaces the bug.
//!
//! Each operator produces an interleaved-complex buffer per shell pair:
//!   out[(j*ni_sp + i)*2 + {0:re, 1:im}]
//! where ni_sp = CINTcgto_spinor(shls[0]), nj_sp = CINTcgto_spinor(shls[1]).
//! Layout convention matches libcint c2s_sf_1e (column-major: bra fastest).
//!
//! Vendor parity is double-gated: it only runs under `--features cpu` AND env
//! `CINTX_ORACLE_BUILD_VENDOR=1` (which makes build.rs set `has_vendor_libcint`).
//! Without both, the vendor bodies are cfg'd out and the test compiles to a no-op
//! plus the non-vendor smoke tests.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

const N_SHELLS: usize = 2;
const N_ATOMS: usize = 2;

// ─────────────────────────────────────────────────────────────────────────────
// Asymmetric p×d fixture — PTR_ENV_START-aligned, spinor shells (kappa=0).
//
// Guarantees a NON-SQUARE p⁺×d⁺ cross block with nci=3, ncj=6 and i!=j:
//   - 2 atoms at DIFFERENT coordinates.
//   - shell 0 = p (l=1, nci=3), shell 1 = d (l=2, ncj=6), on distinct atoms with
//     distinct exponents/coefficients.
//
// Why DIFFERENT angular momenta and not two p shells: the scalar OVERLAP Cartesian
// p×p block is intrinsically transpose-SYMMETRIC (S_{μν} == S_{νμ} for two p shells,
// regardless of centers/exponents), so a ket-major↔bra-major misread is invisible
// even with fully general geometry — empirically the cart block came back symmetric
// to ~1e-16. A NON-SQUARE 3×6 block cannot be its own transpose: reading the
// ket-major (`block[cj*nci+ci]`, 6 outer × 3 inner) device buffer as if bra-major
// (`cart[bra*ncj+ket]`, 3 outer × 6 inner) addresses entirely different elements,
// so the orientation bug becomes a guaranteed, unambiguous mismatch. This is the
// p×d analogue of jtd's asymmetric ipnuc/iprinv gradient blocks.
// env[0..PTR_ENV_START) is reserved for libcint global slots; user data follows.
// ─────────────────────────────────────────────────────────────────────────────

fn build_two_p_spinor() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Atom A at origin, atom B displaced — distinct centers, fully general (all three
    // components distinct and nonzero) so no accidental geometric symmetry survives.
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.7531_f64, 1.4307, 1.1078];

    // Shell 0: p triple (distinct exponents + coeffs).
    let p0_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    let p0_coeff = [0.15591627_f64, 0.60768372, 0.39195739];
    // Shell 1: a distinct d set (different exponents/coeffs) so the p×d cross block is
    // non-square (3×6) and value-asymmetric.
    let p1_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p1_coeff = [0.15432897_f64, 0.53532814, 0.44463454];

    // Reserve the libcint global slots [0..PTR_ENV_START); all user data follows.
    let mut env = vec![0.0_f64; PTR_ENV_START];

    let a_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_coord_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p0_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p0_exp);
    let p0_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p0_coeff);

    let p1_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p1_exp);
    let p1_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p1_coeff);

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
    // Shell 0: p on atom 0.
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 1;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[0 * BAS_SLOTS + PTR_EXP] = p0_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = p0_coeff_ptr;
    // Shell 1: distinct d on atom 1 (l=2 → non-square 3×6 cross block).
    bas[1 * BAS_SLOTS + ATOM_OF] = 1;
    bas[1 * BAS_SLOTS + ANG_OF] = 2;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[1 * BAS_SLOTS + PTR_EXP] = p1_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = p1_coeff_ptr;

    (atm, bas, env)
}

/// Number of spinor components for ang `l` with kappa==0: 4*l + 2. (s=2, p=6)
fn spinor_len_kappa0(l: i32) -> usize {
    (4 * l + 2) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// nctr>1 (general-contraction) variant of the asymmetric p×d spinor fixture
// (quick task 260601-aty). Same NON-SQUARE p(l=1)×d(l=2) on distinct atoms, but
// NCTR_OF=2 on each shell with 3 primitives → two contraction columns of DISTINCT
// coefficients. The env coeff block is COLUMN-major: env[ci*nprim + ip] (the libcint
// convention — see memory raw_nctr_coeff_transpose), so the two columns are laid out
// contiguously, first column's 3 prims then second column's 3 prims.
//
// vendor_CINTcgto_spinor returns nctr*(4l+2), so the cintx collector sizes each shell
// block as nctr*spinor_len_kappa0(l) too, keeping block sizes and global stitch offsets
// contraction-major-consistent with vendor. The whole point of nctr>1 here: each shell
// expands into nctr blocks of (4l+2) spinors that compose contraction-major
// (i_global = ci*di + i_sp), and a transposed/contraction-minor scatter would mismatch.
// ─────────────────────────────────────────────────────────────────────────────

const NCTR2: i32 = 2;

fn build_two_p_spinor_nctr2() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.7531_f64, 1.4307, 1.1078];

    // Shell 0: p, 3 primitives, 2 contraction columns (distinct coeffs per column).
    let p0_exp = [5.0331513_f64, 1.1695961, 0.3803890];
    // COLUMN-major coeff block: [col0_p0, col0_p1, col0_p2, col1_p0, col1_p1, col1_p2].
    let p0_coeff = [
        0.15591627_f64,
        0.60768372,
        0.39195739, // column 0
        0.41098730,
        0.22072503,
        0.81763176, // column 1 (distinct)
    ];
    // Shell 1: d, 3 primitives, 2 contraction columns (distinct coeffs per column).
    let p1_exp = [3.4252509_f64, 0.6239137, 0.1688554];
    let p1_coeff = [
        0.15432897_f64,
        0.53532814,
        0.44463454, // column 0
        0.70211493,
        0.39850911,
        0.12300719, // column 1 (distinct)
    ];

    let mut env = vec![0.0_f64; PTR_ENV_START];

    let a_coord_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_coord_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    let p0_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p0_exp);
    let p0_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p0_coeff);

    let p1_exp_ptr = env.len() as i32;
    env.extend_from_slice(&p1_exp);
    let p1_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&p1_coeff);

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
    // Shell 0: p on atom 0, nctr=2.
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 1;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = NCTR2;
    bas[0 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[0 * BAS_SLOTS + PTR_EXP] = p0_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = p0_coeff_ptr;
    // Shell 1: d on atom 1, nctr=2.
    bas[1 * BAS_SLOTS + ATOM_OF] = 1;
    bas[1 * BAS_SLOTS + ANG_OF] = 2;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = NCTR2;
    bas[1 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[1 * BAS_SLOTS + PTR_EXP] = p1_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = p1_coeff_ptr;

    (atm, bas, env)
}

/// nctr-aware per-shell spinor dim: nctr * (4l+2) at kappa==0, matching
/// vendor_CINTcgto_spinor. Each shell expands into nctr contraction-major blocks.
fn shell_nsp_nctr(bas: &[i32], s: usize) -> usize {
    let l = bas[s * BAS_SLOTS + ANG_OF];
    let nctr = bas[s * BAS_SLOTS + NCTR_OF] as usize;
    nctr * spinor_len_kappa0(l)
}

/// cintx collector, nctr-aware: block sizes and stitch offsets use nctr*(4l+2).
fn collect_cintx_spinor_nctr(api_id: RawApiId, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let shell_nsp: Vec<usize> = (0..N_SHELLS).map(|s| shell_nsp_nctr(bas, s)).collect();
    let n_sp: usize = shell_nsp.iter().sum();

    let mut matrix = vec![0.0_f64; n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nsp[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nsp[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = ni * nj * 2;
            let mut out = vec![0.0_f64; n_elem];

            // SAFETY: atm/bas/env are well-formed by construction; shls are valid.
            unsafe {
                eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
                    .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
            }

            stitch_block(&mut matrix, &out, ni, nj, n_sp, row_offset, col_offset);
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx collector: full scalar spinor matrix via eval_raw (interleaved complex).
//
// Returns a flat `Vec<f64>` of shape `[n_sp * n_sp * 2]` (complex-interleaved
// column-major / bra fastest) stitched from each shell-pair block.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_cintx_spinor(api_id: RawApiId, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsp: Vec<usize> = ang.iter().map(|&l| spinor_len_kappa0(l)).collect();
    let n_sp: usize = shell_nsp.iter().sum();

    let mut matrix = vec![0.0_f64; n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nsp[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nsp[sj];
            let shls = [si as i32, sj as i32];
            // Per shell pair: ni*nj complex × 2 (re/im).
            let n_elem = ni * nj * 2;
            let mut out = vec![0.0_f64; n_elem];

            // SAFETY: atm/bas/env are well-formed by construction; shls are valid.
            unsafe {
                eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
                    .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
            }

            stitch_block(&mut matrix, &out, ni, nj, n_sp, row_offset, col_offset);
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

/// Stitch one complex-interleaved shell-pair block (`out`) into the full matrix.
/// The block is column-major (bra fastest): out[(jj*ni + ii)*2 + {0:re,1:im}].
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

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collector (only available when has_vendor_libcint).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
fn collect_vendor_spinor<F>(vendor_fn: F, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsp: Vec<usize> = ang.iter().map(|&l| spinor_len_kappa0(l)).collect();
    let n_sp: usize = shell_nsp.iter().sum();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut matrix = vec![0.0_f64; n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nsp[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nsp[sj];
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

/// Vendor collector, nctr-aware (only available when has_vendor_libcint).
/// vendor_CINTcgto_spinor returns nctr*(4l+2), which we mirror with shell_nsp_nctr.
#[cfg(has_vendor_libcint)]
fn collect_vendor_spinor_nctr<F>(vendor_fn: F, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let shell_nsp: Vec<usize> = (0..N_SHELLS).map(|s| shell_nsp_nctr(bas, s)).collect();
    let n_sp: usize = shell_nsp.iter().sum();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut matrix = vec![0.0_f64; n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nsp[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nsp[sj];
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
// Parity helpers (copied verbatim from one_electron_grad_spinor_parity.rs).
// ─────────────────────────────────────────────────────────────────────────────

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
        "{label}: spinor matrix is all-zero (zero-fill regression)"
    );
}

/// Self-documenting guard: the fixture MUST present a NON-SQUARE p⁺×d⁺ cross block
/// (shell 0 l==1 → nci==3, shell 1 l==2 → ncj==6, on distinct atoms). A non-square
/// block cannot be its own transpose, so a ket-major↔bra-major misread is guaranteed
/// to mismatch. If this ever regresses to a square (l0==l1) block, the parity proof
/// loses its teeth (a square overlap p×p block is transpose-symmetric and hides the bug).
#[allow(dead_code)]
fn assert_fixture_asymmetric(bas: &[i32]) {
    let l0 = bas[0 * BAS_SLOTS + ANG_OF];
    let l1 = bas[1 * BAS_SLOTS + ANG_OF];
    let a0 = bas[0 * BAS_SLOTS + ATOM_OF];
    let a1 = bas[1 * BAS_SLOTS + ATOM_OF];
    assert_eq!(l0, 1, "shell 0 must be a p shell (l==1) so nci==3 > 1");
    assert_eq!(l1, 2, "shell 1 must be a d shell (l==2) so ncj==6 > 1");
    assert_ne!(
        l0, l1,
        "the cross block must be NON-SQUARE (nci != ncj) so ket-major != bra-major"
    );
    assert_ne!(
        a0, a1,
        "the two shells must sit on distinct atoms so the cross block is value-asymmetric"
    );
}

/// nctr>1 fixture guard: BOTH shells must have NCTR_OF==2 (general contraction is
/// actually exercised) AND l0 != l1 (the angular block is NON-SQUARE so a transposed
/// or contraction-minor scatter is a guaranteed mismatch — square blocks hide it).
#[allow(dead_code)]
fn assert_fixture_nctr_gt1(bas: &[i32]) {
    let n0 = bas[0 * BAS_SLOTS + NCTR_OF];
    let n1 = bas[1 * BAS_SLOTS + NCTR_OF];
    let l0 = bas[0 * BAS_SLOTS + ANG_OF];
    let l1 = bas[1 * BAS_SLOTS + ANG_OF];
    assert_eq!(n0, NCTR2, "shell 0 must be general-contracted (nctr==2)");
    assert_eq!(n1, NCTR2, "shell 1 must be general-contracted (nctr==2)");
    assert_ne!(
        l0, l1,
        "the cross block must be NON-SQUARE (l0 != l1) so a transposed/contraction-minor \
         scatter mismatches"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke tests (always run when cpu feature active).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ovlp_spinor_evaluates() {
    let (atm, bas, env) = build_two_p_spinor();
    assert_fixture_asymmetric(&bas);
    let mat = collect_cintx_spinor(RawApiId::INT1E_OVLP_SPINOR, &atm, &bas, &env);
    // n_sp for p+d (kappa=0): 6 + 10 = 16 → 16*16*2 = 512.
    assert_eq!(mat.len(), 16 * 16 * 2, "ovlp_spinor matrix size 16*16*2");
    assert_any_nonzero(&mat, "int1e_ovlp_spinor cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_kin_spinor_evaluates() {
    let (atm, bas, env) = build_two_p_spinor();
    assert_fixture_asymmetric(&bas);
    let mat = collect_cintx_spinor(RawApiId::INT1E_KIN_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 16 * 16 * 2);
    assert_any_nonzero(&mat, "int1e_kin_spinor cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_nuc_spinor_evaluates() {
    let (atm, bas, env) = build_two_p_spinor();
    assert_fixture_asymmetric(&bas);
    let mat = collect_cintx_spinor(RawApiId::INT1E_NUC_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 16 * 16 * 2);
    assert_any_nonzero(&mat, "int1e_nuc_spinor cintx");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint + cpu feature).
// Asymmetric p⁺×p⁺ cross block — surfaces the cart→spinor orientation bug.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ovlp_spinor_asym_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_two_p_spinor();
    assert_fixture_asymmetric(&bas);

    let vendor = collect_vendor_spinor(vendor_ffi::vendor_int1e_ovlp_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor(RawApiId::INT1E_OVLP_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ovlp_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ovlp_spinor vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ovlp_spinor: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_kin_spinor_asym_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_two_p_spinor();
    assert_fixture_asymmetric(&bas);

    let vendor = collect_vendor_spinor(vendor_ffi::vendor_int1e_kin_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor(RawApiId::INT1E_KIN_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_kin_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_kin_spinor vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_kin_spinor: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_nuc_spinor_asym_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_two_p_spinor();
    assert_fixture_asymmetric(&bas);

    let vendor = collect_vendor_spinor(vendor_ffi::vendor_int1e_nuc_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor(RawApiId::INT1E_NUC_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_nuc_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_nuc_spinor vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_nuc_spinor: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// nctr>1 (general-contraction) tests — quick task 260601-aty.
// p(nctr=2) × d(nctr=2) NON-SQUARE block; n_sp = 2*6 + 2*10 = 32 → 32*32*2.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ovlp_spinor_nctr2_evaluates() {
    let (atm, bas, env) = build_two_p_spinor_nctr2();
    assert_fixture_nctr_gt1(&bas);
    let mat = collect_cintx_spinor_nctr(RawApiId::INT1E_OVLP_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 32 * 32 * 2, "ovlp_spinor nctr2 matrix size 32*32*2");
    assert_any_nonzero(&mat, "int1e_ovlp_spinor nctr2 cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_kin_spinor_nctr2_evaluates() {
    let (atm, bas, env) = build_two_p_spinor_nctr2();
    assert_fixture_nctr_gt1(&bas);
    let mat = collect_cintx_spinor_nctr(RawApiId::INT1E_KIN_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 32 * 32 * 2);
    assert_any_nonzero(&mat, "int1e_kin_spinor nctr2 cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_nuc_spinor_nctr2_evaluates() {
    let (atm, bas, env) = build_two_p_spinor_nctr2();
    assert_fixture_nctr_gt1(&bas);
    let mat = collect_cintx_spinor_nctr(RawApiId::INT1E_NUC_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 32 * 32 * 2);
    assert_any_nonzero(&mat, "int1e_nuc_spinor nctr2 cintx");
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ovlp_spinor_nctr2_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_two_p_spinor_nctr2();
    assert_fixture_nctr_gt1(&bas);

    let vendor =
        collect_vendor_spinor_nctr(vendor_ffi::vendor_int1e_ovlp_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor_nctr(RawApiId::INT1E_OVLP_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ovlp_spinor nctr2 cintx");
    assert_any_nonzero(&vendor, "int1e_ovlp_spinor nctr2 vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ovlp_spinor nctr2: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_kin_spinor_nctr2_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_two_p_spinor_nctr2();
    assert_fixture_nctr_gt1(&bas);

    let vendor = collect_vendor_spinor_nctr(vendor_ffi::vendor_int1e_kin_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor_nctr(RawApiId::INT1E_KIN_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_kin_spinor nctr2 cintx");
    assert_any_nonzero(&vendor, "int1e_kin_spinor nctr2 vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_kin_spinor nctr2: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_nuc_spinor_nctr2_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_two_p_spinor_nctr2();
    assert_fixture_nctr_gt1(&bas);

    let vendor = collect_vendor_spinor_nctr(vendor_ffi::vendor_int1e_nuc_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor_nctr(RawApiId::INT1E_NUC_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_nuc_spinor nctr2 cintx");
    assert_any_nonzero(&vendor, "int1e_nuc_spinor nctr2 vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_nuc_spinor nctr2: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

