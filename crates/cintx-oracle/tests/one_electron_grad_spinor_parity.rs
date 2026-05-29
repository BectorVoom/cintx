//! Oracle parity tests for the four spinor int1e GRADIENT operators on H2O/STO-3G:
//!   int1e_ipovlp_spinor, int1e_ipkin_spinor, int1e_ipnuc_spinor, int1e_iprinv_spinor.
//!
//! These operators were previously rejected with `UnsupportedApi` (Risk R5 / D-03).
//! quick task 260529-jtd wires the existing on-device 3-component Cartesian gradient
//! through the host-side spin-free cart→spinor transform `cart_to_spinor_sf_2d`
//! (applied per component), mirroring the SCALAR spinor 1e path.
//!
//! Each operator produces a 3-component, component-leading, interleaved-complex buffer:
//!   out[comp * ni_sp * nj_sp * 2 + (j*ni_sp + i)*2 + {0:re, 1:im}]
//! where ni_sp = CINTcgto_spinor(shls[0]), nj_sp = CINTcgto_spinor(shls[1]).
//!
//! Layout convention matches libcint c2s_sf_1e per component (column-major: bra fastest).
//!
//! Env layout: PTR_ENV_START-aligned (all user data at env[20..]) so the iprinv rinv
//! origin at env[PTR_RINV_ORIG=4..6] never collides with atom coordinates — copied from
//! one_electron_nuc_grad_parity.rs. Shells use kappa=0 (both GT+LT spinor blocks) — the
//! standard non-relativistic spinor convention also exercised by the scalar spinor path.
//!
//! Vendor parity is double-gated: it only runs under `--features cpu` AND env
//! `CINTX_ORACLE_BUILD_VENDOR=1` (which makes build.rs set `has_vendor_libcint`).
//! Without both, the vendor bodies are cfg'd out and the test compiles to a no-op
//! plus the non-vendor smoke tests.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, KAPPA_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF,
    POINT_NUC, PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId,
    eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

const N_SHELLS: usize = 5;
const N_ATOMS: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G fixture — PTR_ENV_START-aligned, spinor shells (kappa=0).
//
// Identical primitive data to one_electron_nuc_grad_parity.rs, but every bas row
// sets KAPPA_OF = 0 so the shells are spinor shells with both GT (j=l+1/2) and
// LT (j=l-1/2) blocks. env[0..PTR_ENV_START) is reserved for libcint global slots
// (PTR_RINV_ORIG=4..6), so writing the rinv origin never clobbers atom coords.
// ─────────────────────────────────────────────────────────────────────────────

fn build_h2o_sto3g_spinor() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    // Reserve the libcint global slots [0..PTR_ENV_START); all user data follows.
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

    let mut atm = vec![0_i32; N_ATOMS * ATM_SLOTS];
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

    let mut bas = vec![0_i32; N_SHELLS * BAS_SLOTS];
    // O 1s
    bas[0 * BAS_SLOTS + ATOM_OF] = 0;
    bas[0 * BAS_SLOTS + ANG_OF] = 0;
    bas[0 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[0 * BAS_SLOTS + NCTR_OF] = 1;
    bas[0 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[0 * BAS_SLOTS + PTR_EXP] = o1s_exp_ptr;
    bas[0 * BAS_SLOTS + PTR_COEFF] = o1s_coeff_ptr;
    // O 2s
    bas[1 * BAS_SLOTS + ATOM_OF] = 0;
    bas[1 * BAS_SLOTS + ANG_OF] = 0;
    bas[1 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[1 * BAS_SLOTS + NCTR_OF] = 1;
    bas[1 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[1 * BAS_SLOTS + PTR_EXP] = o2s_exp_ptr;
    bas[1 * BAS_SLOTS + PTR_COEFF] = o2s_coeff_ptr;
    // O 2p
    bas[2 * BAS_SLOTS + ATOM_OF] = 0;
    bas[2 * BAS_SLOTS + ANG_OF] = 1;
    bas[2 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[2 * BAS_SLOTS + NCTR_OF] = 1;
    bas[2 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[2 * BAS_SLOTS + PTR_EXP] = o2p_exp_ptr;
    bas[2 * BAS_SLOTS + PTR_COEFF] = o2p_coeff_ptr;
    // H1 1s
    bas[3 * BAS_SLOTS + ATOM_OF] = 1;
    bas[3 * BAS_SLOTS + ANG_OF] = 0;
    bas[3 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[3 * BAS_SLOTS + NCTR_OF] = 1;
    bas[3 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[3 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[3 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;
    // H2 1s
    bas[4 * BAS_SLOTS + ATOM_OF] = 2;
    bas[4 * BAS_SLOTS + ANG_OF] = 0;
    bas[4 * BAS_SLOTS + NPRIM_OF] = 3;
    bas[4 * BAS_SLOTS + NCTR_OF] = 1;
    bas[4 * BAS_SLOTS + KAPPA_OF] = 0;
    bas[4 * BAS_SLOTS + PTR_EXP] = h1s_exp_ptr;
    bas[4 * BAS_SLOTS + PTR_COEFF] = h1s_coeff_ptr;

    (atm, bas, env)
}

/// Number of spinor components for ang `l` with kappa==0: 4*l + 2.
fn spinor_len_kappa0(l: i32) -> usize {
    (4 * l + 2) as usize
}

/// Read nucleus `c`'s [x, y, z] coordinates from the env via atm[PTR_COORD].
fn nucleus_coord(atm: &[i32], env: &[f64], c: usize) -> [f64; 3] {
    let ptr = atm[c * ATM_SLOTS + PTR_COORD] as usize;
    [env[ptr], env[ptr + 1], env[ptr + 2]]
}

/// Set the rinv origin into env[PTR_RINV_ORIG..PTR_RINV_ORIG+3] (returns a fresh env).
#[allow(dead_code)]
fn env_with_rinv_origin(env: &[f64], origin: [f64; 3]) -> Vec<f64> {
    let mut e = env.to_vec();
    e[PTR_RINV_ORIG] = origin[0];
    e[PTR_RINV_ORIG + 1] = origin[1];
    e[PTR_RINV_ORIG + 2] = origin[2];
    e
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx collector: full spinor gradient matrix via eval_raw (interleaved complex).
//
// Returns a flat `Vec<f64>` of shape `[3 * n_sp * n_sp * 2]` (component-leading,
// then complex-interleaved column-major) stitched from each shell-pair block.
// ─────────────────────────────────────────────────────────────────────────────

fn collect_cintx_spinor_grad(api_id: RawApiId, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsp: Vec<usize> = ang.iter().map(|&l| spinor_len_kappa0(l)).collect();
    let n_sp: usize = shell_nsp.iter().sum();

    // 3 components × (n_sp × n_sp complex) × 2 (re/im).
    let mut matrix = vec![0.0_f64; 3 * n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nsp[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nsp[sj];
            let shls = [si as i32, sj as i32];
            // Per shell pair: 3 components × ni*nj complex × 2.
            let n_elem = 3 * ni * nj * 2;
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

/// Stitch one component-leading, complex-interleaved shell-pair block (`out`) into
/// the full matrix. Within each component, the block is column-major (bra fastest):
///   out[comp * ni*nj*2 + (jj*ni + ii)*2 + {0:re,1:im}].
fn stitch_block(
    matrix: &mut [f64],
    out: &[f64],
    ni: usize,
    nj: usize,
    n_sp: usize,
    row_offset: usize,
    col_offset: usize,
) {
    for comp in 0..3usize {
        let comp_off_src = comp * ni * nj * 2;
        let comp_off_dst = comp * n_sp * n_sp * 2;
        for jj in 0..nj {
            for ii in 0..ni {
                let src = comp_off_src + (jj * ni + ii) * 2;
                let row = row_offset + ii;
                let col = col_offset + jj;
                let dst = comp_off_dst + (col * n_sp + row) * 2;
                matrix[dst] = out[src];
                matrix[dst + 1] = out[src + 1];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collector (only available when has_vendor_libcint).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
fn collect_vendor_spinor_grad<F>(vendor_fn: F, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nsp: Vec<usize> = ang.iter().map(|&l| spinor_len_kappa0(l)).collect();
    let n_sp: usize = shell_nsp.iter().sum();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut matrix = vec![0.0_f64; 3 * n_sp * n_sp * 2];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nsp[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nsp[sj];
            let shls: [i32; 2] = [si as i32, sj as i32];
            let mut out = vec![0.0_f64; 3 * ni * nj * 2];

            vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);

            stitch_block(&mut matrix, &out, ni, nj, n_sp, row_offset, col_offset);
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity helpers
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
        "{label}: gradient matrix is all-zero (zero-fill regression)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke tests (always run when cpu feature active).
// These prove the spinor gradient evaluates (no UnsupportedApi) and is non-trivial.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_spinor_evaluates() {
    let (atm, bas, env) = build_h2o_sto3g_spinor();
    let mat = collect_cintx_spinor_grad(RawApiId::INT1E_IPOVLP_SPINOR, &atm, &bas, &env);
    // n_sp for H2O STO-3G spinor kappa=0: O1s=2, O2s=2, O2p=6, H1=2, H2=2 → 14.
    assert_eq!(mat.len(), 3 * 14 * 14 * 2, "ipovlp_spinor matrix size 3*14*14*2");
    assert_any_nonzero(&mat, "int1e_ipovlp_spinor cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipkin_spinor_evaluates() {
    let (atm, bas, env) = build_h2o_sto3g_spinor();
    let mat = collect_cintx_spinor_grad(RawApiId::INT1E_IPKIN_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 3 * 14 * 14 * 2);
    assert_any_nonzero(&mat, "int1e_ipkin_spinor cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipnuc_spinor_evaluates() {
    let (atm, bas, env) = build_h2o_sto3g_spinor();
    let mat = collect_cintx_spinor_grad(RawApiId::INT1E_IPNUC_SPINOR, &atm, &bas, &env);
    assert_eq!(mat.len(), 3 * 14 * 14 * 2);
    assert_any_nonzero(&mat, "int1e_ipnuc_spinor cintx");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_iprinv_spinor_evaluates() {
    let (atm, bas, env) = build_h2o_sto3g_spinor();
    let origin = nucleus_coord(&atm, &env, 0);
    let env_c = env_with_rinv_origin(&env, origin);
    let mat = collect_cintx_spinor_grad(RawApiId::INT1E_IPRINV_SPINOR, &atm, &bas, &env_c);
    assert_eq!(mat.len(), 3 * 14 * 14 * 2);
    assert_any_nonzero(&mat, "int1e_iprinv_spinor cintx");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint + cpu feature).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_spinor_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_spinor();

    let vendor = collect_vendor_spinor_grad(vendor_ffi::vendor_int1e_ipovlp_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor_grad(RawApiId::INT1E_IPOVLP_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipovlp_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ipovlp_spinor vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipovlp_spinor: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipkin_spinor_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_spinor();

    let vendor = collect_vendor_spinor_grad(vendor_ffi::vendor_int1e_ipkin_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor_grad(RawApiId::INT1E_IPKIN_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipkin_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ipkin_spinor vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipkin_spinor: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipnuc_spinor_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_spinor();

    let vendor = collect_vendor_spinor_grad(vendor_ffi::vendor_int1e_ipnuc_spinor, &atm, &bas, &env);
    let cintx = collect_cintx_spinor_grad(RawApiId::INT1E_IPNUC_SPINOR, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipnuc_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ipnuc_spinor vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipnuc_spinor: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_iprinv_spinor_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_spinor();

    // Sweep the rinv origin over every nucleus to prove single-origin parity per atom.
    for c in 0..N_ATOMS {
        let origin = nucleus_coord(&atm, &env, c);
        let env_c = env_with_rinv_origin(&env, origin);

        let vendor =
            collect_vendor_spinor_grad(vendor_ffi::vendor_int1e_iprinv_spinor, &atm, &bas, &env_c);
        let cintx = collect_cintx_spinor_grad(RawApiId::INT1E_IPRINV_SPINOR, &atm, &bas, &env_c);

        assert_any_nonzero(&cintx, "int1e_iprinv_spinor cintx");
        assert_any_nonzero(&vendor, "int1e_iprinv_spinor vendor");

        let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
        assert_eq!(
            mismatches, 0,
            "int1e_iprinv_spinor (origin@atom {c}): {mismatches} mismatches vs vendored libcint at atol={ATOL}"
        );
    }
}
