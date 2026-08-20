//! Oracle parity tests for int1e_ipnuc and int1e_iprinv (cart + sph): H2O STO-3G.
//!
//! Validates 3-component nuclear-gradient output vs vendored libcint 6.1.3 at atol=1e-12.
//! Covers GRAD-05 (int1e_ipnuc) and GRAD-06 (int1e_iprinv) plan requirements.
//!
//! - int1e_ipnuc: hcore nuclear-attraction derivative, `∂/∂Ai` on the bra center,
//!   summed over ALL nuclei with the `-Z_C` charge factor.
//! - int1e_iprinv: per-atom Hellmann–Feynman force, `∂/∂Ai` at a SINGLE rinv origin
//!   (env[PTR_RINV_ORIG..+3]) with factor `+1.0` (no `-Z_C`). The iprinv tests sweep
//!   the rinv origin over EACH nucleus to prove single-origin parity vs libcint per atom.
//!
//! Env layout: PTR_ENV_START-aligned (all user data at env[20..]) so setting the rinv
//! origin at env[PTR_RINV_ORIG=4..6] never collides with atom coordinates. (The shared
//! `one_electron_grad_parity.rs` fixture packs coords at env[0..9], which would clobber
//! the rinv slot — not reusable for iprinv. We use a dedicated env-start fixture here.)
//!
//! These tests require the `cpu` feature (cubecl cpu backend).

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

const N_SHELLS: usize = 5;
const N_ATOMS: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G fixture — PTR_ENV_START-aligned (all user data at env[20..]).
//
// libcint reserves env[0..PTR_ENV_START=20] for global slots (PTR_RINV_ORIG=4..6,
// PTR_RINV_ZETA=7, PTR_RANGE_OMEGA=8, PTR_F12_ZETA=9, ...). The iprinv tests write
// the rinv origin into env[4..6]; placing atom coords there (as the env[0]-based
// shared fixture does) would clobber them. Reserving env[0..20] keeps the two
// disjoint and matches libcint's documented env layout.
// ─────────────────────────────────────────────────────────────────────────────

fn build_h2o_sto3g_envstart() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

/// Read nucleus `c`'s [x, y, z] coordinates from the env via atm[PTR_COORD].
fn nucleus_coord(atm: &[i32], env: &[f64], c: usize) -> [f64; 3] {
    let ptr = atm[c * ATM_SLOTS + PTR_COORD] as usize;
    [env[ptr], env[ptr + 1], env[ptr + 2]]
}

/// Set the rinv origin into env[PTR_RINV_ORIG..PTR_RINV_ORIG+3] (returns a fresh env).
fn env_with_rinv_origin(env: &[f64], origin: [f64; 3]) -> Vec<f64> {
    let mut e = env.to_vec();
    e[PTR_RINV_ORIG] = origin[0];
    e[PTR_RINV_ORIG + 1] = origin[1];
    e[PTR_RINV_ORIG + 2] = origin[2];
    e
}

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient matrix collectors (component-leading [3 * n_ao * n_ao])
// ─────────────────────────────────────────────────────────────────────────────

fn collect_1e_grad_matrix(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    sph: bool,
) -> Vec<f64> {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang
        .iter()
        .map(|&l| if sph { nsph(l) } else { ncart(l) })
        .collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; 3 * n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = 3 * ni * nj;
            let mut out = vec![0.0_f64; n_elem];

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

            // out is component-leading [comp * ni * nj + n]; column-major within block (n = jj*ni+ii).
            for comp in 0..3usize {
                let comp_off_src = comp * ni * nj;
                let comp_off_dst = comp * n_ao * n_ao;
                for jj in 0..nj {
                    for ii in 0..ni {
                        let src = comp_off_src + jj * ni + ii;
                        let row = row_offset + ii;
                        let col = col_offset + jj;
                        matrix[comp_off_dst + row * n_ao + col] = out[src];
                    }
                }
            }

            col_offset += nj;
        }
        row_offset += ni;
    }

    matrix
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_1e_grad_matrix<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    sph: bool,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang
        .iter()
        .map(|&l| if sph { nsph(l) } else { ncart(l) })
        .collect();
    let n_ao: usize = shell_nao.iter().sum();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut matrix = vec![0.0_f64; 3 * n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..N_SHELLS {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..N_SHELLS {
            let nj = shell_nao[sj];
            let shls: [i32; 2] = [si as i32, sj as i32];
            let mut out = vec![0.0_f64; 3 * ni * nj];

            vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);

            for comp in 0..3usize {
                let comp_off_src = comp * ni * nj;
                let comp_off_dst = comp * n_ao * n_ao;
                for jj in 0..nj {
                    for ii in 0..ni {
                        let src = comp_off_src + jj * ni + ii;
                        let row = row_offset + ii;
                        let col = col_offset + jj;
                        matrix[comp_off_dst + row * n_ao + col] = out[src];
                    }
                }
            }

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
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

#[allow(dead_code)]
fn max_abs_diff(reference: &[f64], observed: &[f64]) -> f64 {
    reference
        .iter()
        .zip(observed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(
        any_nonzero,
        "{label}: gradient matrix is all-zero (zero-fill regression)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor determinism tests (always run when cpu feature active)
// ─────────────────────────────────────────────────────────────────────────────

/// int1e_ipnuc_sph: determinism + nonzero sentinel.
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipnuc_sph_determinism() {
    let (atm, bas, env) = build_h2o_sto3g_envstart();
    let mat1 = collect_1e_grad_matrix(RawApiId::INT1E_IPNUC_SPH, &atm, &bas, &env, true);
    let mat2 = collect_1e_grad_matrix(RawApiId::INT1E_IPNUC_SPH, &atm, &bas, &env, true);
    assert_eq!(
        mat1.len(),
        3 * 7 * 7,
        "ipnuc_sph matrix size should be 3*7*7=147"
    );
    for (a, b) in mat1.iter().zip(mat2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "int1e_ipnuc_sph must be bit-identical across two calls"
        );
    }
    assert_any_nonzero(&mat1, "int1e_ipnuc_sph");
}

/// int1e_iprinv_sph: determinism + origin-sensitivity + nonzero sentinel.
/// Sweeping the rinv origin to a different nucleus must change the output (origin consumed).
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_iprinv_sph_origin_sensitivity() {
    let (atm, bas, env) = build_h2o_sto3g_envstart();
    let origin_o = nucleus_coord(&atm, &env, 0);
    let origin_h1 = nucleus_coord(&atm, &env, 1);

    let env_o = env_with_rinv_origin(&env, origin_o);
    let env_h1 = env_with_rinv_origin(&env, origin_h1);

    let mat_o = collect_1e_grad_matrix(RawApiId::INT1E_IPRINV_SPH, &atm, &bas, &env_o, true);
    let mat_o2 = collect_1e_grad_matrix(RawApiId::INT1E_IPRINV_SPH, &atm, &bas, &env_o, true);
    let mat_h1 = collect_1e_grad_matrix(RawApiId::INT1E_IPRINV_SPH, &atm, &bas, &env_h1, true);

    // Determinism for a fixed origin.
    for (a, b) in mat_o.iter().zip(mat_o2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "iprinv_sph must be bit-identical for a fixed origin"
        );
    }
    assert_any_nonzero(&mat_o, "int1e_iprinv_sph @ O");
    // Different origins → different result.
    let any_diff = mat_o
        .iter()
        .zip(mat_h1.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(
        any_diff,
        "iprinv: O vs H1 rinv origins must produce different output (origin consumed)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint)
// ─────────────────────────────────────────────────────────────────────────────

/// int1e_ipnuc_sph: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
/// (Sum over all nuclei with -Z_C — a missing atom or sign error is caught by the gate, T-21-04-02.)
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipnuc_sph_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_envstart();

    let vendor =
        collect_vendor_1e_grad_matrix(vendor_ffi::vendor_int1e_ipnuc_sph, &atm, &bas, &env, true);
    let cintx = collect_1e_grad_matrix(RawApiId::INT1E_IPNUC_SPH, &atm, &bas, &env, true);

    assert_any_nonzero(&cintx, "int1e_ipnuc_sph cintx");
    assert_any_nonzero(&vendor, "int1e_ipnuc_sph vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    eprintln!(
        "int1e_ipnuc_sph worst |diff| = {:.3e}",
        max_abs_diff(&vendor, &cintx)
    );
    assert_eq!(
        mismatches, 0,
        "int1e_ipnuc_sph: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

/// int1e_ipnuc_cart: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipnuc_cart_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_envstart();

    let vendor =
        collect_vendor_1e_grad_matrix(vendor_ffi::vendor_int1e_ipnuc_cart, &atm, &bas, &env, false);
    let cintx = collect_1e_grad_matrix(RawApiId::INT1E_IPNUC_CART, &atm, &bas, &env, false);

    assert_any_nonzero(&cintx, "int1e_ipnuc_cart cintx");
    assert_any_nonzero(&vendor, "int1e_ipnuc_cart vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    eprintln!(
        "int1e_ipnuc_cart worst |diff| = {:.3e}",
        max_abs_diff(&vendor, &cintx)
    );
    assert_eq!(
        mismatches, 0,
        "int1e_ipnuc_cart: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

/// int1e_iprinv_sph: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
/// Sweeps the rinv origin over EACH nucleus to prove single-origin parity per atom (T-21-04-03).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_iprinv_sph_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_envstart();

    let mut worst = 0.0_f64;
    for c in 0..N_ATOMS {
        let origin = nucleus_coord(&atm, &env, c);
        // Set env[PTR_RINV_ORIG..+3] for BOTH the cintx and vendor paths.
        let env_c = env_with_rinv_origin(&env, origin);

        let vendor = collect_vendor_1e_grad_matrix(
            vendor_ffi::vendor_int1e_iprinv_sph,
            &atm,
            &bas,
            &env_c,
            true,
        );
        let cintx = collect_1e_grad_matrix(RawApiId::INT1E_IPRINV_SPH, &atm, &bas, &env_c, true);

        assert_any_nonzero(&cintx, &format!("int1e_iprinv_sph cintx @ nucleus {c}"));
        assert_any_nonzero(&vendor, &format!("int1e_iprinv_sph vendor @ nucleus {c}"));

        let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
        worst = worst.max(max_abs_diff(&vendor, &cintx));
        assert_eq!(
            mismatches, 0,
            "int1e_iprinv_sph @ nucleus {c} (origin {origin:?}): {mismatches} mismatches at atol={ATOL}"
        );
    }
    eprintln!("int1e_iprinv_sph per-nucleus sweep worst |diff| = {worst:.3e}");
}

/// int1e_iprinv_cart: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
/// Sweeps the rinv origin over EACH nucleus.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_iprinv_cart_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g_envstart();

    let mut worst = 0.0_f64;
    for c in 0..N_ATOMS {
        let origin = nucleus_coord(&atm, &env, c);
        let env_c = env_with_rinv_origin(&env, origin);

        let vendor = collect_vendor_1e_grad_matrix(
            vendor_ffi::vendor_int1e_iprinv_cart,
            &atm,
            &bas,
            &env_c,
            false,
        );
        let cintx = collect_1e_grad_matrix(RawApiId::INT1E_IPRINV_CART, &atm, &bas, &env_c, false);

        assert_any_nonzero(&cintx, &format!("int1e_iprinv_cart cintx @ nucleus {c}"));
        assert_any_nonzero(&vendor, &format!("int1e_iprinv_cart vendor @ nucleus {c}"));

        let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
        worst = worst.max(max_abs_diff(&vendor, &cintx));
        assert_eq!(
            mismatches, 0,
            "int1e_iprinv_cart @ nucleus {c} (origin {origin:?}): {mismatches} mismatches at atol={ATOL}"
        );
    }
    eprintln!("int1e_iprinv_cart per-nucleus sweep worst |diff| = {worst:.3e}");
}
