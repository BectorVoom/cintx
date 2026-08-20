//! Oracle parity tests for int1e_ipovlp and int1e_ipkin (cart + sph): H2O STO-3G.
//!
//! Validates 3-component gradient output vs vendored libcint 6.1.3 at atol=1e-12.
//! Covers GRAD-03 (int1e_ipovlp) and GRAD-04 (int1e_ipkin) plan requirements.
//!
//! These tests require the `cpu` feature (cubecl cpu backend).

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G fixture (copied verbatim from one_electron_parity.rs)
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

const N_SHELLS: usize = 5;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient matrix collectors
// ─────────────────────────────────────────────────────────────────────────────

/// Collect 3-component gradient matrix from cintx via eval_raw.
///
/// Returns a flat `Vec<f64>` of shape `[3 * n_ao * n_ao]` in component-leading layout:
/// `out[comp * n_ao * n_ao + row * n_ao + col]`.
///
/// For each shell pair (si, sj), eval_raw fills 3 * ni * nj elements in component-leading
/// order (the planner allocates 3-component staging because component_rank="3" in the
/// manifest). We stitch each shell-pair block into the full matrix per component.
fn collect_1e_grad_sph_matrix(api_id: RawApiId, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();
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

            // SAFETY: atm/bas/env are well-formed by construction. shls are valid.
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

            // Stitch into full matrix: out is component-leading [comp * ni * nj + n].
            // Within each component block, layout is column-major: n = sj_idx * ni + si_idx.
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

/// Collect 3-component gradient matrix from cintx via eval_raw (Cartesian representation).
fn collect_1e_grad_cart_matrix(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| ncart(l)).collect();
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
                .unwrap_or_else(|e| panic!("eval_raw cart failed for shells ({si},{sj}): {e:?}"));
            }

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
// Vendor gradient matrix collectors (only available when has_vendor_libcint)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
fn collect_vendor_1e_grad_sph_matrix<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nsph(l)).collect();
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

            // Vendor libcint writes component-leading: out[comp * ni * nj + n].
            // Column-major within each component block: n = jj * ni + ii.
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
fn collect_vendor_1e_grad_cart_matrix<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let ang: Vec<i32> = (0..N_SHELLS).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| ncart(l)).collect();
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

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor tests (always run when cpu feature active)
// ─────────────────────────────────────────────────────────────────────────────

/// Nonzero sentinel: assert that the gradient matrix has at least one element > threshold.
/// Guards against zero-fill regressions where a stub kernel would silently pass parity.
fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(
        any_nonzero,
        "{label}: gradient matrix is all-zero (zero-fill regression)"
    );
}

/// int1e_ipovlp_sph: determinism check (two calls must be bit-identical).
/// Also validates component count: 3 * n_ao * n_ao elements for H2O STO-3G.
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_sph_determinism() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api = RawApiId::INT1E_IPOVLP_SPH;

    let mat1 = collect_1e_grad_sph_matrix(api, &atm, &bas, &env);
    let mat2 = collect_1e_grad_sph_matrix(api, &atm, &bas, &env);

    // Expected: 3 * 7 * 7 = 147 elements (n_ao=7 for H2O STO-3G sph)
    assert_eq!(
        mat1.len(),
        3 * 7 * 7,
        "ipovlp_sph matrix size should be 3*7*7=147"
    );

    for (a, b) in mat1.iter().zip(mat2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "int1e_ipovlp_sph must be bit-identical across two calls"
        );
    }
    assert_any_nonzero(&mat1, "int1e_ipovlp_sph");
}

/// int1e_ipovlp_cart: determinism check.
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_cart_determinism() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api = RawApiId::INT1E_IPOVLP_CART;

    let mat1 = collect_1e_grad_cart_matrix(api, &atm, &bas, &env);
    let mat2 = collect_1e_grad_cart_matrix(api, &atm, &bas, &env);

    // Expected: 3 * 9 * 9 = 243 elements (n_ao_cart=9: O-1s=1,O-2s=1,O-2p=3,H1-1s=1,H2-1s=1 → 7 sph, 9 cart)
    // Actually H2O STO-3G cart: s+s+p+s+s = 1+1+3+1+1=7 (p cart = 3 = p sph for l=1)
    assert_eq!(
        mat1.len(),
        3 * 7 * 7,
        "ipovlp_cart matrix size should be 3*7*7=147"
    );

    for (a, b) in mat1.iter().zip(mat2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "int1e_ipovlp_cart must be bit-identical across two calls"
        );
    }
    assert_any_nonzero(&mat1, "int1e_ipovlp_cart");
}

/// int1e_ipkin_sph: determinism check.
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipkin_sph_determinism() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api = RawApiId::INT1E_IPKIN_SPH;

    let mat1 = collect_1e_grad_sph_matrix(api, &atm, &bas, &env);
    let mat2 = collect_1e_grad_sph_matrix(api, &atm, &bas, &env);

    assert_eq!(
        mat1.len(),
        3 * 7 * 7,
        "ipkin_sph matrix size should be 3*7*7=147"
    );

    for (a, b) in mat1.iter().zip(mat2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "int1e_ipkin_sph must be bit-identical across two calls"
        );
    }
    assert_any_nonzero(&mat1, "int1e_ipkin_sph");
}

/// int1e_ipkin_cart: determinism check.
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipkin_cart_determinism() {
    let (atm, bas, env) = build_h2o_sto3g();
    let api = RawApiId::INT1E_IPKIN_CART;

    let mat1 = collect_1e_grad_cart_matrix(api, &atm, &bas, &env);
    let mat2 = collect_1e_grad_cart_matrix(api, &atm, &bas, &env);

    assert_eq!(
        mat1.len(),
        3 * 7 * 7,
        "ipkin_cart matrix size should be 3*7*7=147"
    );

    for (a, b) in mat1.iter().zip(mat2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "int1e_ipkin_cart must be bit-identical across two calls"
        );
    }
    assert_any_nonzero(&mat1, "int1e_ipkin_cart");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint)
// ─────────────────────────────────────────────────────────────────────────────

/// int1e_ipovlp_sph: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_sph_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g();

    let vendor =
        collect_vendor_1e_grad_sph_matrix(vendor_ffi::vendor_int1e_ipovlp_sph, &atm, &bas, &env);
    let cintx = collect_1e_grad_sph_matrix(RawApiId::INT1E_IPOVLP_SPH, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipovlp_sph cintx");
    assert_any_nonzero(&vendor, "int1e_ipovlp_sph vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipovlp_sph: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

/// int1e_ipovlp_cart: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_cart_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g();

    let vendor =
        collect_vendor_1e_grad_cart_matrix(vendor_ffi::vendor_int1e_ipovlp_cart, &atm, &bas, &env);
    let cintx = collect_1e_grad_cart_matrix(RawApiId::INT1E_IPOVLP_CART, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipovlp_cart cintx");
    assert_any_nonzero(&vendor, "int1e_ipovlp_cart vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipovlp_cart: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

/// int1e_ipkin_sph: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipkin_sph_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g();

    let vendor =
        collect_vendor_1e_grad_sph_matrix(vendor_ffi::vendor_int1e_ipkin_sph, &atm, &bas, &env);
    let cintx = collect_1e_grad_sph_matrix(RawApiId::INT1E_IPKIN_SPH, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipkin_sph cintx");
    assert_any_nonzero(&vendor, "int1e_ipkin_sph vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipkin_sph: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}

/// int1e_ipkin_cart: byte-identity parity vs vendored libcint 6.1.3 at atol=1e-12.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipkin_cart_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_h2o_sto3g();

    let vendor =
        collect_vendor_1e_grad_cart_matrix(vendor_ffi::vendor_int1e_ipkin_cart, &atm, &bas, &env);
    let cintx = collect_1e_grad_cart_matrix(RawApiId::INT1E_IPKIN_CART, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipkin_cart cintx");
    assert_any_nonzero(&vendor, "int1e_ipkin_cart vendor");

    let mismatches = count_mismatches(&vendor, &cintx, ATOL, RTOL);
    assert_eq!(
        mismatches, 0,
        "int1e_ipkin_cart: {mismatches} mismatches vs vendored libcint at atol={ATOL}"
    );
}
