//! Oracle parity tests for the Phase 25 HESS-01 rank-9 1e Hessian families
//! (`int1e_ipipovlp`, `int1e_ipipnuc`, `int1e_ipipkin`, `int1e_ipiprinv`),
//! cart + sph: H2O STO-3G.
//!
//! Each family is the second nabla applied to the bra of the Phase-23 first-order
//! 1e engine (∇²bra, `ng[]={2,0,0,0,...}`, component_rank=9). Validates the
//! 9-component (3×3 second-derivative tensor) output vs vendored libcint 6.1.3 at
//! atol=1e-12, every component.
//!
//! D-09 transpose discipline: the parity harness (`hess1e_ipip`) compares the
//! FULL multi-shell matrix (collected over every shell pair, including the
//! NON-SQUARE p×s / s×p blocks of the H2O STO-3G basis), AND a dedicated single
//! NON-SQUARE p×d block, so a transposed bra↔ket layout cannot pass.
//!
//! Requires the `cpu` feature (cubecl cpu backend); vendor parity additionally
//! requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint` cfg). Without
//! BOTH flags, only the determinism tests run and vendor parity SILENTLY SKIPS.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ATM_SLOTS, ANG_OF, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_ENV_START, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, NUC_MOD_OF, POINT_NUC, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;
const NCOMP: usize = 9;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G fixture — PTR_ENV_START-aligned (all user data at env[20..]) so the
// PTR_RINV_ORIG=4..6 slot stays clean for the ipiprinv family. Contains O-1s,
// O-2s, O-2p (l=1), H-1s, H-1s — so cross shell pairs naturally exercise
// NON-SQUARE blocks (p×s = 3×1, s×p = 1×3) in the full-matrix collector.
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

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    atm[CHARGE_OF] = 8;
    atm[PTR_COORD] = o_coord_ptr;
    atm[NUC_MOD_OF] = POINT_NUC;
    atm[PTR_ZETA] = zeta_ptr;
    atm[ATM_SLOTS + CHARGE_OF] = 1;
    atm[ATM_SLOTS + PTR_COORD] = h1_coord_ptr;
    atm[ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    atm[2 * ATM_SLOTS + CHARGE_OF] = 1;
    atm[2 * ATM_SLOTS + PTR_COORD] = h2_coord_ptr;
    atm[2 * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
    atm[2 * ATM_SLOTS + PTR_ZETA] = zeta_ptr;

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

const N_SHELLS: usize = 5;

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// A dedicated NON-SQUARE (p × d) two-shell fixture (D-09 Pitfall 4). A square
// block (p×p) is transpose-symmetric and would hide a bra↔ket swap; p (l=1, 3/5
// funcs) × d (l=2, 6/9 funcs) is strictly non-square in BOTH cart and sph.
// ─────────────────────────────────────────────────────────────────────────────

fn build_pd_nonsquare() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Two atoms a short distance apart, one p shell on atom 0, one d shell on atom 1.
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [0.0_f64, 0.0, 1.2];

    let p_exp = [1.8_f64, 0.5];
    let p_coeff = [0.35_f64, 0.65];
    let d_exp = [2.2_f64, 0.7];
    let d_coeff = [0.40_f64, 0.60];

    // PTR_ENV_START-aligned so PTR_RINV_ORIG=4..6 stays clean for ipiprinv.
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
    // shell 0: p on atom 0 (bra)
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 1;
    bas[NPRIM_OF] = 2;
    bas[NCTR_OF] = 1;
    bas[PTR_EXP] = p_exp_ptr;
    bas[PTR_COEFF] = p_coeff_ptr;
    // shell 1: d on atom 1 (ket)
    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 2;
    bas[BAS_SLOTS + NPRIM_OF] = 2;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = d_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = d_coeff_ptr;

    (atm, bas, env)
}

/// Set the rinv origin into env[PTR_RINV_ORIG..PTR_RINV_ORIG+3] (returns a fresh
/// env). `int1e_ipiprinv` is a single-origin 1/r Hessian at the rinv center; the
/// origin MUST be a nonzero off-bra/off-ket point so parity exercises the
/// shifted-center Rys path (not the trivial origin-at-bra case).
#[allow(dead_code)]
fn env_with_rinv_origin(env: &[f64], origin: [f64; 3]) -> Vec<f64> {
    let mut e = env.to_vec();
    e[PTR_RINV_ORIG] = origin[0];
    e[PTR_RINV_ORIG + 1] = origin[1];
    e[PTR_RINV_ORIG + 2] = origin[2];
    e
}

// ─────────────────────────────────────────────────────────────────────────────
// 9-component matrix collectors (cintx via eval_raw / vendor via FFI)
// ─────────────────────────────────────────────────────────────────────────────

fn collect_9c_matrix(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let n_shells = bas.len() / BAS_SLOTS;
    let ang: Vec<i32> = (0..n_shells).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nf(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; NCOMP * n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..n_shells {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..n_shells {
            let nj = shell_nao[sj];
            let shls = [si as i32, sj as i32];
            let n_elem = NCOMP * ni * nj;
            let mut out = vec![0.0_f64; n_elem];

            // SAFETY: atm/bas/env well-formed by construction; shls valid.
            unsafe {
                eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
                    .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
            }

            for comp in 0..NCOMP {
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
fn collect_vendor_9c_matrix<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let n_shells = bas.len() / BAS_SLOTS;
    let ang: Vec<i32> = (0..n_shells).map(|s| bas[s * BAS_SLOTS + ANG_OF]).collect();
    let shell_nao: Vec<usize> = ang.iter().map(|&l| nf(l)).collect();
    let n_ao: usize = shell_nao.iter().sum();
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;

    let mut matrix = vec![0.0_f64; NCOMP * n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..n_shells {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..n_shells {
            let nj = shell_nao[sj];
            let shls: [i32; 2] = [si as i32, sj as i32];
            let mut out = vec![0.0_f64; NCOMP * ni * nj];

            vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);

            for comp in 0..NCOMP {
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
// Determinism tests (always run under cpu) — pin the 9-component count and the
// NON-SQUARE p×d block shape.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism(api_sph: RawApiId, api_cart: RawApiId, label: &str) {
    let (atm, bas, env) = build_h2o_sto3g();
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        let m1 = collect_9c_matrix(api, &atm, &bas, &env, nf);
        let m2 = collect_9c_matrix(api, &atm, &bas, &env, nf);
        for (a, b) in m1.iter().zip(m2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}_{rep} must be bit-identical");
        }
        assert_any_nonzero(&m1, &format!("{label}_{rep}"));
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipovlp_determinism() {
    determinism(RawApiId::INT1E_IPIPOVLP_SPH, RawApiId::INT1E_IPIPOVLP_CART, "int1e_ipipovlp");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipnuc_determinism() {
    determinism(RawApiId::INT1E_IPIPNUC_SPH, RawApiId::INT1E_IPIPNUC_CART, "int1e_ipipnuc");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipkin_determinism() {
    determinism(RawApiId::INT1E_IPIPKIN_SPH, RawApiId::INT1E_IPIPKIN_CART, "int1e_ipipkin");
}

#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipiprinv_determinism() {
    determinism(RawApiId::INT1E_IPIPRINV_SPH, RawApiId::INT1E_IPIPRINV_CART, "int1e_ipiprinv");
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity (require has_vendor_libcint + cpu). `hess1e_ipip` runs the FULL
// multi-shell matrix parity AND the dedicated NON-SQUARE p×d single-block parity
// for one family; `hess1e_ipip_nonsquare_block` is the explicit p×d gate.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
fn hess1e_ipip<FS, FC>(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: FS,
    vendor_cart: FC,
    label: &str,
    rinv_origin: Option<[f64; 3]>,
) where
    FS: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    FC: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    // (1) Full H2O STO-3G matrix — includes NON-SQUARE p×s / s×p cross blocks.
    let (atm, bas, env0) = build_h2o_sto3g();
    let env = match rinv_origin {
        Some(o) => env_with_rinv_origin(&env0, o),
        None => env0,
    };

    let vendor_s = collect_vendor_9c_matrix(&vendor_sph, &atm, &bas, &env, nsph);
    let cintx_s = collect_9c_matrix(api_sph, &atm, &bas, &env, nsph);
    assert_any_nonzero(&cintx_s, &format!("{label}_sph cintx"));
    assert_any_nonzero(&vendor_s, &format!("{label}_sph vendor"));
    let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_sph (h2o): {mm} mismatches vs vendored libcint at atol={ATOL}");

    let vendor_c = collect_vendor_9c_matrix(&vendor_cart, &atm, &bas, &env, ncart);
    let cintx_c = collect_9c_matrix(api_cart, &atm, &bas, &env, ncart);
    assert_any_nonzero(&cintx_c, &format!("{label}_cart cintx"));
    assert_any_nonzero(&vendor_c, &format!("{label}_cart vendor"));
    let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_cart (h2o): {mm} mismatches vs vendored libcint at atol={ATOL}");

    // (2) Dedicated NON-SQUARE p×d block (D-09 Pitfall 4) — a transposed layout
    //     cannot survive a bra(p) ≠ ket(d) block.
    let (atm2, bas2, env2_0) = build_pd_nonsquare();
    let env2 = match rinv_origin {
        Some(o) => env_with_rinv_origin(&env2_0, o),
        None => env2_0,
    };

    let vendor_s2 = collect_vendor_9c_matrix(&vendor_sph, &atm2, &bas2, &env2, nsph);
    let cintx_s2 = collect_9c_matrix(api_sph, &atm2, &bas2, &env2, nsph);
    assert_any_nonzero(&cintx_s2, &format!("{label}_sph p×d cintx"));
    let mm = count_mismatches(&vendor_s2, &cintx_s2, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_sph (p×d non-square): {mm} mismatches at atol={ATOL}");

    let vendor_c2 = collect_vendor_9c_matrix(&vendor_cart, &atm2, &bas2, &env2, ncart);
    let cintx_c2 = collect_9c_matrix(api_cart, &atm2, &bas2, &env2, ncart);
    assert_any_nonzero(&cintx_c2, &format!("{label}_cart p×d cintx"));
    let mm = count_mismatches(&vendor_c2, &cintx_c2, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_cart (p×d non-square): {mm} mismatches at atol={ATOL}");
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipovlp_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    hess1e_ipip(
        RawApiId::INT1E_IPIPOVLP_SPH,
        RawApiId::INT1E_IPIPOVLP_CART,
        vendor_ffi::vendor_int1e_ipipovlp_sph,
        vendor_ffi::vendor_int1e_ipipovlp_cart,
        "int1e_ipipovlp",
        None,
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipnuc_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    hess1e_ipip(
        RawApiId::INT1E_IPIPNUC_SPH,
        RawApiId::INT1E_IPIPNUC_CART,
        vendor_ffi::vendor_int1e_ipipnuc_sph,
        vendor_ffi::vendor_int1e_ipipnuc_cart,
        "int1e_ipipnuc",
        None,
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipkin_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    hess1e_ipip(
        RawApiId::INT1E_IPIPKIN_SPH,
        RawApiId::INT1E_IPIPKIN_CART,
        vendor_ffi::vendor_int1e_ipipkin_sph,
        vendor_ffi::vendor_int1e_ipipkin_cart,
        "int1e_ipipkin",
        None,
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipiprinv_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    hess1e_ipip(
        RawApiId::INT1E_IPIPRINV_SPH,
        RawApiId::INT1E_IPIPRINV_CART,
        vendor_ffi::vendor_int1e_ipiprinv_sph,
        vendor_ffi::vendor_int1e_ipiprinv_cart,
        "int1e_ipiprinv",
        // Single rinv origin off both the bra and ket centers (exercises the
        // shifted-center Rys path; same origin fed to cintx and vendor).
        Some([0.0, 0.3, 0.6]),
    );
}
