//! Oracle parity tests for the Phase 23 cluster-B 3c1e derivative pair
//! (`int3c1e_ip1`, `int3c1e_iprinv`), cart + sph — DRV1-03.
//!
//! - `int3c1e_ip1`   = ∇ on bra i of the 3-center OVERLAP (no Rys roots).
//! - `int3c1e_iprinv` = ∇ on bra i of the 3-center rinv-COULOMB (Rys-driven);
//!   reads env[PTR_RINV_ORIG..+3] (env[4..6]) which the fixture sets non-zero.
//!
//! Validates 3-component output vs vendored libcint 6.1.3 at atol=1e-12 on a
//! NON-SQUARE shell triple (p × s × s, and s × p × s) so a transposed layout
//! cannot pass. Requires the `cpu` feature (cubecl cpu backend); vendor parity
//! additionally requires `CINTX_ORACLE_BUILD_VENDOR=1` (the `has_vendor_libcint`
//! cfg).

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_EXP, PTR_RINV_ORIG, PTR_ZETA, RawApiId, eval_raw,
};

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 0.0;
const NCOMP: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// H2O STO-3G fixture (verbatim shell/coefficient layout from the cluster-C test).
// A non-zero rinv origin is written into env[PTR_RINV_ORIG..+3] for the iprinv
// path; the overlap (ip1) path ignores it.
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
    // Reserve the libcint global-parameter region (env[0..PTR_ENV_START]) so the
    // rinv origin lives in env[4..6] and user data starts at env[20].
    env.resize(20, 0.0);
    // Non-zero rinv origin for int3c1e_iprinv (env[4..6]).
    env[PTR_RINV_ORIG] = 0.30;
    env[PTR_RINV_ORIG + 1] = -0.45;
    env[PTR_RINV_ORIG + 2] = 0.60;

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

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// NON-SQUARE shell triples: the bra (i) and ket (j) angular momenta differ so a
/// transposed i↔j layout cannot pass. Shell 2 is the O 2p (l=1); shells 0/1/3/4
/// are s-shells. `(2,0,3)` is p×s×s, `(0,2,3)` is s×p×s — both non-square.
const TRIPLES: [(usize, usize, usize); 2] = [(2, 0, 3), (0, 2, 3)];

// ─────────────────────────────────────────────────────────────────────────────
// Per-triple rank-3 collectors (cintx via eval_raw / vendor via FFI).
// Layout: out[comp * ni*nj*nk + (k*nj + j)*ni + i].
// ─────────────────────────────────────────────────────────────────────────────

fn shell_ang(bas: &[i32], s: usize) -> i32 {
    bas[s * BAS_SLOTS + ANG_OF]
}

fn collect_cintx_triple(
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    triple: (usize, usize, usize),
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let (si, sj, sk) = triple;
    let ni = nf(shell_ang(bas, si));
    let nj = nf(shell_ang(bas, sj));
    let nk = nf(shell_ang(bas, sk));
    let n_elem = NCOMP * ni * nj * nk;
    let mut out = vec![0.0_f64; n_elem];
    let shls = [si as i32, sj as i32, sk as i32];
    // SAFETY: atm/bas/env well-formed by construction; shls valid.
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
        .unwrap_or_else(|e| panic!("eval_raw failed for triple {triple:?}: {e:?}"));
    }
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_triple<F>(
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    triple: (usize, usize, usize),
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let (si, sj, sk) = triple;
    let ni = nf(shell_ang(bas, si));
    let nj = nf(shell_ang(bas, sj));
    let nk = nf(shell_ang(bas, sk));
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let mut out = vec![0.0_f64; NCOMP * ni * nj * nk];
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
    assert!(
        any_nonzero,
        "{label}: buffer is all-zero (zero-fill regression)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinism tests (always run under cpu) — pin the 3-component element count.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
fn determinism(api_sph: RawApiId, api_cart: RawApiId, label: &str) {
    let (atm, bas, env) = build_h2o_sto3g();
    for (api, nf, rep) in [
        (api_sph, nsph as fn(i32) -> usize, "sph"),
        (api_cart, ncart as fn(i32) -> usize, "cart"),
    ] {
        for &triple in TRIPLES.iter() {
            let m1 = collect_cintx_triple(api, &atm, &bas, &env, triple, nf);
            let m2 = collect_cintx_triple(api, &atm, &bas, &env, triple, nf);
            let (si, sj, sk) = triple;
            let expect =
                NCOMP * nf(shell_ang(&bas, si)) * nf(shell_ang(&bas, sj)) * nf(shell_ang(&bas, sk));
            assert_eq!(
                m1.len(),
                expect,
                "{label}_{rep} {triple:?}: element count = 3*ni*nj*nk"
            );
            for (a, b) in m1.iter().zip(m2.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label}_{rep} {triple:?} must be bit-identical"
                );
            }
            assert_any_nonzero(&m1, &format!("{label}_{rep} {triple:?}"));
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_ip1_determinism() {
    determinism(
        RawApiId::INT3C1E_IP1_SPH,
        RawApiId::INT3C1E_IP1_CART,
        "int3c1e_ip1",
    );
}

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_iprinv_determinism() {
    determinism(
        RawApiId::INT3C1E_IPRINV_SPH,
        RawApiId::INT3C1E_IPRINV_CART,
        "int3c1e_iprinv",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// fff fail-closed assertion (D-13 / VALIDATION Manual-Only): the iprinv launcher
// rejects an fff shell tuple (nroots = (3+1 + 3 + 3)/2 + 1 = 6 > 5) with
// UnsupportedApi rather than truncating or panicking. ip1 (no Rys) is exempt.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_iprinv_fff_fails_closed() {
    use cintx_core::cintxRsError;

    // f-f-f H2 fixture: three f-shells (l=3) on three atoms; iprinv → nroots 6.
    let mut env = Vec::<f64>::new();
    env.resize(20, 0.0);
    env[PTR_RINV_ORIG] = 0.1;
    env[PTR_RINV_ORIG + 1] = 0.2;
    env[PTR_RINV_ORIG + 2] = 0.3;
    let a_coord = [0.0_f64, 0.0, 0.0];
    let b_coord = [1.4_f64, 0.0, 0.0];
    let c_coord = [0.0_f64, 1.4, 0.0];
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&a_coord);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&b_coord);
    let c_ptr = env.len() as i32;
    env.extend_from_slice(&c_coord);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.push(1.0); // single primitive
    let coeff_ptr = env.len() as i32;
    env.push(1.0);

    let mut atm = vec![0_i32; 3 * ATM_SLOTS];
    for (n, &ptr) in [a_ptr, b_ptr, c_ptr].iter().enumerate() {
        atm[n * ATM_SLOTS + CHARGE_OF] = 1;
        atm[n * ATM_SLOTS + PTR_COORD] = ptr;
        atm[n * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[n * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    for n in 0..3 {
        bas[n * BAS_SLOTS + ATOM_OF] = n as i32;
        bas[n * BAS_SLOTS + ANG_OF] = 3; // f-shell
        bas[n * BAS_SLOTS + NPRIM_OF] = 1;
        bas[n * BAS_SLOTS + NCTR_OF] = 1;
        bas[n * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[n * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }

    let ni = ncart(3);
    let mut out = vec![0.0_f64; NCOMP * ni * ni * ni];
    let shls = [0_i32, 1, 2];
    // SAFETY: atm/bas/env well-formed; the call must fail closed (no write past guard).
    let res = unsafe {
        eval_raw(
            RawApiId::INT3C1E_IPRINV_CART,
            Some(&mut out),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    };
    match res {
        Err(cintxRsError::UnsupportedApi { .. }) => {}
        other => panic!("int3c1e_iprinv fff must fail closed with UnsupportedApi, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint + cpu).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
fn vendor_parity<FS, FC>(
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: FS,
    vendor_cart: FC,
    label: &str,
) where
    FS: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    FC: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let (atm, bas, env) = build_h2o_sto3g();

    for &triple in TRIPLES.iter() {
        let vendor_s = collect_vendor_triple(&vendor_sph, &atm, &bas, &env, triple, nsph);
        let cintx_s = collect_cintx_triple(api_sph, &atm, &bas, &env, triple, nsph);
        assert_any_nonzero(&cintx_s, &format!("{label}_sph {triple:?} cintx"));
        assert_any_nonzero(&vendor_s, &format!("{label}_sph {triple:?} vendor"));
        let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
        assert_eq!(
            mm, 0,
            "{label}_sph {triple:?}: {mm} mismatches vs vendored libcint at atol={ATOL}"
        );

        let vendor_c = collect_vendor_triple(&vendor_cart, &atm, &bas, &env, triple, ncart);
        let cintx_c = collect_cintx_triple(api_cart, &atm, &bas, &env, triple, ncart);
        assert_any_nonzero(&cintx_c, &format!("{label}_cart {triple:?} cintx"));
        assert_any_nonzero(&vendor_c, &format!("{label}_cart {triple:?} vendor"));
        let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
        assert_eq!(
            mm, 0,
            "{label}_cart {triple:?}: {mm} mismatches vs vendored libcint at atol={ATOL}"
        );
    }
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_ip1_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT3C1E_IP1_SPH,
        RawApiId::INT3C1E_IP1_CART,
        vendor_ffi::vendor_int3c1e_ip1_sph,
        vendor_ffi::vendor_int3c1e_ip1_cart,
        "int3c1e_ip1",
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_iprinv_h2o_sto3g_parity() {
    use cintx_oracle::vendor_ffi;
    vendor_parity(
        RawApiId::INT3C1E_IPRINV_SPH,
        RawApiId::INT3C1E_IPRINV_CART,
        vendor_ffi::vendor_int3c1e_iprinv_sph,
        vendor_ffi::vendor_int3c1e_iprinv_cart,
        "int3c1e_iprinv",
    );
}
