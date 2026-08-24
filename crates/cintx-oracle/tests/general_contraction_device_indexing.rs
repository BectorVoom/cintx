//! Does the device 2e path index general-contraction coefficients correctly?
//!
//! Reading the source suggests it does not: the device kernel uses
//! `coeff_i[pi * nctr_i + ci]` (`two_electron.rs:1229`) while the host fallback
//! for the *same uploaded buffer* uses `coeff_i[ci * n_prim_i + pi]`
//! (`two_electron.rs:4470`) — the libcint `env` layout — and
//! `run_2e_scalar_device` uploads the slice verbatim with no transpose.
//!
//! The two agree only when `nctr == 1` (the `is_uncontracted` fast path short-
//! circuits it) or `nprim == nctr`. This test settles the question empirically
//! rather than by inspection, with a shell that has `nprim = 3, nctr = 2` and
//! low enough angular momentum to stay on the device path (`nroots <= 5`).
//!
//! def2-SVP and def2-TZVP are fully segmented (`nctr == 1` everywhere), so this
//! is **not** a def2 defect — it would affect generally contracted bases such as
//! cc-pVXZ.
//!
//! The same question is asked of every other device family that reads a
//! contraction coefficient buffer — `int2c2e`, `int3c2e` and the `int1e_*`
//! trio — for the same reason: `nctr == 1` fixtures cannot distinguish a
//! correct index from a transposed or collapsed one.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_oracle::vendor_ffi;

/// Two atoms; shell 0 is a general contraction (nprim=3, nctr=2) on atom 0,
/// shell 1 a plain uncontracted s shell on atom 1.
fn build_general_contraction() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0_f64; PTR_ENV_START];

    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let a_coord = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.0, 0.0]);
    let b_coord = env.len() as i32;
    env.extend_from_slice(&[0.0, 0.0, 1.4]);

    // nprim = 3, nctr = 2. Coefficients are contraction-major
    // (`coeff[ic * nprim + ip]`), the libcint `env` layout — and deliberately
    // asymmetric so a transposed read gives a different answer.
    let exps = [6.0_f64, 1.5, 0.4];
    let coeffs = [
        0.20_f64, 0.55, 0.30, // contraction 0
        -0.10, 0.35, 0.80, // contraction 1
    ];
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&exps);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&coeffs);

    let s_exp = [0.9_f64];
    let s_coeff = [1.0_f64];
    let s_exp_ptr = env.len() as i32;
    env.extend_from_slice(&s_exp);
    let s_coeff_ptr = env.len() as i32;
    env.extend_from_slice(&s_coeff);

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for (index, ptr) in [(0_usize, a_coord), (1, b_coord)] {
        atm[index * ATM_SLOTS + CHARGE_OF] = 1;
        atm[index * ATM_SLOTS + PTR_COORD] = ptr;
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = zeta_ptr;
    }

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    bas[ATOM_OF] = 0;
    bas[ANG_OF] = 0;
    bas[NPRIM_OF] = 3;
    bas[NCTR_OF] = 2;
    bas[PTR_EXP] = exp_ptr;
    bas[PTR_COEFF] = coeff_ptr;

    bas[BAS_SLOTS + ATOM_OF] = 1;
    bas[BAS_SLOTS + ANG_OF] = 0;
    bas[BAS_SLOTS + NPRIM_OF] = 1;
    bas[BAS_SLOTS + NCTR_OF] = 1;
    bas[BAS_SLOTS + PTR_EXP] = s_exp_ptr;
    bas[BAS_SLOTS + PTR_COEFF] = s_coeff_ptr;

    (atm, bas, env)
}

#[test]
fn general_contraction_2e_matches_vendor_on_device_path() {
    let (atm, bas, env) = build_general_contraction();

    // Shell 0 has nctr=2 -> 2 spherical AOs; shell 1 has 1.
    // (0 0 | 0 0) with l all zero => nroots = 1, well inside the device path.
    let shls = [0_i32, 0, 0, 0];
    let len = 2 * 2 * 2 * 2;

    let mut expected = vec![0.0_f64; len];
    vendor_ffi::vendor_int2e_sph(&mut expected, &shls, &atm, 2, &bas, 2, &env);

    let mut actual = vec![0.0_f64; len];
    unsafe {
        eval_raw(
            RawApiId::INT2E_SPH,
            Some(&mut actual),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .expect("general-contraction 2e should evaluate");

    let max_diff = expected
        .iter()
        .zip(&actual)
        .map(|(e, a)| (e - a).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff < 1e-12,
        "general contraction (nprim=3, nctr=2) on the device 2e path disagrees \
         with vendored libcint by {max_diff:.3e}\n  vendor: {expected:?}\n  cintx:  {actual:?}"
    );
}

/// Control: the same geometry with `nctr == 1` must agree, proving any failure
/// above is specific to general contraction and not to the fixture.
#[test]
fn segmented_contraction_2e_matches_vendor() {
    let (atm, mut bas, env) = build_general_contraction();
    bas[NCTR_OF] = 1; // read only the first contraction column

    let shls = [0_i32, 0, 0, 0];
    let len = 1;

    let mut expected = vec![0.0_f64; len];
    vendor_ffi::vendor_int2e_sph(&mut expected, &shls, &atm, 2, &bas, 2, &env);

    let mut actual = vec![0.0_f64; len];
    unsafe {
        eval_raw(
            RawApiId::INT2E_SPH,
            Some(&mut actual),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .expect("segmented 2e should evaluate");

    assert!(
        (expected[0] - actual[0]).abs() < 1e-12,
        "segmented control failed: vendor={} cintx={}",
        expected[0],
        actual[0]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Same question, other families
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate `api` through cintx and compare against a vendor reference of the
/// same length, returning the largest absolute difference.
fn compare(
    api: RawApiId,
    shls: &[i32],
    len: usize,
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
    vendor: impl FnOnce(&mut [f64]),
) -> f64 {
    let mut expected = vec![0.0_f64; len];
    vendor(&mut expected);

    let mut actual = vec![0.0_f64; len];
    unsafe {
        eval_raw(
            api,
            Some(&mut actual),
            None,
            shls,
            atm,
            bas,
            env,
            None,
            None,
        )
    }
    .expect("evaluation should succeed");

    let _ = (natm, nbas);
    expected
        .iter()
        .zip(&actual)
        .map(|(e, a)| (e - a).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn general_contraction_2c2e_matches_vendor() {
    let (atm, bas, env) = build_general_contraction();
    // Shell 0 has nctr=2 -> 2 AOs, shell 1 has 1.
    let shls = [0_i32, 0];
    let len = 2 * 2;
    let max_diff = compare(
        RawApiId::INT2C2E_SPH,
        &shls,
        len,
        &atm,
        2,
        &bas,
        2,
        &env,
        |out| {
            vendor_ffi::vendor_int2c2e_sph(out, &[0, 0], &atm, 2, &bas, 2, &env);
        },
    );
    assert!(
        max_diff < 1e-12,
        "general contraction (nprim=3, nctr=2) on the 2c2e path disagrees with \
         vendored libcint by {max_diff:.3e}"
    );
}

#[test]
fn general_contraction_3c2e_matches_vendor() {
    let (atm, bas, env) = build_general_contraction();
    let shls = [0_i32, 0, 0];
    let len = 2 * 2 * 2;
    let max_diff = compare(
        RawApiId::INT3C2E_SPH,
        &shls,
        len,
        &atm,
        2,
        &bas,
        2,
        &env,
        |out| {
            vendor_ffi::vendor_int3c2e_sph(out, &[0, 0, 0], &atm, 2, &bas, 2, &env);
        },
    );
    assert!(
        max_diff < 1e-12,
        "general contraction (nprim=3, nctr=2) on the 3c2e path disagrees with \
         vendored libcint by {max_diff:.3e}"
    );
}

#[test]
fn general_contraction_1e_matches_vendor() {
    let (atm, bas, env) = build_general_contraction();
    let shls = [0_i32, 0];
    let len = 2 * 2;

    for (label, api, vendor) in [
        (
            "int1e_ovlp_sph",
            RawApiId::INT1E_OVLP_SPH,
            vendor_ffi::vendor_int1e_ovlp_sph
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        ),
        (
            "int1e_kin_sph",
            RawApiId::INT1E_KIN_SPH,
            vendor_ffi::vendor_int1e_kin_sph
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        ),
        (
            "int1e_nuc_sph",
            RawApiId::INT1E_NUC_SPH,
            vendor_ffi::vendor_int1e_nuc_sph
                as fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
        ),
    ] {
        let max_diff = compare(api, &shls, len, &atm, 2, &bas, 2, &env, |out| {
            vendor(out, &[0, 0], &atm, 2, &bas, 2, &env);
        });
        assert!(
            max_diff < 1e-12,
            "general contraction (nprim=3, nctr=2) on the {label} path disagrees \
             with vendored libcint by {max_diff:.3e}"
        );
    }
}
