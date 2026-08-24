//! Spin-dependent two-electron gradient gap oracle gates.
//!
//! W4-00 hygiene (gradient-gap-wave-4-PLAN.md §3):
//!   - `ATOL` is RULE 4's `1e-12`, not the looser `2e-11` this file shipped with.
//!   - the fixture is a `d`-shell `nctr=2` quartet, not a `p`-shell one. At
//!     `ng = {2,1,0,0,...}` a `d` quartet gives `nroots = (4+3+2+2)/2 + 1 = 6`,
//!     which crosses `MAX_DEVICE_NROOTS` and exercises the host Rys route that a
//!     `p`-shell fixture never reaches.
//!   - every family is measured before anything asserts, so one red family no
//!     longer hides the other five.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

/// Angular momentum of every shell in the quartet (`d`).
const ANG: i32 = 2;
/// Contractions per shell (general contraction, `nctr > 1`).
const NCTR: usize = 2;
/// Spinor functions per shell: `(4l + 2) * nctr`.
const NSPINOR: usize = (4 * ANG as usize + 2) * NCTR;

fn fixture() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    let a_ptr = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b_ptr = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let zeta_ptr = env.len() as i32;
    env.push(0.0);
    let exp_ptr = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let coeff_ptr = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.35, 0.8]);
    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (offset, charge, coord) in [(0, 6, a_ptr), (ATM_SLOTS, 8, b_ptr)] {
        atm[offset + CHARGE_OF] = charge;
        atm[offset + PTR_COORD] = coord;
        atm[offset + NUC_MOD_OF] = POINT_NUC;
        atm[offset + PTR_ZETA] = zeta_ptr;
    }
    let mut bas = vec![0; 4 * BAS_SLOTS];
    for shell in 0..4 {
        let offset = shell * BAS_SLOTS;
        bas[offset + ATOM_OF] = (shell % 2) as i32;
        bas[offset + ANG_OF] = ANG;
        bas[offset + NPRIM_OF] = 2;
        bas[offset + NCTR_OF] = NCTR as i32;
        bas[offset + PTR_EXP] = exp_ptr;
        bas[offset + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

type VendorFn = fn(&mut [f64], &[i32; 4], &[i32], i32, &[i32], i32, &[f64]) -> i32;

/// Evaluate one family through cintx and through vendored libcint and return the
/// max absolute deviation. Never asserts — the caller reports every family.
#[cfg(has_vendor_libcint)]
fn measure(symbol: &str, rank: usize, vendor_fn: VendorFn) -> f64 {
    let (atm, bas, env) = fixture();
    let shls = [0, 1, 2, 3];
    let len = rank * NSPINOR.pow(4) * 2;

    let symbol_static: &'static str = Box::leak(symbol.to_owned().into_boxed_str());
    let mut ours = vec![0.0; len];
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol_static),
            Some(&mut ours),
            None,
            &shls,
            &atm,
            &bas,
            &env,
            None,
            None,
        )
        .unwrap();
    }

    let mut reference = vec![0.0; len];
    assert_ne!(
        vendor_fn(&mut reference, &shls, &atm, 2, &bas, 4, &env),
        0,
        "{symbol}: vendored libcint returned an empty shell block"
    );

    ours.iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max)
}

/// Assert over a whole family table at once, reporting every measurement.
#[cfg(has_vendor_libcint)]
fn assert_all_green(label: &str, measured: &[(&str, f64)]) {
    let mut report = String::new();
    let mut failed = 0usize;
    for (symbol, max_abs) in measured {
        let verdict = if *max_abs <= ATOL { "ok  " } else { "FAIL" };
        if *max_abs > ATOL {
            failed += 1;
        }
        report.push_str(&format!("\n  {verdict} {symbol:32} max_abs={max_abs:.3e}"));
    }
    assert!(
        failed == 0,
        "{label}: {failed}/{} families exceed atol={ATOL:.0e}{report}",
        measured.len()
    );
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_two_electron_sigma_gradient_spinor() {
    use cintx_oracle::vendor_ffi as vendor;
    let cases: [(&str, usize, VendorFn); 6] = [
        (
            "int2e_ipspsp1_spinor",
            3,
            vendor::vendor_int2e_ipspsp1_spinor,
        ),
        (
            "int2e_ip1spsp2_spinor",
            3,
            vendor::vendor_int2e_ip1spsp2_spinor,
        ),
        (
            "int2e_ipspsp1spsp2_spinor",
            3,
            vendor::vendor_int2e_ipspsp1spsp2_spinor,
        ),
        (
            "int2e_ipsrsr1_spinor",
            3,
            vendor::vendor_int2e_ipsrsr1_spinor,
        ),
        (
            "int2e_ip1srsr2_spinor",
            3,
            vendor::vendor_int2e_ip1srsr2_spinor,
        ),
        (
            "int2e_ipsrsr1srsr2_spinor",
            3,
            vendor::vendor_int2e_ipsrsr1srsr2_spinor,
        ),
    ];
    let measured: Vec<(&str, f64)> = cases
        .into_iter()
        .map(|(symbol, rank, vendor_fn)| (symbol, measure(symbol, rank, vendor_fn)))
        .collect();
    assert_all_green("2e sigma gradients", &measured);
}

/// `int2e_spsp2` is a Phase-29 REL-03 family, not a Wave-4 one, but it is the base
/// of `int2e_ip1spsp2` and shares its `(c2s_sf_2e1, c2s_si_2e2)` transform pairing.
/// It is gated here because Phase 29 left it out of `rel_2e_sigma_parity.rs`.
#[cfg(has_vendor_libcint)]
#[test]
fn vendor_electron_two_sigma_base_spinor() {
    use cintx_oracle::vendor_ffi::vendor_int2e_spsp2_spinor;
    let measured = [(
        "int2e_spsp2_spinor",
        measure("int2e_spsp2_spinor", 1, vendor_int2e_spsp2_spinor),
    )];
    assert_all_green("2e sigma base", &measured);
}
