//! **A recorded defect, not a passing gate.** `int3c1e_p2` evaluates plain
//! `int3c1e`: the `p2` operator is dropped.
//!
//! # How it survived
//!
//! `OracleRawInputs::sample` puts all four shells on one atom at the origin
//! with `l = 0, 1, 0, 1`, so the arity-3 tuple is `(0, 1, 0)`. On that geometry
//! `int3c1e_p2` is **identically zero**, and so is `int3c1e`. The manifest gate
//! compares them and finds agreement — the failure mode where "both sides
//! returned zero" is indistinguishable from "both sides are right". The family
//! is stamped oracle-covered on the strength of it.
//!
//! It surfaced when `def2_speed_precision_plan.md` D4 added a second fixture
//! set at a real two-centre geometry and swept the manifest on it.
//!
//! # What is actually wrong
//!
//! cintx returns, bit for bit, the value of the corresponding `int3c1e`. That
//! is asserted below, because "wrong by 80%" is a symptom and "the operator is
//! not applied" is the diagnosis — and a fix has to change the first assertion,
//! not merely reduce the number in the second.
//!
//! # Why these tests are `#[ignore]`d
//!
//! They fail. They are checked in so the defect is reproducible and cannot be
//! rediscovered from scratch, and they are ignored so they do not turn a known,
//! unrelated gap into a red suite for every other change. Deleting the
//! `#[ignore]` is part of fixing `int3c1e_p2`, not a separate cleanup.
//!
//! ```text
//! CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu \
//!   --test int3c1e_p2_operator_defect -- --ignored --nocapture
//! ```

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};
use cintx_oracle::vendor_ffi;

/// Three single-primitive shells: `i` and `j` on centre 0, `k` on centre 1,
/// 2.2 bohr away. Any two-centre arrangement shows the defect; this one is the
/// `OracleRawInputs::def2_high_order` geometry, which is where it was found.
fn fixture(ls: [i32; 3]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]];
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut coord_ptr = [0_i32; 2];
    for (index, coord) in coords.iter().enumerate() {
        coord_ptr[index] = env.len() as i32;
        env.extend_from_slice(coord);
    }

    let mut atm = vec![0_i32; 2 * ATM_SLOTS];
    for index in 0..2 {
        atm[index * ATM_SLOTS + CHARGE_OF] = 8;
        atm[index * ATM_SLOTS + PTR_COORD] = coord_ptr[index];
        atm[index * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;
        atm[index * ATM_SLOTS + PTR_ZETA] = 0;
    }

    let atoms = [0, 0, 1];
    let exponents = [1.40, 0.85, 1.10];
    let mut bas = vec![0_i32; 3 * BAS_SLOTS];
    for index in 0..3 {
        let exp_ptr = env.len() as i32;
        env.push(exponents[index]);
        let coeff_ptr = env.len() as i32;
        env.push(1.0);
        bas[index * BAS_SLOTS + ATOM_OF] = atoms[index];
        bas[index * BAS_SLOTS + ANG_OF] = ls[index];
        bas[index * BAS_SLOTS + NPRIM_OF] = 1;
        bas[index * BAS_SLOTS + NCTR_OF] = 1;
        bas[index * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[index * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn evaluate(symbol: &'static str, ls: [i32; 3]) -> (Vec<f64>, Vec<f64>) {
    let (atm, bas, env) = fixture(ls);
    let len: usize = ls.iter().map(|&l| ncart(l)).product();
    let mut expected = vec![0.0_f64; len];
    match symbol {
        "int3c1e_cart" => {
            vendor_ffi::vendor_int3c1e_cart(&mut expected, &[0, 1, 2], &atm, 2, &bas, 3, &env)
        }
        "int3c1e_p2_cart" => {
            vendor_ffi::vendor_int3c1e_p2_cart(&mut expected, &[0, 1, 2], &atm, 2, &bas, 3, &env)
        }
        other => panic!("unhandled symbol {other}"),
    };
    let mut actual = vec![0.0_f64; len];
    // SAFETY: `actual` is sized from the same Cartesian AO counts the vendor
    // writes for these shells.
    unsafe {
        eval_raw(
            RawApiId::Symbol(symbol),
            Some(&mut actual),
            None,
            &[0, 1, 2],
            &atm,
            &bas,
            &env,
            None,
            None,
        )
    }
    .unwrap_or_else(|e| panic!("{symbol} failed: {e}"));
    (expected, actual)
}

/// **The diagnosis.** cintx's `int3c1e_p2` is bit-identical to its own
/// `int3c1e` — the operator is not applied at all.
///
/// Bit-identity, not a tolerance: two different operators agreeing to 1e-15
/// would be a coincidence worth investigating, but agreeing exactly is a
/// statement about which code ran.
#[test]
#[ignore = "records a known defect: int3c1e_p2 evaluates int3c1e (see the module docs)"]
fn int3c1e_p2_returns_the_plain_int3c1e_value() {
    for ls in [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 4]] {
        let (_, p2) = evaluate("int3c1e_p2_cart", ls);
        let (_, plain) = evaluate("int3c1e_cart", ls);
        let identical = p2.iter().zip(&plain).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!(
            "  l={ls:?}: {identical}/{} elements of int3c1e_p2 are bit-identical to int3c1e",
            p2.len()
        );
        assert_eq!(
            identical,
            p2.len(),
            "l={ls:?}: int3c1e_p2 is not simply int3c1e here, so the diagnosis in \
             this file's module docs is incomplete and needs revisiting before \
             anything is fixed"
        );
    }
}

/// **The symptom**, and the assertion a fix has to make pass.
///
/// `int3c1e` itself is included as the control: it agrees with the vendor to
/// machine precision at every angular momentum here, so the geometry is not the
/// problem.
#[test]
#[ignore = "records a known defect: int3c1e_p2 evaluates int3c1e (see the module docs)"]
fn int3c1e_p2_matches_vendor() {
    const ATOL: f64 = 1e-12;

    for symbol in ["int3c1e_cart", "int3c1e_p2_cart"] {
        for ls in [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 4]] {
            let (expected, actual) = evaluate(symbol, ls);
            let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let worst = expected
                .iter()
                .zip(&actual)
                .map(|(e, a)| (e - a).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "  {symbol} l={ls:?}: block peak={scale:.4e} worst|diff|={worst:.4e} \
                 ({:.2e} of peak)",
                worst / scale.max(f64::MIN_POSITIVE)
            );
            assert!(
                scale > 1e-6,
                "{symbol} l={ls:?}: the vendor block peaks at {scale:.3e}, which is \
                 too near zero for agreement to mean anything — pick a different \
                 fixture rather than trusting this row"
            );
            assert!(
                worst <= ATOL,
                "{symbol} l={ls:?}: worst |diff| {worst:.4e} exceeds atol={ATOL:e} \
                 on a block whose peak is {scale:.4e}"
            );
        }
    }
}
