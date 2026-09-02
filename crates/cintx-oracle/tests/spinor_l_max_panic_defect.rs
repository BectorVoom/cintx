//! **A recorded defect, not a passing gate.** A spinor shell above `l = 4`
//! makes the Cartesian-to-spinor transform **panic** instead of returning a
//! typed refusal.
//!
//! # The exact failure
//!
//! ```text
//! thread '...' panicked at crates/cintx-cubecl/src/transform/c2spinor.rs:1088:
//! cart_to_spinor_sf_2d: l=5 > 4 not supported
//! ```
//!
//! reached through `eval_raw` -> `launch_two_electron` ->
//! `cart_to_spinor_sf_4d` -> `apply_2d_spinor_zf`. Nothing between the public
//! entry point and that `panic!` rejects the shell, so an ordinary caller with
//! an `h` function and a spinor symbol gets an unwind rather than an error it
//! can handle.
//!
//! # Why this is a contract problem and not just a table limit
//!
//! `cintx_core::SPHERIC_L_MAX` exists for exactly this shape of gap on the
//! *spherical* side, and its own doc says why: "the transform used to return
//! `0.0` above `l = 4`, so an out-of-range spherical shell came back entirely
//! zeroed with an `Ok` status." That was fixed with a per-shell validation
//! against the table's real ceiling. The spinor side has the same finite table
//! and no such constant — the failure mode was changed from silent zeros to a
//! panic, which is better, but a panic is still not the typed refusal the
//! project's error contract promises.
//!
//! The shape of a fix: a `SPINOR_L_MAX` alongside `SPHERIC_L_MAX`, validated
//! per shell against its own representation, and `apply_2d_spinor_zf`'s
//! catch-all returning an error rather than unwinding.
//!
//! # How it was found
//!
//! `def2_speed_precision_plan.md` D4 added a fixture set carrying an `h` shell
//! and swept the manifest on it. `OracleRawInputs::def2_high_order` excludes the
//! spinor representation for this reason, and says so.
//!
//! # Why this test is `#[ignore]`d
//!
//! It fails — by panicking, which is the point. It is checked in so the defect
//! is reproducible, and ignored so a known, unrelated gap does not turn the
//! suite red for every other change. Removing the `#[ignore]` is part of fixing
//! it.
//!
//! ```text
//! cargo test -p cintx-oracle --features cpu \
//!   --test spinor_l_max_panic_defect -- --ignored --nocapture
//! ```

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

/// Two single-primitive shells at angular momentum `l`, on two centres.
fn fixture(l: i32) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    let mut bas = vec![0_i32; 2 * BAS_SLOTS];
    for index in 0..2 {
        let exp_ptr = env.len() as i32;
        env.push(1.2 + 0.2 * index as f64);
        let coeff_ptr = env.len() as i32;
        env.push(1.0);
        bas[index * BAS_SLOTS + ATOM_OF] = index as i32;
        bas[index * BAS_SLOTS + ANG_OF] = l;
        bas[index * BAS_SLOTS + NPRIM_OF] = 1;
        bas[index * BAS_SLOTS + NCTR_OF] = 1;
        bas[index * BAS_SLOTS + PTR_EXP] = exp_ptr;
        bas[index * BAS_SLOTS + PTR_COEFF] = coeff_ptr;
    }
    (atm, bas, env)
}

/// Spinor AO count for one shell, `2 * (2l + 1)` interleaved real/imaginary
/// halves aside — the buffer is sized generously because the call is expected
/// to fail before it writes anything.
fn generous_len(l: i32) -> usize {
    let n = 4 * (2 * l + 1) as usize;
    n * n
}

/// **The gate a fix has to make pass.** Above the spinor table's ceiling the
/// call must return a typed error, not unwind.
///
/// `l = 4` is included as the control: it is inside the table, so it must
/// succeed. If that ever starts failing, the problem is the fixture, not the
/// ceiling.
#[test]
#[ignore = "records a known defect: the spinor transform panics above l=4 instead of refusing"]
fn spinor_above_the_table_ceiling_refuses_instead_of_panicking() {
    for l in [4_i32, 5, 6] {
        let (atm, bas, env) = fixture(l);
        let mut out = vec![0.0_f64; generous_len(l)];
        // SAFETY: `out` is sized well past the spinor block extent for these
        // shells, so a write inside the contract cannot overrun it.
        let status = unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_spinor"),
                Some(&mut out),
                None,
                &[0, 1, 0, 1],
                &atm,
                &bas,
                &env,
                None,
                None,
            )
        };
        println!("  l={l}: {}", if status.is_ok() { "ok" } else { "typed error" });
        if l <= 4 {
            assert!(
                status.is_ok(),
                "l={l} is inside the spinor table, so it must evaluate: {status:?}"
            );
        } else {
            assert!(
                status.is_err(),
                "l={l} is past the spinor transform's table ceiling and must return \
                 a typed refusal — reaching this line at all means the call did not \
                 panic, which would be the fix landing"
            );
        }
    }
}
