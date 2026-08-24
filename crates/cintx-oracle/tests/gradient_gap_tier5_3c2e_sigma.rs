//! W4-05 — `int3c2e_ipspsp1` (3-centre σ·p gradient) byte-identity vs vendored libcint.
//!
//! `int3c2e.c:668`, `ng = {2,1,0,0,3,4,1,3}`, rank 3, spinor only (the manifest-wide
//! σ-family precedent). Electron 1 folds through the new `c2s_si_3c2e1` analogue
//! `cart_to_spinor_si_3c2e1`; the auxiliary index is spherical.
//!
//! Fixture: `d`-shell bra/ket with `nctr = 2` and a `d` auxiliary. At `ng[0..1] = {2,1}`
//! that is `nroots = (4 + 3 + 0 + 2)/2 + 1 = 5`, at the device cap, so the family runs
//! on the host Rys route like its `int2e_ipspsp1` sibling.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;
const RANK: usize = 3;

/// (atm, bas, env) for a 3-shell system: bra i, ket j, auxiliary k.
fn fixture(angs: [i32; 3], nctr: i32) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let mut env = vec![0.0; PTR_ENV_START];
    let a = env.len() as i32;
    env.extend_from_slice(&[-0.4, 0.1, -0.2]);
    let b = env.len() as i32;
    env.extend_from_slice(&[0.5, -0.3, 0.7]);
    let z = env.len() as i32;
    env.push(0.0);
    let e = env.len() as i32;
    env.extend_from_slice(&[1.7, 0.45]);
    let c = env.len() as i32;
    env.extend_from_slice(&[0.7, 0.3, -0.35, 0.8]);
    let mut atm = vec![0; 2 * ATM_SLOTS];
    for (o, ch, co) in [(0, 6, a), (ATM_SLOTS, 8, b)] {
        atm[o + CHARGE_OF] = ch;
        atm[o + PTR_COORD] = co;
        atm[o + NUC_MOD_OF] = POINT_NUC;
        atm[o + PTR_ZETA] = z;
    }
    let mut bas = vec![0; 3 * BAS_SLOTS];
    for s in 0..3 {
        let o = s * BAS_SLOTS;
        bas[o + ATOM_OF] = (s % 2) as i32;
        bas[o + ANG_OF] = angs[s];
        bas[o + NPRIM_OF] = 2;
        bas[o + NCTR_OF] = nctr;
        bas[o + PTR_EXP] = e;
        bas[o + PTR_COEFF] = c;
    }
    (atm, bas, env)
}

#[cfg(has_vendor_libcint)]
#[test]
fn vendor_int3c2e_ipspsp1_spinor() {
    use cintx_oracle::vendor_ffi::vendor_int3c2e_ipspsp1_spinor;

    // Non-square (li != lj) so an i↔j orientation slip cannot cancel.
    let cases: [([i32; 3], i32); 3] = [([1, 1, 1], 1), ([2, 1, 2], 2), ([1, 2, 1], 2)];
    let mut report = String::new();
    let mut failed = 0usize;

    for (angs, nctr) in cases {
        let (atm, bas, env) = fixture(angs, nctr);
        let shls = [0, 1, 2];
        let n = nctr as usize;
        // bra/ket are spinor (4l+2); the auxiliary index is SPHERICAL (2l+1).
        let ni = (4 * angs[0] as usize + 2) * n;
        let nj = (4 * angs[1] as usize + 2) * n;
        let nk = (2 * angs[2] as usize + 1) * n;
        let len = RANK * ni * nj * nk * 2;

        let mut ours = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int3c2e_ipspsp1_spinor"),
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
        let mut vendor = vec![0.0; len];
        assert_ne!(
            vendor_int3c2e_ipspsp1_spinor(&mut vendor, &shls, &atm, 2, &bas, 3, &env),
            0,
            "{angs:?}: vendored libcint returned an empty shell block"
        );
        assert!(
            vendor.iter().any(|v| v.abs() > 1e-14),
            "{angs:?}: vendor reference is all-zero (driver not linked)"
        );

        let max_abs = ours
            .iter()
            .zip(&vendor)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_abs > ATOL {
            failed += 1;
        }
        let verdict = if max_abs <= ATOL { "ok  " } else { "FAIL" };
        report.push_str(&format!(
            "\n  {verdict} angs={angs:?} nctr={nctr} max_abs={max_abs:.3e}"
        ));
    }
    assert!(
        failed == 0,
        "int3c2e_ipspsp1_spinor: {failed}/3 cases exceed atol={ATOL:.0e}{report}"
    );
}
