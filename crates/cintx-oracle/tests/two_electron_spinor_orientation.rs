//! Regression gate for the `cart_to_spinor_sf_4d` `(i,j)` orientation.
//!
//! `cart_to_spinor_sf_4d` takes its Cartesian input i-fastest
//! (`cart[((l*nck + k)*ncj + j)*nci + i]`) but forwards each `(k,l)` slice to
//! `cart_to_spinor_sf_2d`, which reads BRA-major and does NOT own the KET→BRA
//! transpose. The transform therefore has to perform it.
//!
//! While it did not, `int2e_spinor` — `Stability::Stable`, `oracle_covered = true`,
//! shipped — returned the `i↔j` transpose of the correct spinor block: exact for an
//! all-`s` quartet (`nci == ncj == 1`) and wrong by ~3e-3 for every `l > 0` shell.
//! The pre-existing coverage only exercised `s` shells, so the defect was invisible.
//!
//! This gate sweeps NON-SQUARE quartets, where an `(i,j)` transpose cannot cancel.

#![cfg(feature = "cpu")]

use cintx_compat::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, NUC_MOD_OF, POINT_NUC,
    PTR_COEFF, PTR_COORD, PTR_ENV_START, PTR_EXP, PTR_ZETA, RawApiId, eval_raw,
};

const ATOL: f64 = 1e-12;

fn fixture(angs: [i32; 4], nctr: i32) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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
    let mut bas = vec![0; 4 * BAS_SLOTS];
    for s in 0..4 {
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
fn int2e_spinor_matches_vendor_on_nonsquare_quartets() {
    use cintx_oracle::vendor_ffi::vendor_int2e_spinor;

    // The `[0,0,0,0]` case is the one the pre-existing coverage exercised; it is kept
    // so the sweep shows explicitly that it alone cannot detect the transpose.
    let cases: [([i32; 4], i32); 5] = [
        ([0, 0, 0, 0], 1),
        ([1, 1, 1, 1], 1),
        ([1, 2, 0, 1], 1),
        ([2, 1, 1, 0], 1),
        ([2, 1, 1, 2], 2),
    ];

    let mut report = String::new();
    let mut failed = 0usize;
    for (angs, nctr) in cases {
        let (atm, bas, env) = fixture(angs, nctr);
        let dims: Vec<usize> = angs
            .iter()
            .map(|&l| (4 * l as usize + 2) * nctr as usize)
            .collect();
        let len = dims.iter().product::<usize>() * 2;
        let shls = [0, 1, 2, 3];

        let mut ours = vec![0.0; len];
        unsafe {
            eval_raw(
                RawApiId::Symbol("int2e_spinor"),
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
            vendor_int2e_spinor(&mut vendor, &shls, &atm, 2, &bas, 4, &env),
            0,
            "{angs:?}: vendored libcint returned an empty shell block"
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
        "int2e_spinor: {failed}/5 quartets exceed atol={ATOL:.0e}{report}"
    );
}
