//! Pins `cintx-driver`'s locally-declared libcint ABI slot constants against
//! `cintx_compat::raw`.
//!
//! The driver declares them locally so a slot change in the compat crate cannot
//! silently reinterpret `bas` rows here; this test is what makes that safe.

use cintx_driver::basis_view as driver;

#[test]
fn slot_constants_match_compat() {
    use cintx_compat::raw;
    assert_eq!(driver::ATM_SLOTS, raw::ATM_SLOTS);
    assert_eq!(driver::BAS_SLOTS, raw::BAS_SLOTS);
    assert_eq!(driver::ANG_OF, raw::ANG_OF);
    assert_eq!(driver::NPRIM_OF, raw::NPRIM_OF);
    assert_eq!(driver::NCTR_OF, raw::NCTR_OF);
}

/// `quartet_nroots` must reproduce the libcint 2e Rys order used by
/// `build_2e_shape`, since the whole device-eligibility decision rests on it.
#[test]
fn quartet_nroots_matches_libcint_formula() {
    use cintx_driver::BasisView;
    let l = [0_i32, 1, 2, 3, 4];
    let mut bas = vec![0_i32; l.len() * driver::BAS_SLOTS];
    for (shell, &value) in l.iter().enumerate() {
        bas[shell * driver::BAS_SLOTS + driver::ANG_OF] = value;
        bas[shell * driver::BAS_SLOTS + driver::NPRIM_OF] = 1;
        bas[shell * driver::BAS_SLOTS + driver::NCTR_OF] = 1;
    }
    let atm = vec![0_i32; driver::ATM_SLOTS];
    let env = vec![0.0_f64; 8];
    let basis = BasisView::new(&atm, &bas, &env);

    for i in 0..l.len() {
        for j in 0..l.len() {
            for k in 0..l.len() {
                for m in 0..l.len() {
                    let expected = ((l[i] + l[j] + l[k] + l[m]) / 2 + 1) as u32;
                    assert_eq!(basis.quartet_nroots(i, j, k, m), expected);
                }
            }
        }
    }
}
