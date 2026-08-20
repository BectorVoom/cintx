//! Phase 24 MOM-02 vendor parity scaffold: `int1e_rr` (rank 9), `int1e_r2` (rank 1),
//! `int1e_z` (rank 1), `int1e_zz` (rank 1) + their `_origj` variants, cart + sph, on
//! the NON-ZERO gauge-origin H2O/STO-3G fixture.
//!
//! Byte-identity vs vendored libcint 6.1.3 at atol=1e-12 (D-10), NON-SQUARE block
//! (D-07). Each `_origj` family is its OWN vendor symbol / RawApiId (D-02).
//!
//! RED STATE (Nyquist target): references `RawApiId::INT1E_{RR,R2,Z,ZZ}_*` and their
//! `_ORIGJ_*` consts that land in plan 24-02's Task 0. Until then this CRATE does not
//! compile — the intended automated RED→GREEN target, NOT a plan-01 failure.

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[path = "moment_common.rs"]
mod moment_common;

macro_rules! moment_parity_test {
    ($name:ident, $rank:expr, $sph:ident, $cart:ident, $vsph:ident, $vcart:ident, $label:literal) => {
        #[cfg(has_vendor_libcint)]
        #[cfg(feature = "cpu")]
        #[test]
        fn $name() {
            use cintx_compat::raw::RawApiId;
            use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
            use cintx_oracle::vendor_ffi;

            let (atm, bas, env) = build_h2o_sto3g_common_orig();
            moment_common::vendor_parity(
                $rank,
                RawApiId::$sph,
                RawApiId::$cart,
                vendor_ffi::$vsph,
                vendor_ffi::$vcart,
                &atm,
                &bas,
                &env,
                $label,
            );
        }
    };
}

/// `_origj` families: same as `moment_parity_test!` but on the CROSS-center
/// non-square block (H1-1s × O-2p). `_origj` measures position relative to the ket
/// center, so a same-center block gives identically-zero even-moment integrals
/// (vendor included) — the cross-center ket makes the comparison substantive while
/// keeping the block non-square (D-07 transpose gate preserved).
macro_rules! moment_origj_parity_test {
    ($name:ident, $rank:expr, $sph:ident, $cart:ident, $vsph:ident, $vcart:ident, $label:literal) => {
        #[cfg(has_vendor_libcint)]
        #[cfg(feature = "cpu")]
        #[test]
        fn $name() {
            use cintx_compat::raw::RawApiId;
            use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
            use cintx_oracle::vendor_ffi;

            let (atm, bas, env) = build_h2o_sto3g_common_orig();
            moment_common::vendor_parity_at(
                $rank,
                moment_common::cross_center_non_square_shell_pair(),
                RawApiId::$sph,
                RawApiId::$cart,
                vendor_ffi::$vsph,
                vendor_ffi::$vcart,
                &atm,
                &bas,
                &env,
                $label,
            );
        }
    };
}

moment_parity_test!(
    test_int1e_rr_parity,
    9,
    INT1E_RR_SPH,
    INT1E_RR_CART,
    vendor_int1e_rr_sph,
    vendor_int1e_rr_cart,
    "int1e_rr"
);
moment_parity_test!(
    test_int1e_r2_parity,
    1,
    INT1E_R2_SPH,
    INT1E_R2_CART,
    vendor_int1e_r2_sph,
    vendor_int1e_r2_cart,
    "int1e_r2"
);
moment_parity_test!(
    test_int1e_z_parity,
    1,
    INT1E_Z_SPH,
    INT1E_Z_CART,
    vendor_int1e_z_sph,
    vendor_int1e_z_cart,
    "int1e_z"
);
moment_parity_test!(
    test_int1e_zz_parity,
    1,
    INT1E_ZZ_SPH,
    INT1E_ZZ_CART,
    vendor_int1e_zz_sph,
    vendor_int1e_zz_cart,
    "int1e_zz"
);

moment_origj_parity_test!(
    test_int1e_rr_origj_parity,
    9,
    INT1E_RR_ORIGJ_SPH,
    INT1E_RR_ORIGJ_CART,
    vendor_int1e_rr_origj_sph,
    vendor_int1e_rr_origj_cart,
    "int1e_rr_origj"
);
moment_origj_parity_test!(
    test_int1e_r2_origj_parity,
    1,
    INT1E_R2_ORIGJ_SPH,
    INT1E_R2_ORIGJ_CART,
    vendor_int1e_r2_origj_sph,
    vendor_int1e_r2_origj_cart,
    "int1e_r2_origj"
);
moment_origj_parity_test!(
    test_int1e_z_origj_parity,
    1,
    INT1E_Z_ORIGJ_SPH,
    INT1E_Z_ORIGJ_CART,
    vendor_int1e_z_origj_sph,
    vendor_int1e_z_origj_cart,
    "int1e_z_origj"
);
moment_origj_parity_test!(
    test_int1e_zz_origj_parity,
    1,
    INT1E_ZZ_ORIGJ_SPH,
    INT1E_ZZ_ORIGJ_CART,
    vendor_int1e_zz_origj_sph,
    vendor_int1e_zz_origj_cart,
    "int1e_zz_origj"
);
