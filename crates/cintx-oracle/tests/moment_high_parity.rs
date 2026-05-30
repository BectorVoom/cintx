//! Phase 24 MOM-03 vendor parity scaffold: `int1e_rrr` (rank 27), `int1e_rrrr`
//! (rank 81), `int1e_r4` (rank 1) + `int1e_r4_origj` (rank 1), cart + sph, on the
//! NON-ZERO gauge-origin H2O/STO-3G fixture.
//!
//! Byte-identity vs vendored libcint 6.1.3 at atol=1e-12 (D-10), NON-SQUARE block
//! (D-07). The rank-81 `rrrr` family exercises the existing staging at maximum rank
//! (D-03; FND-06 OOM hardening stays Phase 25).
//!
//! NOTE: there is intentionally NO `int1e_rrr_origj` / `int1e_rrrr_origj` test —
//! those symbols DO NOT EXIST in libcint 6.1.3 (confirmed by grep of
//! src/autocode/intor1.c). Only `r4_origj` has a vendor target in this file.
//!
//! RED STATE (Nyquist target): references `RawApiId::INT1E_{RRR,RRRR,R4}_*` and
//! `INT1E_R4_ORIGJ_*` consts that land in plan 24-02's Task 0. Until then this CRATE
//! does not compile — the intended automated RED→GREEN target, NOT a plan-01 failure.

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

/// `_origj` families on the CROSS-center non-square block (see moment_common docs).
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
    test_int1e_rrr_parity, 27,
    INT1E_RRR_SPH, INT1E_RRR_CART,
    vendor_int1e_rrr_sph, vendor_int1e_rrr_cart, "int1e_rrr"
);
moment_parity_test!(
    test_int1e_rrrr_parity, 81,
    INT1E_RRRR_SPH, INT1E_RRRR_CART,
    vendor_int1e_rrrr_sph, vendor_int1e_rrrr_cart, "int1e_rrrr"
);
moment_parity_test!(
    test_int1e_r4_parity, 1,
    INT1E_R4_SPH, INT1E_R4_CART,
    vendor_int1e_r4_sph, vendor_int1e_r4_cart, "int1e_r4"
);
moment_origj_parity_test!(
    test_int1e_r4_origj_parity, 1,
    INT1E_R4_ORIGJ_SPH, INT1E_R4_ORIGJ_CART,
    vendor_int1e_r4_origj_sph, vendor_int1e_r4_origj_cart, "int1e_r4_origj"
);
