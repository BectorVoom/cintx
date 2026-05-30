//! Phase 24 MOM-04 vendor parity scaffold: `int1e_p4` (rank 1), `int1e_irp` (rank 9),
//! `int1e_rinv` (rank 1), `int1e_drinv` (rank 3), cart + sph.
//!
//! Byte-identity vs vendored libcint 6.1.3 at atol=1e-12 (D-10), NON-SQUARE block
//! (D-07).
//!
//! Fixtures / origins:
//!   - `p4` (∇⁴ overlap-derivative engine, no Rys) and `irp` (i·r×∇, reads
//!     PTR_COMMON_ORIG) use the NON-ZERO gauge-origin fixture `build_h2o_sto3g_common_orig`.
//!   - `rinv` / `drinv` are single-center 1/r potentials reading `env[PTR_RINV_ORIG]`
//!     (Phase 24 D-04 / OQ-1 correction — NOT PTR_COMMON_ORIG). The common-orig fixture
//!     leaves `rinv_orig=[0,0,0]`, which is trivially-passing and DISALLOWED. We inject
//!     a NON-ZERO rinv center [0.5,-0.3,0.8] via `env_with_rinv_origin` before parity.
//!
//! RED STATE (Nyquist target): references `RawApiId::INT1E_{P4,IRP,RINV,DRINV}_*`
//! consts that land in plans 24-04/24-05. Until then this CRATE does not compile —
//! the intended automated RED→GREEN target, NOT a plan-01 failure.

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[path = "moment_common.rs"]
mod moment_common;

// ── p4 / irp: common-orig fixture (gauge origin, reads PTR_COMMON_ORIG) ──────────

macro_rules! moment_common_orig_test {
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

moment_common_orig_test!(
    test_int1e_p4_parity, 1,
    INT1E_P4_SPH, INT1E_P4_CART,
    vendor_int1e_p4_sph, vendor_int1e_p4_cart, "int1e_p4"
);
moment_common_orig_test!(
    test_int1e_irp_parity, 9,
    INT1E_IRP_SPH, INT1E_IRP_CART,
    vendor_int1e_irp_sph, vendor_int1e_irp_cart, "int1e_irp"
);

// ── rinv / drinv: NON-ZERO rinv center (reads PTR_RINV_ORIG, D-04/OQ-1) ──────────

macro_rules! moment_rinv_test {
    ($name:ident, $rank:expr, $sph:ident, $cart:ident, $vsph:ident, $vcart:ident, $label:literal) => {
        #[cfg(has_vendor_libcint)]
        #[cfg(feature = "cpu")]
        #[test]
        fn $name() {
            use cintx_compat::raw::RawApiId;
            use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
            use cintx_oracle::vendor_ffi;

            let (atm, bas, env0) = build_h2o_sto3g_common_orig();
            // D-04 / OQ-1: a ZERO rinv origin is trivially-passing and disallowed.
            // Inject a NON-ZERO single-center 1/r origin.
            let env = moment_common::env_with_rinv_origin(&env0, [0.5, -0.3, 0.8]);
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

moment_rinv_test!(
    test_int1e_rinv_parity, 1,
    INT1E_RINV_SPH, INT1E_RINV_CART,
    vendor_int1e_rinv_sph, vendor_int1e_rinv_cart, "int1e_rinv"
);
moment_rinv_test!(
    test_int1e_drinv_parity, 3,
    INT1E_DRINV_SPH, INT1E_DRINV_CART,
    vendor_int1e_drinv_sph, vendor_int1e_drinv_cart, "int1e_drinv"
);
