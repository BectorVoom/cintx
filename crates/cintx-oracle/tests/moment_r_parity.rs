//! Phase 24 MOM-01 vendor parity scaffold: `int1e_r` (rank 3) and `int1e_r_origj`
//! (rank 3), cart + sph, on the NON-ZERO gauge-origin H2O/STO-3G fixture.
//!
//! Validates byte-identity vs vendored libcint 6.1.3 at atol=1e-12 (D-10) using a
//! NON-SQUARE bra×ket block (D-07). The `_origj` variant reads the ket shell-j
//! center instead of `env[PTR_COMMON_ORIG]` (D-02), but is its OWN vendor symbol.
//!
//! RED STATE (Nyquist target): these tests reference `RawApiId::INT1E_R_*` /
//! `INT1E_R_ORIGJ_*` consts that land in plan 24-02's Task 0. Until then this CRATE
//! does not compile — that is the intended automated RED→GREEN target, NOT a failure
//! of plan 24-01. Plan 24-02's first task verifies `cargo test --no-run`. Under the
//! vendor gate, once the consts + kernel land, these go GREEN; without the gate they
//! skip cleanly (the `has_vendor_libcint` cfg gates the whole `#[test]`).

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[path = "moment_common.rs"]
mod moment_common;

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_r_h2o_sto3g_common_orig_parity() {
    use cintx_compat::raw::RawApiId;
    use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g_common_orig();
    moment_common::vendor_parity(
        3,
        RawApiId::INT1E_R_SPH,
        RawApiId::INT1E_R_CART,
        vendor_ffi::vendor_int1e_r_sph,
        vendor_ffi::vendor_int1e_r_cart,
        &atm,
        &bas,
        &env,
        "int1e_r",
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_r_origj_h2o_sto3g_common_orig_parity() {
    use cintx_compat::raw::RawApiId;
    use cintx_oracle::fixtures::build_h2o_sto3g_common_orig;
    use cintx_oracle::vendor_ffi;

    let (atm, bas, env) = build_h2o_sto3g_common_orig();
    moment_common::vendor_parity(
        3,
        RawApiId::INT1E_R_ORIGJ_SPH,
        RawApiId::INT1E_R_ORIGJ_CART,
        vendor_ffi::vendor_int1e_r_origj_sph,
        vendor_ffi::vendor_int1e_r_origj_cart,
        &atm,
        &bas,
        &env,
        "int1e_r_origj",
    );
}
