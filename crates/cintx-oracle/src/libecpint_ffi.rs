//! Optional libecpint secondary-oracle FFI shim (Phase 19 D-02 REVISED).
//!
//! libecpint (Shaw & Hill, *J. Chem. Phys.* **147**, 074108, 2017, MIT;
//! <https://github.com/robashaw/libecpint>) is a NON-BLOCKING cross-check oracle.
//! This module is gated `#![cfg(has_libecpint_oracle)]`, and that cfg is emitted by
//! `crates/cintx-oracle/build.rs` ONLY when `CINTX_LIBECPINT_ORACLE=1` AND libecpint
//! plus an operator-supplied `extern "C"` shim are reachable on the host (mirroring the
//! Phase 16 ROCm opt-in `CINTX_ROCM_ORACLE=1`). When the env var is unset — the default
//! build — this module does not exist and the crate compiles exactly as before
//! (no regression; threat T-19-27 mitigated).
//!
//! libecpint is C++17 and exposes no stable C API of its own, so the FFI crosses an
//! `extern "C"` shim layer (a small `.cpp` the operator supplies via
//! `CINTX_LIBECPINT_SHIM`, compiled in `build.rs`). The Rust side declares the C
//! signatures below and provides safe-ish wrappers that take slices and call the
//! `unsafe extern "C"` symbols — mirroring `vendor_ffi.rs`'s wrapper shape.
//!
//! The cross-check is **informational only** (atol≈1e-9): libecpint and PySCF `nr_ecp`
//! use different internal recurrence + quadrature conventions, so byte-identity is NOT
//! expected. Failures NEVER block CI. See
//! `.planning/notes/ecp-libecpint-crosscheck.md` for provenance, the tolerance
//! rationale, and the operator activation steps.
//!
//! # FFI contract expected of the operator shim
//!
//! The shim `.cpp` (compiled into `cintx_libecpint_shim`) must export the two C symbols
//! declared below. Each evaluates the ECP integral matrix for ONE AO shell pair on the
//! ECP center, in the same atm/bas/env layout cintx-compat uses, and writes `ni*nj`
//! f64s (Cartesian or spherical per the symbol) into `out` in libecpint's native
//! ordering. The Rust collector (`ecp_libecpint_crosscheck_parity.rs`) applies any
//! convention/normalization adapter needed to map that ordering onto cintx's
//! row-major `[ao_i, ao_j]` layout. Return value mirrors libcint's status convention
//! (non-zero = wrote a non-trivial block, 0 = zero by symmetry); the cross-check does
//! not gate on the status, only on the matrix values within the informational envelope.

#![cfg(has_libecpint_oracle)]

use std::os::raw::c_int;
use std::ptr;

#[allow(non_camel_case_types)]
mod ffi {
    use std::os::raw::c_int;

    unsafe extern "C" {
        /// Operator-supplied extern "C" shim around libecpint's Cartesian ECP
        /// integral entry point for a single AO shell pair on the ECP center.
        ///
        /// `out` must hold `ni*nj` f64s where `nX = (lX+1)(lX+2)/2`. The shim reads
        /// the ECP basis from `env[AS_ECPBAS_OFFSET]`/`env[AS_NECPBAS]` exactly as
        /// PySCF's `ECPscalar_cart` does, so the cintx collector can reuse the same
        /// combined-bas + env wiring as the PySCF vendor collector.
        pub fn cintx_libecpint_ecp_cart(
            out: *mut f64,
            shls: *const c_int,
            atm: *const c_int,
            natm: c_int,
            bas: *const c_int,
            nbas: c_int,
            env: *const f64,
        ) -> c_int;

        /// Spherical analog of [`cintx_libecpint_ecp_cart`].
        ///
        /// `out` must hold `ni*nj` f64s where `nX = 2*lX+1`.
        pub fn cintx_libecpint_ecp_sph(
            out: *mut f64,
            shls: *const c_int,
            atm: *const c_int,
            natm: c_int,
            bas: *const c_int,
            nbas: c_int,
            env: *const f64,
        ) -> c_int;
    }
}

/// Evaluate the Cartesian ECP integral block for one AO shell pair via libecpint.
///
/// `out` must be pre-allocated with `ni*nj` elements (Cartesian component counts).
/// `shls` is `[i, j]` (the two AO shell indices on the ECP center). `atm`/`bas`/`env`
/// follow the cintx-compat layout; `env[AS_ECPBAS_OFFSET]`/`env[AS_NECPBAS]` must point
/// at the ECP basis slab (same wiring as the PySCF `nr_ecp` vendor collector).
///
/// Informational cross-check only (atol≈1e-9); never blocks CI (D-02).
pub fn libecpint_ecp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    let _ = ptr::null::<()>(); // keep std::ptr import meaningful across cfg variants
    unsafe {
        ffi::cintx_libecpint_ecp_cart(
            out.as_mut_ptr(),
            shls.as_ptr() as *const c_int,
            atm.as_ptr() as *const c_int,
            natm,
            bas.as_ptr() as *const c_int,
            nbas,
            env.as_ptr(),
        )
    }
}

/// Spherical analog of [`libecpint_ecp_cart`].
///
/// `out` must be pre-allocated with `ni*nj` elements (spherical component counts,
/// `2*l+1` per shell). Informational cross-check only (atol≈1e-9); never blocks CI.
pub fn libecpint_ecp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::cintx_libecpint_ecp_sph(
            out.as_mut_ptr(),
            shls.as_ptr() as *const c_int,
            atm.as_ptr() as *const c_int,
            natm,
            bas.as_ptr() as *const c_int,
            nbas,
            env.as_ptr(),
        )
    }
}
