//! Safe wrappers around vendored libcint 6.1.3 FFI for oracle comparison.
//!
//! Only available when built with CINTX_ORACLE_BUILD_VENDOR=1.
//! All functions use the same atm/bas/env layout as cintx_compat::raw.

#![cfg(has_vendor_libcint)]

#[allow(non_camel_case_types, non_upper_case_globals, dead_code, non_snake_case, improper_ctypes)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/oracle_bindings.rs"));
}

use std::ptr;

/// Evaluate int1e_ovlp_sph for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj elements where ni=CINTcgto_spheric(shls[0])
/// and nj=CINTcgto_spheric(shls[1]).
///
/// Returns the number of output elements (or 0 if the integral is zero by symmetry).
pub fn vendor_int1e_ovlp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ovlp_sph(
            out.as_mut_ptr(),
            ptr::null_mut(), // dims = NULL means use default
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(), // opt = NULL
            ptr::null_mut(), // cache = NULL (let libcint allocate)
        )
    }
}

/// Evaluate int1e_kin_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_kin_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_kin_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Evaluate int1e_nuc_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_nuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_nuc_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Get the number of spherical AOs for a given shell index from vendored libcint.
pub fn vendor_cgto_spheric(bas_id: i32, bas: &[i32]) -> i32 {
    unsafe { ffi::CINTcgto_spheric(bas_id, bas.as_ptr() as *mut i32) }
}

/// Evaluate int2e_sph for a single shell quartet using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj*nk*nl elements where
/// nX = CINTcgto_spheric(shls[X], bas).
///
/// `shls` is `[i, j, k, l]` — four shell indices (4-center 2-electron integral).
///
/// Returns the number of output elements (or 0 if the integral is zero by symmetry).
pub fn vendor_int2e_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_sph(
            out.as_mut_ptr(),
            ptr::null_mut(), // dims = NULL means use default
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(), // opt = NULL
            ptr::null_mut(), // cache = NULL (let libcint allocate)
        )
    }
}

/// Evaluate int2c2e_sph for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with ni*nk elements where
/// nX = CINTcgto_spheric(shls[X], bas).
///
/// `shls` is `[i, k]` — two shell indices (2-center 2-electron integral).
pub fn vendor_int2c2e_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Evaluate int3c1e_sph for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj*nk elements where
/// nX = CINTcgto_spheric(shls[X], bas).
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 1-electron integral).
pub fn vendor_int3c1e_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Evaluate int3c2e_sph for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj*nk elements where
/// nX = CINTcgto_spheric(shls[X], bas).
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral).
pub fn vendor_int3c2e_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

// ---- Helper symbol vendor FFI wrappers ----
// Integer-returning helpers (exact equality comparison per D-02).

/// Number of Cartesian basis functions for angular momentum l.
pub fn vendor_CINTlen_cart(l: i32) -> i32 {
    unsafe { ffi::CINTlen_cart(l) }
}

/// Number of spinor basis functions for shell bas_id.
pub fn vendor_CINTlen_spinor(bas_id: i32, bas: &[i32]) -> i32 {
    unsafe { ffi::CINTlen_spinor(bas_id, bas.as_ptr() as *mut i32) }
}

/// Number of contracted Cartesian GTOs for shell bas_id.
pub fn vendor_CINTcgto_cart(bas_id: i32, bas: &[i32]) -> i32 {
    unsafe { ffi::CINTcgto_cart(bas_id, bas.as_ptr() as *mut i32) }
}

/// Number of contracted spherical GTOs for shell bas_id (alias for CINTcgto_spheric).
pub fn vendor_CINTcgto_spheric(bas_id: i32, bas: &[i32]) -> i32 {
    unsafe { ffi::CINTcgto_spheric(bas_id, bas.as_ptr() as *mut i32) }
}

/// Number of contracted spinor GTOs for shell bas_id.
pub fn vendor_CINTcgto_spinor(bas_id: i32, bas: &[i32]) -> i32 {
    unsafe { ffi::CINTcgto_spinor(bas_id, bas.as_ptr() as *mut i32) }
}

/// Total number of spherical primitive GTOs across all nbas shells.
pub fn vendor_CINTtot_pgto_spheric(bas: &[i32], nbas: i32) -> i32 {
    unsafe { ffi::CINTtot_pgto_spheric(bas.as_ptr() as *mut i32, nbas) }
}

/// Total number of spinor primitive GTOs across all nbas shells.
pub fn vendor_CINTtot_pgto_spinor(bas: &[i32], nbas: i32) -> i32 {
    unsafe { ffi::CINTtot_pgto_spinor(bas.as_ptr() as *mut i32, nbas) }
}

/// Total number of Cartesian contracted GTOs across all nbas shells.
pub fn vendor_CINTtot_cgto_cart(bas: &[i32], nbas: i32) -> i32 {
    unsafe { ffi::CINTtot_cgto_cart(bas.as_ptr() as *mut i32, nbas) }
}

/// Total number of spherical contracted GTOs across all nbas shells.
pub fn vendor_CINTtot_cgto_spheric(bas: &[i32], nbas: i32) -> i32 {
    unsafe { ffi::CINTtot_cgto_spheric(bas.as_ptr() as *mut i32, nbas) }
}

/// Total number of spinor contracted GTOs across all nbas shells.
pub fn vendor_CINTtot_cgto_spinor(bas: &[i32], nbas: i32) -> i32 {
    unsafe { ffi::CINTtot_cgto_spinor(bas.as_ptr() as *mut i32, nbas) }
}

/// Write Cartesian AO offsets into ao_loc[0..=nbas] (nbas+1 elements required).
pub fn vendor_CINTshells_cart_offset(ao_loc: &mut [i32], bas: &[i32], nbas: i32) {
    unsafe {
        ffi::CINTshells_cart_offset(ao_loc.as_mut_ptr(), bas.as_ptr() as *mut i32, nbas);
    }
}

/// Write spherical AO offsets into ao_loc[0..=nbas] (nbas+1 elements required).
pub fn vendor_CINTshells_spheric_offset(ao_loc: &mut [i32], bas: &[i32], nbas: i32) {
    unsafe {
        ffi::CINTshells_spheric_offset(ao_loc.as_mut_ptr(), bas.as_ptr() as *mut i32, nbas);
    }
}

/// Write spinor AO offsets into ao_loc[0..=nbas] (nbas+1 elements required).
pub fn vendor_CINTshells_spinor_offset(ao_loc: &mut [i32], bas: &[i32], nbas: i32) {
    unsafe {
        ffi::CINTshells_spinor_offset(ao_loc.as_mut_ptr(), bas.as_ptr() as *mut i32, nbas);
    }
}

/// GTO normalization constant for angular momentum n and exponent a.
/// Float-returning — compare at atol=1e-12 per D-02.
pub fn vendor_CINTgto_norm(n: i32, a: f64) -> f64 {
    unsafe { ffi::CINTgto_norm(n, a) }
}

// ---- Transform symbol vendor FFI wrapper ----
// Direct buffer comparison for at least one transform symbol per HELP-02.

/// Cart-to-spherical transform for bra index.
///
/// Writes the spherical representation into `sph`. The returned pointer points
/// to `sph` (C convention). We ignore it and use the output slice directly.
pub fn vendor_CINTc2s_bra_sph(sph: &mut [f64], nket: i32, cart: &[f64], l: i32) {
    unsafe {
        ffi::CINTc2s_bra_sph(sph.as_mut_ptr(), nket, cart.as_ptr() as *mut f64, l);
    }
}
