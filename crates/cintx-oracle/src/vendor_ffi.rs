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
