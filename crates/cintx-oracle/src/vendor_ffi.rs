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

/// Evaluate int3c2e_cart for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, Cartesian).
pub fn vendor_int3c2e_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_cart(
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

/// Evaluate int4c1e_sph for a single shell quartet using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj elements where
/// nX = CINTcgto_spheric(shls[X], bas). The 4c1e integral traces over the
/// k=l auxiliary center, so the output dimension is ni*nj not ni*nj*nk*nl.
///
/// `shls` is `[i, j, k, l]` — four shell indices (4-center 1-electron integral).
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int4c1e_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int4c1e_sph(
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

/// Evaluate int4c1e_cart for a single shell quartet using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj elements where
/// nX = CINTcgto_cart(shls[X], bas).
///
/// `shls` is `[i, j, k, l]` — four shell indices (4-center 1-electron integral).
pub fn vendor_int4c1e_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int4c1e_cart(
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

// ---- Cart integral vendor FFI wrappers ----
// Follow the same pattern as sph wrappers above.

/// Evaluate int1e_ovlp_cart for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj elements where ni=CINTcgto_cart(shls[0])
/// and nj=CINTcgto_cart(shls[1]).
pub fn vendor_int1e_ovlp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ovlp_cart(
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

/// Evaluate int1e_kin_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_kin_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_kin_cart(
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

/// Evaluate int1e_nuc_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_nuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_nuc_cart(
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

/// Evaluate int1e_ipovlp_sph for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_ipovlp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlp_sph(
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

/// Evaluate int1e_ipovlp_cart for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_ipovlp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlp_cart(
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

/// Evaluate int1e_ipkin_sph for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_ipkin_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipkin_sph(
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

/// Evaluate int1e_ipkin_cart for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_ipkin_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipkin_cart(
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

/// Evaluate int1e_ipnuc_sph for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
///
/// int1e_ipnuc is the hcore nuclear-attraction derivative: `∂/∂Ai` on the bra
/// center, summed over ALL nuclei with the `-Z_C` charge factor. The vendor reads
/// the nuclear charges/coords from atm/env; no special env slot is required.
pub fn vendor_int1e_ipnuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipnuc_sph(
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

/// Evaluate int1e_ipnuc_cart for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_ipnuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipnuc_cart(
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

/// Evaluate int1e_iprinv_sph for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
///
/// int1e_iprinv is the per-atom Hellmann–Feynman force term: `∂/∂Ai` evaluated at
/// a SINGLE rinv origin with factor `+1.0` (no `-Z_C`).
///
/// IMPORTANT: the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]`
/// (libcint `PTR_RINV_ORIG = 4`, i.e. `env[4], env[5], env[6]` = x, y, z) to the
/// chosen origin BEFORE calling this function. The same env must be used for the
/// matching cintx `eval_raw` call so both evaluate at the identical origin.
pub fn vendor_int1e_iprinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_iprinv_sph(
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

/// Evaluate int1e_iprinv_cart for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (3 gradient components).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
///
/// IMPORTANT: the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]`
/// (libcint `PTR_RINV_ORIG = 4`) to the chosen origin BEFORE calling this function
/// (see `vendor_int1e_iprinv_sph`).
pub fn vendor_int1e_iprinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_iprinv_cart(
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

/// Evaluate int2e_cart for a single shell quartet using vendored libcint.
///
/// `out` must be pre-allocated with ni*nj*nk*nl elements where
/// nX = CINTcgto_cart(shls[X], bas).
///
/// `shls` is `[i, j, k, l]` — four shell indices (4-center 2-electron integral).
pub fn vendor_int2e_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_cart(
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

/// Evaluate int2e_ip1_sph for a single shell quartet using vendored libcint.
///
/// The two-electron force `∇_A <ij|kl>` (3 gradient components on electron 1).
///
/// `out` must be pre-allocated with `3 * ni*nj*nk*nl` elements where
/// nX = CINTcgto_spheric(shls[X], bas).
///
/// `shls` is `[i, j, k, l]` — four shell indices.
///
/// LAYOUT: libcint writes **component-leading** F-order — `out[comp * (ni*nj*nk*nl) + n]`
/// for comp in 0..3, where the per-component block `n` walks the AO product i-fastest
/// (i.e. `[nl][nk][nj][ni]` with `ni` fastest, matching pyscf-gto `layout_table.rs`).
/// The cintx `int2e_ip1` kernel emits this identical layout, so the byte-identity
/// element-for-element comparison in `two_electron_ip1_parity.rs` IS the F-order /
/// component-leading layout validation (Risk R3).
pub fn vendor_int2e_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ip1_sph(
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

/// Evaluate int2e_ip1_cart for a single shell quartet using vendored libcint.
///
/// Cartesian analog of [`vendor_int2e_ip1_sph`]. `out` must be pre-allocated with
/// `3 * ni*nj*nk*nl` elements where nX = CINTcgto_cart(shls[X], bas).
///
/// LAYOUT: component-leading F-order — `out[comp * (ni*nj*nk*nl) + n]` (same
/// convention as the sph wrapper; see its doc for the R3 layout note).
pub fn vendor_int2e_ip1_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ip1_cart(
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

/// Evaluate int2c2e_cart for a single shell pair using vendored libcint.
///
/// `shls` is `[i, k]` — two shell indices (2-center 2-electron integral).
pub fn vendor_int2c2e_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_cart(
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

/// Evaluate int3c1e_cart for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 1-electron integral).
pub fn vendor_int3c1e_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_cart(
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

/// Evaluate int3c1e_p2_cart for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 1-electron integral, p2 variant).
pub fn vendor_int3c1e_p2_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_p2_cart(
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

/// Evaluate int3c1e_p2_sph for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 1-electron integral, p2 variant, spherical).
pub fn vendor_int3c1e_p2_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_p2_sph(
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

/// Evaluate int3c2e_ip1_cart for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj * nk elements (3 gradient components).
/// Layout: component-leading — out[comp * ni*nj*nk + n] for comp in 0..3.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, ip1 variant).
pub fn vendor_int3c2e_ip1_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ip1_cart(
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

/// Evaluate int3c2e_ip1_sph for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj * nk elements (3 gradient components)
/// where nX = CINTcgto_spheric(shls[X], bas).
/// Layout: component-leading — out[comp * ni*nj*nk + n] for comp in 0..3.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, ip1
/// variant, spherical). This is the REAL `∇_A` first-center derivative reference for
/// the int3c2e_ip1 oracle gate (GRAD-08 / Risk R1) — NOT the plain int3c2e_sph.
pub fn vendor_int3c2e_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ip1_sph(
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
/// libcint's `CINTc2s_bra_sph` does NOT always write into the `sph` argument:
/// for l<2 (s/p, non-PYPZPX) it returns `gcart` WITHOUT touching `gsph`, and the
/// RETURNED `*mut f64` is the authoritative result. For l>=2 it writes `gsph`
/// (ket-blocked) and returns that same pointer. We therefore copy the returned
/// pointer into `sph` so callers always read the correct result.
///
/// The `ret != sph.as_mut_ptr()` guard skips the redundant self-copy for l>=2.
/// For l<2 the returned pointer aliases the `cart` input (which lives across the
/// call), so `std::ptr::copy` (memmove-safe) is sound even on overlap. `n` is
/// clamped to `sph.len()` to prevent any out-of-bounds write.
pub fn vendor_CINTc2s_bra_sph(sph: &mut [f64], nket: i32, cart: &[f64], l: i32) {
    unsafe {
        let ret = ffi::CINTc2s_bra_sph(sph.as_mut_ptr(), nket, cart.as_ptr() as *mut f64, l);
        // nket * nsph(l) = nket * (2l+1)
        let n = (nket.max(0) as usize) * ((2 * l.max(0) + 1) as usize);
        let n = n.min(sph.len());
        if !ret.is_null() && ret != sph.as_mut_ptr() {
            std::ptr::copy(ret, sph.as_mut_ptr(), n);
        }
    }
}

// ---- F12/STG/YP integral vendor FFI wrappers (with-f12 feature) ----
// All 10 F12/STG/YP operators are sph-only (no cart, no spinor representations).
// They require env[PTR_F12_ZETA=9] to be set to a positive zeta value.

/// Evaluate int2e_stg_sph for a single shell quartet using vendored libcint.
///
/// Requires `env[9]` (PTR_F12_ZETA) set to a positive zeta value.
/// `out` must be pre-allocated with ni*nj*nk*nl elements.
pub fn vendor_int2e_stg_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_stg_sph(
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

/// Evaluate int2e_stg_ip1_sph for a single shell quartet using vendored libcint.
///
/// ip1 variant: 3 components × ni*nj*nk*nl elements.
pub fn vendor_int2e_stg_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_stg_ip1_sph(
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

/// Evaluate int2e_stg_ipip1_sph for a single shell quartet using vendored libcint.
///
/// ipip1 variant: 9 components × ni*nj*nk*nl elements.
pub fn vendor_int2e_stg_ipip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_stg_ipip1_sph(
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

/// Evaluate int2e_stg_ipvip1_sph for a single shell quartet using vendored libcint.
///
/// ipvip1 variant: 9 components × ni*nj*nk*nl elements.
pub fn vendor_int2e_stg_ipvip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_stg_ipvip1_sph(
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

/// Evaluate int2e_stg_ip1ip2_sph for a single shell quartet using vendored libcint.
///
/// ip1ip2 variant: 9 components × ni*nj*nk*nl elements.
pub fn vendor_int2e_stg_ip1ip2_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_stg_ip1ip2_sph(
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

/// Evaluate int2e_yp_sph for a single shell quartet using vendored libcint.
///
/// Yukawa-potential variant. Requires env[9] (PTR_F12_ZETA) set to a positive value.
pub fn vendor_int2e_yp_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_yp_sph(
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

/// Evaluate int2e_yp_ip1_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_yp_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_yp_ip1_sph(
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

/// Evaluate int2e_yp_ipip1_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_yp_ipip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_yp_ipip1_sph(
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

/// Evaluate int2e_yp_ipvip1_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_yp_ipvip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_yp_ipvip1_sph(
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

/// Evaluate int2e_yp_ip1ip2_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_yp_ip1ip2_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_yp_ip1ip2_sph(
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

// ---- 1e spinor integral vendor FFI wrappers ----
// Output buffer layout: ni_spinor * nj_spinor complex elements = ni_sp * nj_sp * 2 f64 values
// (interleaved re/im pairs), where ni_sp = CINTcgto_spinor(shls[0]) and
// nj_sp = CINTcgto_spinor(shls[1]).

/// Evaluate int1e_ovlp_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * 2` f64 elements where
/// ni_sp = CINTcgto_spinor(shls[0]) and nj_sp = CINTcgto_spinor(shls[1]).
/// The layout is interleaved real/imaginary pairs for each complex element.
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int1e_ovlp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ovlp_spinor(
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

/// Evaluate int1e_kin_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_kin_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_kin_spinor(
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

/// Evaluate int1e_nuc_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_nuc_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_nuc_spinor(
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

// ---- 1e spinor GRADIENT integral vendor FFI wrappers ----
// These are 3-component gradient operators. Output buffer layout:
//   `3 * ni_sp * nj_sp * 2` f64 values — 3 Cartesian gradient components,
//   each an interleaved real/imaginary spinor block, component-leading
//   (out[comp * ni_sp * nj_sp * 2 + ...]). ni_sp = CINTcgto_spinor(shls[0]),
//   nj_sp = CINTcgto_spinor(shls[1]).

/// Evaluate int1e_ipovlp_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `3 * ni_sp * nj_sp * 2` f64 elements
/// (3 gradient components × interleaved-complex spinor block).
pub fn vendor_int1e_ipovlp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlp_spinor(
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

/// Evaluate int1e_ipkin_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `3 * ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_ipkin_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipkin_spinor(
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

/// Evaluate int1e_ipnuc_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `3 * ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_ipnuc_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipnuc_spinor(
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

/// Evaluate int1e_iprinv_spinor for a single shell pair using vendored libcint.
///
/// The rinv origin must be set in `env[PTR_RINV_ORIG..+3]` by the caller.
/// `out` must be pre-allocated with `3 * ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_iprinv_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_iprinv_spinor(
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

// ---- Multi-center spinor integral vendor FFI wrappers ----
// Output buffer layout: product of spinor component counts × 2 f64 values
// (interleaved re/im pairs), where each nX_sp = CINTcgto_spinor(shls[X]).

/// Evaluate int2e_spinor for a single shell quartet using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * nk_sp * nl_sp * 2` f64 elements
/// where nX_sp = CINTcgto_spinor(shls[X]).
/// The layout is interleaved real/imaginary pairs for each complex element.
///
/// `shls` is `[i, j, k, l]` — four shell indices (4-center 2-electron spinor integral).
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int2e_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spinor(
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

/// Evaluate int2c2e_spinor for a single shell pair using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * 2` f64 elements
/// where ni_sp = CINTcgto_spinor(shls[0]) and nj_sp = CINTcgto_spinor(shls[1]).
/// The layout is interleaved real/imaginary pairs for each complex element.
///
/// `shls` is `[i, k]` — two shell indices (2-center 2-electron spinor integral).
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int2c2e_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_spinor(
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

/// Evaluate int3c1e_spinor for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * nk_sp * 2` f64 elements
/// where nX_sp = CINTcgto_spinor(shls[X]).
/// The layout is interleaved real/imaginary pairs for each complex element.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 1-electron spinor integral).
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int3c1e_spinor(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_spinor(
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

/// Evaluate int3c2e_spinor for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with `ni_sp * nj_sp * nk_sp * 2` f64 elements
/// where nX_sp = CINTcgto_spinor(shls[X]).
/// The layout is interleaved real/imaginary pairs for each complex element.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron spinor integral).
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int3c2e_spinor(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_spinor(
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

// -------------------------------------------------------------------------
// Phase 14 unstable-source family vendor FFI wrappers
// -------------------------------------------------------------------------

/// Evaluate int1e_r2_origi_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r2_origi_sph(out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_r2_origi_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_r4_origi_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r4_origi_sph(out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_r4_origi_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_r2_origi_ip2_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r2_origi_ip2_sph(out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_r2_origi_ip2_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_r4_origi_ip2_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r4_origi_ip2_sph(out: &mut [f64], shls: &[i32; 2], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_r4_origi_ip2_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_grids_sph for a shell pair + grid range using vendored libcint.
/// `shls` is `[i, j, grid_start, grid_end]` where grid indices come from env.
pub fn vendor_int1e_grids_sph(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_grids_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_grids_ip_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_ip_sph(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_grids_ip_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_grids_ipvip_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_ipvip_sph(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_grids_ipvip_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_grids_spvsp_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_spvsp_sph(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_grids_spvsp_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int1e_grids_ipip_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_ipip_sph(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int1e_grids_ipip_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int2e_breit_r1p2_spinor (spinor-only Breit 2e) using vendored libcint.
pub fn vendor_int2e_breit_r1p2_spinor(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int2e_breit_r1p2_spinor(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int2e_breit_r2p2_spinor (spinor-only Breit 2e) using vendored libcint.
pub fn vendor_int2e_breit_r2p2_spinor(out: &mut [f64], shls: &[i32; 4], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int2e_breit_r2p2_spinor(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c1e_r2_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_r2_origk_sph(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c1e_r2_origk_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c1e_r4_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_r4_origk_sph(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c1e_r4_origk_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c1e_r6_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_r6_origk_sph(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c1e_r6_origk_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c1e_ip1_r2_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_ip1_r2_origk_sph(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c1e_ip1_r2_origk_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c1e_ip1_r4_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_ip1_r4_origk_sph(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c1e_ip1_r4_origk_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c1e_ip1_r6_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_ip1_r6_origk_sph(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c1e_ip1_r6_origk_sph(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

/// Evaluate int3c2e_sph_ssc (spin-spin contact 3c2e) for a shell triple using vendored libcint.
pub fn vendor_int3c2e_sph_ssc(out: &mut [f64], shls: &[i32; 3], atm: &[i32], natm: i32, bas: &[i32], nbas: i32, env: &[f64]) -> i32 {
    unsafe { ffi::int3c2e_sph_ssc(out.as_mut_ptr(), ptr::null_mut(), shls.as_ptr() as *mut i32, atm.as_ptr() as *mut i32, natm, bas.as_ptr() as *mut i32, nbas, env.as_ptr() as *mut f64, ptr::null_mut(), ptr::null_mut()) }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 19 PySCF nr_ecp scalar wrappers (D-01 revised: PySCF's nr_ecp is the
// primary ECP byte-identity reference since libcint 6.1.3 upstream ships no
// ECP code).
//
// Source: vendor/pyscf-nr-ecp/src/nr_ecp.c:6179-6266 (ECPscalar_sph + ECPscalar_cart).
// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:366-453 (ECPscalar_ipnuc_*).
//
// IMPORTANT — ECP slab packing convention: PySCF's ECPscalar_* functions
// extract the ECP shell rows from inside the same `bas` table via
//     ecpbas = bas + env[AS_ECPBAS_OFFSET] * BAS_SLOTS
//     necpbas = (int)env[AS_NECPBAS]
// (nr_ecp.c lines 6205-6206 for sph, 6248-6249 for cart). Callers MUST set
// env[AS_ECPBAS_OFFSET=18] to the shell-index where ECP rows START in the
// combined bas table, AND set env[AS_NECPBAS=19] to the ECP row count.
// The cintx-oracle Cu/LANL2DZ fixture (fixtures.rs::build_cu_lanl2dz)
// returns ecpbas as a separate slab — callers wrapping this into a vendor
// call must concatenate (atom_bas ++ ecp_bas) into a single bas table and
// set env[AS_ECPBAS_OFFSET] = atom_bas.len() / BAS_SLOTS before calling.
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate PySCF `ECPscalar_sph` for a single shell pair using the vendored
/// PySCF nr_ecp Type-1 + Type-2 reference. See module rustdoc for ecpbas
/// packing convention.
///
/// Gated `#[cfg(has_vendor_pyscf_nr_ecp)]` — the cfg flag is emitted by
/// `crates/cintx-oracle/build.rs` when `CINTX_ORACLE_BUILD_VENDOR=1`.
#[cfg(has_vendor_pyscf_nr_ecp)]
#[allow(non_snake_case)]
pub fn vendor_ECPscalar_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::ECPscalar_sph(
            out.as_mut_ptr(),
            ptr::null_mut(), // dims = NULL means default
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(), // opt = NULL (ECPOpt)
            ptr::null_mut(), // cache = NULL (let PySCF allocate)
        )
    }
}

/// Evaluate PySCF `ECPscalar_cart` for a single shell pair.
#[cfg(has_vendor_pyscf_nr_ecp)]
#[allow(non_snake_case)]
pub fn vendor_ECPscalar_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::ECPscalar_cart(
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

/// Evaluate PySCF `ECPscalar_ipnuc_sph` — ECP gradient (component_rank=3) sph.
///
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:453-462 (`ECPscalar_ipnuc_sph`
/// → `_sph_factory(_deriv1_cart, .., comp=3, ..)`). The `out` buffer MUST hold
/// `3 * nao_i * nao_j` f64s (component_rank=3), where `nao_X = CINTcgto_spheric`.
/// PySCF writes `[comp ∈ {x,y,z}, dij]` with `comp` slowest-varying and `dij`
/// F-order (`j*di + i`, `i` fastest) — i.e. `[axis, ao_j, ao_i]`. The
/// `debug_assert!` below enforces the buffer-size invariant (T-19-23) before the
/// unsafe FFI call. Plan 19-07 wires this into the gradient parity tests.
#[cfg(has_vendor_pyscf_nr_ecp)]
#[allow(non_snake_case)]
pub fn vendor_ECPscalar_ipnuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    // T-19-23: out must be 3 * nao_i * nao_j f64s. nao = CINTcgto_spheric.
    debug_assert!(
        out.len() % 3 == 0,
        "ECPscalar_ipnuc_sph out buffer must be 3 * nao_i * nao_j (component_rank=3), got len={}",
        out.len()
    );
    unsafe {
        ffi::ECPscalar_ipnuc_sph(
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

/// Evaluate PySCF `ECPscalar_ipnuc_cart` — ECP gradient (component_rank=3) cart.
///
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:366-375 (`ECPscalar_ipnuc_cart`
/// → `_cart_factory(_deriv1_cart, .., comp=3, ..)`). The `out` buffer MUST hold
/// `3 * nao_i * nao_j` f64s (component_rank=3), where `nao_X = CINTcgto_cart`.
/// Same `[comp, ao_j, ao_i]` layout as the sph variant; the `debug_assert!`
/// enforces the buffer-size invariant (T-19-23) before the unsafe FFI call.
#[cfg(has_vendor_pyscf_nr_ecp)]
#[allow(non_snake_case)]
pub fn vendor_ECPscalar_ipnuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    // T-19-23: out must be 3 * nao_i * nao_j f64s. nao = CINTcgto_cart.
    debug_assert!(
        out.len() % 3 == 0,
        "ECPscalar_ipnuc_cart out buffer must be 3 * nao_i * nao_j (component_rank=3), got len={}",
        out.len()
    );
    unsafe {
        ffi::ECPscalar_ipnuc_cart(
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

/// Evaluate PySCF `ECPscalar_iprinv_sph` — per-nucleus ECP force
/// (component_rank=3) sph (21-07, GRAD-09).
///
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:420-453 (`ECPscalar_iprinv_sph`
/// → `_one_shell_ecpbas` selects the single ECP shell on the atom indexed by
/// `env[AS_RINV_ORIG_ATOM]`, then runs the SAME comp=3 `_deriv1_cart` driver as
/// `ipnuc` on that one shell — no all-slot accumulation). The `out` buffer MUST
/// hold `3 * nao_i * nao_j` f64s (`nao = CINTcgto_spheric`), with the same
/// `[axis, ao_j, ao_i]` layout as the ipnuc variant.
///
/// IMPORTANT — the caller MUST set `env[AS_RINV_ORIG_ATOM] = <target atom index>`
/// (slot 17, an INTEGER atom index) before calling, in addition to the
/// `env[AS_ECPBAS_OFFSET]` / `env[AS_NECPBAS]` ECP slab packing the scalar
/// wrappers require. A `shl_id < 0` (no ECP shell on that atom) makes PySCF
/// return 0 without writing `out`.
#[cfg(has_vendor_pyscf_nr_ecp)]
#[allow(non_snake_case)]
pub fn vendor_ECPscalar_iprinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    // T-19-23: out must be 3 * nao_i * nao_j f64s. nao = CINTcgto_spheric.
    debug_assert!(
        out.len() % 3 == 0,
        "ECPscalar_iprinv_sph out buffer must be 3 * nao_i * nao_j (component_rank=3), got len={}",
        out.len()
    );
    unsafe {
        ffi::ECPscalar_iprinv_sph(
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

/// Evaluate PySCF `ECPscalar_iprinv_cart` — per-nucleus ECP force
/// (component_rank=3) cart (21-07, GRAD-09).
///
/// Source: vendor/pyscf-nr-ecp/src/nr_ecp_deriv.c:333-375 (`ECPscalar_iprinv_cart`
/// → `_one_shell_ecpbas` single-atom selection + comp=3 `_deriv1_cart`). The
/// `out` buffer MUST hold `3 * nao_i * nao_j` f64s (`nao = CINTcgto_cart`), same
/// `[axis, ao_j, ao_i]` layout as the ipnuc variant.
///
/// IMPORTANT — the caller MUST set `env[AS_RINV_ORIG_ATOM] = <target atom index>`
/// (slot 17) before calling. See the sph variant for full env-slot notes.
#[cfg(has_vendor_pyscf_nr_ecp)]
#[allow(non_snake_case)]
pub fn vendor_ECPscalar_iprinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    // T-19-23: out must be 3 * nao_i * nao_j f64s. nao = CINTcgto_cart.
    debug_assert!(
        out.len() % 3 == 0,
        "ECPscalar_iprinv_cart out buffer must be 3 * nao_i * nao_j (component_rank=3), got len={}",
        out.len()
    );
    unsafe {
        ffi::ECPscalar_iprinv_cart(
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

// ─────────────────────────────────────────────────────────────────────────────
// FFI ABI smoke test — catches symbol/ABI mismatches between cintx-oracle's
// bindgen output and the vendored PySCF nr_ecp shared object before parity
// tests run.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
mod ecp_ffi_smoke {
    use super::*;

    /// Minimal H-atom + 1 Local ECP row, calls vendor_ECPscalar_sph and
    /// vendor_ECPscalar_cart and asserts the FFI returns without crashing.
    /// The numerical value is not checked here — see safe_api_ecp_parity.rs.
    #[test]
    fn ecpscalar_sph_and_cart_smoke() {
        const ATM_SLOTS: usize = 6;
        const BAS_SLOTS: usize = 8;
        const PTR_ENV_START: usize = 20;
        const AS_ECPBAS_OFFSET: usize = 18;
        const AS_NECPBAS: usize = 19;
        const CHARGE_OF: usize = 0;
        const PTR_COORD: usize = 1;
        const NUC_MOD_OF: usize = 2;
        const PTR_ZETA: usize = 3;
        const POINT_NUC: i32 = 1;
        const ATOM_OF: usize = 0;
        const ANG_OF: usize = 1;
        const NPRIM_OF: usize = 2;
        const NCTR_OF: usize = 3;
        const RADI_POWER: usize = 3; // ECP slot
        const PTR_EXP: usize = 5;
        const PTR_COEFF: usize = 6;

        // Build env: pre-pad to PTR_ENV_START=20.
        let mut env = vec![0.0_f64; PTR_ENV_START];

        let coord_ptr = env.len() as i32;
        env.extend_from_slice(&[0.0_f64, 0.0, 0.0]);
        let zeta_ptr = env.len() as i32;
        env.push(0.0);

        // Single s-shell AO: 1 primitive, exp=1.0, coeff=1.0.
        let ao_exp_ptr = env.len() as i32;
        env.push(1.0);
        let ao_coeff_ptr = env.len() as i32;
        env.push(1.0);

        // ECP Type-1 (Local) row: 1 primitive, exp=1.0, coeff=1.0,
        // radial_power=0.
        let ecp_exp_ptr = env.len() as i32;
        env.push(1.0);
        let ecp_coeff_ptr = env.len() as i32;
        env.push(1.0);

        let mut atm = vec![0_i32; ATM_SLOTS];
        atm[CHARGE_OF] = 1;
        atm[PTR_COORD] = coord_ptr;
        atm[NUC_MOD_OF] = POINT_NUC;
        atm[PTR_ZETA] = zeta_ptr;
        let natm: i32 = 1;

        // bas table: AO row + ECP row (concatenated).
        let mut bas = vec![0_i32; 2 * BAS_SLOTS];
        // AO row at index 0
        bas[ATOM_OF] = 0;
        bas[ANG_OF] = 0;
        bas[NPRIM_OF] = 1;
        bas[NCTR_OF] = 1;
        bas[PTR_EXP] = ao_exp_ptr;
        bas[PTR_COEFF] = ao_coeff_ptr;
        // ECP row at index 1
        bas[BAS_SLOTS + ATOM_OF] = 0;
        bas[BAS_SLOTS + ANG_OF] = -1; // Local channel sentinel
        bas[BAS_SLOTS + NPRIM_OF] = 1;
        bas[BAS_SLOTS + NCTR_OF] = 1;
        bas[BAS_SLOTS + RADI_POWER] = 0;
        bas[BAS_SLOTS + PTR_EXP] = ecp_exp_ptr;
        bas[BAS_SLOTS + PTR_COEFF] = ecp_coeff_ptr;
        let nbas: i32 = 2;

        // Wire env[AS_ECPBAS_OFFSET] = 1 (ECP starts at bas index 1).
        // Wire env[AS_NECPBAS] = 1 (one ECP row).
        env[AS_ECPBAS_OFFSET] = 1.0;
        env[AS_NECPBAS] = 1.0;

        // Evaluate (s-shell × s-shell): output = 1 element (cart) or 1 element (sph).
        let shls = [0_i32, 0_i32];
        let mut out_sph = vec![0.0_f64; 1];
        let _ret_sph = vendor_ECPscalar_sph(&mut out_sph, &shls, &atm, natm, &bas, nbas, &env);

        let mut out_cart = vec![0.0_f64; 1];
        let _ret_cart = vendor_ECPscalar_cart(&mut out_cart, &shls, &atm, natm, &bas, nbas, &env);

        // FFI returned without crashing — smoke test passes. Numerical
        // value validation happens in safe_api_ecp_parity.rs.
        // The returned outputs SHOULD be non-zero (a Type-1 ECP integral
        // for an s-s pair at zero displacement is finite), but we don't
        // assert specifics here.
        let _ = out_sph;
        let _ = out_cart;
    }
}
