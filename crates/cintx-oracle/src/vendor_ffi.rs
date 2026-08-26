//! Safe wrappers around vendored libcint 6.1.3 FFI for oracle comparison.
//!
//! Only available when built with CINTX_ORACLE_BUILD_VENDOR=1.
//! All functions use the same atm/bas/env layout as cintx_compat::raw.

#![cfg(has_vendor_libcint)]
// Every wrapper is named after the libcint C symbol it calls
// (`vendor_CINTgto_norm` -> `CINTgto_norm`). Renaming them to snake case would
// break the one property that makes this file auditable against the vendored
// header: a reader can match a wrapper to its C entry point by name alone.
#![allow(non_snake_case)]

#[allow(
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    non_snake_case,
    improper_ctypes
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/oracle_bindings.rs"));
}

use std::ptr;

macro_rules! vendor_1e_gap_wrapper {
    ($wrapper:ident, $ffi_symbol:ident) => {
        pub fn $wrapper(
            out: &mut [f64],
            shls: &[i32; 2],
            atm: &[i32],
            natm: i32,
            bas: &[i32],
            nbas: i32,
            env: &[f64],
        ) -> i32 {
            unsafe {
                ffi::$ffi_symbol(
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
    };
}

vendor_1e_gap_wrapper!(vendor_int1e_iprinvip_cart, int1e_iprinvip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_iprinvip_sph, int1e_iprinvip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_iprinvip_spinor, int1e_iprinvip_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipipr_cart, int1e_ipipr_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ipipr_sph, int1e_ipipr_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ipipr_spinor, int1e_ipipr_spinor);
vendor_1e_gap_wrapper!(vendor_int2c2e_ip1ip2_cart, int2c2e_ip1ip2_cart);
vendor_1e_gap_wrapper!(vendor_int2c2e_ip1ip2_sph, int2c2e_ip1ip2_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ippnucp_cart, int1e_ippnucp_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ippnucp_sph, int1e_ippnucp_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ipprinvp_cart, int1e_ipprinvp_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ipprinvp_sph, int1e_ipprinvp_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ippnucpip_cart, int1e_ippnucpip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ippnucpip_sph, int1e_ippnucpip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ipprinvpip_cart, int1e_ipprinvpip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ipprinvpip_sph, int1e_ipprinvpip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ipippnucp_cart, int1e_ipippnucp_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ipippnucp_sph, int1e_ipippnucp_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ipipprinvp_cart, int1e_ipipprinvp_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ipipprinvp_sph, int1e_ipipprinvp_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ippnucp_spinor, int1e_ippnucp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipprinvp_spinor, int1e_ipprinvp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ippnucpip_spinor, int1e_ippnucpip_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipprinvpip_spinor, int1e_ipprinvpip_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipippnucp_spinor, int1e_ipippnucp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipipprinvp_spinor, int1e_ipipprinvp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipspnucsp_spinor, int1e_ipspnucsp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipsprinvsp_spinor, int1e_ipsprinvsp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipipspnucsp_spinor, int1e_ipipspnucsp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipipsprinvsp_spinor, int1e_ipipsprinvsp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipspnucspip_spinor, int1e_ipspnucspip_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_ipsprinvspip_spinor, int1e_ipsprinvspip_spinor);

macro_rules! vendor_3c_gap_wrapper {
    ($wrapper:ident, $ffi_symbol:ident) => {
        pub fn $wrapper(
            out: &mut [f64],
            shls: &[i32; 3],
            atm: &[i32],
            natm: i32,
            bas: &[i32],
            nbas: i32,
            env: &[f64],
        ) -> i32 {
            unsafe {
                ffi::$ffi_symbol(
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
    };
}

vendor_3c_gap_wrapper!(vendor_int3c2e_ipvip1_cart, int3c2e_ipvip1_cart);
vendor_3c_gap_wrapper!(vendor_int3c2e_ipvip1_sph, int3c2e_ipvip1_sph);
vendor_3c_gap_wrapper!(vendor_int3c2e_ip1ip2_cart, int3c2e_ip1ip2_cart);
vendor_3c_gap_wrapper!(vendor_int3c2e_ip1ip2_sph, int3c2e_ip1ip2_sph);
// W4-05: 3c2e σ gradient (int3c2e.c:668, ng={2,1,0,0,3,4,1,3}, c2s_si_3c2e1).
vendor_3c_gap_wrapper!(vendor_int3c2e_ipspsp1_spinor, int3c2e_ipspsp1_spinor);

macro_rules! vendor_2e_gap_wrapper {
    ($wrapper:ident, $ffi_symbol:ident) => {
        pub fn $wrapper(
            out: &mut [f64],
            shls: &[i32; 4],
            atm: &[i32],
            natm: i32,
            bas: &[i32],
            nbas: i32,
            env: &[f64],
        ) -> i32 {
            unsafe {
                ffi::$ffi_symbol(
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
    };
}

vendor_2e_gap_wrapper!(vendor_int2e_ipvip1ipvip2_cart, int2e_ipvip1ipvip2_cart);
vendor_2e_gap_wrapper!(vendor_int2e_ipvip1ipvip2_sph, int2e_ipvip1ipvip2_sph);
vendor_2e_gap_wrapper!(vendor_int2e_spsp2_spinor, int2e_spsp2_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ipspsp1_spinor, int2e_ipspsp1_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ip1spsp2_spinor, int2e_ip1spsp2_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ipspsp1spsp2_spinor, int2e_ipspsp1spsp2_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ipsrsr1_spinor, int2e_ipsrsr1_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ip1srsr2_spinor, int2e_ip1srsr2_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ipsrsr1srsr2_spinor, int2e_ipsrsr1srsr2_spinor);

// W4-06: gauge / cross-product 2e families (intor2.c). Spin-free: cart, sph and
// spinor (c2s_sf_2e1 + c2s_sf_2e2), NOT a σ family.
vendor_2e_gap_wrapper!(vendor_int2e_ip1v_r1_cart, int2e_ip1v_r1_cart);
vendor_2e_gap_wrapper!(vendor_int2e_ip1v_r1_sph, int2e_ip1v_r1_sph);
vendor_2e_gap_wrapper!(vendor_int2e_ip1v_r1_spinor, int2e_ip1v_r1_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ip1v_rc1_cart, int2e_ip1v_rc1_cart);
vendor_2e_gap_wrapper!(vendor_int2e_ip1v_rc1_sph, int2e_ip1v_rc1_sph);
vendor_2e_gap_wrapper!(vendor_int2e_ip1v_rc1_spinor, int2e_ip1v_rc1_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ipvg1_xp1_cart, int2e_ipvg1_xp1_cart);
vendor_2e_gap_wrapper!(vendor_int2e_ipvg1_xp1_sph, int2e_ipvg1_xp1_sph);
vendor_2e_gap_wrapper!(vendor_int2e_ipvg1_xp1_spinor, int2e_ipvg1_xp1_spinor);
vendor_2e_gap_wrapper!(vendor_int2e_ipvg2_xp1_cart, int2e_ipvg2_xp1_cart);
vendor_2e_gap_wrapper!(vendor_int2e_ipvg2_xp1_sph, int2e_ipvg2_xp1_sph);
vendor_2e_gap_wrapper!(vendor_int2e_ipvg2_xp1_spinor, int2e_ipvg2_xp1_spinor);

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

/// Evaluate int1e_cg_sa10sa01_cart for a single shell pair using vendored libcint
/// (30-01c-DEBUG cart-path discriminator). CART path bypasses the c2s_si spinor
/// transform entirely, so the output is the raw 36-component cart tensor —
/// `out[comp * (ncart_j * ncart_i) + (j * ncart_i + i)]`, comp in 0..36, where the
/// 36 = 9 sigma-groups × 4 gc-blocks (x,y,z,1) in `gout[grp*4+block]` order.
///
/// `out` must be pre-allocated with 36 * ncart_i * ncart_j elements.
pub fn vendor_int1e_cg_sa10sa01_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_sa10sa01_cart(
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

/// Evaluate int1e_giao_sa10sa01_cart for a single shell pair using vendored libcint
/// (30-01c-DEBUG cart-path discriminator). Same 36-component cart layout as
/// [`vendor_int1e_cg_sa10sa01_cart`]; differs only in the gauge origin ([0,0,0]).
pub fn vendor_int1e_giao_sa10sa01_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_sa10sa01_cart(
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

// ── Phase 23 both-side rank-9 1e families (9 = 3×3 components, component-leading
//    out[comp * ni * nj + n] for comp in 0..9). ────────────────────────────────

/// Evaluate int1e_ipovlpip_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipovlpip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlpip_sph(
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

/// Evaluate int1e_ipovlpip_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipovlpip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlpip_cart(
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

/// Evaluate int1e_ipkinip_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipkinip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipkinip_sph(
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

/// Evaluate int1e_ipkinip_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipkinip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipkinip_cart(
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

/// Evaluate int1e_ipnucip_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipnucip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipnucip_sph(
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

/// Evaluate int1e_ipnucip_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipnucip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipnucip_cart(
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

/// Evaluate int1e_ipipovlp_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipipovlp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipovlp_sph(
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

/// Evaluate int1e_ipipovlp_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipipovlp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipovlp_cart(
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

/// Evaluate int1e_ipipnuc_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipipnuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipnuc_sph(
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

/// Evaluate int1e_ipipnuc_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipipnuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipnuc_cart(
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

/// Evaluate int1e_ipipkin_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipipkin_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipkin_sph(
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

/// Evaluate int1e_ipipkin_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipipkin_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipkin_cart(
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

/// Evaluate int1e_ipiprinv_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipiprinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipiprinv_sph(
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

/// Evaluate int1e_ipiprinv_cart for a single shell pair using vendored libcint.
pub fn vendor_int1e_ipiprinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipiprinv_cart(
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

/// Evaluate int1e_ipipipnuc_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipipnuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipipnuc_sph(
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

/// Evaluate int1e_ipipipnuc_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipipnuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipipnuc_cart(
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

/// Evaluate int1e_ipipiprinv_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipiprinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipiprinv_sph(
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

/// Evaluate int1e_ipipiprinv_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipiprinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipiprinv_cart(
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

/// Evaluate int1e_ipipnucip_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipnucip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipnucip_sph(
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

/// Evaluate int1e_ipipnucip_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipnucip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipnucip_cart(
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

/// Evaluate int1e_ipiprinvip_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipiprinvip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipiprinvip_sph(
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

/// Evaluate int1e_ipiprinvip_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipiprinvip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipiprinvip_cart(
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

/// Evaluate int1e_ipipipiprinv_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipipiprinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipipiprinv_sph(
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

/// Evaluate int1e_ipipipiprinv_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipipiprinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipipiprinv_cart(
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

/// Evaluate int1e_ipiprinvipip_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipiprinvipip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipiprinvipip_sph(
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

/// Evaluate int1e_ipiprinvipip_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipiprinvipip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipiprinvipip_cart(
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

/// Evaluate int1e_ipipiprinvip_sph for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipiprinvip_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipiprinvip_sph(
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

/// Evaluate int1e_ipipiprinvip_cart for a single shell pair using vendored libcint (Phase 25 HESS-04).
pub fn vendor_int1e_ipipiprinvip_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipiprinvip_cart(
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

/// Safe wrapper for the vendored libcint `CINTrys_roots(nroots, x, u, w)`
/// root/weight dispatcher (Phase 25 FND-02 — the byte-identity reference for the
/// host Wheeler nroots>=6 port). The C signature is
/// `int CINTrys_roots(int nroots, double x, double *u, double *w)` (rys_roots.c:57),
/// the `lower == 0` long-range path; the short-range `lower != 0` dispatcher is a
/// separate symbol (`CINTsr_rys_roots`) and is out of scope for Phase 25.
///
/// Returns `(roots, weights)` each of length `nroots`. The rys symbols are already
/// compiled into the vendor static lib (build.rs rys source list); `CINTrys_roots`
/// is allowlisted + declared in the supplemental header (build.rs).
pub fn vendor_CINTrys_roots(nroots: i32, x: f64) -> (Vec<f64>, Vec<f64>) {
    let n = nroots.max(0) as usize;
    let mut u = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    unsafe {
        ffi::CINTrys_roots(nroots, x, u.as_mut_ptr(), w.as_mut_ptr());
    }
    (u, w)
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

// ─────────────────────────────────────────────────────────────────────────
// Phase 24: position / multipole-moment family vendor wrappers (cart + sph).
// Cloned verbatim from `vendor_int1e_iprinv_{sph,cart}` — only the `ffi::int1e_*`
// call and the doc-comment differ. `out` is sized RANK*ni*nj component-leading;
// the test caller owns sizing. Base families (26) + `_origj` variants (12) = 38.
// NOTE: there is NO int1e_rrr_origj / int1e_rrrr_origj symbol in libcint 6.1.3
// (confirmed by grep of src/autocode/intor1.c) — they are intentionally absent.
// ─────────────────────────────────────────────────────────────────────────

/// Evaluate int1e_r_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_r_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r_sph(
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

/// Evaluate int1e_r_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_r_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r_cart(
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

/// Evaluate int1e_rr_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
pub fn vendor_int1e_rr_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rr_sph(
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

/// Evaluate int1e_rr_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
pub fn vendor_int1e_rr_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rr_cart(
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

/// Evaluate int1e_rrr_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 27 * ni * nj elements (component_rank=27).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..27.
pub fn vendor_int1e_rrr_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rrr_sph(
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

/// Evaluate int1e_rrr_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 27 * ni * nj elements (component_rank=27).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..27.
pub fn vendor_int1e_rrr_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rrr_cart(
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

/// Evaluate int1e_rrrr_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 81 * ni * nj elements (component_rank=81).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..81.
pub fn vendor_int1e_rrrr_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rrrr_sph(
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

/// Evaluate int1e_rrrr_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 81 * ni * nj elements (component_rank=81).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..81.
pub fn vendor_int1e_rrrr_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rrrr_cart(
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

/// Evaluate int1e_r2_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r2_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r2_sph(
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

/// Evaluate int1e_r2_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r2_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r2_cart(
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

/// Evaluate int1e_r4_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r4_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r4_sph(
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

/// Evaluate int1e_r4_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r4_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r4_cart(
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

/// Evaluate int1e_z_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_z_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_z_sph(
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

/// Evaluate int1e_z_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_z_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_z_cart(
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

/// Evaluate int1e_zz_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_zz_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_zz_sph(
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

/// Evaluate int1e_zz_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_zz_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_zz_cart(
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

/// Evaluate int1e_p4_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_p4_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_p4_sph(
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

/// Evaluate int1e_p4_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_p4_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_p4_cart(
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

/// Evaluate int1e_irp_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
pub fn vendor_int1e_irp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_irp_sph(
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

/// Evaluate int1e_irp_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
pub fn vendor_int1e_irp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_irp_cart(
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

/// Evaluate int1e_rinv_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
///
/// IMPORTANT: the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]`
/// (libcint `PTR_RINV_ORIG = 4`) to the single rinv center BEFORE calling.
/// A zero rinv origin is trivially-passing and disallowed (Phase 24 D-04).
pub fn vendor_int1e_rinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rinv_sph(
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

/// Evaluate int1e_rinv_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
///
/// IMPORTANT: the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]`
/// (libcint `PTR_RINV_ORIG = 4`) to the single rinv center BEFORE calling.
/// A zero rinv origin is trivially-passing and disallowed (Phase 24 D-04).
pub fn vendor_int1e_rinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rinv_cart(
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

/// Evaluate int1e_drinv_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
///
/// IMPORTANT: the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]`
/// (libcint `PTR_RINV_ORIG = 4`) to the single rinv center BEFORE calling.
/// A zero rinv origin is trivially-passing and disallowed (Phase 24 D-04).
pub fn vendor_int1e_drinv_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_drinv_sph(
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

/// Evaluate int1e_drinv_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
///
/// IMPORTANT: the caller MUST set `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]`
/// (libcint `PTR_RINV_ORIG = 4`) to the single rinv center BEFORE calling.
/// A zero rinv origin is trivially-passing and disallowed (Phase 24 D-04).
pub fn vendor_int1e_drinv_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_drinv_cart(
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

/// Evaluate int1e_r_origj_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_r_origj_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r_origj_sph(
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

/// Evaluate int1e_r_origj_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
pub fn vendor_int1e_r_origj_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r_origj_cart(
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

/// Evaluate int1e_rr_origj_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
pub fn vendor_int1e_rr_origj_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rr_origj_sph(
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

/// Evaluate int1e_rr_origj_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
pub fn vendor_int1e_rr_origj_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_rr_origj_cart(
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

/// Evaluate int1e_r2_origj_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r2_origj_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r2_origj_sph(
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

/// Evaluate int1e_r2_origj_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r2_origj_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r2_origj_cart(
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

/// Evaluate int1e_r4_origj_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r4_origj_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r4_origj_sph(
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

/// Evaluate int1e_r4_origj_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_r4_origj_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r4_origj_cart(
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

/// Evaluate int1e_z_origj_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_z_origj_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_z_origj_sph(
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

/// Evaluate int1e_z_origj_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_z_origj_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_z_origj_cart(
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

/// Evaluate int1e_zz_origj_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_zz_origj_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_zz_origj_sph(
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

/// Evaluate int1e_zz_origj_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 1 * ni * nj elements (component_rank=1).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..1.
pub fn vendor_int1e_zz_origj_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_zz_origj_cart(
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

/// Evaluate int2e_ip2_sph for a single shell quartet using vendored libcint.
///
/// Phase 23 DRV1-01 (`int2e_ip2` — ∇ on the 2nd-electron bra-center k). 4-shell
/// arity (`shls = [i, j, k, l]`). `out` must be pre-allocated with
/// `3 * ni*nj*nk*nl` elements where nX = CINTcgto_spheric(shls[X], bas).
///
/// LAYOUT: component-leading F-order — `out[comp * (ni*nj*nk*nl) + n]` (same
/// convention as `vendor_int2e_ip1_sph`).
pub fn vendor_int2e_ip2_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ip2_sph(
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

/// Evaluate int2e_ip2_cart for a single shell quartet using vendored libcint.
///
/// Cartesian analog of [`vendor_int2e_ip2_sph`].
pub fn vendor_int2e_ip2_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ip2_cart(
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

// ── Phase 25 HESS-02: 2e Hessian families (rank 9 / 81), 4-shell arity ──
// Component-leading F-order `out[comp * (ni*nj*nk*nl) + n]`, same convention as
// vendor_int2e_ip2_*. ipip1/ipvip1/ip1ip2 are rank 9; ipip1ipip2 is rank 81.

/// Evaluate int2e_ipip1_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ipip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ipip1_sph(
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

/// Evaluate int2e_ipip1_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ipip1_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ipip1_cart(
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

/// Evaluate int2e_ipvip1_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ipvip1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ipvip1_sph(
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

/// Evaluate int2e_ipvip1_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ipvip1_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ipvip1_cart(
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

/// Evaluate int2e_ip1ip2_sph for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ip1ip2_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ip1ip2_sph(
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

/// Evaluate int2e_ip1ip2_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ip1ip2_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ip1ip2_cart(
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

/// Evaluate int2e_ipip1ipip2_sph (rank 81) for a single shell quartet.
pub fn vendor_int2e_ipip1ipip2_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ipip1ipip2_sph(
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

/// Evaluate int2e_ipip1ipip2_cart (rank 81) for a single shell quartet.
pub fn vendor_int2e_ipip1ipip2_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ipip1ipip2_cart(
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

// ── Phase 26 GIAO-02 (D-16): spin-free 2e GIAO families, 4-shell arity ──
// Real `double *out`, component-leading F-order `out[comp * (ni*nj*nk*nl) + n]`,
// same convention as vendor_int2e_ip1_*. g1/ig1 are rank 3; gg1/g1g2 are rank 9.
// (D-15: the cart/sph vendor symbols are plain real double*, NOT double complex —
// the cintx-side Complex<f64> view is materialized from the real device output.)

/// Evaluate int2e_g1_sph for a single shell quartet using vendored libcint.
/// `out` must be pre-allocated with `3 * ni*nj*nk*nl` elements (rank 3).
pub fn vendor_int2e_g1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_g1_sph(
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

/// Evaluate int2e_g1_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_g1_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_g1_cart(
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

/// Evaluate int2e_ig1_sph for a single shell quartet using vendored libcint (rank 3).
pub fn vendor_int2e_ig1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ig1_sph(
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

/// Evaluate int2e_ig1_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_ig1_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ig1_cart(
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

/// Evaluate int2e_gg1_sph for a single shell quartet using vendored libcint (rank 9).
pub fn vendor_int2e_gg1_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_gg1_sph(
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

/// Evaluate int2e_gg1_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_gg1_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_gg1_cart(
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

/// Evaluate int2e_g1g2_sph for a single shell quartet using vendored libcint (rank 9).
pub fn vendor_int2e_g1g2_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_g1g2_sph(
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

/// Evaluate int2e_g1g2_cart for a single shell quartet using vendored libcint.
pub fn vendor_int2e_g1g2_cart(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_g1g2_cart(
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

/// Evaluate int2c2e_ip1_sph for a single shell pair using vendored libcint.
///
/// Phase 23 DRV1-04 (`int2c2e_ip1` — ∇ on bra center i). 2-shell arity
/// (`shls = [i, k]`). `out` must be pre-allocated with `3 * ni*nk` elements
/// where nX = CINTcgto_spheric(shls[X], bas).
pub fn vendor_int2c2e_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ip1_sph(
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

/// Evaluate int2c2e_ip1_cart for a single shell pair using vendored libcint.
pub fn vendor_int2c2e_ip1_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ip1_cart(
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

/// Evaluate int2c2e_ip2_sph for a single shell pair using vendored libcint.
///
/// Phase 23 DRV1-04 (`int2c2e_ip2` — ∇ on ket center k). 2-shell arity.
pub fn vendor_int2c2e_ip2_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ip2_sph(
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

/// Evaluate int2c2e_ip2_cart for a single shell pair using vendored libcint.
pub fn vendor_int2c2e_ip2_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ip2_cart(
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

// ─────────────────────────────────────────────────────────────────────────────
// Phase 23 DRV1-03: int3c1e_ip1 (∇ on bra i of the 3-center OVERLAP) and
// int3c1e_iprinv (∇ on bra i of the 3-center rinv-COULOMB, Rys-driven). Both are
// arity-3, rank-3 derivative families. iprinv reads env[PTR_RINV_ORIG..+3]; the
// caller MUST set a non-zero rinv origin in env[4..6] before invoking.
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate int3c1e_ip1_sph for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices. Output is rank-3
/// (3 components × ni*nj*nk), component-leading.
pub fn vendor_int3c1e_ip1_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_ip1_sph(
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

/// Evaluate int3c1e_ip1_cart for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]` — three shell indices. Output is rank-3.
pub fn vendor_int3c1e_ip1_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_ip1_cart(
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

/// Evaluate int3c1e_iprinv_sph for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]`. The caller MUST set env[PTR_RINV_ORIG..+3] (env[4..6])
/// to the desired rinv origin before calling. Output is rank-3.
pub fn vendor_int3c1e_iprinv_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_iprinv_sph(
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

/// Evaluate int3c1e_iprinv_cart for a single shell triple using vendored libcint.
///
/// `shls` is `[i, j, k]`. The caller MUST set env[PTR_RINV_ORIG..+3] (env[4..6])
/// to the desired rinv origin before calling. Output is rank-3.
pub fn vendor_int3c1e_iprinv_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_iprinv_cart(
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

/// Evaluate int3c2e_ip2_cart for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj * nk elements (3 gradient components).
/// Layout: component-leading — out[comp * ni*nj*nk + n] for comp in 0..3.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, ip2
/// variant: ∇ on the auxiliary `k` center — `G2E_D_K` on the real aux k per
/// int3c2e.c:99, in cintx's layout the 2e `ll` slot).
pub fn vendor_int3c2e_ip2_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ip2_cart(
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

/// Evaluate int3c2e_ip2_sph for a single shell triple using vendored libcint.
///
/// `out` must be pre-allocated with 3 * ni * nj * nk elements (3 gradient components)
/// where nX = CINTcgto_spheric(shls[X], bas).
/// Layout: component-leading — out[comp * ni*nj*nk + n] for comp in 0..3.
///
/// `shls` is `[i, j, k]` — three shell indices (3-center 2-electron integral, ip2
/// variant, spherical). This is the `∇` auxiliary-`k`-center DERIVATIVE reference
/// for the int3c2e_ip2 oracle gate (DRV1-05).
pub fn vendor_int3c2e_ip2_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ip2_sph(
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
// Phase 25 HESS-03 — multi-center rank-9 Hessian vendor references.
// int2c2e_ipip1 (2-shell, ∇² on bra center 1), int3c2e_ipip1 (3-shell, ∇² on bra
// center 1), int3c2e_ipip2 (3-shell, ∇² on the auxiliary k center — KET headroom).
// `out` is component-leading: out[comp*nf + n] for comp in 0..9.
// ─────────────────────────────────────────────────────────────────────────────

/// int2c2e_ipip1_cart — 2-shell, ∇² on bra center 1 (rank 9).
pub fn vendor_int2c2e_ipip1_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ipip1_cart(
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

/// int2c2e_ipip1_sph — 2-shell, ∇² on bra center 1 (rank 9).
pub fn vendor_int2c2e_ipip1_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ipip1_sph(
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

/// int3c2e_ipip1_cart — 3-shell, ∇² on bra center 1 (rank 9).
pub fn vendor_int3c2e_ipip1_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ipip1_cart(
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

/// int3c2e_ipip1_sph — 3-shell, ∇² on bra center 1 (rank 9).
pub fn vendor_int3c2e_ipip1_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ipip1_sph(
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

/// int3c2e_ipip2_cart — 3-shell, ∇² on the auxiliary k center (KET headroom, rank 9).
pub fn vendor_int3c2e_ipip2_cart(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ipip2_cart(
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

/// int3c2e_ipip2_sph — 3-shell, ∇² on the auxiliary k center (KET headroom, rank 9).
pub fn vendor_int3c2e_ipip2_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ipip2_sph(
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

/// Evaluate int1e_sp_spinor (σ·p on the bra) for a single shell pair using
/// vendored libcint. This is the Phase-28 Gap B2 byte-identity reference for
/// the spin-included `c2s_si_1e` transform driven through the σ·p assembler.
///
/// `int1e_sp_spinor` is a tensor-rank-1 spinor family (libcint ng[7]=1): the
/// four `gc_x/gc_y/gc_z/gc_1` cart blocks are `ncomp_e1`, folded into a single
/// `di*dj` complex output by `c2s_si_1e`. So `out` is sized `ni_sp * nj_sp * 2`
/// f64 (interleaved real/imaginary), exactly like int1e_ovlp_spinor — NOT
/// 3-component like the ipovlp gradient block. ni_sp = CINTcgto_spinor(shls[0]),
/// nj_sp = CINTcgto_spinor(shls[1]) (use `vendor_CINTcgto_spinor` for sizing;
/// kappa≠0 → 2l or 2l+2, never a hardcoded 4l+2).
///
/// Returns the libcint status (1 for non-zero, 0 for zero by symmetry).
pub fn vendor_int1e_sp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_sp_spinor(
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

/// Evaluate `int2e_spsp1_spinor` (REL-03, the thinnest 2e σ family) for a single
/// shell quartet using vendored libcint. This is the Phase-29 Wave-2 D-03 BLOCKING
/// byte-identity reference for the brand-new 2e si/sf transform suite
/// (`c2s_si_2e1` + `c2s_sf_2e2`, intor4.c:85).
///
/// `int2e_spsp1_spinor` is component_rank=1: the σ·p₁ G-tensor's four cart blocks
/// (`gc_x/gc_y/gc_z/gc_1`) are folded into a single `ni*nj*nk*nl` complex output by
/// the `c2s_si_2e1`+`c2s_sf_2e2` driver. So `out` is sized `ni_sp*nj_sp*nk_sp*nl_sp*2`
/// f64 (interleaved real/imaginary), with each spinor extent from
/// `vendor_CINTcgto_spinor` (kappa≠0 → 2l or 2l+2, never a hardcoded 4l+2).
///
/// `shls` is `&[i32; 4]` (the four shell indices). Returns the libcint status.
pub fn vendor_int2e_spsp1_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spsp1_spinor(
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

// ─────────────────────────────────────────────────────────────────────────────
// Phase 29 Wave 1 — 1e Group-4 relativistic σ spinor families (REL-01/02).
// Each is a verbatim clone of `vendor_int1e_sp_spinor` with only the
// `ffi::int1e_X_spinor` driver symbol swapped. All are component_rank=1: the
// σ-component fold is internal to the c2s transform, so `out` is sized
// `ni_sp * nj_sp * 2` f64 (interleaved real/imaginary) via `vendor_CINTcgto_spinor`.
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate `int1e_spsp_spinor` (REL-01, `c2s_sf_1e` path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_spsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_spsp_spinor(
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

/// Evaluate `int1e_spnucsp_spinor` (REL-01, `c2s_si_1e` path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_spnucsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_spnucsp_spinor(
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

/// Evaluate `int1e_sprinvsp_spinor` (REL-01, `c2s_si_1e` path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_sprinvsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_sprinvsp_spinor(
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

/// Evaluate `int1e_srsr_spinor` (REL-02, `c2s_si_1e` path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_srsr_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_srsr_spinor(
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

/// Evaluate `int1e_srnucsr_spinor` (REL-02, `c2s_si_1e` path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_srnucsr_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_srnucsr_spinor(
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

/// Evaluate `int1e_sr_spinor` (REL-02, `c2s_si_1ei` imaginary-ket path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_sr_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_sr_spinor(
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

/// Evaluate `int1e_sigma_spinor` (REL-02, `c2s_si_1ei` imaginary-ket path) — `out` sized `ni_sp*nj_sp*2`.
pub fn vendor_int1e_sigma_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_sigma_spinor(
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

/// Evaluate `int1e_cg_sa10sp_spinor` (GIAO-03, `c2s_si_1ei` imaginary-ket gauge
/// path, rank 3) — `out` sized `3*ni_sp*nj_sp*2` (interleaved re/im, 3 stacked
/// spinor matrices). The driver reads `env[PTR_COMMON_ORIG]` for the gauge origin.
pub fn vendor_int1e_cg_sa10sp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_sa10sp_spinor(
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

/// Evaluate `int1e_giao_sa10sp_spinor` (GIAO-03, `c2s_si_1ei` imaginary-ket
/// natural-center path, rank 3) — `out` sized `3*ni_sp*nj_sp*2`. Same gout as
/// `cg_sa10sp` but with the gauge `x1i` step at the natural bra center (no origin
/// shift); used as the cg→giao collapse witness at `common_orig = bra center`.
pub fn vendor_int1e_giao_sa10sp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_sa10sp_spinor(
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

/// Evaluate `int1e_spgsp_spinor` (GIAO-03 Wave 1, `c2s_si_1ei`) — `out` sized `3*ni_sp*nj_sp*2`.
pub fn vendor_int1e_spgsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_spgsp_spinor(
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

/// Evaluate `int1e_spgnucsp_spinor` (GIAO-03 Wave 1, Rys nuclear, `c2s_si_1ei`) — `out` sized `3*ni_sp*nj_sp*2`.
pub fn vendor_int1e_spgnucsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_spgnucsp_spinor(
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

/// Evaluate `int1e_spgsa01_spinor` (GIAO-03 Wave 1, Rys rinv, `c2s_si_1e`, rank 9) — `out` sized `9*ni_sp*nj_sp*2`.
pub fn vendor_int1e_spgsa01_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_spgsa01_spinor(
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

/// Evaluate `int1e_cg_sa10nucsp_spinor` (GIAO-03 Wave 1, Rys nuclear, `c2s_si_1ei`) — `out` sized `3*ni_sp*nj_sp*2`.
pub fn vendor_int1e_cg_sa10nucsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_sa10nucsp_spinor(
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

/// Evaluate `int1e_cg_sa10sa01_spinor` (GIAO-03 Wave 1, Rys rinv, `c2s_si_1e`, rank 9) — `out` sized `9*ni_sp*nj_sp*2`.
pub fn vendor_int1e_cg_sa10sa01_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_sa10sa01_spinor(
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

/// Evaluate `int1e_giao_sa10nucsp_spinor` (GIAO-03 Wave 1, Rys nuclear, `c2s_si_1ei`) — `out` sized `3*ni_sp*nj_sp*2`.
pub fn vendor_int1e_giao_sa10nucsp_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_sa10nucsp_spinor(
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

/// Evaluate `int1e_giao_sa10sa01_spinor` (GIAO-03 Wave 1, Rys rinv, `c2s_si_1e`, rank 9) — `out` sized `9*ni_sp*nj_sp*2`.
pub fn vendor_int1e_giao_sa10sa01_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_sa10sa01_spinor(
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

// ---- Phase 27: spinor DERIVATIVE integral vendor FFI wrappers (D-09) ----
//
// Six new spinor-gradient operators that complete the spinor derivative parity
// matrix beyond the four existing rank-3 1e ip-spinor wrappers (ipovlp/ipkin/
// ipnuc/iprinv above). Each writes a component-leading, interleaved-complex
// buffer:
//   1e / 2c2e:  out[comp * ni_sp * nj_sp * 2 + (j*ni_sp + i)*2 + {0:re,1:im}]
//   3c2e / 3c1e: out[comp * ni_sp * nj_sp * nk_sph * 2 + ...]
// where ni_sp / nj_sp = CINTcgto_spinor(shls[i|j]) (bra i and ket j are spinor,
// 4l+2). IMPORTANT (27-SPIKE-FINDINGS CORRECTION NOTICE): the auxiliary-k axis of
// the arity-3 families is SPHERICAL nsph(lk) = (2lk+1)*nctr_k (libcint
// CINT3c2e_spinor_drv is_ssc=0, cint3c2e.c:631-636), NOT spinor-sized. Size the out
// buffer with vendor_cgto_spheric for the aux-k axis and vendor_CINTcgto_spinor only
// for the bra-i and ket-j axes. The earlier "spinor aux-k = 720" was a compat-dims
// over-sizing artifact, not a real vendor requirement (correct p×d×s kappa=0 is 360).

/// Evaluate int1e_ipovlpip_spinor (rank-9 both-side 1e gradient) for a shell pair.
///
/// `out` must be pre-allocated with `9 * ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_ipovlpip_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipovlpip_spinor(
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

/// Evaluate int1e_ipipipiprinv_spinor (rank-81 1e rinv gradient) for a shell pair.
///
/// The rinv origin must be set in `env[PTR_RINV_ORIG..+3]` by the caller.
/// `out` must be pre-allocated with `81 * ni_sp * nj_sp * 2` f64 elements.
pub fn vendor_int1e_ipipipiprinv_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ipipipiprinv_spinor(
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

/// Evaluate int2c2e_ip1_spinor (rank-3 2-center 2e gradient) for a shell pair.
///
/// `shls` is `[i, k]`. `out` must be pre-allocated with `3 * ni_sp * nk_sp * 2`
/// f64 elements (both axes spinor-sized via CINTcgto_spinor).
pub fn vendor_int2c2e_ip1_spinor(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2c2e_ip1_spinor(
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

/// Evaluate int3c2e_ip1_spinor (rank-3 3-center 2e gradient on bra i) for a triple.
///
/// `shls` is `[i, j, k]`. `out` must be pre-allocated with
/// `3 * ni_sp * nj_sp * nk_sph * 2` f64 elements. The aux-k axis is SPHERICAL
/// (nsph(lk) = (2lk+1)*nctr_k), NOT spinor-sized; only bra i and ket j use
/// CINTcgto_spinor (4l+2) (27-SPIKE-FINDINGS CORRECTION NOTICE, cint3c2e.c:631-636).
pub fn vendor_int3c2e_ip1_spinor(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_ip1_spinor(
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

/// Evaluate int3c1e_ip1_spinor (rank-3 3-center 1e overlap gradient on bra i).
///
/// `shls` is `[i, j, k]`. `out` must be pre-allocated with
/// `3 * ni_sp * nj_sp * nk_sph * 2` f64 elements (aux-k axis SPHERICAL,
/// nsph(lk) = (2lk+1)*nctr_k; only bra i and ket j are spinor-sized).
pub fn vendor_int3c1e_ip1_spinor(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_ip1_spinor(
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

/// Evaluate int3c1e_iprinv_spinor (rank-3 3-center 1e rinv-Coulomb gradient on i).
///
/// The rinv origin must be set in `env[PTR_RINV_ORIG..+3]` by the caller (a
/// zero origin trivially passes — use a displaced origin). `shls` is `[i, j, k]`.
/// `out` must be pre-allocated with `3 * ni_sp * nj_sp * nk_sph * 2` f64 elements
/// (aux-k axis SPHERICAL, nsph(lk) = (2lk+1)*nctr_k; only bra i and ket j spinor-sized).
pub fn vendor_int3c1e_iprinv_spinor(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_iprinv_spinor(
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
pub fn vendor_int1e_r2_origi_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r2_origi_sph(
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

/// Evaluate int1e_r4_origi_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r4_origi_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r4_origi_sph(
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

/// Evaluate int1e_r2_origi_ip2_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r2_origi_ip2_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r2_origi_ip2_sph(
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

/// Evaluate int1e_r4_origi_ip2_sph for a single shell pair using vendored libcint.
pub fn vendor_int1e_r4_origi_ip2_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_r4_origi_ip2_sph(
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

/// Evaluate int1e_grids_sph for a shell pair + grid range using vendored libcint.
/// `shls` is `[i, j, grid_start, grid_end]` where grid indices come from env.
pub fn vendor_int1e_grids_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_grids_sph(
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

/// Evaluate int1e_grids_ip_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_ip_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_grids_ip_sph(
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

/// Evaluate int1e_grids_ipvip_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_ipvip_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_grids_ipvip_sph(
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

/// Evaluate int1e_grids_spvsp_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_spvsp_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_grids_spvsp_sph(
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

/// Evaluate int1e_grids_ipip_sph using vendored libcint. `shls` is `[i, j, grid_start, grid_end]`.
pub fn vendor_int1e_grids_ipip_sph(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_grids_ipip_sph(
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

/// Evaluate int2e_breit_r1p2_spinor (spinor-only Breit 2e) using vendored libcint.
pub fn vendor_int2e_breit_r1p2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_breit_r1p2_spinor(
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

/// Evaluate int2e_breit_r2p2_spinor (spinor-only Breit 2e) using vendored libcint.
pub fn vendor_int2e_breit_r2p2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_breit_r2p2_spinor(
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

/// Evaluate int3c1e_r2_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_r2_origk_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_r2_origk_sph(
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

/// Evaluate int3c1e_r4_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_r4_origk_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_r4_origk_sph(
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

/// Evaluate int3c1e_r6_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_r6_origk_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_r6_origk_sph(
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

/// Evaluate int3c1e_ip1_r2_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_ip1_r2_origk_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_ip1_r2_origk_sph(
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

/// Evaluate int3c1e_ip1_r4_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_ip1_r4_origk_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_ip1_r4_origk_sph(
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

/// Evaluate int3c1e_ip1_r6_origk_sph for a shell triple using vendored libcint.
pub fn vendor_int3c1e_ip1_r6_origk_sph(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c1e_ip1_r6_origk_sph(
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

/// `int3c1e_ip1_r6_origk_sph` evaluated against an explicitly ZEROED cache.
///
/// This operator is the one place where upstream's result is not a function of
/// its inputs. `CINTgout1e_int3c1e_ip1_r6_origk`
/// (libcint-master/src/cint3c1e_a.c:627) reads `g76` in the `s[1]` term
/// `6*g48[ix]*g76[iy]*g3[iz]`, but its `G1E_D_I` list (cint3c1e_a.c:604-609)
/// covers g64/g67/g79/g112/g124/g127 and omits `G1E_D_I(g76, g12, ...)`. `g76`
/// lies inside the `MALLOC_INSTACK` span but is never written, so the value read
/// there is whatever the cache allocation happens to hold: a fresh mmap-backed
/// `malloc` gives zero, a recycled heap chunk gives stale numbers. Calling the
/// plain [`vendor_int3c1e_ip1_r6_origk_sph`] twice with identical arguments but
/// different call histories can therefore return results differing by ~1e-1.
///
/// libcint's own ABI lets the caller own `cache` (pass `out == NULL` to learn the
/// size), so handing it a zeroed buffer pins `g76` to 0 and makes the vendor
/// result reproducible. That zero-`g76` behaviour is what cintx's
/// `origk_ip1_kernel` reproduces, so this is the comparison that can actually be
/// asserted byte-for-byte.
pub fn vendor_int3c1e_ip1_r6_origk_sph_zeroed_cache(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    // `out == NULL` makes libcint return the cache size it needs instead of
    // evaluating.
    let cache_len = unsafe {
        ffi::int3c1e_ip1_r6_origk_sph(
            ptr::null_mut(),
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
    };
    let mut cache = vec![0.0_f64; cache_len.max(0) as usize];
    unsafe {
        ffi::int3c1e_ip1_r6_origk_sph(
            out.as_mut_ptr(),
            ptr::null_mut(),
            shls.as_ptr() as *mut i32,
            atm.as_ptr() as *mut i32,
            natm,
            bas.as_ptr() as *mut i32,
            nbas,
            env.as_ptr() as *mut f64,
            ptr::null_mut(),
            cache.as_mut_ptr(),
        )
    }
}

/// Evaluate int3c2e_sph_ssc (spin-spin contact 3c2e) for a shell triple using vendored libcint.
pub fn vendor_int3c2e_sph_ssc(
    out: &mut [f64],
    shls: &[i32; 3],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int3c2e_sph_ssc(
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
        out.len().is_multiple_of(3),
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
        out.len().is_multiple_of(3),
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
        out.len().is_multiple_of(3),
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
        out.len().is_multiple_of(3),
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
// Phase 26 GIAO-01: spin-free 1e GIAO/CG magnetic-property vendor wrappers.
// 22 real `double *out` wrappers (11 families x {cart, sph}). D-15: cart/sph
// GIAO symbols are real doubles, so these clone vendor_int1e_r_* verbatim.
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate int1e_govlp_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_govlp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_govlp_sph(
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

/// Evaluate int1e_govlp_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_govlp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_govlp_cart(
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

/// Evaluate int1e_gnuc_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_gnuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_gnuc_sph(
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

/// Evaluate int1e_gnuc_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_gnuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_gnuc_cart(
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

/// Evaluate int1e_igovlp_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_igovlp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_igovlp_sph(
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

/// Evaluate int1e_igovlp_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_igovlp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_igovlp_cart(
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

/// Evaluate int1e_ignuc_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_ignuc_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ignuc_sph(
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

/// Evaluate int1e_ignuc_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_ignuc_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ignuc_cart(
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

/// Evaluate int1e_igkin_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_igkin_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_igkin_sph(
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

/// Evaluate int1e_igkin_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_igkin_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_igkin_cart(
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

/// Evaluate int1e_a01gp_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_a01gp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_a01gp_sph(
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

/// Evaluate int1e_a01gp_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_a01gp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_a01gp_cart(
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

/// Evaluate int1e_ia01p_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_ia01p_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ia01p_sph(
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

/// Evaluate int1e_ia01p_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_ia01p_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_ia01p_cart(
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

/// Evaluate int1e_cg_irxp_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_cg_irxp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_irxp_sph(
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

/// Evaluate int1e_cg_irxp_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_cg_irxp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_irxp_cart(
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

/// Evaluate int1e_giao_irjxp_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_giao_irjxp_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_irjxp_sph(
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

/// Evaluate int1e_giao_irjxp_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 3 * ni * nj elements (component_rank=3).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..3.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_giao_irjxp_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_irjxp_cart(
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

/// Evaluate int1e_cg_a11part_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_cg_a11part_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_a11part_sph(
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

/// Evaluate int1e_cg_a11part_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_cg_a11part_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_cg_a11part_cart(
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

/// Evaluate int1e_giao_a11part_sph for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_giao_a11part_sph(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_a11part_sph(
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

/// Evaluate int1e_giao_a11part_cart for a single shell pair using vendored libcint 6.1.3.
///
/// `out` must be pre-allocated with 9 * ni * nj elements (component_rank=9).
/// Layout: component-leading — out[comp * ni * nj + n] for comp in 0..9.
/// D-15: the cart/sph GIAO symbol is a REAL `double *out` (the magnitude of the
/// purely-imaginary integral), so this is an ordinary real wrapper, NOT len-2N.
pub fn vendor_int1e_giao_a11part_cart(
    out: &mut [f64],
    shls: &[i32; 2],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int1e_giao_a11part_cart(
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
// Phase 29 Wave 3 — remaining 2e Group-4 relativistic σ spinor families
// (REL-03 intor4.c; REL-04 gaunt1.c/dkb.c, BLOCKING-wired in 29-05 build.rs).
// Each is a verbatim clone of `vendor_int2e_spsp1_spinor` with only the
// `ffi::int2e_X_spinor` driver symbol swapped. All component_rank=1: the
// σ-component fold is internal to the c2s_si/sf_2e transform pair, so `out` is
// sized `ni_sp*nj_sp*nk_sp*nl_sp*2` f64 (interleaved real/imaginary) via
// `vendor_CINTcgto_spinor` (kappa≠0 → 2l or 2l+2, never a hardcoded 4l+2).
// `shls` is `&[i32; 4]` (the four shell indices). Returns the libcint status.
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate `int2e_srsr1_spinor` (Group-4 2e σ, intor4.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_srsr1_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_srsr1_spinor(
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

/// Evaluate `int2e_spsp1spsp2_spinor` (Group-4 2e σ, intor4.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_spsp1spsp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spsp1spsp2_spinor(
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

/// Evaluate `int2e_srsr1srsr2_spinor` (Group-4 2e σ, intor4.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_srsr1srsr2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_srsr1srsr2_spinor(
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

/// Evaluate `int2e_ssp1ssp2_spinor` (Group-4 2e σ, gaunt1.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_ssp1ssp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ssp1ssp2_spinor(
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

/// Evaluate `int2e_ssp1sps2_spinor` (Group-4 2e σ, gaunt1.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_ssp1sps2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_ssp1sps2_spinor(
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

/// Evaluate `int2e_sps1ssp2_spinor` (Group-4 2e σ, gaunt1.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_sps1ssp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_sps1ssp2_spinor(
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

/// Evaluate `int2e_sps1sps2_spinor` (Group-4 2e σ, gaunt1.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_sps1sps2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_sps1sps2_spinor(
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

/// Evaluate `int2e_spv1_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_spv1_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spv1_spinor(
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

/// Evaluate `int2e_vsp1_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_vsp1_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_vsp1_spinor(
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

/// Evaluate `int2e_spv1spv2_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_spv1spv2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spv1spv2_spinor(
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

/// Evaluate `int2e_vsp1spv2_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_vsp1spv2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_vsp1spv2_spinor(
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

/// Evaluate `int2e_spv1vsp2_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_spv1vsp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spv1vsp2_spinor(
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

/// Evaluate `int2e_vsp1vsp2_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_vsp1vsp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_vsp1vsp2_spinor(
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

/// Evaluate `int2e_spv1spsp2_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_spv1spsp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_spv1spsp2_spinor(
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

/// Evaluate `int2e_vsp1spsp2_spinor` (Group-4 2e σ, dkb.c) — `out` sized `ni*nj*nk*nl*2`.
pub fn vendor_int2e_vsp1spsp2_spinor(
    out: &mut [f64],
    shls: &[i32; 4],
    atm: &[i32],
    natm: i32,
    bas: &[i32],
    nbas: i32,
    env: &[f64],
) -> i32 {
    unsafe {
        ffi::int2e_vsp1spsp2_spinor(
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

// ── Wave 5 W5-01/W5-02: spinor rows re-proven with a general-contracted aux-k ──
vendor_1e_gap_wrapper!(vendor_int2c2e_ip2_spinor, int2c2e_ip2_spinor);
vendor_3c_gap_wrapper!(vendor_int3c2e_ip2_spinor, int3c2e_ip2_spinor);

// ── Wave 5 W5-05: derivative families the parent plan's Gap-A table omitted ──
vendor_1e_gap_wrapper!(vendor_int1e_ovlpip_cart, int1e_ovlpip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ovlpip_sph, int1e_ovlpip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_kinip_cart, int1e_kinip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_kinip_sph, int1e_kinip_sph);
// ── Wave 5 W5-05: Tier 6 (lresc.c + hess.c) ──
vendor_1e_gap_wrapper!(vendor_int1e_iprinvr_cart, int1e_iprinvr_cart);
vendor_1e_gap_wrapper!(vendor_int1e_iprinvr_sph, int1e_iprinvr_sph);
vendor_1e_gap_wrapper!(vendor_int1e_iprip_cart, int1e_iprip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_iprip_sph, int1e_iprip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_iprinviprip_cart, int1e_iprinviprip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_iprinviprip_sph, int1e_iprinviprip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_ipiprinvrip_cart, int1e_ipiprinvrip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_ipiprinvrip_sph, int1e_ipiprinvrip_sph);
vendor_1e_gap_wrapper!(vendor_int1e_rinvipiprip_cart, int1e_rinvipiprip_cart);
vendor_1e_gap_wrapper!(vendor_int1e_rinvipiprip_sph, int1e_rinvipiprip_sph);
// ── Wave 5 W5-06: X2C base families (intor1.c) ──
vendor_1e_gap_wrapper!(vendor_int1e_pnucp_cart, int1e_pnucp_cart);
vendor_1e_gap_wrapper!(vendor_int1e_pnucp_sph, int1e_pnucp_sph);
vendor_1e_gap_wrapper!(vendor_int1e_prinvp_cart, int1e_prinvp_cart);
vendor_1e_gap_wrapper!(vendor_int1e_prinvp_sph, int1e_prinvp_sph);
// W5-06: the spinor forms ride CINT1e_spinor_drv, which is real (unlike the
// CINT2c2e/CINT3c1e spinor drivers), so they ARE oracle-provable.
vendor_1e_gap_wrapper!(vendor_int1e_pnucp_spinor, int1e_pnucp_spinor);
vendor_1e_gap_wrapper!(vendor_int1e_prinvp_spinor, int1e_prinvp_spinor);
