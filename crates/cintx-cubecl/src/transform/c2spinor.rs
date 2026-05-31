//! Cartesian-to-spinor (c2spinor) transform functions.
//!
//! Implements the four variants of the spinor transform corresponding to
//! libcint `CINTc2s_ket_spinor_*` functions, using CG coupling coefficients
//! from `c2spinor_coeffs`.
//!
//! Output layout convention (for nd spinor components):
//!   - Alpha upper component: gsp[i*2] = re, gsp[i*2+1] = im, for i in 0..nd
//!   - Beta lower component: gsp[(nd+i)*2] = re, gsp[(nd+i)*2+1] = im, for i in 0..nd
//!   Total buffer size: 2 * nd * 2 = 4*nd f64 values.
//!
//! For kappa == 0, both GT (j=l+1/2) and LT (j=l-1/2) blocks are applied,
//! with GT written first (rows 0..nd_gt) and LT next (rows nd_gt..nd_gt+nd_lt).

use cintx_core::{CintFloat, cintxRsError};
use super::c2s::ncart;
use super::c2spinor_coeffs as cj;

/// Number of spinor components for angular momentum l and quantum number kappa.
///
/// Mirrors libcint `_len_spinor`:
///   - kappa < 0: j = l+1/2 → 2*l+2 components
///   - kappa > 0: j = l-1/2 → 2*l components
///   - kappa == 0: both blocks → 4*l+2 components
pub fn spinor_len(l: u8, kappa: i32) -> usize {
    if kappa < 0 {
        2 * l as usize + 2
    } else if kappa > 0 {
        2 * l as usize
    } else {
        4 * l as usize + 2
    }
}

/// Internal: apply the sf (scalar-field, spin-free) accumulation for one block.
///
/// sf formula from `CINTc2s_ket_spinor_sf1`:
///   gspaz_re += caR * v1
///   gspaz_im += caI * v1
///   gspbz_re += cbR * v1
///   gspbz_im += cbI * v1
///
/// Writes nd spinor rows starting at gsp[offset_alpha..] and gsp[offset_beta..].
fn apply_sf_block<F: CintFloat>(
    gsp: &mut [F],
    cart: &[f64],
    coeff_r: &[&[f64]],
    coeff_i: &[&[f64]],
    nd: usize,
    nf: usize,
    nd_total: usize,
    row_offset: usize,
) {
    for i in 0..nd {
        let row_r = coeff_r[i];
        let row_i = coeff_i[i];
        let mut sa_re = 0.0f64;
        let mut sa_im = 0.0f64;
        let mut sb_re = 0.0f64;
        let mut sb_im = 0.0f64;
        for n in 0..nf {
            let v1 = cart[n];
            let ca_r = row_r[n];
            let ca_i = row_i[n];
            let cb_r = row_r[nf + n];
            let cb_i = row_i[nf + n];
            sa_re += ca_r * v1;
            sa_im += ca_i * v1;
            sb_re += cb_r * v1;
            sb_im += cb_i * v1;
        }
        let out_i = row_offset + i;
        gsp[out_i * 2] = F::from_f64_lossy(sa_re);
        gsp[out_i * 2 + 1] = F::from_f64_lossy(sa_im);
        gsp[(nd_total + out_i) * 2] = F::from_f64_lossy(sb_re);
        gsp[(nd_total + out_i) * 2 + 1] = F::from_f64_lossy(sb_im);
    }
}

/// Internal: apply the iket_sf accumulation for one block.
///
/// iket_sf = multiply sf output by i: (re, im) → (-im, re)
/// Formula from `CINTc2s_iket_spinor_sf1`:
///   gspaz_re -= caI * v1
///   gspaz_im += caR * v1
///   gspbz_re -= cbI * v1
///   gspbz_im += cbR * v1
fn apply_iket_sf_block<F: CintFloat>(
    gsp: &mut [F],
    cart: &[f64],
    coeff_r: &[&[f64]],
    coeff_i: &[&[f64]],
    nd: usize,
    nf: usize,
    nd_total: usize,
    row_offset: usize,
) {
    for i in 0..nd {
        let row_r = coeff_r[i];
        let row_i = coeff_i[i];
        let mut sa_re = 0.0f64;
        let mut sa_im = 0.0f64;
        let mut sb_re = 0.0f64;
        let mut sb_im = 0.0f64;
        for n in 0..nf {
            let v1 = cart[n];
            let ca_r = row_r[n];
            let ca_i = row_i[n];
            let cb_r = row_r[nf + n];
            let cb_i = row_i[nf + n];
            sa_re -= ca_i * v1;
            sa_im += ca_r * v1;
            sb_re -= cb_i * v1;
            sb_im += cb_r * v1;
        }
        let out_i = row_offset + i;
        gsp[out_i * 2] = F::from_f64_lossy(sa_re);
        gsp[out_i * 2 + 1] = F::from_f64_lossy(sa_im);
        gsp[(nd_total + out_i) * 2] = F::from_f64_lossy(sb_re);
        gsp[(nd_total + out_i) * 2 + 1] = F::from_f64_lossy(sb_im);
    }
}

/// Internal: apply the si (spin-included) accumulation for one block.
///
/// si formula from `CINTc2s_ket_spinor_si1`:
///   gspaz_re += caR*v1 - caI*vz + cbR*vy - cbI*vx
///   gspaz_im += caI*v1 + caR*vz + cbI*vy + cbR*vx
///   gspbz_re += cbR*v1 + cbI*vz - caR*vy - caI*vx
///   gspbz_im += cbI*v1 - cbR*vz - caI*vy + caR*vx
#[allow(clippy::too_many_arguments)]
fn apply_si_block<F: CintFloat>(
    gsp: &mut [F],
    cart_v1: &[f64],
    cart_vx: &[f64],
    cart_vy: &[f64],
    cart_vz: &[f64],
    coeff_r: &[&[f64]],
    coeff_i: &[&[f64]],
    nd: usize,
    nf: usize,
    nd_total: usize,
    row_offset: usize,
) {
    for i in 0..nd {
        let row_r = coeff_r[i];
        let row_i = coeff_i[i];
        let mut sa_re = 0.0f64;
        let mut sa_im = 0.0f64;
        let mut sb_re = 0.0f64;
        let mut sb_im = 0.0f64;
        for n in 0..nf {
            let v1 = cart_v1[n];
            let vx = cart_vx[n];
            let vy = cart_vy[n];
            let vz = cart_vz[n];
            let ca_r = row_r[n];
            let ca_i = row_i[n];
            let cb_r = row_r[nf + n];
            let cb_i = row_i[nf + n];
            sa_re += ca_r * v1 - ca_i * vz + cb_r * vy - cb_i * vx;
            sa_im += ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
            sb_re += cb_r * v1 + cb_i * vz - ca_r * vy - ca_i * vx;
            sb_im += cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
        }
        let out_i = row_offset + i;
        gsp[out_i * 2] = F::from_f64_lossy(sa_re);
        gsp[out_i * 2 + 1] = F::from_f64_lossy(sa_im);
        gsp[(nd_total + out_i) * 2] = F::from_f64_lossy(sb_re);
        gsp[(nd_total + out_i) * 2 + 1] = F::from_f64_lossy(sb_im);
    }
}

/// Internal: apply the iket_si accumulation for one block.
///
/// iket_si = multiply si output by i: (re, im) → (-im, re)
/// Formula from `CINTc2s_iket_spinor_si1`:
///   gspaz_re -= caI*v1 + caR*vz + cbI*vy + cbR*vx
///   gspaz_im += caR*v1 - caI*vz + cbR*vy - cbI*vx
///   gspbz_re -= cbI*v1 - cbR*vz - caI*vy + caR*vx
///   gspbz_im += cbR*v1 + cbI*vz - caR*vy - caI*vx
#[allow(clippy::too_many_arguments)]
fn apply_iket_si_block<F: CintFloat>(
    gsp: &mut [F],
    cart_v1: &[f64],
    cart_vx: &[f64],
    cart_vy: &[f64],
    cart_vz: &[f64],
    coeff_r: &[&[f64]],
    coeff_i: &[&[f64]],
    nd: usize,
    nf: usize,
    nd_total: usize,
    row_offset: usize,
) {
    for i in 0..nd {
        let row_r = coeff_r[i];
        let row_i = coeff_i[i];
        let mut sa_re = 0.0f64;
        let mut sa_im = 0.0f64;
        let mut sb_re = 0.0f64;
        let mut sb_im = 0.0f64;
        for n in 0..nf {
            let v1 = cart_v1[n];
            let vx = cart_vx[n];
            let vy = cart_vy[n];
            let vz = cart_vz[n];
            let ca_r = row_r[n];
            let ca_i = row_i[n];
            let cb_r = row_r[nf + n];
            let cb_i = row_i[nf + n];
            sa_re -= ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
            sa_im += ca_r * v1 - ca_i * vz + cb_r * vy - cb_i * vx;
            sb_re -= cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
            sb_im += cb_r * v1 + cb_i * vz - ca_r * vy - ca_i * vx;
        }
        let out_i = row_offset + i;
        gsp[out_i * 2] = F::from_f64_lossy(sa_re);
        gsp[out_i * 2 + 1] = F::from_f64_lossy(sa_im);
        gsp[(nd_total + out_i) * 2] = F::from_f64_lossy(sb_re);
        gsp[(nd_total + out_i) * 2 + 1] = F::from_f64_lossy(sb_im);
    }
}

/// Retrieve GT block (j=l+1/2, kappa<0) coefficient rows for angular momentum l.
///
/// Returns (real_rows, imag_rows) as slices of rows, where each row has 2*nf entries.
fn gt_coeff_rows(l: u8) -> (Vec<&'static [f64]>, Vec<&'static [f64]>) {
    match l {
        0 => (
            cj::CJ_GT_L0_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_GT_L0_I.iter().map(|r| r.as_ref()).collect(),
        ),
        1 => (
            cj::CJ_GT_L1_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_GT_L1_I.iter().map(|r| r.as_ref()).collect(),
        ),
        2 => (
            cj::CJ_GT_L2_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_GT_L2_I.iter().map(|r| r.as_ref()).collect(),
        ),
        3 => (
            cj::CJ_GT_L3_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_GT_L3_I.iter().map(|r| r.as_ref()).collect(),
        ),
        4 => (
            cj::CJ_GT_L4_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_GT_L4_I.iter().map(|r| r.as_ref()).collect(),
        ),
        _ => (vec![], vec![]),
    }
}

/// Retrieve LT block (j=l-1/2, kappa>0) coefficient rows for angular momentum l.
fn lt_coeff_rows(l: u8) -> (Vec<&'static [f64]>, Vec<&'static [f64]>) {
    match l {
        0 => (
            cj::CJ_LT_L0_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_LT_L0_I.iter().map(|r| r.as_ref()).collect(),
        ),
        1 => (
            cj::CJ_LT_L1_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_LT_L1_I.iter().map(|r| r.as_ref()).collect(),
        ),
        2 => (
            cj::CJ_LT_L2_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_LT_L2_I.iter().map(|r| r.as_ref()).collect(),
        ),
        3 => (
            cj::CJ_LT_L3_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_LT_L3_I.iter().map(|r| r.as_ref()).collect(),
        ),
        4 => (
            cj::CJ_LT_L4_R.iter().map(|r| r.as_ref()).collect(),
            cj::CJ_LT_L4_I.iter().map(|r| r.as_ref()).collect(),
        ),
        _ => (vec![], vec![]),
    }
}

/// Cart-to-spinor scalar-field (sf) transform.
///
/// Corresponds to `CINTc2s_ket_spinor_sf1` in libcint.
///
/// `gsp`: output buffer of length `2 * spinor_len(l, kappa) * 2` f64.
///        (nd complex spinor components × 2 spinors × 2 real/imag = 4*nd f64)
/// `cart`: input cartesian buffer of length ncart(l).
/// `l`: angular momentum.
/// `kappa`: spinor quantum number (<0 → GT block, >0 → LT block, ==0 → both).
pub fn cart_to_spinor_sf<F: CintFloat>(
    gsp: &mut [F],
    cart: &[f64],
    l: u8,
    kappa: i32,
) -> Result<(), cintxRsError> {
    let nf = ncart(l);
    if cart.len() != nf {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "c2spinor_sf",
            detail: format!("cart length {} != ncart({}) = {}", cart.len(), l, nf),
        });
    }
    let nd = spinor_len(l, kappa);
    let required = 4 * nd;
    if gsp.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: gsp.len(),
        });
    }

    if kappa < 0 {
        let (rr, ri) = gt_coeff_rows(l);
        apply_sf_block(gsp, cart, &rr, &ri, nd, nf, nd, 0);
    } else if kappa > 0 {
        let (rr, ri) = lt_coeff_rows(l);
        apply_sf_block(gsp, cart, &rr, &ri, nd, nf, nd, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        // Matches libcint: CINTc2s_ket_spinor_sf1 uses LT pointer which over-reads
        // into the GT table for kappa=0. LT immediately precedes GT in g_trans memory.
        let nd_lt = 2 * l as usize;
        let nd_gt = 2 * l as usize + 2;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        if nd_lt > 0 {
            apply_sf_block(gsp, cart, &rr_lt, &ri_lt, nd_lt, nf, nd, 0);
        }
        apply_sf_block(gsp, cart, &rr_gt, &ri_gt, nd_gt, nf, nd, nd_lt);
    }
    Ok(())
}

/// Cart-to-spinor iket scalar-field transform (multiply by i).
///
/// Corresponds to `CINTc2s_iket_spinor_sf1` in libcint.
/// Same signature as `cart_to_spinor_sf` but output is multiplied by i:
/// (re, im) → (-im, re).
pub fn cart_to_spinor_iket_sf<F: CintFloat>(
    gsp: &mut [F],
    cart: &[f64],
    l: u8,
    kappa: i32,
) -> Result<(), cintxRsError> {
    let nf = ncart(l);
    if cart.len() != nf {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "c2spinor_iket_sf",
            detail: format!("cart length {} != ncart({}) = {}", cart.len(), l, nf),
        });
    }
    let nd = spinor_len(l, kappa);
    let required = 4 * nd;
    if gsp.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: gsp.len(),
        });
    }

    if kappa < 0 {
        let (rr, ri) = gt_coeff_rows(l);
        apply_iket_sf_block(gsp, cart, &rr, &ri, nd, nf, nd, 0);
    } else if kappa > 0 {
        let (rr, ri) = lt_coeff_rows(l);
        apply_iket_sf_block(gsp, cart, &rr, &ri, nd, nf, nd, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        let nd_lt = 2 * l as usize;
        let nd_gt = 2 * l as usize + 2;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        if nd_lt > 0 {
            apply_iket_sf_block(gsp, cart, &rr_lt, &ri_lt, nd_lt, nf, nd, 0);
        }
        apply_iket_sf_block(gsp, cart, &rr_gt, &ri_gt, nd_gt, nf, nd, nd_lt);
    }
    Ok(())
}

/// Cart-to-spinor spin-included (si) transform with Pauli coupling.
///
/// Corresponds to `CINTc2s_ket_spinor_si1` in libcint.
///
/// `gsp`: output buffer of length `4 * spinor_len(l, kappa)` f64.
/// `cart_v1`: scalar component cartesian buffer (length ncart(l)).
/// `cart_vx`: x Pauli component cartesian buffer (length ncart(l)).
/// `cart_vy`: y Pauli component cartesian buffer (length ncart(l)).
/// `cart_vz`: z Pauli component cartesian buffer (length ncart(l)).
pub fn cart_to_spinor_si<F: CintFloat>(
    gsp: &mut [F],
    cart_v1: &[f64],
    cart_vx: &[f64],
    cart_vy: &[f64],
    cart_vz: &[f64],
    l: u8,
    kappa: i32,
) -> Result<(), cintxRsError> {
    let nf = ncart(l);
    for (name, buf) in [
        ("v1", cart_v1),
        ("vx", cart_vx),
        ("vy", cart_vy),
        ("vz", cart_vz),
    ] {
        if buf.len() != nf {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "c2spinor_si",
                detail: format!("cart_{name} length {0} != ncart({l}) = {nf}", buf.len()),
            });
        }
    }
    let nd = spinor_len(l, kappa);
    let required = 4 * nd;
    if gsp.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: gsp.len(),
        });
    }

    if kappa < 0 {
        let (rr, ri) = gt_coeff_rows(l);
        apply_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr, &ri, nd, nf, nd, 0);
    } else if kappa > 0 {
        let (rr, ri) = lt_coeff_rows(l);
        apply_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr, &ri, nd, nf, nd, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        let nd_lt = 2 * l as usize;
        let nd_gt = 2 * l as usize + 2;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        if nd_lt > 0 {
            apply_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_lt, &ri_lt, nd_lt, nf, nd, 0);
        }
        apply_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_gt, &ri_gt, nd_gt, nf, nd, nd_lt);
    }
    Ok(())
}

/// Cart-to-spinor iket spin-included transform (multiply by i).
///
/// Corresponds to `CINTc2s_iket_spinor_si1` in libcint.
/// Same as `cart_to_spinor_si` but output is multiplied by i.
pub fn cart_to_spinor_iket_si<F: CintFloat>(
    gsp: &mut [F],
    cart_v1: &[f64],
    cart_vx: &[f64],
    cart_vy: &[f64],
    cart_vz: &[f64],
    l: u8,
    kappa: i32,
) -> Result<(), cintxRsError> {
    let nf = ncart(l);
    for (name, buf) in [
        ("v1", cart_v1),
        ("vx", cart_vx),
        ("vy", cart_vy),
        ("vz", cart_vz),
    ] {
        if buf.len() != nf {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "c2spinor_iket_si",
                detail: format!("cart_{name} length {0} != ncart({l}) = {nf}", buf.len()),
            });
        }
    }
    let nd = spinor_len(l, kappa);
    let required = 4 * nd;
    if gsp.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: gsp.len(),
        });
    }

    if kappa < 0 {
        let (rr, ri) = gt_coeff_rows(l);
        apply_iket_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr, &ri, nd, nf, nd, 0);
    } else if kappa > 0 {
        let (rr, ri) = lt_coeff_rows(l);
        apply_iket_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr, &ri, nd, nf, nd, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        let nd_lt = 2 * l as usize;
        let nd_gt = 2 * l as usize + 2;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        if nd_lt > 0 {
            apply_iket_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_lt, &ri_lt, nd_lt, nf, nd, 0);
        }
        apply_iket_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_gt, &ri_gt, nd_gt, nf, nd, nd_lt);
    }
    Ok(())
}

/// Full 2D cart-to-spinor transform for 1e integrals (scalar-field, spin-free).
///
/// Implements libcint `c2s_sf_1e`: a two-step transform that converts the
/// contracted Cartesian matrix `cart[nci × ncj]` into the spinor matrix
/// stored as interleaved complex in `staging`.
///
/// Algorithm (matching libcint `c2s_sf_1e`):
/// 1. Bra step (`a_bra_cart2spinor_sf`): for each ket Cartesian column, apply
///    the bra CG transform with sign-flipped imaginary: `saI += -caI * v1`.
///    Produces a complex intermediate `tmp[di_bra × ncj]`.
/// 2. Ket step (`a_ket_cart2spinor`): apply the ket CG transform (complex multiply)
///    over the 2*ncj ket-Cartesian indices (alpha+beta coefficient blocks).
///    Produces the output `out[di_bra × dj_ket]` complex.
/// 3. Store as column-major interleaved: `staging[(j*di + i)*2] = re`, `+1 = im`.
///
/// # Parameters
/// - `staging`: output buffer, must have at least `di * dj * 2` f64 elements
/// - `cart`: Cartesian input buffer, row-major: `cart[i_cart * ncj + j_cart]`
/// - `li`, `kappa_i`: bra angular momentum and kappa
/// - `lj`, `kappa_j`: ket angular momentum and kappa
///
/// # Kappa dispatch
/// When kappa == 0, both GT (j=l+1/2) and LT (j=l-1/2) blocks are applied.
/// The convention is: kappa_i < 0 → GT bra block, kappa_i > 0 → LT bra block,
/// kappa_i == 0 → both blocks concatenated (GT first). Same for ket.
///
/// # Signs
/// The bra transform uses the conjugate convention from libcint:
///   `saI += -caI * v1` (negative imaginary part of bra coefficient).
pub fn cart_to_spinor_sf_2d<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);

    if cart.len() < nci * ncj {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "c2spinor_sf_2d",
            detail: format!(
                "cart buffer length {} < nci*ncj = {}*{} = {}",
                cart.len(), nci, ncj, nci * ncj
            ),
        });
    }
    let required = di * dj * 2;
    if staging.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: staging.len(),
        });
    }

    // ── Step 1: Bra transform ──────────────────────────────────────────────
    // a_bra_cart2spinor_sf: gctr[j * nci + n] → tmp[alpha|beta, j * di + i]
    // tmp_alpha_R/I: [di × ncj] complex (interleaved separately, not interleaved re/im)
    // tmp_beta_R/I:  [di × ncj] complex
    // Indexing: tmp_alpha[j * di + i], tmp_beta[j * di + i]
    //
    // Sign convention (libcint a_bra_cart2spinor_sf):
    //   saI += -caI * v1   (minus sign on imaginary part)
    let mut tmp_alpha_r = vec![0.0f64; di * ncj];
    let mut tmp_alpha_i = vec![0.0f64; di * ncj];
    let mut tmp_beta_r = vec![0.0f64; di * ncj];
    let mut tmp_beta_i = vec![0.0f64; di * ncj];

    apply_bra_sf_block_all_kappa(
        &mut tmp_alpha_r, &mut tmp_alpha_i,
        &mut tmp_beta_r, &mut tmp_beta_i,
        cart, nci, ncj, di, li, kappa_i as i32,
    );

    // ── Step 2: Ket transform ──────────────────────────────────────────────
    // a_ket_cart2spinor: complex (cR + i*cI) applied over 2*ncj ket indices
    // Input layout: gcartR[j + n*di] where j=bra-spinor-index, n=ket-cart-index
    //   n ∈ [0..ncj]:    reads tmp_alpha[n*di + j]
    //   n ∈ [ncj..2*ncj]: reads tmp_beta[(n-ncj)*di + j]
    // coeff[ket_spinor_row][2*ncj] — first ncj = alpha, next ncj = beta
    //
    // Output: tmp2[di × dj] complex stored as column-major (j_ket outer, i_bra inner)
    //   tmp2[j_sp * di + i_sp] = complex spinor value
    let mut out_r = vec![0.0f64; di * dj];
    let mut out_i = vec![0.0f64; di * dj];

    apply_ket_transform(
        &mut out_r, &mut out_i,
        &tmp_alpha_r, &tmp_alpha_i,
        &tmp_beta_r, &tmp_beta_i,
        di, ncj, dj, lj, kappa_j as i32,
    );

    // ── Step 3: Write column-major interleaved to staging ─────────────────
    // zcopy_ij: staging[(j*di + i)*2] = re, [(j*di+i)*2+1] = im
    // ni=di, nj=dj: output is column-major, j-spinor outer, i-spinor inner
    // Cast from f64 intermediates to F at the output boundary via from_f64_lossy.
    for j in 0..dj {
        for i in 0..di {
            let out_idx = j * di + i;
            staging[out_idx * 2] = F::from_f64_lossy(out_r[j * di + i]);
            staging[out_idx * 2 + 1] = F::from_f64_lossy(out_i[j * di + i]);
        }
    }

    Ok(())
}

/// Host-side spin-included (si) 2D cart→spinor transform — libcint `c2s_si_1e`.
///
/// This is the σ-coupled bra+ket driver matching libcint `c2s_si_1e` (`cart2sph.c:4947`):
/// a Pauli-σ mix on the bra (`a_bra_cart2spinor_si`, via [`apply_bra_si_block`]) and the
/// ORDINARY ket transform (`a_ket_cart2spinor`, via [`apply_ket_transform`] reused
/// verbatim — `c2s_si_1e`'s ket is identical to `c2s_sf_1e`'s; the ket is NOT symmetrized).
///
/// It is the structural analog of [`cart_to_spinor_sf_2d`] with ONLY the bra step swapped
/// for the four-cart-input σ-mix. It consumes the four Cartesian G-tensor blocks
/// `gc_x/gc_y/gc_z` (Pauli-σ components) + `gc_1` (scalar) the σ·p assembler emits.
///
/// # Buffer sizing (Spike Target D)
/// All spinor sizing comes from [`spinor_len`] (kappa==0→4l+2, kappa<0→2l+2 GT,
/// kappa>0→2l LT) — never a hardcoded `4l+2`. Output is interleaved complex
/// `[re,im,…]`, column-major (ket-spinor outer, bra-spinor inner), length `di*dj*2`.
///
/// # Orientation (Pitfall 4 / Phase-27 D-06)
/// The transform OWNS the KET→BRA transpose: device cart blocks arrive KET-major
/// (`block[j*nci+i]`); the bra step reads BRA-major (`block[i*ncj+j]`). Each of the four
/// `gc_*` blocks is transposed independently before the bra fold.
///
/// # Errors
/// Validates buffer sizes BEFORE any write (OOM-safe stop, no partial writes):
/// undersized `gc_*` → [`cintxRsError::ChunkPlanFailed`]; undersized `staging` →
/// [`cintxRsError::BufferTooSmall`].
#[allow(clippy::too_many_arguments)]
pub fn cart_to_spinor_si_2d<F: CintFloat>(
    staging: &mut [F],
    gc_x: &[f64],
    gc_y: &[f64],
    gc_z: &[f64],
    gc_1: &[f64],
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);

    // ── Buffer guards (no writes before these pass) ────────────────────────────
    let need = nci * ncj;
    for (name, block) in [
        ("gc_x", gc_x), ("gc_y", gc_y), ("gc_z", gc_z), ("gc_1", gc_1),
    ] {
        if block.len() < need {
            return Err(cintxRsError::ChunkPlanFailed {
                from: "c2spinor_si_2d",
                detail: format!(
                    "{} block length {} < nci*ncj = {}*{} = {}",
                    name, block.len(), nci, ncj, need
                ),
            });
        }
    }
    let required = di * dj * 2;
    if staging.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: staging.len(),
        });
    }

    // ── Stage 0: KET→BRA transpose for each of the four gc blocks ──────────────
    // Device cart blocks are KET-major block[j*nci+i]; the bra step reads BRA-major
    // block[i*ncj+j]. Latent on square blocks (nci==ncj); the kappa fixture is
    // non-square (p×d) precisely to surface this.
    let transpose_ket_to_bra = |src: &[f64]| -> Vec<f64> {
        let mut dst = vec![0.0f64; nci * ncj];
        for i in 0..nci {
            for j in 0..ncj {
                dst[i * ncj + j] = src[j * nci + i];
            }
        }
        dst
    };
    let bm_x = transpose_ket_to_bra(gc_x);
    let bm_y = transpose_ket_to_bra(gc_y);
    let bm_z = transpose_ket_to_bra(gc_z);
    let bm_1 = transpose_ket_to_bra(gc_1);

    // ── Stage 1: Bra σ-mix (a_bra_cart2spinor_si) ──────────────────────────────
    let mut tmp_alpha_r = vec![0.0f64; di * ncj];
    let mut tmp_alpha_i = vec![0.0f64; di * ncj];
    let mut tmp_beta_r = vec![0.0f64; di * ncj];
    let mut tmp_beta_i = vec![0.0f64; di * ncj];

    apply_bra_si_block(
        &mut tmp_alpha_r, &mut tmp_alpha_i,
        &mut tmp_beta_r, &mut tmp_beta_i,
        &bm_x, &bm_y, &bm_z, &bm_1,
        nci, ncj, di, li, kappa_i as i32,
    );

    // ── Stage 2: Ordinary ket transform (REUSE verbatim — c2s_si_1e ket == sf) ──
    let mut out_r = vec![0.0f64; di * dj];
    let mut out_i = vec![0.0f64; di * dj];

    apply_ket_transform(
        &mut out_r, &mut out_i,
        &tmp_alpha_r, &tmp_alpha_i,
        &tmp_beta_r, &tmp_beta_i,
        di, ncj, dj, lj, kappa_j as i32,
    );

    // ── Stage 3: Column-major interleaved zcopy to staging ─────────────────────
    for j in 0..dj {
        for i in 0..di {
            let out_idx = j * di + i;
            staging[out_idx * 2] = F::from_f64_lossy(out_r[j * di + i]);
            staging[out_idx * 2 + 1] = F::from_f64_lossy(out_i[j * di + i]);
        }
    }

    Ok(())
}

/// Bra step of the 2D c2spinor_sf transform for all kappa cases.
///
/// Matches `a_bra_cart2spinor_sf` in libcint `cart2sph.c`.
///
/// Kappa==0 ordering: LT first (rows 0..nd_lt), then GT (rows nd_lt..nd).
/// This mirrors libcint's memory-layout trick: for kappa>=0, `a_bra_cart2spinor_sf`
/// uses the LT coeff pointer which for kappa=0 reads LT rows 0..nd_lt first, then
/// continues reading into the GT table (which immediately follows LT in memory).
///
/// Sign convention: imaginary coefficient applied with MINUS: `saI += -caI * v1`.
fn apply_bra_sf_block_all_kappa(
    alpha_r: &mut [f64],
    alpha_i: &mut [f64],
    beta_r: &mut [f64],
    beta_i: &mut [f64],
    cart: &[f64],
    nci: usize,
    ncj: usize,
    di: usize,
    li: u8,
    kappa_i: i32,
) {
    let (coeff_gt_r, coeff_gt_i, coeff_lt_r, coeff_lt_i) = bra_coeff_refs(li);

    if kappa_i < 0 {
        apply_bra_block(alpha_r, alpha_i, beta_r, beta_i,
                        cart, nci, ncj, di, coeff_gt_r, coeff_gt_i, 0);
    } else if kappa_i > 0 {
        apply_bra_block(alpha_r, alpha_i, beta_r, beta_i,
                        cart, nci, ncj, di, coeff_lt_r, coeff_lt_i, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        // This matches libcint's implicit memory ordering where kappa=0 uses
        // the LT pointer and over-reads into the GT table region.
        let nd_lt = 2 * li as usize;
        let nd_gt = 2 * li as usize + 2;
        if nd_lt > 0 {
            apply_bra_block(alpha_r, alpha_i, beta_r, beta_i,
                            cart, nci, ncj, nd_lt, coeff_lt_r, coeff_lt_i, 0);
        }
        apply_bra_block(alpha_r, alpha_i, beta_r, beta_i,
                        cart, nci, ncj, nd_gt, coeff_gt_r, coeff_gt_i, nd_lt);
    }
}

/// Apply bra spinor transform for one kappa block.
///
/// Writes `nd` spinor rows starting at `row_offset` in the alpha/beta buffers.
/// Each column j ∈ [0..ncj] of `cart` (the ket Cartesian index) is processed.
/// Layout: `alpha_r[j * di_total + row_offset + i]` for i ∈ [0..nd], j ∈ [0..ncj].
///
/// Coefficients: `coeff_r/i[spinor_row * (2*nci) + n]` for n ∈ [0..nci] (alpha)
///              `coeff_r/i[spinor_row * (2*nci) + nci + n]` for n ∈ [0..nci] (beta)
/// Sign: `saI += -caI * v1` (conjugate of bra spinor).
fn apply_bra_block(
    alpha_r: &mut [f64],
    alpha_i: &mut [f64],
    beta_r: &mut [f64],
    beta_i: &mut [f64],
    cart: &[f64],
    nci: usize,
    ncj: usize,
    nd: usize,
    coeff_r: &[f64],
    coeff_i: &[f64],
    row_offset: usize,
) {
    // di_total is the total number of bra spinor components (for indexing into output buffers)
    let di_total = alpha_r.len() / ncj;
    for j in 0..ncj {
        for i in 0..nd {
            let out_idx = j * di_total + (row_offset + i);
            let mut sa_r = 0.0f64;
            let mut sa_i = 0.0f64;
            let mut sb_r = 0.0f64;
            let mut sb_i = 0.0f64;
            for n in 0..nci {
                // cart is bra × ket row-major: cart[bra_n * ncj + ket_j]
                // libcint gctr[j*nf+n] with j=ket, n=bra — so read cart[n * ncj + j]
                let v1 = cart[n * ncj + j];
                let ca_r = coeff_r[i * 2 * nci + n];
                let ca_i = coeff_i[i * 2 * nci + n];
                let cb_r = coeff_r[i * 2 * nci + nci + n];
                let cb_i = coeff_i[i * 2 * nci + nci + n];
                // Sign: saI += -caI * v1 (libcint conjugate convention)
                sa_r += ca_r * v1;
                sa_i += -ca_i * v1;
                sb_r += cb_r * v1;
                sb_i += -cb_i * v1;
            }
            alpha_r[out_idx] = sa_r;
            alpha_i[out_idx] = sa_i;
            beta_r[out_idx] = sb_r;
            beta_i[out_idx] = sb_i;
        }
    }
}

/// Get flat coefficient slices for bra transform.
/// Returns (gt_r, gt_i, lt_r, lt_i) as flat slices.
fn bra_coeff_refs(l: u8) -> (&'static [f64], &'static [f64], &'static [f64], &'static [f64]) {
    match l {
        0 => (
            cj::CJ_GT_L0_R.as_flattened(),
            cj::CJ_GT_L0_I.as_flattened(),
            cj::CJ_LT_L0_R.as_flattened(),
            cj::CJ_LT_L0_I.as_flattened(),
        ),
        1 => (
            cj::CJ_GT_L1_R.as_flattened(),
            cj::CJ_GT_L1_I.as_flattened(),
            cj::CJ_LT_L1_R.as_flattened(),
            cj::CJ_LT_L1_I.as_flattened(),
        ),
        2 => (
            cj::CJ_GT_L2_R.as_flattened(),
            cj::CJ_GT_L2_I.as_flattened(),
            cj::CJ_LT_L2_R.as_flattened(),
            cj::CJ_LT_L2_I.as_flattened(),
        ),
        3 => (
            cj::CJ_GT_L3_R.as_flattened(),
            cj::CJ_GT_L3_I.as_flattened(),
            cj::CJ_LT_L3_R.as_flattened(),
            cj::CJ_LT_L3_I.as_flattened(),
        ),
        4 => (
            cj::CJ_GT_L4_R.as_flattened(),
            cj::CJ_GT_L4_I.as_flattened(),
            cj::CJ_LT_L4_R.as_flattened(),
            cj::CJ_LT_L4_I.as_flattened(),
        ),
        _ => panic!("cart_to_spinor_sf_2d: l={l} > 4 not supported"),
    }
}

/// Spin-included (si) bra step of the 2D `c2s_si_1e` transform — all kappa cases.
///
/// This is the σ-coupled bra analog of [`apply_bra_sf_block_all_kappa`]: it consumes
/// FOUR Cartesian blocks (`gc_1` scalar + `gc_x/gc_y/gc_z` Pauli-σ) instead of one and
/// applies the `a_bra_cart2spinor_si` accumulation (libcint `cart2sph.c:3958-3961`).
///
/// **Sign convention (THE landmine):** this transcribes `a_bra_cart2spinor_si`, NOT
/// [`apply_si_block`]. The 2D `c2s_si_1e` path uses `a_bra_cart2spinor_si`, whose signs
/// differ from `CINTc2s_ket_spinor_si1` (the single-block helper `apply_si_block` ports)
/// on three of four cross/imaginary terms. Do NOT delegate to `apply_si_block`.
///
/// Each block is read BRA-major as `block[n * ncj + j]` (n=bra cart, j=ket cart); the
/// KET→BRA orientation transpose is owned by the caller [`cart_to_spinor_si_2d`].
///
/// Kappa==0 ordering mirrors `apply_bra_sf_block_all_kappa`: LT rows first, GT rows next.
#[allow(clippy::too_many_arguments)]
fn apply_bra_si_block(
    alpha_r: &mut [f64],
    alpha_i: &mut [f64],
    beta_r: &mut [f64],
    beta_i: &mut [f64],
    gc_x: &[f64],
    gc_y: &[f64],
    gc_z: &[f64],
    gc_1: &[f64],
    nci: usize,
    ncj: usize,
    di: usize,
    li: u8,
    kappa_i: i32,
) {
    let (coeff_gt_r, coeff_gt_i, coeff_lt_r, coeff_lt_i) = bra_coeff_refs(li);

    if kappa_i < 0 {
        apply_bra_si_block_one(alpha_r, alpha_i, beta_r, beta_i,
                               gc_x, gc_y, gc_z, gc_1, nci, ncj, di,
                               coeff_gt_r, coeff_gt_i, 0);
    } else if kappa_i > 0 {
        apply_bra_si_block_one(alpha_r, alpha_i, beta_r, beta_i,
                               gc_x, gc_y, gc_z, gc_1, nci, ncj, di,
                               coeff_lt_r, coeff_lt_i, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        let nd_lt = 2 * li as usize;
        let nd_gt = 2 * li as usize + 2;
        if nd_lt > 0 {
            apply_bra_si_block_one(alpha_r, alpha_i, beta_r, beta_i,
                                   gc_x, gc_y, gc_z, gc_1, nci, ncj, nd_lt,
                                   coeff_lt_r, coeff_lt_i, 0);
        }
        apply_bra_si_block_one(alpha_r, alpha_i, beta_r, beta_i,
                               gc_x, gc_y, gc_z, gc_1, nci, ncj, nd_gt,
                               coeff_gt_r, coeff_gt_i, nd_lt);
    }
}

/// Apply the σ-coupled (si) bra transform for one kappa block.
///
/// Writes `nd` spinor rows starting at `row_offset` in the alpha/beta buffers, copying
/// the loop/index/coeff-layout structure of [`apply_bra_block`] but accumulating the
/// `a_bra_cart2spinor_si` signs over the four `gc_*` blocks.
///
/// Accumulation (libcint `cart2sph.c:3958-3961`, transcribed verbatim):
/// ```text
///   sa_r +=  ca_r * v1 + ca_i * vz - cb_r * vy + cb_i * vx;
///   sa_i += -ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
///   sb_r +=  cb_r * v1 - cb_i * vz + ca_r * vy + ca_i * vx;
///   sb_i += -cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
/// ```
#[allow(clippy::too_many_arguments)]
fn apply_bra_si_block_one(
    alpha_r: &mut [f64],
    alpha_i: &mut [f64],
    beta_r: &mut [f64],
    beta_i: &mut [f64],
    gc_x: &[f64],
    gc_y: &[f64],
    gc_z: &[f64],
    gc_1: &[f64],
    nci: usize,
    ncj: usize,
    nd: usize,
    coeff_r: &[f64],
    coeff_i: &[f64],
    row_offset: usize,
) {
    // di_total = total bra spinor rows (for indexing into the output buffers).
    let di_total = alpha_r.len() / ncj;
    for j in 0..ncj {
        for i in 0..nd {
            let out_idx = j * di_total + (row_offset + i);
            let mut sa_r = 0.0f64;
            let mut sa_i = 0.0f64;
            let mut sb_r = 0.0f64;
            let mut sb_i = 0.0f64;
            for n in 0..nci {
                // BRA-major read: block[bra_n * ncj + ket_j].
                let v1 = gc_1[n * ncj + j];
                let vx = gc_x[n * ncj + j];
                let vy = gc_y[n * ncj + j];
                let vz = gc_z[n * ncj + j];
                let ca_r = coeff_r[i * 2 * nci + n];
                let ca_i = coeff_i[i * 2 * nci + n];
                let cb_r = coeff_r[i * 2 * nci + nci + n];
                let cb_i = coeff_i[i * 2 * nci + nci + n];
                // a_bra_cart2spinor_si signs — NOT apply_si_block's signs.
                sa_r += ca_r * v1 + ca_i * vz - cb_r * vy + cb_i * vx;
                sa_i += -ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
                sb_r += cb_r * v1 - cb_i * vz + ca_r * vy + ca_i * vx;
                sb_i += -cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
            }
            alpha_r[out_idx] = sa_r;
            alpha_i[out_idx] = sa_i;
            beta_r[out_idx] = sb_r;
            beta_i[out_idx] = sb_i;
        }
    }
}

/// Ket step of the 2D c2spinor_sf transform.
///
/// Matches `a_ket_cart2spinor` in libcint `cart2sph.c`.
/// Applies complex CG coefficient multiplication over the 2*ncj ket-Cartesian
/// indices (alpha + beta blocks of the intermediate) to produce the spinor output.
///
/// Input layout:
///   `alpha_r/i[n * di + j]` for ket-cart n ∈ [0..ncj], bra-spinor j ∈ [0..di]
///   `beta_r/i[n * di + j]` for ket-cart n ∈ [0..ncj], bra-spinor j ∈ [0..di]
///
/// Output layout: `out_r/i[ket_sp_i * di + j]` (column-major: ket-spinor outer, bra-spinor inner)
///
/// Coefficient layout: `coeff[ket_spinor_row * (2*ncj) + n]`
///   n ∈ [0..ncj]: alpha part, n ∈ [ncj..2*ncj]: beta part
///
/// Complex multiply: `out += (cR + i*cI) * (gR + i*gI)` for each n, j
fn apply_ket_transform(
    out_r: &mut [f64],
    out_i: &mut [f64],
    alpha_r: &[f64],
    alpha_i: &[f64],
    beta_r: &[f64],
    beta_i: &[f64],
    di: usize,
    ncj: usize,
    dj: usize,
    lj: u8,
    kappa_j: i32,
) {
    let nf2 = 2 * ncj; // total coefficient columns (alpha + beta)
    let (coeff_gt_r, coeff_gt_i, coeff_lt_r, coeff_lt_i) = bra_coeff_refs(lj);

    // Determine which blocks to apply and their row offsets in the output
    let blocks: &[(&[f64], &[f64], usize, usize)] = match kappa_j.cmp(&0) {
        std::cmp::Ordering::Less => &[(coeff_gt_r, coeff_gt_i, dj, 0)],
        std::cmp::Ordering::Greater => &[(coeff_lt_r, coeff_lt_i, dj, 0)],
        std::cmp::Ordering::Equal => {
            // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
            // Matches libcint ordering via memory layout: LT pointer over-reads into GT.
            let nd_lt = 2 * lj as usize;
            let nd_gt = 2 * lj as usize + 2;
            if nd_lt > 0 {
                apply_ket_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i,
                               di, ncj, nd_lt, nf2, coeff_lt_r, coeff_lt_i, 0);
            }
            apply_ket_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i,
                           di, ncj, nd_gt, nf2, coeff_gt_r, coeff_gt_i, nd_lt);
            return;
        }
    };

    for &(coeff_r, coeff_i, nd, row_off) in blocks {
        apply_ket_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i,
                       di, ncj, nd, nf2, coeff_r, coeff_i, row_off);
    }
}

/// Apply one block of the ket spinor transform.
///
/// `nd`: number of ket spinor components in this block.
/// `row_off`: starting row in the output for this block.
fn apply_ket_block(
    out_r: &mut [f64],
    out_i: &mut [f64],
    alpha_r: &[f64],
    alpha_i: &[f64],
    beta_r: &[f64],
    beta_i: &[f64],
    di: usize,
    ncj: usize,
    nd: usize,
    nf2: usize,
    coeff_r: &[f64],
    coeff_i: &[f64],
    row_off: usize,
) {
    for i in 0..nd {
        // zero the output rows for this ket spinor component
        for j in 0..di {
            out_r[(row_off + i) * di + j] = 0.0;
            out_i[(row_off + i) * di + j] = 0.0;
        }
        for n in 0..nf2 {
            let cr = coeff_r[i * nf2 + n];
            let ci = coeff_i[i * nf2 + n];
            if cr == 0.0 && ci == 0.0 {
                continue;
            }
            // Read from alpha (n < ncj) or beta (n >= ncj) intermediate buffer
            let (gr_col, gi_col) = if n < ncj {
                (&alpha_r[n * di..(n + 1) * di], &alpha_i[n * di..(n + 1) * di])
            } else {
                (&beta_r[(n - ncj) * di..(n - ncj + 1) * di],
                 &beta_i[(n - ncj) * di..(n - ncj + 1) * di])
            };
            // Complex multiply: (cR + i*cI) * (gR + i*gI) = (cR*gR - cI*gI) + i*(cI*gR + cR*gI)
            for j in 0..di {
                let gr = gr_col[j];
                let gi = gi_col[j];
                out_r[(row_off + i) * di + j] += cr * gr - ci * gi;
                out_i[(row_off + i) * di + j] += ci * gr + cr * gi;
            }
        }
    }
}

/// Full 4D cart-to-spinor scalar-field transform for 2e (4-center) integrals.
///
/// Implements the two-step libcint `c2s_sf_2e1` + `c2s_sf_2e2` transform that
/// converts a contracted Cartesian 4-center integral buffer to spinor form.
///
/// Algorithm:
/// Step 1 (`c2s_sf_2e1`): Transform (i,j) bra/ket pair to spinor, keeping (k,l) Cartesian.
///   - Input: `cart[nck * ncl * nci * ncj]` with (i innermost, j next, k and l outermost).
///     NOTE: In libcint the cart buffer is indexed as `gctr[kl_idx * nci * ncj + ij_idx]`
///     (k,l outer, i,j inner).
///   - For each (k,l) pair: apply bra transform on i, ket transform on j.
///   - Intermediate: `opij[dk * dl * di * dj]` complex interleaved, where
///     di = spinor_len(li, kappa_i), dj = spinor_len(lj, kappa_j).
///
/// Step 2 (`c2s_sf_2e2`): Transform (k,l) pair to spinor on the complex intermediate.
///   - For each (i_sp, j_sp) spinor pair: apply bra-zf transform on k, ket transform on l.
///   - Output layout: `staging[(((l_sp * dk + k_sp) * dj + j_sp) * di + i_sp) * 2]` = re, +1 = im.
///     (i innermost, l outermost — column-major matching `zcopy_iklj`)
///
/// # Parameters
/// - `staging`: output buffer, size `di * dj * dk * dl * 2`
/// - `cart`: Cartesian input, size `nci * ncj * nck * ncl`
///   Layout: i innermost, l outermost: `cart[((l*nck+k)*ncj+j)*nci+i]`
pub fn cart_to_spinor_sf_4d<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    li: u8, kappa_i: i16,
    lj: u8, kappa_j: i16,
    lk: u8, kappa_k: i16,
    ll: u8, kappa_l: i16,
) -> Result<(), cintxRsError> {
    use super::c2s::ncart;

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let ncl = ncart(ll);

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let dk = spinor_len(lk, kappa_k as i32);
    let dl = spinor_len(ll, kappa_l as i32);

    let expected_cart = nci * ncj * nck * ncl;
    if cart.len() < expected_cart {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "c2spinor_sf_4d",
            detail: format!(
                "cart buffer length {} < nci*ncj*nck*ncl = {}*{}*{}*{} = {}",
                cart.len(), nci, ncj, nck, ncl, expected_cart
            ),
        });
    }
    let required = di * dj * dk * dl * 2;
    if staging.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: staging.len(),
        });
    }

    // ── Step 1: transform (i,j) pair for each (k,l) Cartesian combination ─
    // For each kl slice of size [nci * ncj], apply cart_to_spinor_sf_2d.
    // Result: opij[nck * ncl * di * dj * 2] complex interleaved
    // Index: opij[((l * nck + k) * dj * di + j_sp * di + i_sp) * 2] = re, +1 = im
    let mut opij = vec![0.0f64; nck * ncl * di * dj * 2];

    let ij_stride = di * dj; // complex elements per (k,l) slice
    for l_cart in 0..ncl {
        for k_cart in 0..nck {
            let kl_offset = (l_cart * nck + k_cart) * nci * ncj;
            let cart_slice = &cart[kl_offset..kl_offset + nci * ncj];
            let opij_offset = (l_cart * nck + k_cart) * ij_stride * 2;
            let opij_slice = &mut opij[opij_offset..opij_offset + ij_stride * 2];
            // Intermediate step 1 stays in f64 (opij is Vec<f64>).
            cart_to_spinor_sf_2d::<f64>(opij_slice, cart_slice, li, kappa_i, lj, kappa_j)?;
        }
    }

    // ── Step 2: transform (k,l) pair over the complex intermediate ──────────
    // The intermediate opij has shape [ncl * nck * dj * di] complex elements
    // For each spinor pair (i_sp, j_sp), apply bra-zf on k and ket on l.
    //
    // libcint c2s_sf_2e2: a_bra1_cart2spinor_zf for k, a_ket1_cart2spinor for l
    // The "1" variants have stride arguments, treating the (i,j) spinor block as columns.
    //
    // Output: staging[(((l_sp * dk + k_sp) * dj + j_sp) * di + i_sp) * 2]
    // We iterate: for each ij_sp in [0..di*dj], apply 2D transform to complex kl data.

    // Zero out staging
    for v in staging[..required].iter_mut() {
        *v = F::from_f64_lossy(0.0);
    }

    // For each (j_sp, i_sp) spinor pair from step 1, build a complex [nck * ncl] vector
    // and apply the 2D spinor transform (k,l) → (dk, dl) complex.
    // The opij buffer is indexed as: opij[((l_cart * nck + k_cart) * dj * di + j_sp * di + i_sp) * 2]
    // We want: for each (i_sp, j_sp) — a complex-valued [nck][ncl] "Cartesian" matrix.
    //
    // cart2spinor step 2 uses a_bra1_cart2spinor_zf (ZF = zero-field complex version)
    // which multiplies a complex input by a complex coefficient:
    //   out_R += cR * vR - cI * vI
    //   out_I += cR * vI + cI * vR
    // This differs from step 1's conjugate convention (saI += -caI * v1).

    let mut kl_re = vec![0.0f64; nck * ncl];
    let mut kl_im = vec![0.0f64; nck * ncl];
    let mut spinor_out_r = vec![0.0f64; dk * dl];
    let mut spinor_out_i = vec![0.0f64; dk * dl];

    for j_sp in 0..dj {
        for i_sp in 0..di {
            // Extract complex [nck * ncl] slice for this (i_sp, j_sp) pair
            for l_cart in 0..ncl {
                for k_cart in 0..nck {
                    let src_idx = ((l_cart * nck + k_cart) * dj * di + j_sp * di + i_sp) * 2;
                    kl_re[l_cart * nck + k_cart] = opij[src_idx];
                    kl_im[l_cart * nck + k_cart] = opij[src_idx + 1];
                }
            }

            // Apply bra-zf on k (2D transform with complex coefficients)
            // Then ket on l — both using complex multiply convention.
            // This mirrors apply_bra_sf (but complex input) then apply_ket.
            apply_2d_spinor_zf(
                &mut spinor_out_r, &mut spinor_out_i,
                &kl_re, &kl_im,
                nck, ncl, dk, dl, lk, kappa_k as i32, ll, kappa_l as i32,
            );

            // Store result: staging[(((l_sp * dk + k_sp) * dj + j_sp) * di + i_sp) * 2]
            // Cast from f64 intermediates to F at the output boundary.
            for l_sp in 0..dl {
                for k_sp in 0..dk {
                    let dst_idx = (((l_sp * dk + k_sp) * dj + j_sp) * di + i_sp) * 2;
                    staging[dst_idx] = F::from_f64_lossy(spinor_out_r[l_sp * dk + k_sp]);
                    staging[dst_idx + 1] = F::from_f64_lossy(spinor_out_i[l_sp * dk + k_sp]);
                }
            }
        }
    }

    Ok(())
}

/// Apply 2D spinor transform matching libcint `c2s_sf_2e2` step 2 algorithm.
///
/// Used in step 2 of `cart_to_spinor_sf_4d` where the input is complex (from
/// the step-1 spinor transform on the (i,j) pair).
///
/// Algorithm:
/// 1. `bra1_zf` on k: for each k_sp, apply `conj(coeff_k) * v` to each k_cart
///    independently for each l_cart column. Produces alpha and beta spinor-k
///    outputs: `tmp_alpha[k_sp, l_cart]` and `tmp_beta[k_sp, l_cart]`.
/// 2. `ket1` on l: for each l_sp, combine alpha and beta from step 1 using
///    `coeff_l_alpha * tmp_alpha + coeff_l_beta * tmp_beta`. Regular complex multiply.
///
/// This exactly mirrors `a_bra1_cart2spinor_zf` + `a_ket1_cart2spinor` in libcint.
#[allow(clippy::too_many_arguments)]
fn apply_2d_spinor_zf(
    out_r: &mut [f64],
    out_i: &mut [f64],
    kl_re: &[f64],
    kl_im: &[f64],
    nck: usize,
    ncl: usize,
    dk: usize,
    dl: usize,
    lk: u8, kappa_k: i32,
    ll: u8, kappa_l: i32,
) {
    // Zero output
    for v in out_r.iter_mut() { *v = 0.0; }
    for v in out_i.iter_mut() { *v = 0.0; }

    // Step 1: bra1_zf on k — produces alpha and beta k-spinor blocks.
    // tmp_alpha/beta: [dk * ncl] each — indexed as [k_sp * ncl + l_cart]
    let mut tmp_alpha_r = vec![0.0f64; dk * ncl];
    let mut tmp_alpha_i = vec![0.0f64; dk * ncl];
    let mut tmp_beta_r  = vec![0.0f64; dk * ncl];
    let mut tmp_beta_i  = vec![0.0f64; dk * ncl];

    let (coeff_k_gt_r, coeff_k_gt_i, coeff_k_lt_r, coeff_k_lt_i) = bra_coeff_refs(lk);
    apply_bra1_zf_block_all_kappa(
        &mut tmp_alpha_r, &mut tmp_alpha_i,
        &mut tmp_beta_r,  &mut tmp_beta_i,
        kl_re, kl_im,
        nck, ncl, dk, lk, kappa_k,
        coeff_k_gt_r, coeff_k_gt_i, coeff_k_lt_r, coeff_k_lt_i,
    );

    // Step 2: ket1 on l — combines alpha and beta, transforms l_cart → l_sp.
    let (coeff_l_gt_r, coeff_l_gt_i, coeff_l_lt_r, coeff_l_lt_i) = bra_coeff_refs(ll);
    apply_ket1_block_all_kappa(
        out_r, out_i,
        &tmp_alpha_r, &tmp_alpha_i,
        &tmp_beta_r,  &tmp_beta_i,
        dk, ncl, dl, ll, kappa_l,
        coeff_l_gt_r, coeff_l_gt_i, coeff_l_lt_r, coeff_l_lt_i,
    );
}

/// bra1_zf block: apply `conj(coeff) * v` to transform k_cart → k_spinor for each l_cart.
///
/// Mirrors `a_bra1_cart2spinor_zf` in libcint: for each k_sp row, multiply complex input
/// by CONJUGATE of the CG coefficient. Produces separate alpha and beta outputs.
///
/// Convention (conjugate multiply): `re += cR*vR + cI*vI`, `im += cR*vI - cI*vR`
/// (i.e., `conj(c) * v` not `c * v`).
///
/// Input `kl_re/im`: complex [ncl * nck] (l_cart outer, k_cart inner) — indexing [l*nck+k].
/// Output `alpha/beta_r/i`: [dk * ncl] (k_spinor outer, l_cart inner) — indexing [k_sp*ncl+l].
#[allow(clippy::too_many_arguments)]
fn apply_bra1_zf_block_all_kappa(
    alpha_r: &mut [f64],
    alpha_i: &mut [f64],
    beta_r:  &mut [f64],
    beta_i:  &mut [f64],
    kl_re: &[f64],
    kl_im: &[f64],
    nck: usize,
    ncl: usize,
    dk: usize,
    lk: u8,
    kappa_k: i32,
    coeff_gt_r: &[f64],
    coeff_gt_i: &[f64],
    coeff_lt_r: &[f64],
    coeff_lt_i: &[f64],
) {
    // Initialize outputs to zero
    for v in alpha_r.iter_mut() { *v = 0.0; }
    for v in alpha_i.iter_mut() { *v = 0.0; }
    for v in beta_r.iter_mut()  { *v = 0.0; }
    for v in beta_i.iter_mut()  { *v = 0.0; }

    if kappa_k < 0 {
        apply_bra1_zf_block(alpha_r, alpha_i, beta_r, beta_i, kl_re, kl_im, nck, ncl, dk,
                            coeff_gt_r, coeff_gt_i, 0);
    } else if kappa_k > 0 {
        apply_bra1_zf_block(alpha_r, alpha_i, beta_r, beta_i, kl_re, kl_im, nck, ncl, dk,
                            coeff_lt_r, coeff_lt_i, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        // Matches libcint ordering: a_bra1_cart2spinor_zf uses LT pointer which
        // over-reads into the GT table in memory for kappa=0.
        let nd_lt = 2 * lk as usize;
        let nd_gt = 2 * lk as usize + 2;
        if nd_lt > 0 {
            apply_bra1_zf_block(alpha_r, alpha_i, beta_r, beta_i, kl_re, kl_im, nck, ncl, nd_lt,
                                coeff_lt_r, coeff_lt_i, 0);
        }
        apply_bra1_zf_block(alpha_r, alpha_i, beta_r, beta_i, kl_re, kl_im, nck, ncl, nd_gt,
                            coeff_gt_r, coeff_gt_i, nd_lt);
    }
}

/// bra1_zf block for one kappa sub-block: conj(coeff) * v, maps k_cart → k_spinor per l_cart.
///
/// Mirrors libcint `a_bra1_cart2spinor_zf` for one (alpha or beta) sub-block.
/// Convention: `re += cR*vR + cI*vI`, `im += cR*vI - cI*vR` (conjugate of coeff).
///
/// Input `kl_re/im`: [ncl * nck] complex (l_cart outer, k_cart inner).
/// Output `alpha/beta_r/i`: [dk_total * ncl] (k_spinor outer, l_cart inner), k_spinor written
///   starting at row_off, for nd k-spinor rows.
///   Index: `alpha_r[k_sp * ncl + l_cart]` for k_sp in [row_off..row_off+nd].
#[allow(clippy::too_many_arguments)]
fn apply_bra1_zf_block(
    alpha_r: &mut [f64],
    alpha_i: &mut [f64],
    beta_r:  &mut [f64],
    beta_i:  &mut [f64],
    kl_re: &[f64],
    kl_im: &[f64],
    nck: usize,
    ncl: usize,
    nd: usize,
    coeff_r: &[f64],
    coeff_i: &[f64],
    row_off: usize,
) {
    // dk_total = alpha_r.len() / ncl  (total k_spinor rows)
    let dk_total = if ncl > 0 { alpha_r.len() / ncl } else { 0 };
    for l_cart in 0..ncl {
        for k_sp in 0..nd {
            let out_idx = (row_off + k_sp) * ncl + l_cart;
            let mut sa_r = 0.0f64;
            let mut sa_i = 0.0f64;
            let mut sb_r = 0.0f64;
            let mut sb_i = 0.0f64;
            for n in 0..nck {
                // coeff layout: [k_sp][2*nck] where first nck = alpha, next nck = beta
                let ca_r = coeff_r[k_sp * 2 * nck + n];
                let ca_i = coeff_i[k_sp * 2 * nck + n];
                let cb_r = coeff_r[k_sp * 2 * nck + nck + n];
                let cb_i = coeff_i[k_sp * 2 * nck + nck + n];
                let vr = kl_re[l_cart * nck + n];
                let vi = kl_im[l_cart * nck + n];
                // Conjugate multiply: conj(c) * v = (cR*vR + cI*vI) + i*(cR*vI - cI*vR)
                sa_r += ca_r * vr + ca_i * vi;
                sa_i += ca_r * vi - ca_i * vr;
                sb_r += cb_r * vr + cb_i * vi;
                sb_i += cb_r * vi - cb_i * vr;
            }
            alpha_r[out_idx] = sa_r;
            alpha_i[out_idx] = sa_i;
            beta_r[out_idx]  = sb_r;
            beta_i[out_idx]  = sb_i;
        }
    }
    let _ = dk_total; // suppress warning if unused
}

/// ket1 transform: combine alpha and beta from bra1_zf output, transform l_cart → l_sp.
///
/// Mirrors libcint `a_ket1_cart2spinor`: for each l_sp, sum over l_cart:
///   `out += coeff_alpha[l_sp, l_cart] * alpha[k_sp, l_cart] + coeff_beta[l_sp, l_cart] * beta[k_sp, l_cart]`
/// Uses regular complex multiply for both terms.
///
/// Input `alpha/beta_r/i`: [dk * ncl] (k_sp outer, l_cart inner) — index [k_sp*ncl + l_cart].
/// Output `out_r/i`: [dl * dk] (l_sp outer, k_sp inner) — index [(row_off+l_sp)*dk + k_sp].
#[allow(clippy::too_many_arguments)]
fn apply_ket1_block_all_kappa(
    out_r: &mut [f64],
    out_i: &mut [f64],
    alpha_r: &[f64],
    alpha_i: &[f64],
    beta_r:  &[f64],
    beta_i:  &[f64],
    dk: usize,
    ncl: usize,
    dl: usize,
    ll: u8,
    kappa_l: i32,
    coeff_gt_r: &[f64],
    coeff_gt_i: &[f64],
    coeff_lt_r: &[f64],
    coeff_lt_i: &[f64],
) {
    // Zero output
    for v in out_r.iter_mut() { *v = 0.0; }
    for v in out_i.iter_mut() { *v = 0.0; }
    if kappa_l < 0 {
        apply_ket1_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i, dk, ncl, dl,
                         coeff_gt_r, coeff_gt_i, 0);
    } else if kappa_l > 0 {
        apply_ket1_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i, dk, ncl, dl,
                         coeff_lt_r, coeff_lt_i, 0);
    } else {
        // kappa == 0: LT first (rows 0..nd_lt), GT second (rows nd_lt..nd).
        // Matches libcint ordering: a_ket1_cart2spinor uses LT pointer which
        // over-reads into the GT table in memory for kappa=0.
        let nd_lt = 2 * ll as usize;
        let nd_gt = 2 * ll as usize + 2;
        if nd_lt > 0 {
            apply_ket1_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i, dk, ncl, nd_lt,
                             coeff_lt_r, coeff_lt_i, 0);
        }
        apply_ket1_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i, dk, ncl, nd_gt,
                         coeff_gt_r, coeff_gt_i, nd_lt);
    }
}

/// ket1 block for one kappa sub-block: combines alpha+beta, transforms l_cart → l_spinor.
///
/// For each l_sp and k_sp:
///   out += ca*alpha[k_sp, l_cart] + cb*beta[k_sp, l_cart]  (regular c*v multiply)
///
/// Coefficients are [nd][2*ncl]: rows 0..nd for l-spinor rows, columns: first ncl = alpha, next ncl = beta.
#[allow(clippy::too_many_arguments)]
fn apply_ket1_block(
    out_r: &mut [f64],
    out_i: &mut [f64],
    alpha_r: &[f64],
    alpha_i: &[f64],
    beta_r:  &[f64],
    beta_i:  &[f64],
    dk: usize,
    ncl: usize,
    nd: usize,
    coeff_r: &[f64],
    coeff_i: &[f64],
    row_off: usize,
) {
    for l_sp in 0..nd {
        for k_sp in 0..dk {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for n in 0..ncl {
                // Coefficient columns: first ncl = alpha, next ncl = beta
                let ca_r = coeff_r[l_sp * 2 * ncl + n];
                let ca_i = coeff_i[l_sp * 2 * ncl + n];
                let cb_r = coeff_r[l_sp * 2 * ncl + ncl + n];
                let cb_i = coeff_i[l_sp * 2 * ncl + ncl + n];
                let ga_r = alpha_r[k_sp * ncl + n];
                let ga_i = alpha_i[k_sp * ncl + n];
                let gb_r = beta_r[k_sp * ncl + n];
                let gb_i = beta_i[k_sp * ncl + n];
                // Regular complex multiply: ca*ga + cb*gb
                re += ca_r * ga_r - ca_i * ga_i + cb_r * gb_r - cb_i * gb_i;
                im += ca_r * ga_i + ca_i * ga_r + cb_r * gb_i + cb_i * gb_r;
            }
            out_r[(row_off + l_sp) * dk + k_sp] += re;
            out_i[(row_off + l_sp) * dk + k_sp] += im;
        }
    }
}

/// Full 3D cart-to-spinor transform for 3c2e integrals.
///
/// Implements libcint `c2s_sf_3c2e1`: sph transform on auxiliary k, then
/// spinor bra+ket transform on (i, j).
///
/// Algorithm:
/// 1. Apply cart-to-sph on k-index: `cart[nci * ncj * nck]` → `tmp[nci * ncj * nsk]`.
/// 2. Apply bra spinor transform on i-index (over nsk * ncj "columns").
/// 3. Apply ket spinor transform on j-index.
/// 4. Store as column-major interleaved: `staging[(k_sph * dj * di + j_sp * di + i_sp) * 2]`.
///
/// # Parameters
/// - `staging`: output buffer, size `di * dj * nsk * 2` (nsk = 2*lk+1 spherical k components)
/// - `cart`: Cartesian input `[nck * ncj * nci]` (k outermost, i innermost)
/// - `li`, `kappa_i`: bra shell angular momentum and kappa
/// - `lj`, `kappa_j`: ket shell angular momentum and kappa
/// - `lk`: auxiliary shell angular momentum (no kappa — transforms to spherical)
pub fn cart_to_spinor_sf_3c2e<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    li: u8, kappa_i: i16,
    lj: u8, kappa_j: i16,
    lk: u8,
) -> Result<(), cintxRsError> {
    use super::c2s::{ncart, nsph};

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    let nsk = nsph(lk);

    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);

    let expected_cart = nci * ncj * nck;
    if cart.len() < expected_cart {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "c2spinor_sf_3c2e",
            detail: format!(
                "cart buffer length {} < nci*ncj*nck = {}*{}*{} = {}",
                cart.len(), nci, ncj, nck, expected_cart
            ),
        });
    }
    let required = di * dj * nsk * 2;
    if staging.len() < required {
        return Err(cintxRsError::BufferTooSmall {
            required,
            provided: staging.len(),
        });
    }

    // ── Step 1: cart-to-sph on k-index ──────────────────────────────────────
    // Input: cart[nck * ncj * nci] (k outermost, i innermost)
    // Output: sph_k[nsk * ncj * nci]
    // For each (j, i) pair, apply c2s on k-axis.
    let mut sph_k = vec![0.0f64; nsk * ncj * nci];
    for j in 0..ncj {
        for i in 0..nci {
            for mk in 0..nsk {
                let mut sum = 0.0f64;
                for ck in 0..nck {
                    let cart_idx = (ck * ncj + j) * nci + i;
                    sum += c2s_k_coeff(lk, mk, ck) * cart[cart_idx];
                }
                sph_k[(mk * ncj + j) * nci + i] = sum;
            }
        }
    }

    // ── Step 2+3: apply 2D spinor transform (i,j) for each k_sph slice ────
    // Input per k: sph_k[(mk * ncj + j) * nci + i] — KET-major (k outer, j=ket middle,
    // i=bra inner). `cart_to_spinor_sf_2d` reads its cart input BRA-major as
    // `cart[bra_n * ncj + ket_j]` (apply_bra_block L693). So the per-k slice must be
    // transposed KET→BRA before the spin-free fold. This is the D-06 orientation
    // transpose (it lives in the transform layer, never the launcher); it was latent
    // because no NON-SQUARE 3c2e spinor block (nci != ncj) was ever exercised — for a
    // square block the two layouts coincide and the bug is invisible (27-04).
    let mut bra_major = vec![0.0f64; ncj * nci];
    for mk in 0..nsk {
        let slice_start = mk * ncj * nci;
        let cart_slice = &sph_k[slice_start..slice_start + ncj * nci];
        // ket-major sph_k[j*nci + i]  →  bra-major bra_major[i*ncj + j]
        for j in 0..ncj {
            for i in 0..nci {
                bra_major[i * ncj + j] = cart_slice[j * nci + i];
            }
        }
        let staging_start = mk * di * dj * 2;
        let staging_slice = &mut staging[staging_start..staging_start + di * dj * 2];
        cart_to_spinor_sf_2d(staging_slice, &bra_major, li, kappa_i, lj, kappa_j)?;
    }

    Ok(())
}

/// Retrieve a single cart-to-sph coefficient for the k auxiliary index transform.
fn c2s_k_coeff(l: u8, m_row: usize, cart_col: usize) -> f64 {
    use super::c2s::{C2S_L0, C2S_L1, C2S_L2, C2S_L3, C2S_L4};
    match l {
        0 => C2S_L0[m_row][cart_col],
        1 => C2S_L1[m_row][cart_col],
        2 => C2S_L2[m_row][cart_col],
        3 => C2S_L3[m_row][cart_col],
        4 => C2S_L4[m_row][cart_col],
        _ => 0.0,
    }
}

/// Derivative (multi-component) spin-free cart→spinor transform for arity-2 families
/// (1e gradients/Hessians: ipovlp, ipkin, ipnuc, iprinv, and their higher-order siblings).
///
/// This is the SINGLE audited place that owns the KET→BRA orientation transpose (D-06).
/// No launcher may own that transpose again — the scalar-spinor orientation bug
/// (project memory `1e family fully on-device + spinor orientation fixed`) came from a
/// launcher doing it incorrectly on a square block. Here it is centralized and regression-
/// anchored to a NON-SQUARE p×d block.
///
/// # Layout contract
/// - `cart`: device-native, component-leading, KET-major per (comp, contraction) sub-block:
///   `cart[(ci*nctr_j + cj)*total_len + comp*block_len + jc*nci + ic]`, `total_len = ncomp*block_len`,
///   `block_len = nci*ncj` (matches the cart/sph nctr>1 scatter in one_electron.rs L9897-9916).
/// - `staging`: component-outer interleaved-complex spinor output:
///   `staging[comp*spinor_block + (j_global*ni_full + i_global)*2 + {0:re, 1:im}]`,
///   `spinor_block = ni_full*nj_full*2`, `ni_full = nctr_i*di`, `nj_full = nctr_j*dj`,
///   `di = spinor_len(li, kappa_i)`, `dj = spinor_len(lj, kappa_j)`.
/// - nctr>1 composes contraction-MAJOR: `i_global = ci*di + ic`, `j_global = cj*dj + jc`
///   (D-08 / spike D4). The env coeff column→row transpose lives in the launcher, not here;
///   this wrapper consumes the already-emitted device cart blocks, so no coefficient transpose
///   leaks to the output.
///
/// # Fail-closed (FND-06)
/// Sizes are checked ONCE upfront from `ncomp`/`nctr_*`; on mismatch a typed error is returned
/// BEFORE any write. There are NO `if dst < staging.len()` scatter guards (monolithic-writer
/// contract).
#[allow(clippy::too_many_arguments)]
pub fn cart_to_spinor_sf_derivative_2d<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    ncomp: usize,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    nctr_i: usize,
    nctr_j: usize,
) -> Result<(), cintxRsError> {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let block_len = nci * ncj;
    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_full = nctr_i * di;
    let nj_full = nctr_j * dj;
    let spinor_block = ni_full * nj_full * 2; // D-07 component-outer stride
    let total_len = ncomp * block_len; // per (ci,cj) component-leading cart extent

    // ── FAIL-CLOSED upfront (FND-06): size-check once, then write unconditionally ──
    let cart_required = ncomp * block_len * nctr_i * nctr_j;
    if cart.len() < cart_required {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "c2spinor_sf_derivative_2d",
            detail: format!(
                "cart buffer length {} < ncomp*block_len*nctr_i*nctr_j = {}*{}*{}*{} = {}",
                cart.len(), ncomp, block_len, nctr_i, nctr_j, cart_required
            ),
        });
    }
    let staging_required = ncomp * spinor_block;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // Scratch reused across (comp, ci, cj) iterations.
    let mut block_bra_major = vec![0.0f64; block_len];
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * 2];

    for comp in 0..ncomp {
        let comp_base = comp * spinor_block;
        for ci in 0..nctr_i {
            for cj in 0..nctr_j {
                // Device-native KET-major sub-block for this (comp, ci, cj).
                let src_base = (ci * nctr_j + cj) * total_len + comp * block_len;
                let block = &cart[src_base..src_base + block_len];

                // D-06: KET→BRA transpose so cart_to_spinor_sf_2d reads bra-major.
                for ic in 0..nci {
                    for jc in 0..ncj {
                        block_bra_major[ic * ncj + jc] = block[jc * nci + ic];
                    }
                }

                cart_to_spinor_sf_2d::<F>(
                    &mut scratch, &block_bra_major, li, kappa_i, lj, kappa_j,
                )?;

                // Scatter the di*dj*2 spinor block into the contraction-major position.
                // scratch is column-major: scratch[(j*di + i)*2 + {re,im}].
                for j in 0..dj {
                    let j_global = cj * dj + j;
                    for i in 0..di {
                        let i_global = ci * di + i;
                        let src = (j * di + i) * 2;
                        let dst = comp_base + (j_global * ni_full + i_global) * 2;
                        staging[dst] = scratch[src];
                        staging[dst + 1] = scratch[src + 1];
                    }
                }
            }
        }
    }

    Ok(())
}

/// Internal: shared per-(comp,k) `[ket][bra]` spherical-aux-k derivative fold for arity-3
/// families. Both the 3c2e wrapper and the int3c1e thin sibling delegate here — they consume
/// the SAME device/host cart layout family (`[comp][k][ket][bra]`, KET-major bra-fastest) and
/// fold it the SAME way; only the producing code path differs (D-11 / D3).
///
/// AUX-K is SPHERICAL: `nsk = nsph(lk) = (2lk+1)` (libcint `CINT3c2e_spinor_drv` is_ssc=0,
/// cint3c2e.c:631-636). Only bra i and ket j are spinor-sized (4l+2). NEVER reconcile aux-k up
/// to `CINTcgto_spinor` — that produced the disproven 720 (27-SPIKE-FINDINGS ⚠ CORRECTION NOTICE).
///
/// Fail-closed (FND-06): sizes checked ONCE upfront; no `if dst < len` scatter guards.
#[allow(clippy::too_many_arguments)]
fn cart_to_spinor_sf_derivative_3c_impl<F: CintFloat>(
    from: &'static str,
    staging: &mut [F],
    cart: &[f64],
    ncomp: usize,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    lk: u8,
    nctr_i: usize,
    nctr_j: usize,
) -> Result<(), cintxRsError> {
    use super::c2s::nsph;

    let nci = ncart(li);
    let ncj = ncart(lj);
    let nck = ncart(lk);
    // SPHERICAL aux-k — IDENTICAL to what the inner cart_to_spinor_sf_3c2e computes (L1293).
    let nsk = nsph(lk);
    let di = spinor_len(li, kappa_i as i32);
    let dj = spinor_len(lj, kappa_j as i32);
    let ni_full = nctr_i * di;
    let nj_full = nctr_j * dj;
    let kblock = nck * ncj * nci; // per-component cart extent for one (ci,cj) sub-block
    let total_len = ncomp * kblock;
    // Component-outer stride includes the SPHERICAL k axis (the SAME nsk the inner uses).
    let comp_stride = ni_full * nj_full * nsk * 2;

    // ── FAIL-CLOSED upfront (FND-06) ──
    let cart_required = ncomp * kblock * nctr_i * nctr_j;
    if cart.len() < cart_required {
        return Err(cintxRsError::ChunkPlanFailed {
            from,
            detail: format!(
                "cart buffer length {} < ncomp*nci*ncj*nck*nctr_i*nctr_j = {}*{}*{}*{}*{}*{} = {}",
                cart.len(), ncomp, nci, ncj, nck, nctr_i, nctr_j, cart_required
            ),
        });
    }
    let staging_required = ncomp * comp_stride;
    if staging.len() < staging_required {
        return Err(cintxRsError::BufferTooSmall {
            required: staging_required,
            provided: staging.len(),
        });
    }

    // Scratch for one (comp,ci,cj) inner 3c2e fold: di*dj*nsk*2 (the inner's required size).
    let mut scratch = vec![F::from_f64_lossy(0.0); di * dj * nsk * 2];

    for comp in 0..ncomp {
        let comp_base = comp * comp_stride;
        for ci in 0..nctr_i {
            for cj in 0..nctr_j {
                let src_base = (ci * nctr_j + cj) * total_len + comp * kblock;
                let block = &cart[src_base..src_base + kblock];

                // Inner transform owns the per-(comp,k) cart→sph(k) + KET→BRA + sf_2d fold.
                cart_to_spinor_sf_3c2e::<F>(
                    &mut scratch, block, li, kappa_i, lj, kappa_j, lk,
                )?;

                // scratch layout (per inner): scratch[mk*di*dj*2 + (j*di + i)*2 + {re,im}].
                // Scatter into contraction-major output with the SPHERICAL k axis preserved.
                for mk in 0..nsk {
                    for j in 0..dj {
                        let j_global = cj * dj + j;
                        for i in 0..di {
                            let i_global = ci * di + i;
                            let src = (mk * di * dj + j * di + i) * 2;
                            let dst = comp_base
                                + (mk * ni_full * nj_full + j_global * ni_full + i_global) * 2;
                            staging[dst] = scratch[src];
                            staging[dst + 1] = scratch[src + 1];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Derivative (multi-component) spin-free cart→spinor transform for arity-3 2-electron
/// families (`int3c2e_ip1_spinor` / `int3c2e_ip2_spinor`; ip1 and ip2 share this shape).
///
/// Loops the already-byte-identity-proven inner `cart_to_spinor_sf_3c2e` `ncomp` times with
/// `comp_stride = ni_full*nj_full*nsph(lk)*2`. The KET→BRA transpose and the cart→sph(k) fold
/// live inside that inner transform (per-(comp,k) granularity, D-11 spike).
///
/// AUX-K IS SPHERICAL `nsph(lk) = (2lk+1)*nctr_k` — never `CINTcgto_spinor`. For p×d×s kappa=0
/// nctr=1 ncomp=3 the buffer is 360, NOT 720 (27-SPIKE-FINDINGS ⚠ CORRECTION NOTICE).
///
/// nctr>1 composes contraction-major on bra i / ket j only (`i_global = ci*di + ic`); the aux-k
/// stays a single spherical axis. Fail-closed (FND-06): sizes checked upfront, no scatter guards.
#[allow(clippy::too_many_arguments)]
pub fn cart_to_spinor_sf_derivative_3c2e<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    ncomp: usize,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    lk: u8,
    nctr_i: usize,
    nctr_j: usize,
) -> Result<(), cintxRsError> {
    cart_to_spinor_sf_derivative_3c_impl::<F>(
        "c2spinor_sf_derivative_3c2e",
        staging, cart, ncomp, li, kappa_i, lj, kappa_j, lk, nctr_i, nctr_j,
    )
}

/// THIN SIBLING (D3 decision): derivative spin-free cart→spinor transform for the
/// `int3c1e_ip1` / `int3c1e_iprinv` spinor gradients. NOT the shared `_3c2e` wrapper because
/// the int3c1e launcher produces its own host-side `out_buf` (a DIFFERENT code path:
/// host scatter, not the device kernel + `cart_to_spinor_sf_3c2e`). The fold math is identical
/// (per-(comp,k) `[ket][bra]`, SPHERICAL aux-k), so this sibling delegates to the same
/// internal implementation while keeping the 3c2e wrapper's device-cart precondition decoupled
/// from the 3c1e host scatter.
///
/// AUX-K IS SPHERICAL `nsph(lk)` (int3c1e_spinor sizes aux-k spherically exactly as int3c2e does);
/// never `CINTcgto_spinor`. Fail-closed (FND-06). iprinv differs only in the gout (reads
/// `env[PTR_RINV_ORIG]`) — that lives in the launcher, not here.
#[allow(clippy::too_many_arguments)]
pub fn cart_to_spinor_sf_derivative_3c1e<F: CintFloat>(
    staging: &mut [F],
    cart: &[f64],
    ncomp: usize,
    li: u8,
    kappa_i: i16,
    lj: u8,
    kappa_j: i16,
    lk: u8,
    nctr_i: usize,
    nctr_j: usize,
) -> Result<(), cintxRsError> {
    cart_to_spinor_sf_derivative_3c_impl::<F>(
        "c2spinor_sf_derivative_3c1e",
        staging, cart, ncomp, li, kappa_i, lj, kappa_j, lk, nctr_i, nctr_j,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn check_close(a: f64, b: f64, label: &str) {
        assert!(
            (a - b).abs() < TOL,
            "{}: got {:.15e}, expected {:.15e}, diff={:.3e}",
            label, a, b, (a - b).abs()
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  apply_bra_si_block: a_bra_cart2spinor_si sign convention (THE landmine)
    // ──────────────────────────────────────────────────────────────────────────

    /// Hand-derived l=1 (nf=3), kappa=-1 (GT, nd=4), single ket column (ncj=1).
    ///
    /// Fills gc_1/gc_x/gc_y/gc_z with 4 distinct cart vectors, pulls the CG coeffs
    /// CJ_GT_L1_R/I, and computes (saR,saI,sbR,sbI) by an INDEPENDENT transcription of
    /// the `a_bra_cart2spinor_si` equations (cart2sph.c:3958-3961), asserting the
    /// `apply_bra_si_block` output matches to 1e-14. Also guards that the output is NOT
    /// what `apply_si_block` (the WRONG single-block sign convention) would produce.
    #[test]
    fn apply_bra_si_block_l1_kappa_neg1_hand_derived() {
        let li: u8 = 1;
        let nci = ncart(li); // 3
        let ncj = 1usize;
        let di = spinor_len(li, -1); // GT → 4
        assert_eq!(nci, 3);
        assert_eq!(di, 4);

        // Four distinct cart vectors (length nci*ncj = 3), BRA-major (ncj=1).
        let gc_1 = [1.0, 2.0, 3.0];
        let gc_x = [0.5, -1.5, 2.5];
        let gc_y = [-0.25, 0.75, -1.25];
        let gc_z = [4.0, -2.0, 1.0];

        // CG coeffs flattened: row i has [alpha(0..nci), beta(nci..2nci)] = 6 entries.
        let coeff_r = cj::CJ_GT_L1_R.as_flattened();
        let coeff_i = cj::CJ_GT_L1_I.as_flattened();

        // ── Independent reference using the a_bra_cart2spinor_si equations ──────
        let mut ref_ar = vec![0.0f64; di * ncj];
        let mut ref_ai = vec![0.0f64; di * ncj];
        let mut ref_br = vec![0.0f64; di * ncj];
        let mut ref_bi = vec![0.0f64; di * ncj];
        for j in 0..ncj {
            for i in 0..di {
                let mut sa_r = 0.0;
                let mut sa_i = 0.0;
                let mut sb_r = 0.0;
                let mut sb_i = 0.0;
                for n in 0..nci {
                    let v1 = gc_1[n * ncj + j];
                    let vx = gc_x[n * ncj + j];
                    let vy = gc_y[n * ncj + j];
                    let vz = gc_z[n * ncj + j];
                    let ca_r = coeff_r[i * 2 * nci + n];
                    let ca_i = coeff_i[i * 2 * nci + n];
                    let cb_r = coeff_r[i * 2 * nci + nci + n];
                    let cb_i = coeff_i[i * 2 * nci + nci + n];
                    sa_r += ca_r * v1 + ca_i * vz - cb_r * vy + cb_i * vx;
                    sa_i += -ca_i * v1 + ca_r * vz + cb_i * vy + cb_r * vx;
                    sb_r += cb_r * v1 - cb_i * vz + ca_r * vy + ca_i * vx;
                    sb_i += -cb_i * v1 - cb_r * vz - ca_i * vy + ca_r * vx;
                }
                let idx = j * di + i;
                ref_ar[idx] = sa_r;
                ref_ai[idx] = sa_i;
                ref_br[idx] = sb_r;
                ref_bi[idx] = sb_i;
            }
        }

        // ── Function under test ────────────────────────────────────────────────
        let mut a_r = vec![0.0f64; di * ncj];
        let mut a_i = vec![0.0f64; di * ncj];
        let mut b_r = vec![0.0f64; di * ncj];
        let mut b_i = vec![0.0f64; di * ncj];
        apply_bra_si_block(
            &mut a_r, &mut a_i, &mut b_r, &mut b_i,
            &gc_x, &gc_y, &gc_z, &gc_1, nci, ncj, di, li, -1,
        );

        for idx in 0..di * ncj {
            assert!((a_r[idx] - ref_ar[idx]).abs() < 1e-14, "alpha_r[{idx}]");
            assert!((a_i[idx] - ref_ai[idx]).abs() < 1e-14, "alpha_i[{idx}]");
            assert!((b_r[idx] - ref_br[idx]).abs() < 1e-14, "beta_r[{idx}]");
            assert!((b_i[idx] - ref_bi[idx]).abs() < 1e-14, "beta_i[{idx}]");
        }

        // ── Sign-discrepancy guard: bra-si output ≠ apply_si_block output ───────
        // apply_si_block uses the WRONG (CINTc2s_ket_spinor_si1) sign convention for
        // this 2D path. Build a (coeff-row-major) reference and prove they differ.
        let coeff_r_rows: Vec<&[f64]> = (0..di)
            .map(|i| &coeff_r[i * 2 * nci..(i + 1) * 2 * nci])
            .collect();
        let coeff_i_rows: Vec<&[f64]> = (0..di)
            .map(|i| &coeff_i[i * 2 * nci..(i + 1) * 2 * nci])
            .collect();
        // apply_si_block expects per-block cart vectors of length nf (=nci), single column.
        let mut si_out = vec![0.0f64; 4 * di];
        apply_si_block(
            &mut si_out, &gc_1, &gc_x, &gc_y, &gc_z,
            &coeff_r_rows, &coeff_i_rows, di, nci, di, 0,
        );
        // apply_si_block writes interleaved re/im: alpha rows [0..di], beta rows [di..2di].
        let mut any_diff = false;
        for i in 0..di {
            let si_alpha_r = si_out[i * 2];
            if (si_alpha_r - ref_ar[i]).abs() > 1e-12 {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "apply_bra_si_block must NOT match apply_si_block's (wrong) sign convention"
        );
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  cart_to_spinor_si_2d: sizing, fail-closed guards, non-square round-trip
    // ──────────────────────────────────────────────────────────────────────────

    /// kappa≠0 sizing the si_2d transform must honor (Spike Target D).
    #[test]
    fn si_2d_spinor_len_kappa_nonzero() {
        assert_eq!(spinor_len(1, 1), 2); // p LT (j=1/2) → 2*l
        assert_eq!(spinor_len(2, -1), 6); // d GT (j=5/2) → 2*l+2
    }

    /// Undersized staging → BufferTooSmall with required == di*dj*2; no partial writes.
    #[test]
    fn si_2d_undersized_staging_fails_closed() {
        // p(kappa=-1, di=4) × d(kappa=-1, dj=6): required = 4*6*2 = 48.
        let li = 1u8;
        let lj = 2u8;
        let nci = ncart(li);
        let ncj = ncart(lj);
        let gc = vec![1.0f64; nci * ncj];
        let mut staging = vec![f64::NAN; 10]; // too small
        let sentinel = staging.clone();
        let err = cart_to_spinor_si_2d::<f64>(
            &mut staging, &gc, &gc, &gc, &gc, li, -1, lj, -1,
        )
        .unwrap_err();
        match err {
            cintxRsError::BufferTooSmall { required, provided } => {
                assert_eq!(required, 4 * 6 * 2);
                assert_eq!(provided, 10);
            }
            other => panic!("expected BufferTooSmall, got {other:?}"),
        }
        // No partial writes: staging untouched.
        for (a, b) in staging.iter().zip(sentinel.iter()) {
            assert!(a.is_nan() && b.is_nan());
        }
    }

    /// Undersized cart block → ChunkPlanFailed; no partial writes.
    #[test]
    fn si_2d_undersized_cart_fails_closed() {
        let li = 1u8;
        let lj = 2u8;
        let nci = ncart(li);
        let ncj = ncart(lj);
        let full = vec![1.0f64; nci * ncj];
        let short = vec![1.0f64; nci * ncj - 1]; // gc_y too small
        let di = spinor_len(li, -1);
        let dj = spinor_len(lj, -1);
        let mut staging = vec![f64::NAN; di * dj * 2];
        let err = cart_to_spinor_si_2d::<f64>(
            &mut staging, &full, &short, &full, &full, li, -1, lj, -1,
        )
        .unwrap_err();
        assert!(matches!(err, cintxRsError::ChunkPlanFailed { from: "c2spinor_si_2d", .. }));
        // No partial writes.
        assert!(staging.iter().all(|v| v.is_nan()));
    }

    /// Non-square p(kappa=-1)×d(kappa=-1) round-trip: finite output, length di*dj*2.
    #[test]
    fn si_2d_nonsquare_pd_roundtrip_finite() {
        let li = 1u8;
        let lj = 2u8;
        let nci = ncart(li); // 3
        let ncj = ncart(lj); // 6
        let di = spinor_len(li, -1); // 4
        let dj = spinor_len(lj, -1); // 6
        // Distinct, non-symmetric ket-major cart blocks to exercise the transpose.
        let mk = |seed: f64| -> Vec<f64> {
            (0..nci * ncj).map(|n| seed + n as f64 * 0.137).collect()
        };
        let gc_x = mk(0.3);
        let gc_y = mk(-0.7);
        let gc_z = mk(1.1);
        let gc_1 = vec![0.0f64; nci * ncj]; // int1e_sp scalar slot is zero
        let mut staging = vec![0.0f64; di * dj * 2];
        cart_to_spinor_si_2d::<f64>(
            &mut staging, &gc_x, &gc_y, &gc_z, &gc_1, li, -1, lj, -1,
        )
        .expect("si_2d should succeed on well-sized buffers");
        assert_eq!(staging.len(), di * dj * 2);
        assert!(staging.iter().all(|v| v.is_finite()));
        // Output must be non-trivial (σ-mix of non-zero Pauli blocks).
        assert!(staging.iter().any(|&v| v.abs() > 1e-9));
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  spinor_len dispatch tests
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn spinor_len_l0_kappa_neg1() {
        assert_eq!(spinor_len(0, -1), 2); // 2*0+2
    }

    #[test]
    fn spinor_len_l1_kappa_neg1() {
        assert_eq!(spinor_len(1, -1), 4); // 2*1+2 (gt, j=3/2)
    }

    #[test]
    fn spinor_len_l1_kappa_pos1() {
        assert_eq!(spinor_len(1, 1), 2); // 2*1 (lt, j=1/2)
    }

    #[test]
    fn spinor_len_l1_kappa_0() {
        assert_eq!(spinor_len(1, 0), 6); // 4*1+2 = 6
    }

    #[test]
    fn spinor_len_l2_kappa_neg1() {
        assert_eq!(spinor_len(2, -1), 6); // 2*2+2
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  s-shell (l=0) sf value-correctness tests
    // ──────────────────────────────────────────────────────────────────────────

    /// s-shell (l=0), kappa=-1, cart=[1.0]:
    ///
    /// CJ_GT_L0_R = [[0, 1], [1, 0]], CJ_GT_L0_I = [[0, 0], [0, 0]]
    /// Row 0: ca_r=0, ca_i=0, cb_r=1, cb_i=0
    ///   sa_re = ca_r*v1 = 0, sa_im = ca_i*v1 = 0
    ///   sb_re = cb_r*v1 = 1, sb_im = cb_i*v1 = 0
    /// Row 1: ca_r=1, ca_i=0, cb_r=0, cb_i=0
    ///   sa_re = 1, sa_im = 0
    ///   sb_re = 0, sb_im = 0
    ///
    /// nd=2, total output = 4*nd = 8 f64
    /// gsp[0..3] = alpha: [row0_re, row0_im, row1_re, row1_im] = [0, 0, 1, 0]
    /// gsp[4..7] = beta:  [row0_re, row0_im, row1_re, row1_im] = [1, 0, 0, 0]
    #[test]
    fn sf_s_shell_kappa_neg1_cart_one() {
        let cart = [1.0f64];
        let nd = spinor_len(0, -1); // 2
        let mut gsp = vec![0.0f64; 4 * nd]; // 8
        cart_to_spinor_sf(&mut gsp, &cart, 0, -1).expect("sf s-shell kappa=-1 should succeed");
        // alpha component
        check_close(gsp[0], 0.0, "alpha[0] re (row 0)");
        check_close(gsp[1], 0.0, "alpha[0] im (row 0)");
        check_close(gsp[2], 1.0, "alpha[1] re (row 1)");
        check_close(gsp[3], 0.0, "alpha[1] im (row 1)");
        // beta component (offset by nd=2 complex values = 4 f64)
        check_close(gsp[4], 1.0, "beta[0] re (row 0)");
        check_close(gsp[5], 0.0, "beta[0] im (row 0)");
        check_close(gsp[6], 0.0, "beta[1] re (row 1)");
        check_close(gsp[7], 0.0, "beta[1] im (row 1)");
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  p-shell (l=1) sf value-correctness tests
    // ──────────────────────────────────────────────────────────────────────────

    /// p-shell (l=1), kappa=-1 (gt block), cart=[1,0,0] (px only).
    ///
    /// GT row 0: row_r=[0,0,0, 0.7071...,0,0], row_i=[0,0,0,0,-0.7071..,0]
    ///   only n=0 contributes (v1=1 at n=0): ca_r=0, ca_i=0, cb_r=0.7071, cb_i=0
    ///   sa_re=0, sa_im=0, sb_re=0.7071, sb_im=0
    /// GT row 1: row_r=[0.4082,0,0, 0,0,0.8165], row_i=[0,-0.4082,0, 0,0,0]
    ///   n=0: ca_r=0.4082, ca_i=0, cb_r=0 (row_r[3]=0), ...
    ///   Wait: for l=1 nf=3 so row[0..3]=alpha, row[3..6]=beta
    ///   GT row 1 = [0.408248.., 0, 0,  0, 0, 0.816496..]
    ///     alpha half: [0.408248, 0, 0], beta half: [0, 0, 0.816496]
    ///   n=0: ca_r=0.408248, ca_i=0 (row_i[1][0]=0), cb_r=0, cb_i=0
    ///   sa_re=0.408248, sa_im=0, sb_re=0, sb_im=0
    #[test]
    fn sf_p_shell_kappa_neg1_cart_px() {
        let cart = [1.0f64, 0.0, 0.0]; // px=1, py=0, pz=0
        let nd = spinor_len(1, -1); // 4
        let mut gsp = vec![0.0f64; 4 * nd]; // 16
        cart_to_spinor_sf(&mut gsp, &cart, 1, -1).expect("sf p-shell kappa=-1");

        // GT row 0: alpha half [0,0,0], beta half [0.7071,0,0]
        // n=0 only: ca_r=0, ca_i=0, cb_r=0.7071, cb_i=0 => sa_re=0,sa_im=0,sb_re=0.7071,sb_im=0
        check_close(gsp[0], 0.0, "alpha[0] re");
        check_close(gsp[1], 0.0, "alpha[0] im");
        check_close(gsp[4 * nd / 2], 0.7071067811865476, "beta[0] re");  // beta starts at index 4*nd/2 = 4*2=8
        check_close(gsp[4 * nd / 2 + 1], 0.0, "beta[0] im");
    }

    /// p-shell (l=1), kappa=+1 (lt block): nd=2 (2*1 LT components).
    #[test]
    fn sf_p_shell_kappa_pos1_lt_block() {
        let cart = [1.0f64, 0.0, 0.0]; // px=1
        let nd = spinor_len(1, 1); // 2
        let mut gsp = vec![0.0f64; 4 * nd]; // 8
        cart_to_spinor_sf(&mut gsp, &cart, 1, 1).expect("sf p-shell kappa=+1");
        // LT row 0: row_r=[-0.5773,0,0, 0,0,0.5773], row_i=[0,0.5773,0, 0,0,0]
        // n=0: ca_r=-0.5773, ca_i=0, cb_r=0, cb_i=0
        // sa_re=-0.5773, sa_im=0, sb_re=0, sb_im=0
        check_close(gsp[0], -0.5773502691896257, "lt alpha[0] re");
        check_close(gsp[1], 0.0, "lt alpha[0] im");
        // beta starts at nd*2=4
        check_close(gsp[4], 0.0, "lt beta[0] re");
    }

    /// p-shell (l=1), kappa=0: nd=6 (GT 4 + LT 2).
    #[test]
    fn sf_p_shell_kappa_0_both_blocks() {
        let cart = [1.0f64, 0.0, 0.0]; // px=1
        let nd = spinor_len(1, 0); // 6
        assert_eq!(nd, 6);
        let mut gsp = vec![0.0f64; 4 * nd]; // 24
        cart_to_spinor_sf(&mut gsp, &cart, 1, 0).expect("sf p-shell kappa=0");
        // GT block (rows 0..4) written, LT block (rows 4..6) written
        // Non-trivial: just check buffer size and no panic
        assert_eq!(gsp.len(), 24);
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  iket_sf: conjugation test (re,im) -> (-im,re)
    // ──────────────────────────────────────────────────────────────────────────

    /// iket_sf should produce output where (re,im) -> (-im,re) compared to sf,
    /// for a case where both re and im of sf are non-zero.
    ///
    /// Use l=2 gt (kappa=-1) with a cart input that produces non-zero re AND im
    /// in at least one component.
    #[test]
    fn iket_sf_vs_sf_sign_relationship() {
        let cart: Vec<f64> = vec![1.0, 0.5, 0.3, 0.7, 0.2, 0.9]; // d-shell: 6 cart components
        let l = 2u8;
        let kappa = -1i32;
        let nd = spinor_len(l, kappa);

        let mut gsp_sf = vec![0.0f64; 4 * nd];
        let mut gsp_iket = vec![0.0f64; 4 * nd];

        cart_to_spinor_sf(&mut gsp_sf, &cart, l, kappa).unwrap();
        cart_to_spinor_iket_sf(&mut gsp_iket, &cart, l, kappa).unwrap();

        // For each complex value (re, im) in sf, iket should give (-im, re)
        for k in 0..(2 * nd) {
            let sf_re = gsp_sf[k * 2];
            let sf_im = gsp_sf[k * 2 + 1];
            let iket_re = gsp_iket[k * 2];
            let iket_im = gsp_iket[k * 2 + 1];
            check_close(iket_re, -sf_im, &format!("iket_re[{k}] = -sf_im"));
            check_close(iket_im, sf_re, &format!("iket_im[{k}] = sf_re"));
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  si vs sf: si should differ when Pauli components non-zero
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn si_differs_from_sf_with_pauli() {
        // p-shell with non-zero vx/vy/vz should produce different output than sf
        let nf = 3usize;
        let v1 = vec![1.0f64, 0.5, 0.3];
        let vx = vec![0.2f64, 0.4, 0.1];
        let vy = vec![0.3f64, 0.1, 0.5];
        let vz = vec![0.1f64, 0.2, 0.4];
        let l = 1u8;
        let kappa = -1i32;
        let nd = spinor_len(l, kappa);

        let mut gsp_sf = vec![0.0f64; 4 * nd];
        let mut gsp_si = vec![0.0f64; 4 * nd];

        cart_to_spinor_sf(&mut gsp_sf, &v1, l, kappa).unwrap();
        cart_to_spinor_si(&mut gsp_si, &v1, &vx, &vy, &vz, l, kappa).unwrap();

        // At least one element must differ
        let differs = gsp_sf.iter().zip(gsp_si.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(differs, "si with non-zero Pauli should differ from sf");
        let _ = nf; // suppress unused warning
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  iket_si: verify (re,im) -> (-im,re) relationship with si
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn iket_si_vs_si_sign_relationship() {
        let v1 = vec![1.0f64, 0.5, 0.3, 0.7, 0.2, 0.9];
        let vx = vec![0.2f64, 0.4, 0.1, 0.3, 0.7, 0.5];
        let vy = vec![0.3f64, 0.1, 0.5, 0.2, 0.4, 0.8];
        let vz = vec![0.1f64, 0.2, 0.4, 0.6, 0.3, 0.7];
        let l = 2u8;
        let kappa = -1i32;
        let nd = spinor_len(l, kappa);

        let mut gsp_si = vec![0.0f64; 4 * nd];
        let mut gsp_iket_si = vec![0.0f64; 4 * nd];

        cart_to_spinor_si(&mut gsp_si, &v1, &vx, &vy, &vz, l, kappa).unwrap();
        cart_to_spinor_iket_si(&mut gsp_iket_si, &v1, &vx, &vy, &vz, l, kappa).unwrap();

        for k in 0..(2 * nd) {
            let si_re = gsp_si[k * 2];
            let si_im = gsp_si[k * 2 + 1];
            let iket_re = gsp_iket_si[k * 2];
            let iket_im = gsp_iket_si[k * 2 + 1];
            check_close(iket_re, -si_im, &format!("iket_si_re[{k}] = -si_im"));
            check_close(iket_im, si_re, &format!("iket_si_im[{k}] = si_re"));
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Error handling
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn sf_rejects_wrong_cart_length() {
        let mut gsp = vec![0.0f64; 8];
        let result = cart_to_spinor_sf(&mut gsp, &[1.0, 2.0], 0, -1); // ncart(0)=1 but gave 2
        assert!(result.is_err());
    }

    #[test]
    fn sf_rejects_small_output_buffer() {
        let mut gsp = vec![0.0f64; 3]; // need 8 for l=0,kappa=-1
        let result = cart_to_spinor_sf(&mut gsp, &[1.0], 0, -1);
        assert!(result.is_err());
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  cart_to_spinor_sf_4d tests
    // ──────────────────────────────────────────────────────────────────────────

    /// All s-shells (l=0, kappa=-1): simplest 4-center case.
    /// nci=ncj=nck=ncl=1, di=dj=dk=dl=2. Output size = 2*2*2*2*2 = 32 f64.
    #[test]
    fn sf_4d_ssss_kappa_neg1_output_size() {
        let cart = vec![1.0f64]; // 1*1*1*1 = 1 element
        let di = spinor_len(0, -1); // 2
        let dj = spinor_len(0, -1);
        let dk = spinor_len(0, -1);
        let dl = spinor_len(0, -1);
        let required = di * dj * dk * dl * 2; // 32
        let mut staging = vec![0.0f64; required];
        cart_to_spinor_sf_4d(
            &mut staging, &cart,
            0, -1, 0, -1, 0, -1, 0, -1,
        ).expect("4d ssss kappa=-1 should succeed");
        assert_eq!(staging.len(), required);
    }

    /// 4d ssss with all kappa=-1 and cart=[1.0]: output should be non-zero.
    #[test]
    fn sf_4d_ssss_kappa_neg1_nonzero() {
        let cart = vec![1.0f64];
        let required = spinor_len(0, -1).pow(4) * 2;
        let mut staging = vec![0.0f64; required];
        cart_to_spinor_sf_4d(
            &mut staging, &cart,
            0, -1, 0, -1, 0, -1, 0, -1,
        ).expect("sf_4d should succeed");
        let nonzero = staging.iter().filter(|&&v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "4d ssss spinor output should be non-zero, got all zeros");
    }

    /// Output size for p-shell quartet (l=1, kappa=-1): di=dj=dk=dl=4, size=4^4*2=512.
    #[test]
    fn sf_4d_pppp_kappa_neg1_output_size() {
        let nci: usize = 3; // ncart(1)
        let cart = vec![0.1f64; nci * nci * nci * nci]; // random non-zero
        let di = spinor_len(1, -1); // 4
        let required = di.pow(4) * 2; // 512
        let mut staging = vec![0.0f64; required];
        cart_to_spinor_sf_4d(
            &mut staging, &cart,
            1, -1, 1, -1, 1, -1, 1, -1,
        ).expect("sf_4d pppp should succeed");
        assert_eq!(staging.len(), required);
        let nonzero = staging.iter().filter(|&&v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "pppp spinor output should be non-zero");
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  cart_to_spinor_sf_3c2e tests
    // ──────────────────────────────────────────────────────────────────────────

    /// s-shells for i,j (l=0 kappa=-1) and s-shell for k (l=0): output size = 2*2*1*2 = 8 f64.
    #[test]
    fn sf_3c2e_sss_output_size() {
        use super::super::c2s::nsph;
        let cart = vec![1.0f64]; // nci*ncj*nck = 1
        let di = spinor_len(0, -1); // 2
        let dj = spinor_len(0, -1); // 2
        let nsk = nsph(0);          // 1
        let required = di * dj * nsk * 2; // 8
        let mut staging = vec![0.0f64; required];
        cart_to_spinor_sf_3c2e(
            &mut staging, &cart,
            0, -1, 0, -1, 0,
        ).expect("3c2e sss should succeed");
        assert_eq!(staging.len(), required);
    }

    /// sss with cart=[1.0]: output should be non-zero.
    #[test]
    fn sf_3c2e_sss_nonzero() {
        use super::super::c2s::nsph;
        let cart = vec![1.0f64];
        let di = spinor_len(0, -1);
        let dj = spinor_len(0, -1);
        let nsk = nsph(0);
        let required = di * dj * nsk * 2;
        let mut staging = vec![0.0f64; required];
        cart_to_spinor_sf_3c2e(
            &mut staging, &cart,
            0, -1, 0, -1, 0,
        ).expect("3c2e sss should succeed");
        let nonzero = staging.iter().filter(|&&v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "3c2e sss spinor output should be non-zero");
    }

    /// p-shell k: output has nsk=3 k-sph components, each with di*dj complex spinors.
    #[test]
    fn sf_3c2e_ssp_k_output_size() {
        use super::super::c2s::nsph;
        let nci: usize = 1; let ncj: usize = 1; let nck: usize = 3; // ncart(1)
        let cart = vec![0.5f64; nci * ncj * nck];
        let di = spinor_len(0, -1); // 2
        let dj = spinor_len(0, -1); // 2
        let nsk = nsph(1); // 3
        let required = di * dj * nsk * 2; // 24
        let mut staging = vec![0.0f64; required];
        cart_to_spinor_sf_3c2e(
            &mut staging, &cart,
            0, -1, 0, -1, 1,
        ).expect("3c2e s,s,p should succeed");
        assert_eq!(staging.len(), required);
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  RED tests: generic cart_to_spinor_sf_2d<F: CintFloat>
    //  These reference the turbofish ::<f64> and ::<f32> which don't compile
    //  until cart_to_spinor_sf_2d is made generic (Task 2 PART B GREEN).
    // ──────────────────────────────────────────────────────────────────────────

    /// T04-3a: cart_to_spinor_sf_2d::<f64> writes byte-identical output to pre-generic version.
    ///
    /// s-s pair (li=lj=0, kappa_i=kappa_j=-1): cart=[1.0], di=dj=2.
    /// Expected staging (from existing sf_s_shell_kappa_neg1_cart_one test):
    ///   staging[0..3] = [0.0, 0.0, 1.0, 0.0] (alpha), staging[4..7] = [1.0, 0.0, 0.0, 0.0] (beta)
    ///   Actually for 2D: staging[(j*di+i)*2] = re, +1 = im for di=dj=2, dj=dj
    ///   The 2D function produces di*dj*2 = 2*2*2 = 8 values.
    ///
    /// RED: compile fails until cart_to_spinor_sf_2d accepts a type parameter.
    #[test]
    fn test_c2spinor_sf_2d_f64_byte_identical() {
        // s-s pair with simple cart input
        let cart = vec![1.0f64]; // ncart(0) * ncart(0) = 1
        let di = spinor_len(0, -1); // 2
        let dj = spinor_len(0, -1); // 2
        let required = di * dj * 2; // 8

        // f64 path via turbofish — RED: fails to compile until <F: CintFloat> added
        let mut staging_f64 = vec![0.0_f64; required];
        cart_to_spinor_sf_2d::<f64>(&mut staging_f64, &cart, 0, -1, 0, -1)
            .expect("f64 2d spinor sf should succeed");

        // All values must be finite
        for (i, &v) in staging_f64.iter().enumerate() {
            assert!(v.is_finite(), "staging_f64[{i}] = {v} is not finite");
        }
        // At least one non-zero
        let nonzero = staging_f64.iter().filter(|&&v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "2d spinor sf s-s should produce non-zero output");
    }

    /// T04-3b: cart_to_spinor_sf_2d::<f32> writes finite f32 values without panic.
    ///
    /// RED: compile fails until cart_to_spinor_sf_2d accepts a type parameter.
    #[test]
    fn test_c2spinor_sf_2d_f32_finite() {
        let cart = vec![1.0f64]; // ncart(0) * ncart(0) = 1
        let di = spinor_len(0, -1); // 2
        let dj = spinor_len(0, -1); // 2
        let required = di * dj * 2; // 8

        // f32 path — RED: fails to compile until <F: CintFloat> added
        let mut staging_f32 = vec![0.0_f32; required];
        cart_to_spinor_sf_2d::<f32>(&mut staging_f32, &cart, 0, -1, 0, -1)
            .expect("f32 2d spinor sf should succeed");

        for (i, &v) in staging_f32.iter().enumerate() {
            assert!(v.is_finite(), "staging_f32[{i}] = {v} is not finite");
        }
        let nonzero = staging_f32.iter().filter(|&&v| v.abs() > 1e-5f32).count();
        assert!(nonzero > 0, "f32 2d spinor sf s-s should produce non-zero output");
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  cart_to_spinor_sf_derivative_2d tests (Task 1)
    // ──────────────────────────────────────────────────────────────────────────

    /// Build a deterministic KET-major derivative cart buffer `[comp][ket][bra]`
    /// for nctr=1: cart[comp*block_len + jc*nci + ic].
    fn make_deriv_cart_nctr1(ncomp: usize, nci: usize, ncj: usize) -> Vec<f64> {
        let block_len = nci * ncj;
        let mut cart = vec![0.0f64; ncomp * block_len];
        for comp in 0..ncomp {
            for jc in 0..ncj {
                for ic in 0..nci {
                    // distinct, non-zero per (comp,ic,jc)
                    cart[comp * block_len + jc * nci + ic] =
                        1.0 + comp as f64 + 0.5 * ic as f64 + 0.25 * jc as f64;
                }
            }
        }
        cart
    }

    /// (a) ncomp=3, nctr=1, NON-SQUARE p×d, kappa=0: wrapper output must equal a manual
    /// replay of the inline rank-3 transform (one_electron.rs L9937-9965) byte-for-byte.
    #[test]
    fn derivative_2d_rank3_matches_inline() {
        let (li, lj): (u8, u8) = (1, 2); // p × d (NON-SQUARE)
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li); // 3
        let ncj = ncart(lj); // 6
        let block_len = nci * ncj; // 18
        let di = spinor_len(li, ki as i32); // 6
        let dj = spinor_len(lj, kj as i32); // 10
        let spinor_block = di * dj * 2;
        let ncomp = 3usize;

        let cart = make_deriv_cart_nctr1(ncomp, nci, ncj);

        // Manual replay of the inline rank-3 transform.
        let mut expected = vec![0.0f64; ncomp * spinor_block];
        for comp in 0..ncomp {
            let block = &cart[comp * block_len..comp * block_len + block_len];
            let mut block_bra_major = vec![0.0f64; block_len];
            for ic in 0..nci {
                for jc in 0..ncj {
                    block_bra_major[ic * ncj + jc] = block[jc * nci + ic];
                }
            }
            cart_to_spinor_sf_2d::<f64>(
                &mut expected[comp * spinor_block..comp * spinor_block + spinor_block],
                &block_bra_major,
                li, ki, lj, kj,
            ).unwrap();
        }

        let mut got = vec![0.0f64; ncomp * spinor_block];
        cart_to_spinor_sf_derivative_2d::<f64>(
            &mut got, &cart, ncomp, li, ki, lj, kj, 1, 1,
        ).expect("derivative_2d rank3 should succeed");

        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            check_close(*g, *e, &format!("derivative_2d_rank3[{idx}]"));
        }
    }

    /// (b) ncomp=9: output splits into exactly 9 non-overlapping all-nonzero di*dj*2 slices
    /// (component-truncation guard — no trailing zero slice).
    #[test]
    fn derivative_2d_rank9_no_trailing_zero() {
        let (li, lj): (u8, u8) = (1, 2);
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li);
        let ncj = ncart(lj);
        let di = spinor_len(li, ki as i32);
        let dj = spinor_len(lj, kj as i32);
        let spinor_block = di * dj * 2;
        let ncomp = 9usize;

        let cart = make_deriv_cart_nctr1(ncomp, nci, ncj);
        let mut got = vec![0.0f64; ncomp * spinor_block];
        cart_to_spinor_sf_derivative_2d::<f64>(
            &mut got, &cart, ncomp, li, ki, lj, kj, 1, 1,
        ).expect("derivative_2d rank9 should succeed");

        for comp in 0..ncomp {
            let slice = &got[comp * spinor_block..comp * spinor_block + spinor_block];
            let nonzero = slice.iter().filter(|&&v| v.abs() > 1e-15).count();
            assert!(nonzero > 0, "component {comp} slice is all-zero (truncation landmine)");
        }
    }

    /// (c) nctr_i=2: output length is ncomp*(nctr_i*di)*(nctr_j*dj)*2 and contraction-major
    /// composition places ci=1 at i_global = di..2*di (no coefficient transpose leaks).
    #[test]
    fn derivative_2d_nctr2_sizing() {
        let (li, lj): (u8, u8) = (1, 2);
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li);
        let ncj = ncart(lj);
        let block_len = nci * ncj;
        let di = spinor_len(li, ki as i32);
        let dj = spinor_len(lj, kj as i32);
        let ncomp = 3usize;
        let nctr_i = 2usize;
        let nctr_j = 1usize;
        let ni_full = nctr_i * di;
        let nj_full = nctr_j * dj;
        let spinor_block = ni_full * nj_full * 2;
        let total_len = ncomp * block_len;

        // KET-major cart with contraction sub-blocks: base = (ci*nctr_j+cj)*total_len + comp*block_len.
        let mut cart = vec![0.0f64; nctr_i * nctr_j * total_len];
        for ci in 0..nctr_i {
            for cj in 0..nctr_j {
                for comp in 0..ncomp {
                    let base = (ci * nctr_j + cj) * total_len + comp * block_len;
                    for jc in 0..ncj {
                        for ic in 0..nci {
                            // ci=0 sub-block deliberately ZERO; ci=1 non-zero so we can locate it.
                            cart[base + jc * nci + ic] = if ci == 0 {
                                0.0
                            } else {
                                1.0 + comp as f64 + 0.5 * ic as f64 + 0.25 * jc as f64
                            };
                        }
                    }
                }
            }
        }

        let mut got = vec![0.0f64; ncomp * spinor_block];
        cart_to_spinor_sf_derivative_2d::<f64>(
            &mut got, &cart, ncomp, li, ki, lj, kj, nctr_i, nctr_j,
        ).expect("derivative_2d nctr2 should succeed");

        assert_eq!(got.len(), ncomp * spinor_block, "nctr2 output length mismatch");

        // ci=0 region (i_global in 0..di) must be all-zero; ci=1 region (di..2di) non-zero.
        let comp = 0usize;
        let mut ci0_nonzero = 0usize;
        let mut ci1_nonzero = 0usize;
        for jg in 0..nj_full {
            for ig in 0..ni_full {
                let v = got[comp * spinor_block + (jg * ni_full + ig) * 2];
                let im = got[comp * spinor_block + (jg * ni_full + ig) * 2 + 1];
                let mag = v.abs() + im.abs();
                if ig < di {
                    if mag > 1e-15 { ci0_nonzero += 1; }
                } else if mag > 1e-15 {
                    ci1_nonzero += 1;
                }
            }
        }
        assert_eq!(ci0_nonzero, 0, "ci=0 (zero cart sub-block) should map to zero output");
        assert!(ci1_nonzero > 0, "ci=1 sub-block should populate i_global in di..2*di");
    }

    /// (d) staging too small: returns BufferTooSmall BEFORE any write (sentinel survives).
    #[test]
    fn derivative_2d_staging_too_small_fails_closed() {
        let (li, lj): (u8, u8) = (1, 2);
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li);
        let ncj = ncart(lj);
        let ncomp = 3usize;
        let cart = make_deriv_cart_nctr1(ncomp, nci, ncj);

        // Deliberately undersized staging with a sentinel at [0].
        let mut staging = vec![0.0f64; 4];
        staging[0] = 12345.0;
        let res = cart_to_spinor_sf_derivative_2d::<f64>(
            &mut staging, &cart, ncomp, li, ki, lj, kj, 1, 1,
        );
        assert!(
            matches!(res, Err(cintxRsError::BufferTooSmall { .. })),
            "undersized staging must return BufferTooSmall, got {res:?}"
        );
        assert_eq!(staging[0], 12345.0, "sentinel overwritten — wrote before size check");
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  cart_to_spinor_sf_derivative_3c2e / _3c1e tests (Task 2)
    //  Aux-k axis is SPHERICAL nsph(lk) — the canonical p×d×s nctr=1 kappa=0 ncomp=3
    //  buffer is 360, NOT 720 (27-SPIKE-FINDINGS ⚠ CORRECTION NOTICE).
    // ──────────────────────────────────────────────────────────────────────────

    /// Build a deterministic KET-major derivative 3c cart buffer `[comp][k][ket][bra]`
    /// for nctr=1: cart[comp*kblock + ck*(ncj*nci) + jc*nci + ic], kblock = nck*ncj*nci.
    fn make_deriv_cart_3c_nctr1(ncomp: usize, nci: usize, ncj: usize, nck: usize) -> Vec<f64> {
        let kblock = nck * ncj * nci;
        let mut cart = vec![0.0f64; ncomp * kblock];
        for comp in 0..ncomp {
            for ck in 0..nck {
                for jc in 0..ncj {
                    for ic in 0..nci {
                        let idx = comp * kblock + (ck * ncj + jc) * nci + ic;
                        cart[idx] = 1.0 + comp as f64 + 0.5 * ic as f64
                            + 0.25 * jc as f64 + 0.1 * ck as f64;
                    }
                }
            }
        }
        cart
    }

    /// (a) ncomp=3, NON-SQUARE p×d ket + s aux, nctr=1, kappa=0:
    /// output length = 3*(nctr_i*di)*(nctr_j*dj)*nsph(lk)*2, split into 3 non-overlapping
    /// all-nonzero comp_stride slices, AND the canonical total is exactly 360 (3*6*10*1*2),
    /// NEVER 720.
    #[test]
    fn derivative_3c2e_rank3_layout() {
        use super::super::c2s::nsph;
        let (li, lj, lk): (u8, u8, u8) = (1, 2, 0); // p × d × s aux
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li); // 3
        let ncj = ncart(lj); // 6
        let nck = ncart(lk); // 1
        let di = spinor_len(li, ki as i32); // 6
        let dj = spinor_len(lj, kj as i32); // 10
        let nsk = nsph(lk); // 1 (SPHERICAL aux-k)
        let ncomp = 3usize;
        let comp_stride = di * dj * nsk * 2; // 6*10*1*2 = 120
        let total = ncomp * comp_stride; // 360, NOT 720

        // Canonical figure assertion: 360 not 720.
        assert_eq!(total, 360, "canonical p×d×s ncomp=3 buffer must be 360, not 720");

        let cart = make_deriv_cart_3c_nctr1(ncomp, nci, ncj, nck);
        let mut got = vec![0.0f64; total];
        cart_to_spinor_sf_derivative_3c2e::<f64>(
            &mut got, &cart, ncomp, li, ki, lj, kj, lk, 1, 1,
        ).expect("derivative_3c2e rank3 should succeed");

        assert_eq!(got.len(), total, "3c2e output length mismatch");
        for comp in 0..ncomp {
            let slice = &got[comp * comp_stride..comp * comp_stride + comp_stride];
            let nonzero = slice.iter().filter(|&&v| v.abs() > 1e-15).count();
            assert!(nonzero > 0, "3c2e component {comp} slice is all-zero (truncation landmine)");
        }
    }

    /// (b) staging too small: returns BufferTooSmall BEFORE any write (sentinel survives).
    #[test]
    fn derivative_3c2e_staging_too_small_fails_closed() {
        let (li, lj, lk): (u8, u8, u8) = (1, 2, 0);
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li);
        let ncj = ncart(lj);
        let nck = ncart(lk);
        let ncomp = 3usize;
        let cart = make_deriv_cart_3c_nctr1(ncomp, nci, ncj, nck);

        let mut staging = vec![0.0f64; 4];
        staging[0] = 98765.0;
        let res = cart_to_spinor_sf_derivative_3c2e::<f64>(
            &mut staging, &cart, ncomp, li, ki, lj, kj, lk, 1, 1,
        );
        assert!(
            matches!(res, Err(cintxRsError::BufferTooSmall { .. })),
            "undersized staging must return BufferTooSmall, got {res:?}"
        );
        assert_eq!(staging[0], 98765.0, "sentinel overwritten — wrote before size check");
    }

    /// (c) the int3c1e thin sibling shares the SPHERICAL aux-k contract and the same fold.
    #[test]
    fn derivative_3c1e_rank3_spherical_auxk() {
        use super::super::c2s::nsph;
        let (li, lj, lk): (u8, u8, u8) = (1, 2, 0);
        let (ki, kj): (i16, i16) = (0, 0);
        let nci = ncart(li);
        let ncj = ncart(lj);
        let nck = ncart(lk);
        let di = spinor_len(li, ki as i32);
        let dj = spinor_len(lj, kj as i32);
        let nsk = nsph(lk);
        let ncomp = 3usize;
        let comp_stride = di * dj * nsk * 2;
        let total = ncomp * comp_stride;

        let cart = make_deriv_cart_3c_nctr1(ncomp, nci, ncj, nck);
        let mut got = vec![0.0f64; total];
        cart_to_spinor_sf_derivative_3c1e::<f64>(
            &mut got, &cart, ncomp, li, ki, lj, kj, lk, 1, 1,
        ).expect("derivative_3c1e rank3 should succeed");
        assert_eq!(got.len(), total);
        let nonzero = got.iter().filter(|&&v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "3c1e sibling output should be non-zero");
    }
}
