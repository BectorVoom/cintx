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

use cintx_core::cintxRsError;
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
fn apply_sf_block(
    gsp: &mut [f64],
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
        gsp[out_i * 2] = sa_re;
        gsp[out_i * 2 + 1] = sa_im;
        gsp[(nd_total + out_i) * 2] = sb_re;
        gsp[(nd_total + out_i) * 2 + 1] = sb_im;
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
fn apply_iket_sf_block(
    gsp: &mut [f64],
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
        gsp[out_i * 2] = sa_re;
        gsp[out_i * 2 + 1] = sa_im;
        gsp[(nd_total + out_i) * 2] = sb_re;
        gsp[(nd_total + out_i) * 2 + 1] = sb_im;
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
fn apply_si_block(
    gsp: &mut [f64],
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
        gsp[out_i * 2] = sa_re;
        gsp[out_i * 2 + 1] = sa_im;
        gsp[(nd_total + out_i) * 2] = sb_re;
        gsp[(nd_total + out_i) * 2 + 1] = sb_im;
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
fn apply_iket_si_block(
    gsp: &mut [f64],
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
        gsp[out_i * 2] = sa_re;
        gsp[out_i * 2 + 1] = sa_im;
        gsp[(nd_total + out_i) * 2] = sb_re;
        gsp[(nd_total + out_i) * 2 + 1] = sb_im;
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
pub fn cart_to_spinor_sf(
    gsp: &mut [f64],
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
        // kappa == 0: GT first, then LT
        let nd_gt = 2 * l as usize + 2;
        let nd_lt = 2 * l as usize;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        apply_sf_block(gsp, cart, &rr_gt, &ri_gt, nd_gt, nf, nd, 0);
        apply_sf_block(gsp, cart, &rr_lt, &ri_lt, nd_lt, nf, nd, nd_gt);
    }
    Ok(())
}

/// Cart-to-spinor iket scalar-field transform (multiply by i).
///
/// Corresponds to `CINTc2s_iket_spinor_sf1` in libcint.
/// Same signature as `cart_to_spinor_sf` but output is multiplied by i:
/// (re, im) → (-im, re).
pub fn cart_to_spinor_iket_sf(
    gsp: &mut [f64],
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
        let nd_gt = 2 * l as usize + 2;
        let nd_lt = 2 * l as usize;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        apply_iket_sf_block(gsp, cart, &rr_gt, &ri_gt, nd_gt, nf, nd, 0);
        apply_iket_sf_block(gsp, cart, &rr_lt, &ri_lt, nd_lt, nf, nd, nd_gt);
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
pub fn cart_to_spinor_si(
    gsp: &mut [f64],
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
        let nd_gt = 2 * l as usize + 2;
        let nd_lt = 2 * l as usize;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        apply_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_gt, &ri_gt, nd_gt, nf, nd, 0);
        apply_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_lt, &ri_lt, nd_lt, nf, nd, nd_gt);
    }
    Ok(())
}

/// Cart-to-spinor iket spin-included transform (multiply by i).
///
/// Corresponds to `CINTc2s_iket_spinor_si1` in libcint.
/// Same as `cart_to_spinor_si` but output is multiplied by i.
pub fn cart_to_spinor_iket_si(
    gsp: &mut [f64],
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
        let nd_gt = 2 * l as usize + 2;
        let nd_lt = 2 * l as usize;
        debug_assert_eq!(nd, nd_gt + nd_lt);
        let (rr_gt, ri_gt) = gt_coeff_rows(l);
        let (rr_lt, ri_lt) = lt_coeff_rows(l);
        apply_iket_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_gt, &ri_gt, nd_gt, nf, nd, 0);
        apply_iket_si_block(gsp, cart_v1, cart_vx, cart_vy, cart_vz, &rr_lt, &ri_lt, nd_lt, nf, nd, nd_gt);
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
pub fn cart_to_spinor_sf_2d(
    staging: &mut [f64],
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
    for j in 0..dj {
        for i in 0..di {
            let out_idx = j * di + i;
            staging[out_idx * 2] = out_r[j * di + i];
            staging[out_idx * 2 + 1] = out_i[j * di + i];
        }
    }

    Ok(())
}

/// Bra step of the 2D c2spinor_sf transform for all kappa cases.
///
/// Matches `a_bra_cart2spinor_sf` in libcint `cart2sph.c`.
/// For kappa==0, applies GT first (rows 0..nd_gt), then LT (rows nd_gt..nd).
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
        // kappa == 0: GT first, LT second
        let nd_gt = 2 * li as usize + 2;
        apply_bra_block(alpha_r, alpha_i, beta_r, beta_i,
                        cart, nci, ncj, nd_gt, coeff_gt_r, coeff_gt_i, 0);
        let nd_lt = 2 * li as usize;
        if nd_lt > 0 {
            apply_bra_block(alpha_r, alpha_i, beta_r, beta_i,
                            cart, nci, ncj, nd_lt, coeff_lt_r, coeff_lt_i, nd_gt);
        }
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
            // Use static arrays to avoid lifetime issues
            // Apply inline for kappa==0 case
            let nd_gt = 2 * lj as usize + 2;
            let nd_lt = 2 * lj as usize;
            apply_ket_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i,
                           di, ncj, nd_gt, nf2, coeff_gt_r, coeff_gt_i, 0);
            if nd_lt > 0 {
                apply_ket_block(out_r, out_i, alpha_r, alpha_i, beta_r, beta_i,
                               di, ncj, nd_lt, nf2, coeff_lt_r, coeff_lt_i, nd_gt);
            }
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

/// Staging spinor transform (placeholder for executor path).
///
/// The real per-shell transform is done via `cart_to_spinor_sf` and friends
/// with explicit l and kappa parameters. This staging function is kept for
/// API compatibility; it performs no transformation (no-op).
///
/// TODO: Wire executor to pass l and kappa so this can call the correct variant.
pub fn cart_to_spinor_interleaved_staging(staging: &mut [f64]) -> Result<(), cintxRsError> {
    let _ = staging;
    Ok(())
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
    //  staging no-op
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn staging_is_noop() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        cart_to_spinor_interleaved_staging(&mut data).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
