//! Cartesian-to-spherical (c2s) transform coefficients and functions.
//!
//! Condon-Shortley coefficient matrices extracted from libcint `cart2sph.c`
//! `g_trans_cart2sph[]` array. The layout is:
//!   - rows = spherical components (2l+1), ordered m = -l, -l+1, ..., 0, ..., l
//!   - cols = cartesian components (l+1)(l+2)/2, in libcint ordering
//!
//! Reference: H. B. Schlegel and M. J. Frisch, Int. J. Quant. Chem., 54(1995), 83-87.

use cintx_core::cintxRsError;

// ──────────────────────────────────────────────────────────────────────────
//  Helper dimension functions
// ──────────────────────────────────────────────────────────────────────────

/// Number of Cartesian components for angular momentum l: (l+1)(l+2)/2
pub fn ncart(l: u8) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}

/// Number of spherical components for angular momentum l: 2l+1
pub fn nsph(l: u8) -> usize {
    2 * l as usize + 1
}

// ──────────────────────────────────────────────────────────────────────────
//  Condon-Shortley coefficient matrices (from libcint g_trans_cart2sph[])
// ──────────────────────────────────────────────────────────────────────────

/// s-shell (l=0): 1 sph x 1 cart.
/// g_trans_cart2sph offset 0, 1 element.
pub const C2S_L0: [[f64; 1]; 1] = [[1.0]];

/// p-shell (l=1): 3 sph x 3 cart (px, py, pz ordering).
///
/// From libcint `cart2sph.c` `g_trans_cart2sph[]` p-shell section (default, no PYPZPX):
///   sph[0] = px  -> [1, 0, 0]
///   sph[1] = py  -> [0, 1, 0]
///   sph[2] = pz  -> [0, 0, 1]
///
/// Libcint uses (px, py, pz) as the spherical p ordering — this is the identity
/// transform from Cartesian (px, py, pz) to spherical. The CINTcommon_fac_sp(1)
/// prefactor (0.4886) is applied externally in the primitive loop, not here.
pub const C2S_L1: [[f64; 3]; 3] = [
    // sph[0] = px
    [1.0, 0.0, 0.0],
    // sph[1] = py
    [0.0, 1.0, 0.0],
    // sph[2] = pz
    [0.0, 0.0, 1.0],
];

/// d-shell (l=2): 5 sph x 6 cart.
/// g_trans_cart2sph offset 10, 30 elements.
/// Rows: m = -2 (dxy), m = -1 (dyz), m = 0 (dz2), m = +1 (dxz), m = +2 (dx2-y2)
/// Cols: cartesian (xx, xy, xz, yy, yz, zz)
pub const C2S_L2: [[f64; 6]; 5] = [
    // m=-2: dxy
    [0.0, 1.092548430592079070, 0.0, 0.0, 0.0, 0.0],
    // m=-1: dyz
    [0.0, 0.0, 0.0, 0.0, 1.092548430592079070, 0.0],
    // m= 0: dz2
    [-0.315391565252520002, 0.0, 0.0, -0.315391565252520002, 0.0, 0.630783130505040012],
    // m=+1: dxz
    [0.0, 0.0, 1.092548430592079070, 0.0, 0.0, 0.0],
    // m=+2: dx2-y2
    [0.546274215296039535, 0.0, 0.0, -0.546274215296039535, 0.0, 0.0],
];

/// f-shell (l=3): 7 sph x 10 cart.
/// g_trans_cart2sph offset 40, 70 elements.
/// Rows: m = -3..+3
/// Cols: cartesian (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz)
pub const C2S_L3: [[f64; 10]; 7] = [
    // m=-3: fyx2 (f-3)
    [0.0, 1.770130769779930531, 0.0, 0.0, 0.0, 0.0, -0.590043589926643510, 0.0, 0.0, 0.0],
    // m=-2: fxyz (f-2)
    [0.0, 0.0, 0.0, 0.0, 2.890611442640554055, 0.0, 0.0, 0.0, 0.0, 0.0],
    // m=-1: fyz2 (f-1)
    [0.0, -0.457045799464465739, 0.0, 0.0, 0.0, 0.0, -0.457045799464465739, 0.0, 1.828183197857862944, 0.0],
    // m= 0: fz3 (f0)
    [0.0, 0.0, -1.119528997770346170, 0.0, 0.0, 0.0, 0.0, -1.119528997770346170, 0.0, 0.746352665180230782],
    // m=+1: fxz2 (f1)
    [-0.457045799464465739, 0.0, 0.0, -0.457045799464465739, 0.0, 1.828183197857862944, 0.0, 0.0, 0.0, 0.0],
    // m=+2: fzx2 (f2)
    [0.0, 0.0, 1.445305721320277020, 0.0, 0.0, 0.0, 0.0, -1.445305721320277020, 0.0, 0.0],
    // m=+3: fx3 (f3)
    [0.590043589926643510, 0.0, 0.0, -1.770130769779930530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
];

/// g-shell (l=4): 9 sph x 15 cart.
/// g_trans_cart2sph offset 110, 135 elements.
/// Rows: m = -4..+4
/// Cols: cartesian (xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz)
pub const C2S_L4: [[f64; 15]; 9] = [
    // m=-4: gyx3 (g-4)
    [0.0, 2.503342941796704538, 0.0, 0.0, 0.0, 0.0, -2.503342941796704530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // m=-3: gx2yz (g-3)
    [0.0, 0.0, 0.0, 0.0, 5.310392309339791593, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.770130769779930530, 0.0, 0.0, 0.0],
    // m=-2: gxyz2 (g-2)
    [0.0, -0.946174695757560014, 0.0, 0.0, 0.0, 0.0, -0.946174695757560014, 0.0, 5.677048174545360108, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // m=-1: gyz3 (g-1)
    [0.0, 0.0, 0.0, 0.0, -2.007139630671867500, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.007139630671867500, 0.0, 2.676186174229156671, 0.0],
    // m= 0: gz4 (g0)
    [0.317356640745612911, 0.0, 0.0, 0.634713281491225822, 0.0, -2.538853125964903290, 0.0, 0.0, 0.0, 0.0, 0.317356640745612911, 0.0, -2.538853125964903290, 0.0, 0.846284375321634430],
    // m=+1: gxz3 (g1)
    [0.0, 0.0, -2.007139630671867500, 0.0, 0.0, 0.0, 0.0, -2.007139630671867500, 0.0, 2.676186174229156671, 0.0, 0.0, 0.0, 0.0, 0.0],
    // m=+2: gx2z2 (g2)
    [-0.473087347878780002, 0.0, 0.0, 0.0, 0.0, 2.838524087272680054, 0.0, 0.0, 0.0, 0.0, 0.473087347878780009, 0.0, -2.838524087272680050, 0.0, 0.0],
    // m=+3: gzx3 (g3)
    [0.0, 0.0, 1.770130769779930531, 0.0, 0.0, 0.0, 0.0, -5.310392309339791590, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // m=+4: gy4 (g4)
    [0.625835735449176134, 0.0, 0.0, -3.755014412695056800, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.625835735449176134, 0.0, 0.0, 0.0, 0.0],
];

// ──────────────────────────────────────────────────────────────────────────
//  Transform function
// ──────────────────────────────────────────────────────────────────────────

/// Apply cart-to-sph transform for a 1-electron shell pair (li, lj).
///
/// Input `cart_buf`: flat row-major array of shape `[ncart(lj) * ncart(li)]`
///   (j is the outer/slow index, i is the inner/fast index — bra=i, ket=j).
///
/// Output `sph_buf`: flat row-major array of shape `[nsph(lj) * nsph(li)]`.
///
/// The transform applies:
///   1. Bra (i-axis): multiply T[li] (nsph_i x ncart_i) from the left.
///   2. Ket (j-axis): multiply T[lj] (nsph_j x ncart_j) from the left.
///
/// For l=0 both axes are identity (no-op).
pub fn cart_to_sph_1e(cart_buf: &[f64], sph_buf: &mut [f64], li: u8, lj: u8) {
    let nci = ncart(li);
    let ncj = ncart(lj);
    let nsi = nsph(li);
    let nsj = nsph(lj);

    debug_assert_eq!(cart_buf.len(), nci * ncj);
    debug_assert_eq!(sph_buf.len(), nsi * nsj);

    // Step 1: Transform bra (i-axis): T[li] @ cart_buf column-by-column.
    // Intermediate shape: [ncj * nsi] (j is outer, i_sph is inner)
    let mut tmp = vec![0.0f64; ncj * nsi];
    for j in 0..ncj {
        for mi in 0..nsi {
            let mut sum = 0.0;
            for ci in 0..nci {
                sum += c2s_coeff(li, mi, ci) * cart_buf[j * nci + ci];
            }
            tmp[j * nsi + mi] = sum;
        }
    }

    // Step 2: Transform ket (j-axis): T[lj] @ tmp^T row-by-row.
    // Output shape: [nsj * nsi]
    for mj in 0..nsj {
        for mi in 0..nsi {
            let mut sum = 0.0;
            for cj in 0..ncj {
                sum += c2s_coeff(lj, mj, cj) * tmp[cj * nsi + mi];
            }
            sph_buf[mj * nsi + mi] = sum;
        }
    }
}

/// Retrieve a single Condon-Shortley coefficient T[l][m_row][cart_col].
///
/// `l`        : angular momentum
/// `m_row`    : spherical index (0-based, maps to m = -l, ..., +l)
/// `cart_col` : cartesian index (0-based)
///
/// Returns 0.0 for l > 4 (unsupported — caller should validate before calling).
#[inline]
fn c2s_coeff(l: u8, m_row: usize, cart_col: usize) -> f64 {
    match l {
        0 => C2S_L0[m_row][cart_col],
        1 => C2S_L1[m_row][cart_col],
        2 => C2S_L2[m_row][cart_col],
        3 => C2S_L3[m_row][cart_col],
        4 => C2S_L4[m_row][cart_col],
        _ => 0.0,
    }
}

// ──────────────────────────────────────────────────────────────────────────
//  Staging transform (public API compat)
// ──────────────────────────────────────────────────────────────────────────

/// Staging cart-to-sph transform — no-op.
///
/// Real kernels (1e, 2e, etc.) handle cart-to-sph internally using
/// `cart_to_sph_1e()` with per-shell angular momentum info.
/// The generic staging transform is bypassed; calling it is safe and idempotent.
pub fn cart_to_spheric_staging(staging: &mut [f64]) -> Result<(), cintxRsError> {
    let _ = staging;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncart_values() {
        assert_eq!(ncart(0), 1);
        assert_eq!(ncart(1), 3);
        assert_eq!(ncart(2), 6);
        assert_eq!(ncart(3), 10);
        assert_eq!(ncart(4), 15);
    }

    #[test]
    fn nsph_values() {
        assert_eq!(nsph(0), 1);
        assert_eq!(nsph(1), 3);
        assert_eq!(nsph(2), 5);
        assert_eq!(nsph(3), 7);
        assert_eq!(nsph(4), 9);
    }

    #[test]
    fn c2s_l0_identity() {
        assert_eq!(C2S_L0, [[1.0]]);
    }

    #[test]
    fn c2s_l2_d_xy_coefficient() {
        // m=-2, col=1 (xy): dxy coefficient
        let diff = (C2S_L2[0][1] - 1.092548430592079070_f64).abs();
        assert!(diff < 1e-15, "C2S_L2[0][1] diff={diff}");
    }

    #[test]
    fn c2s_l2_dz2_coefficient() {
        // m=0, col=0 (xx): dz2 xx coefficient
        let diff = (C2S_L2[2][0] - (-0.315391565252520002_f64)).abs();
        assert!(diff < 1e-15, "C2S_L2[2][0] diff={diff}");
    }

    #[test]
    fn cart_to_sph_1e_ss_identity() {
        let cart = [1.0_f64];
        let mut sph = [0.0_f64];
        cart_to_sph_1e(&cart, &mut sph, 0, 0);
        assert_eq!(sph, [1.0]);
    }

    #[test]
    fn cart_to_spheric_staging_is_noop() {
        let mut data = vec![1.0, 2.0, 3.0];
        cart_to_spheric_staging(&mut data).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
