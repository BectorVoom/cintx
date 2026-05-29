#![allow(non_snake_case)]

use cintx_core::cintxRsError;
use cintx_cubecl::transform::c2spinor;
use cintx_cubecl::transform::c2s::{c2s_coeff, ncart, nsph};

/// Apply the real per-l bra cart->sph transform, mirroring libcint's
/// `*_bra_cart2spheric` (cart2sph.c). Ket-blocked layout: for each ket block,
/// write `nsph(l)` spherical values (sph-row fastest), advancing `sph` by
/// `nsph(l)` and `cart` by `ncart(l)` per ket.
///
/// For l=0 and l=1 the C2S coefficient table is the identity (non-PYPZPX), so
/// internal callers `CINTc2s_ket_sph` / `CINTc2s_ket_sph1` (which pass l=0) are
/// preserved exactly.
///
/// NOTE on l>4: `c2s::c2s_coeff` returns 0.0 for l>4 (its accessor contract).
/// This transform therefore zeroes the output for l>4; the vendor gate only
/// exercises l in 0..=4. l>4 support is intentionally not added here.
pub fn CINTc2s_bra_sph(
    sph: &mut [f64],
    nket: i32,
    cart: &[f64],
    l: i32,
) -> Result<(), cintxRsError> {
    let lu = l.max(0) as u8;
    let nc = ncart(lu);
    let ns = nsph(lu);
    let nk = nket.max(0) as usize;

    let required_cart = nk * nc;
    if cart.len() < required_cart {
        return Err(cintxRsError::BufferTooSmall {
            required: required_cart,
            provided: cart.len(),
        });
    }
    let required_sph = nk * ns;
    if sph.len() < required_sph {
        return Err(cintxRsError::BufferTooSmall {
            required: required_sph,
            provided: sph.len(),
        });
    }

    for k in 0..nk {
        for m in 0..ns {
            let mut acc = 0.0f64;
            for c in 0..nc {
                acc += c2s_coeff(lu, m, c) * cart[k * nc + c];
            }
            sph[k * ns + m] = acc;
        }
    }
    Ok(())
}

pub fn CINTc2s_ket_sph(
    sph: &mut [f64],
    _nket: i32,
    cart: &[f64],
    _l: i32,
) -> Result<(), cintxRsError> {
    CINTc2s_bra_sph(sph, 0, cart, 0)
}

pub fn CINTc2s_ket_sph1(
    sph: &mut [f64],
    cart: &[f64],
    _lds: i32,
    _ldc: i32,
    _l: i32,
) -> Result<(), cintxRsError> {
    CINTc2s_bra_sph(sph, 0, cart, 0)
}

/// Cart-to-spinor scalar-field ket transform.
///
/// Delegates to `c2spinor::cart_to_spinor_sf` with the l and kappa from the
/// compat signature. The `lds`, `ldc`, `nctr` parameters are from the libcint
/// strided API; for nctr > 1, each contraction block of ncart(l) values in
/// `gcart` is processed in sequence and outputs are concatenated in `gsp`.
pub fn CINTc2s_ket_spinor_sf1(
    gsp: &mut [f64],
    gcart: &[f64],
    _lds: i32,
    _ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    if l < 0 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "CINTc2s_ket_spinor_sf1",
            detail: format!("l={l} must be non-negative"),
        });
    }
    let lu = l as u8;
    let nf = ncart(lu);
    let nctr_usize = nctr.max(1) as usize;

    // Validate gcart has enough data for all contractions
    let required_cart = nf * nctr_usize;
    if gcart.len() < required_cart {
        return Err(cintxRsError::BufferTooSmall {
            required: required_cart,
            provided: gcart.len(),
        });
    }

    // Process all contractions; each contraction produces 4*nd f64 output
    let nd = c2spinor::spinor_len(lu, kappa);
    let out_per_ctr = 4 * nd;
    let required_out = out_per_ctr * nctr_usize;
    if gsp.len() < required_out {
        return Err(cintxRsError::BufferTooSmall {
            required: required_out,
            provided: gsp.len(),
        });
    }

    for k in 0..nctr_usize {
        let cart_slice = &gcart[k * nf..(k + 1) * nf];
        let gsp_slice = &mut gsp[k * out_per_ctr..(k + 1) * out_per_ctr];
        c2spinor::cart_to_spinor_sf(gsp_slice, cart_slice, lu, kappa)?;
    }
    Ok(())
}

/// Cart-to-spinor iket scalar-field transform (multiply by i).
///
/// Delegates to `c2spinor::cart_to_spinor_iket_sf`.
pub fn CINTc2s_iket_spinor_sf1(
    gsp: &mut [f64],
    gcart: &[f64],
    _lds: i32,
    _ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    if l < 0 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "CINTc2s_iket_spinor_sf1",
            detail: format!("l={l} must be non-negative"),
        });
    }
    let lu = l as u8;
    let nf = ncart(lu);
    let nctr_usize = nctr.max(1) as usize;
    let nd = c2spinor::spinor_len(lu, kappa);
    let out_per_ctr = 4 * nd;
    let required_cart = nf * nctr_usize;
    let required_out = out_per_ctr * nctr_usize;

    if gcart.len() < required_cart {
        return Err(cintxRsError::BufferTooSmall { required: required_cart, provided: gcart.len() });
    }
    if gsp.len() < required_out {
        return Err(cintxRsError::BufferTooSmall { required: required_out, provided: gsp.len() });
    }

    for k in 0..nctr_usize {
        let cart_slice = &gcart[k * nf..(k + 1) * nf];
        let gsp_slice = &mut gsp[k * out_per_ctr..(k + 1) * out_per_ctr];
        c2spinor::cart_to_spinor_iket_sf(gsp_slice, cart_slice, lu, kappa)?;
    }
    Ok(())
}

/// Cart-to-spinor spin-included ket transform with Pauli coupling.
///
/// Delegates to `c2spinor::cart_to_spinor_si`.
///
/// The `gcart` buffer is expected to contain four concatenated Pauli components:
///   `gcart[0..nf]` = v1 (scalar), `gcart[nf..2*nf]` = vx,
///   `gcart[2*nf..3*nf]` = vy, `gcart[3*nf..4*nf]` = vz,
/// where nf = ncart(l). For nctr > 1, each contraction block has size 4*nf.
pub fn CINTc2s_ket_spinor_si1(
    gsp: &mut [f64],
    gcart: &[f64],
    _lds: i32,
    _ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    if l < 0 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "CINTc2s_ket_spinor_si1",
            detail: format!("l={l} must be non-negative"),
        });
    }
    let lu = l as u8;
    let nf = ncart(lu);
    let nctr_usize = nctr.max(1) as usize;
    let nd = c2spinor::spinor_len(lu, kappa);
    let out_per_ctr = 4 * nd;
    // si takes 4 Pauli components each of size nf per contraction
    let required_cart = 4 * nf * nctr_usize;
    let required_out = out_per_ctr * nctr_usize;

    if gcart.len() < required_cart {
        return Err(cintxRsError::BufferTooSmall { required: required_cart, provided: gcart.len() });
    }
    if gsp.len() < required_out {
        return Err(cintxRsError::BufferTooSmall { required: required_out, provided: gsp.len() });
    }

    for k in 0..nctr_usize {
        let base = k * 4 * nf;
        let v1 = &gcart[base..base + nf];
        let vx = &gcart[base + nf..base + 2 * nf];
        let vy = &gcart[base + 2 * nf..base + 3 * nf];
        let vz = &gcart[base + 3 * nf..base + 4 * nf];
        let gsp_slice = &mut gsp[k * out_per_ctr..(k + 1) * out_per_ctr];
        c2spinor::cart_to_spinor_si(gsp_slice, v1, vx, vy, vz, lu, kappa)?;
    }
    Ok(())
}

/// Cart-to-spinor iket spin-included transform (multiply by i).
///
/// Delegates to `c2spinor::cart_to_spinor_iket_si`.
/// Same gcart layout as `CINTc2s_ket_spinor_si1`: four Pauli components packed.
pub fn CINTc2s_iket_spinor_si1(
    gsp: &mut [f64],
    gcart: &[f64],
    _lds: i32,
    _ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    if l < 0 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "CINTc2s_iket_spinor_si1",
            detail: format!("l={l} must be non-negative"),
        });
    }
    let lu = l as u8;
    let nf = ncart(lu);
    let nctr_usize = nctr.max(1) as usize;
    let nd = c2spinor::spinor_len(lu, kappa);
    let out_per_ctr = 4 * nd;
    let required_cart = 4 * nf * nctr_usize;
    let required_out = out_per_ctr * nctr_usize;

    if gcart.len() < required_cart {
        return Err(cintxRsError::BufferTooSmall { required: required_cart, provided: gcart.len() });
    }
    if gsp.len() < required_out {
        return Err(cintxRsError::BufferTooSmall { required: required_out, provided: gsp.len() });
    }

    for k in 0..nctr_usize {
        let base = k * 4 * nf;
        let v1 = &gcart[base..base + nf];
        let vx = &gcart[base + nf..base + 2 * nf];
        let vy = &gcart[base + 2 * nf..base + 3 * nf];
        let vz = &gcart[base + 3 * nf..base + 4 * nf];
        let gsp_slice = &mut gsp[k * out_per_ctr..(k + 1) * out_per_ctr];
        c2spinor::cart_to_spinor_iket_si(gsp_slice, v1, vx, vy, vz, lu, kappa)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// l=0 (s-shell) must be the identity. Internal callers CINTc2s_ket_sph /
    /// CINTc2s_ket_sph1 funnel through CINTc2s_bra_sph(.., 0, .., 0); this pins
    /// the path they depend on. Expected PASS even before the fix (stub identity).
    #[test]
    fn bra_sph_l0_identity() {
        let mut out = vec![0.0; 1]; // nsph(0) = 1
        CINTc2s_bra_sph(&mut out, 1, &[7.0], 0).unwrap();
        assert_eq!(out, vec![7.0]);
    }

    /// l=1 (p-shell, non-PYPZPX) is the identity in libcint. Expected PASS even
    /// before the fix (stub identity). 3 cart (ncart(1)) -> 3 sph (nsph(1)).
    #[test]
    fn bra_sph_l1_identity() {
        let mut out = vec![0.0; 3];
        CINTc2s_bra_sph(&mut out, 1, &[1.0, 2.0, 3.0], 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    /// l=2 (d-shell): real cart->sph transform. RED against the identity stub.
    /// cart = [xx, xy, xz, yy, yz, zz]; sph rows m=-2..+2 from C2S_L2.
    #[test]
    fn bra_sph_l2_d_transform() {
        let cart = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut sph = vec![0.0; 5]; // nsph(2) = 5
        CINTc2s_bra_sph(&mut sph, 1, &cart, 2).unwrap();

        let expected = [
            1.092548430592079070 * cart[1],                                   // m=-2 dxy
            1.092548430592079070 * cart[4],                                   // m=-1 dyz
            -0.315391565252520002 * cart[0] - 0.315391565252520002 * cart[3]  // m= 0 dz2
                + 0.630783130505040012 * cart[5],
            1.092548430592079070 * cart[2],                                   // m=+1 dxz
            0.546274215296039535 * cart[0] - 0.546274215296039535 * cart[3],  // m=+2 dx2y2
        ];
        for m in 0..5 {
            assert!(
                (sph[m] - expected[m]).abs() < 1e-12,
                "d-transform sph[{m}] = {} expected {}",
                sph[m],
                expected[m]
            );
        }
    }

    /// l=2 with nket=2: ket-blocked layout (sph row fastest, advance per ket).
    /// RED against the identity stub.
    #[test]
    fn bra_sph_l2_nket2_blocking() {
        // two ket blocks of 6 cart values each
        let cart = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // block 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // block 1
        ];
        let mut sph = vec![0.0; 10]; // 2 * nsph(2)
        CINTc2s_bra_sph(&mut sph, 2, &cart, 2).unwrap();

        let d_transform = |c: &[f64]| -> [f64; 5] {
            [
                1.092548430592079070 * c[1],
                1.092548430592079070 * c[4],
                -0.315391565252520002 * c[0] - 0.315391565252520002 * c[3]
                    + 0.630783130505040012 * c[5],
                1.092548430592079070 * c[2],
                0.546274215296039535 * c[0] - 0.546274215296039535 * c[3],
            ]
        };
        let exp0 = d_transform(&cart[0..6]);
        let exp1 = d_transform(&cart[6..12]);
        for m in 0..5 {
            assert!(
                (sph[m] - exp0[m]).abs() < 1e-12,
                "block0 sph[{m}] = {} expected {}",
                sph[m],
                exp0[m]
            );
            assert!(
                (sph[5 + m] - exp1[m]).abs() < 1e-12,
                "block1 sph[{m}] = {} expected {}",
                sph[5 + m],
                exp1[m]
            );
        }
    }

    /// Verify CINTc2s_ket_spinor_sf1 delegates to cart_to_spinor_sf correctly.
    /// Use l=0, kappa=-1 (s-shell, gt block): expected nd=2, total output 8 f64.
    #[test]
    fn compat_ket_spinor_sf1_delegates_correctly() {
        use cintx_cubecl::transform::c2spinor;

        let cart = [1.0f64];
        let l = 0i32;
        let kappa = -1i32;
        let nd = c2spinor::spinor_len(0, -1); // 2
        let mut gsp_compat = vec![0.0f64; 4 * nd];
        let mut gsp_direct = vec![0.0f64; 4 * nd];

        CINTc2s_ket_spinor_sf1(&mut gsp_compat, &cart, 0, 1, 1, l, kappa).unwrap();
        c2spinor::cart_to_spinor_sf(&mut gsp_direct, &cart, 0, kappa).unwrap();

        for (i, (a, b)) in gsp_compat.iter().zip(gsp_direct.iter()).enumerate() {
            assert!((a - b).abs() < 1e-15, "compat vs direct at [{}]: {} vs {}", i, a, b);
        }
    }

    /// All four variants should produce different outputs for non-trivial p-shell input.
    ///
    /// - sf vs iket_sf: differ in imaginary sign pattern
    /// - sf vs si: differ when Pauli components are non-zero
    /// - si vs iket_si: differ in real/imag sign pattern
    #[test]
    fn compat_all_four_variants_produce_different_output() {
        use cintx_cubecl::transform::c2spinor;
        use cintx_cubecl::transform::c2s::ncart;

        let l = 1i32;
        let kappa = -1i32;
        let lu = l as u8;
        let nf = ncart(lu);
        let nd = c2spinor::spinor_len(lu, kappa);

        // p-shell: [px, py, pz]
        let cart_sf = vec![1.0f64, 0.5, 0.3];
        // si needs 4*nf = 12 values: v1, vx, vy, vz
        let cart_si: Vec<f64> = {
            let mut v = cart_sf.clone(); // v1
            v.extend_from_slice(&[0.2f64, 0.4, 0.1]); // vx
            v.extend_from_slice(&[0.3f64, 0.1, 0.5]); // vy
            v.extend_from_slice(&[0.1f64, 0.2, 0.4]); // vz
            v
        };

        let mut gsp_sf = vec![0.0f64; 4 * nd];
        let mut gsp_iket_sf = vec![0.0f64; 4 * nd];
        let mut gsp_si = vec![0.0f64; 4 * nd];
        let mut gsp_iket_si = vec![0.0f64; 4 * nd];

        CINTc2s_ket_spinor_sf1(&mut gsp_sf, &cart_sf, 0, 1, 1, l, kappa).unwrap();
        CINTc2s_iket_spinor_sf1(&mut gsp_iket_sf, &cart_sf, 0, 1, 1, l, kappa).unwrap();
        CINTc2s_ket_spinor_si1(&mut gsp_si, &cart_si, 0, 1, 1, l, kappa).unwrap();
        CINTc2s_iket_spinor_si1(&mut gsp_iket_si, &cart_si, 0, 1, 1, l, kappa).unwrap();

        // sf vs iket_sf must differ (iket multiplies by i)
        let sf_vs_iket = gsp_sf.iter().zip(gsp_iket_sf.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(sf_vs_iket, "sf and iket_sf should differ");

        // sf vs si must differ when Pauli is non-zero
        let sf_vs_si = gsp_sf.iter().zip(gsp_si.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(sf_vs_si, "sf and si should differ when Pauli components are non-zero");

        // si vs iket_si must differ
        let si_vs_iket = gsp_si.iter().zip(gsp_iket_si.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(si_vs_iket, "si and iket_si should differ");

        let _ = nf; // suppress unused warning
    }
}
