#![allow(non_snake_case)]

use cintx_core::cintxRsError;
use cintx_cubecl::transform::{c2s, c2spinor};
use cintx_cubecl::transform::c2s::ncart;

fn copy_cart_into_target(target: &mut [f64], cart: &[f64]) -> Result<(), cintxRsError> {
    if target.len() < cart.len() {
        return Err(cintxRsError::BufferTooSmall {
            required: cart.len(),
            provided: target.len(),
        });
    }
    target[..cart.len()].copy_from_slice(cart);
    Ok(())
}

pub fn CINTc2s_bra_sph(
    sph: &mut [f64],
    _nket: i32,
    cart: &[f64],
    _l: i32,
) -> Result<(), cintxRsError> {
    copy_cart_into_target(sph, cart)?;
    c2s::cart_to_spheric_staging(&mut sph[..cart.len()])
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

    #[test]
    fn spherical_transform_entry_points_work() {
        // cart_to_spheric_staging is a no-op — real c2s is done per-shell
        // in cart_to_sph_1e(). This test verifies the compat shim copies
        // cart data through without error.
        let mut out = vec![0.0; 4];
        CINTc2s_bra_sph(&mut out, 1, &[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
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
