#![allow(non_snake_case)]

use cintx_core::cintxRsError;
use cintx_cubecl::transform::{c2s, c2spinor};

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

fn spinor_transform(target: &mut [f64], cart: &[f64]) -> Result<(), cintxRsError> {
    copy_cart_into_target(target, cart)?;
    c2spinor::cart_to_spinor_interleaved_staging(&mut target[..cart.len()])
}

pub fn CINTc2s_ket_spinor_sf1(
    gsp: &mut [f64],
    gcart: &[f64],
    _lds: i32,
    _ldc: i32,
    _nctr: i32,
    _l: i32,
    _kappa: i32,
) -> Result<(), cintxRsError> {
    spinor_transform(gsp, gcart)
}

pub fn CINTc2s_iket_spinor_sf1(
    gsp: &mut [f64],
    gcart: &[f64],
    lds: i32,
    ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    CINTc2s_ket_spinor_sf1(gsp, gcart, lds, ldc, nctr, l, kappa)
}

pub fn CINTc2s_ket_spinor_si1(
    gsp: &mut [f64],
    gcart: &[f64],
    lds: i32,
    ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    CINTc2s_ket_spinor_sf1(gsp, gcart, lds, ldc, nctr, l, kappa)
}

pub fn CINTc2s_iket_spinor_si1(
    gsp: &mut [f64],
    gcart: &[f64],
    lds: i32,
    ldc: i32,
    nctr: i32,
    l: i32,
    kappa: i32,
) -> Result<(), cintxRsError> {
    CINTc2s_ket_spinor_sf1(gsp, gcart, lds, ldc, nctr, l, kappa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_transform_entry_points_work() {
        let mut out = vec![0.0; 4];
        CINTc2s_bra_sph(&mut out, 1, &[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        assert_eq!(out, vec![0.5, 1.5, 2.5, 3.5]);
    }

    #[test]
    fn spinor_transform_entry_points_keep_interleaved_layout() {
        let mut out = vec![0.0; 4];
        CINTc2s_ket_spinor_sf1(&mut out, &[1.0, 2.0, 3.0, 5.0], 0, 0, 1, 1, 0).unwrap();
        assert_eq!(out, vec![1.5, -1.5, 4.0, -4.0]);
    }
}
