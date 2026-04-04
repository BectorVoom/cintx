pub mod c2s;
pub mod c2spinor;
pub mod c2spinor_coeffs;

use cintx_core::{Representation, cintxRsError};

pub fn apply_representation_transform(
    representation: Representation,
    staging: &mut [f64],
) -> Result<(), cintxRsError> {
    match representation {
        Representation::Cart => Ok(()),
        Representation::Spheric => c2s::cart_to_spheric_staging(staging),
        // Spinor: the real per-shell transform requires l and kappa passed explicitly.
        // The staging path calls cart_to_spinor_interleaved_staging (no-op) because
        // actual spinor transforms are done via cart_to_spinor_sf et al. with known l/kappa.
        // TODO: extend executor dispatch to pass l and kappa for full spinor oracle paths.
        Representation::Spinor => c2spinor::cart_to_spinor_interleaved_staging(staging),
    }
}
