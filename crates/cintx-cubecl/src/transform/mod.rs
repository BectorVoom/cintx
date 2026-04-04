pub mod c2s;
pub mod c2spinor;
pub mod c2spinor_coeffs;

use cintx_core::{Representation, cintxRsError};

pub fn apply_representation_transform(
    representation: Representation,
    staging: &mut [f64],
) -> Result<(), cintxRsError> {
    match representation {
        Representation::Cart => {
            let _ = staging;
            Ok(())
        }
        Representation::Spheric => c2s::cart_to_spheric_staging(staging),
        // Spinor: apply_representation_transform does NOT support Spinor.
        // Spinor transforms require explicit l and kappa per shell; use
        // cart_to_spinor_sf_2d (for 1e/2c2e), cart_to_spinor_sf_4d (for 2e),
        // or cart_to_spinor_sf_3c2e (for 3c2e) directly in the kernel launcher.
        Representation::Spinor => {
            let _ = staging;
            Err(cintxRsError::UnsupportedApi {
                requested: "apply_representation_transform does not support Spinor — \
                            use explicit cart_to_spinor_sf_2d/4d in kernel launchers".to_owned(),
            })
        }
    }
}
