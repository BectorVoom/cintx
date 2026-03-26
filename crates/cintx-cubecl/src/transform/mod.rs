pub mod c2s;
pub mod c2spinor;

use cintx_core::{cintxRsError, Representation};

pub fn apply_representation_transform(
    representation: Representation,
    staging: &mut [f64],
) -> Result<(), cintxRsError> {
    match representation {
        Representation::Cart => Ok(()),
        Representation::Spheric => c2s::cart_to_spheric_staging(staging),
        Representation::Spinor => c2spinor::cart_to_spinor_interleaved_staging(staging),
    }
}
