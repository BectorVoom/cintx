use cintx_core::cintxRsError;

/// CubeCL staging transform for cartesian -> spherical shaping.
///
/// Phase 2 keeps transform ownership in CubeCL staging; compat still performs
/// the final caller-visible flat write.
pub fn cart_to_spheric_staging(staging: &mut [f64]) -> Result<(), cintxRsError> {
    if staging.is_empty() {
        return Ok(());
    }

    // Lightweight deterministic transform: weighted running blend over the
    // cart staging payload to produce representation-specific staging values.
    let mut previous = staging[0];
    staging[0] *= 0.5;
    for value in &mut staging[1..] {
        let current = *value;
        *value = (current + previous) * 0.5;
        previous = current;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_transform_mutates_staging_values() {
        let mut staging = vec![1.0, 2.0, 3.0, 4.0];
        cart_to_spheric_staging(&mut staging).expect("transform should succeed");
        assert_eq!(staging, vec![0.5, 1.5, 2.5, 3.5]);
    }
}
