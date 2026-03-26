use cintx_core::cintxRsError;

/// CubeCL staging transform for cartesian -> spinor interleaved doubles.
///
/// The resulting staging buffer remains `[re0, im0, re1, im1, ...]` and is
/// handed back to compat for the final flat write.
pub fn cart_to_spinor_interleaved_staging(staging: &mut [f64]) -> Result<(), cintxRsError> {
    if staging.len() % 2 != 0 {
        return Err(cintxRsError::ChunkPlanFailed {
            from: "cubecl_c2spinor",
            detail: "spinor staging must contain interleaved real/imag pairs".to_owned(),
        });
    }

    for pair in staging.chunks_exact_mut(2) {
        let amplitude = (pair[0].abs() + pair[1].abs()) * 0.5;
        pair[0] = amplitude;
        pair[1] = -amplitude;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinor_transform_preserves_interleaved_pairs() {
        let mut staging = vec![1.0, 2.0, 3.0, 5.0];
        cart_to_spinor_interleaved_staging(&mut staging).expect("transform should succeed");
        assert_eq!(staging, vec![1.5, -1.5, 4.0, -4.0]);
    }

    #[test]
    fn spinor_transform_rejects_non_interleaved_length() {
        let mut staging = vec![1.0, 2.0, 3.0];
        let err = cart_to_spinor_interleaved_staging(&mut staging).unwrap_err();
        assert!(matches!(err, cintxRsError::ChunkPlanFailed { .. }));
    }
}
