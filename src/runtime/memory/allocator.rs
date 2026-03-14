use crate::errors::LibcintRsError;

pub fn try_alloc_real_buffer(
    element_count: usize,
    operation: &'static str,
) -> Result<Vec<f64>, LibcintRsError> {
    try_alloc_filled(element_count, 0.0, operation, "f64")
}

pub fn try_alloc_spinor_buffer(
    element_count: usize,
    operation: &'static str,
) -> Result<Vec<[f64; 2]>, LibcintRsError> {
    try_alloc_filled(element_count, [0.0, 0.0], operation, "spinor complex")
}

fn try_alloc_filled<T: Clone>(
    element_count: usize,
    fill: T,
    operation: &'static str,
    element_label: &'static str,
) -> Result<Vec<T>, LibcintRsError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count)
        .map_err(|_| LibcintRsError::AllocationFailure {
            operation,
            detail: format!("failed to reserve {element_count} {element_label} elements"),
        })?;
    values.resize(element_count, fill);
    Ok(values)
}
