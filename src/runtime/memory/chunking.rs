use crate::contracts::{IntegralFamily, OperatorKind};
use crate::errors::LibcintRsError;

pub const DEFAULT_ALIGNMENT_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    pub total_elements: usize,
    pub chunk_elements: usize,
    pub chunk_count: usize,
}

impl ChunkPlan {
    pub const fn is_chunked(self) -> bool {
        self.chunk_count > 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub payload_bytes: usize,
    pub scratch_bytes: usize,
    pub required_bytes: usize,
    pub working_set_bytes: usize,
    pub chunk_plan: ChunkPlan,
}

pub fn compute_scratch_bytes(
    shell_angular_momentum: &[u8],
    primitive_count: usize,
    dims_len: usize,
    operator_kind: OperatorKind,
    family: IntegralFamily,
    feature_flag_count: usize,
) -> Result<usize, LibcintRsError> {
    let angular_complexity =
        shell_angular_momentum
            .iter()
            .try_fold(0usize, |acc, angular_momentum| {
                acc.checked_add(usize::from(*angular_momentum) + 1)
                    .ok_or_else(|| LibcintRsError::InvalidInput {
                        field: "workspace",
                        reason: "angular complexity overflows usize".to_string(),
                    })
            })?;
    let scratch_units = angular_complexity
        .checked_add(primitive_count)
        .and_then(|value| value.checked_add(dims_len))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "scratch unit computation overflows usize".to_string(),
        })?;
    let operator_scale = operator_scale(operator_kind);
    let family_scale = family_scale(family);
    let feature_scale = feature_flag_count.max(1);

    scratch_units
        .checked_mul(16)
        .and_then(|value| value.checked_mul(operator_scale))
        .and_then(|value| value.checked_mul(family_scale))
        .and_then(|value| value.checked_mul(feature_scale))
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "scratch byte computation overflows usize".to_string(),
        })
}

pub fn build_memory_plan(
    element_count: usize,
    element_width_bytes: usize,
    scratch_bytes: usize,
    memory_limit_bytes: Option<usize>,
) -> Result<MemoryPlan, LibcintRsError> {
    let payload_bytes = element_count
        .checked_mul(element_width_bytes)
        .ok_or_else(|| LibcintRsError::InvalidInput {
            field: "workspace",
            reason: "payload byte computation overflows usize".to_string(),
        })?;
    let required_unaligned =
        payload_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "workspace",
                reason: "required byte computation overflows usize".to_string(),
            })?;
    let required_bytes =
        align_up(required_unaligned, DEFAULT_ALIGNMENT_BYTES).ok_or_else(|| {
            LibcintRsError::InvalidInput {
                field: "workspace",
                reason: "required byte alignment overflow".to_string(),
            }
        })?;
    let (chunk_elements, working_set_bytes) = match memory_limit_bytes {
        None => (element_count, required_bytes),
        Some(limit_bytes) if required_bytes <= limit_bytes => (element_count, required_bytes),
        Some(limit_bytes) => {
            let aligned_limit =
                align_down(limit_bytes, DEFAULT_ALIGNMENT_BYTES).ok_or_else(|| {
                    LibcintRsError::InvalidInput {
                        field: "workspace",
                        reason: "memory limit alignment is invalid".to_string(),
                    }
                })?;
            let min_working_unaligned =
                scratch_bytes
                    .checked_add(element_width_bytes)
                    .ok_or_else(|| LibcintRsError::InvalidInput {
                        field: "workspace",
                        reason: "minimum chunk working-set computation overflows usize"
                            .to_string(),
                    })?;
            let min_working_bytes =
                align_up(min_working_unaligned, DEFAULT_ALIGNMENT_BYTES).ok_or_else(|| {
                    LibcintRsError::InvalidInput {
                        field: "workspace",
                        reason: "minimum chunk working-set alignment overflow".to_string(),
                    }
                })?;
            if min_working_bytes > limit_bytes {
                return Err(LibcintRsError::MemoryLimitExceeded {
                    required_bytes,
                    limit_bytes,
                });
            }

            let available_payload = aligned_limit.saturating_sub(scratch_bytes);
            let feasible_chunk_elements = available_payload / element_width_bytes;
            if feasible_chunk_elements == 0 {
                return Err(LibcintRsError::MemoryLimitExceeded {
                    required_bytes,
                    limit_bytes,
                });
            }

            let chunk_elements = feasible_chunk_elements.min(element_count);
            let chunk_payload_bytes = chunk_elements.checked_mul(element_width_bytes).ok_or_else(|| {
                LibcintRsError::InvalidInput {
                    field: "workspace",
                    reason: "chunk payload byte computation overflows usize".to_string(),
                }
            })?;
            let chunk_working_unaligned = scratch_bytes.checked_add(chunk_payload_bytes).ok_or_else(|| {
                LibcintRsError::InvalidInput {
                    field: "workspace",
                    reason: "chunk working-set computation overflows usize".to_string(),
                }
            })?;
            let working_set_bytes = align_up(chunk_working_unaligned, DEFAULT_ALIGNMENT_BYTES)
                .ok_or_else(|| LibcintRsError::InvalidInput {
                    field: "workspace",
                    reason: "chunk working-set alignment overflow".to_string(),
                })?;
            if working_set_bytes > limit_bytes {
                return Err(LibcintRsError::MemoryLimitExceeded {
                    required_bytes,
                    limit_bytes,
                });
            }
            (chunk_elements, working_set_bytes)
        }
    };

    let chunk_count = if chunk_elements == 0 {
        0
    } else {
        element_count.div_ceil(chunk_elements)
    };

    Ok(MemoryPlan {
        payload_bytes,
        scratch_bytes,
        required_bytes,
        working_set_bytes,
        chunk_plan: ChunkPlan {
            total_elements: element_count,
            chunk_elements,
            chunk_count,
        },
    })
}

fn operator_scale(kind: OperatorKind) -> usize {
    match kind {
        OperatorKind::Overlap => 1,
        OperatorKind::Kinetic => 2,
        OperatorKind::NuclearAttraction => 3,
        OperatorKind::ElectronRepulsion => 4,
    }
}

fn family_scale(family: IntegralFamily) -> usize {
    match family {
        IntegralFamily::OneElectron => 2,
        IntegralFamily::TwoCenterTwoElectron => 2,
        IntegralFamily::ThreeCenterOneElectron => 3,
        IntegralFamily::ThreeCenterTwoElectron => 3,
        IntegralFamily::TwoElectron => 4,
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }

    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

fn align_down(value: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }

    Some(value & !(alignment - 1))
}
