use crate::contracts::Representation;
use crate::errors::LibcintRsError;

use super::PlannedExecution;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutElementKind {
    RealF64,
    ComplexF64Pair,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLayout {
    pub representation: Representation,
    pub dims: Vec<usize>,
    pub element_count: usize,
    pub element_width_bytes: usize,
    pub required_bytes: usize,
    pub element_kind: LayoutElementKind,
}

impl OutputLayout {
    pub fn expected_real_elements(&self) -> Option<usize> {
        match self.element_kind {
            LayoutElementKind::RealF64 => Some(self.element_count),
            LayoutElementKind::ComplexF64Pair => None,
        }
    }

    pub fn expected_complex_elements(&self) -> Option<usize> {
        match self.element_kind {
            LayoutElementKind::RealF64 => None,
            LayoutElementKind::ComplexF64Pair => Some(self.element_count),
        }
    }

    pub fn validate_real_buffer_len(&self, provided_len: usize) -> Result<(), LibcintRsError> {
        let Some(expected) = self.expected_real_elements() else {
            return Err(LibcintRsError::UnsupportedRepresentation {
                api: "output.layout.real",
                representation: self.representation.as_str(),
            });
        };
        if provided_len != expected {
            return Err(LibcintRsError::InvalidLayout {
                item: "output_elements",
                expected,
                got: provided_len,
            });
        }
        Ok(())
    }

    pub fn validate_complex_buffer_len(&self, provided_len: usize) -> Result<(), LibcintRsError> {
        let Some(expected) = self.expected_complex_elements() else {
            return Err(LibcintRsError::UnsupportedRepresentation {
                api: "output.layout.complex",
                representation: self.representation.as_str(),
            });
        };
        if provided_len != expected {
            return Err(LibcintRsError::InvalidLayout {
                item: "output_elements",
                expected,
                got: provided_len,
            });
        }
        Ok(())
    }
}

pub fn layout_for_plan(plan: &PlannedExecution) -> OutputLayout {
    OutputLayout {
        representation: plan.request.representation,
        dims: plan.dims.clone(),
        element_count: plan.element_count,
        element_width_bytes: plan.element_width_bytes,
        required_bytes: plan.required_output_bytes,
        element_kind: match plan.request.representation {
            Representation::Spinor => LayoutElementKind::ComplexF64Pair,
            Representation::Cartesian | Representation::Spherical => LayoutElementKind::RealF64,
        },
    }
}
