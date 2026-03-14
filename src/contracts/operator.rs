use super::ContractResult;
use crate::errors::LibcintRsError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegralFamily {
    OneElectron,
    TwoElectron,
    TwoCenterTwoElectron,
    ThreeCenterOneElectron,
    ThreeCenterTwoElectron,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    Overlap,
    Kinetic,
    NuclearAttraction,
    ElectronRepulsion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Operator {
    family: IntegralFamily,
    kind: OperatorKind,
}

impl Operator {
    pub fn new(family: IntegralFamily, kind: OperatorKind) -> ContractResult<Self> {
        if is_supported_pair(family, kind) {
            Ok(Self { family, kind })
        } else {
            Err(LibcintRsError::UnsupportedApi {
                api: "operator",
                reason: "operator kind is not valid for selected integral family",
            })
        }
    }

    pub fn family(&self) -> IntegralFamily {
        self.family
    }

    pub fn kind(&self) -> OperatorKind {
        self.kind
    }
}

fn is_supported_pair(family: IntegralFamily, kind: OperatorKind) -> bool {
    matches!(
        (family, kind),
        (
            IntegralFamily::OneElectron | IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Overlap | OperatorKind::Kinetic | OperatorKind::NuclearAttraction
        ) | (
            IntegralFamily::TwoElectron
                | IntegralFamily::TwoCenterTwoElectron
                | IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion
        )
    )
}
