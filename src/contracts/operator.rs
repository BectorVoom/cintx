use super::{ContractError, ContractResult};

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
        match (family, kind) {
            (IntegralFamily::OneElectron, OperatorKind::ElectronRepulsion)
            | (IntegralFamily::ThreeCenterOneElectron, OperatorKind::ElectronRepulsion) => {
                Err(ContractError::Unsupported {
                    field: "operator",
                    value: "electron repulsion is not valid for selected family",
                })
            }
            _ => Ok(Self { family, kind }),
        }
    }

    pub fn family(&self) -> IntegralFamily {
        self.family
    }

    pub fn kind(&self) -> OperatorKind {
        self.kind
    }
}
