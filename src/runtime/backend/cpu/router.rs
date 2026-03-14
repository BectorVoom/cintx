use crate::contracts::{IntegralFamily, OperatorKind, Representation};
use crate::errors::LibcintRsError;
use crate::runtime::ExecutionRequest;

use super::ffi::CpuKernelSymbol;
use super::spinor_3c1e::{Spinor3c1eAdapter, adapter_route};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpuRouteKey {
    pub family: IntegralFamily,
    pub operator: OperatorKind,
    pub representation: Representation,
}

impl CpuRouteKey {
    pub const fn new(
        family: IntegralFamily,
        operator: OperatorKind,
        representation: Representation,
    ) -> Self {
        Self {
            family,
            operator,
            representation,
        }
    }
}

impl From<&ExecutionRequest> for CpuRouteKey {
    fn from(request: &ExecutionRequest) -> Self {
        Self::new(
            request.operator.family,
            request.operator.kind,
            request.representation,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuRouteTarget {
    Direct(CpuKernelSymbol),
    ThreeCenterOneElectronSpinor(Spinor3c1eAdapter),
}

impl CpuRouteTarget {
    pub fn entry_symbol(self) -> CpuKernelSymbol {
        match self {
            Self::Direct(symbol) => symbol,
            Self::ThreeCenterOneElectronSpinor(adapter) => adapter.driver_symbol,
        }
    }
}

pub fn route_request(request: &ExecutionRequest) -> Result<CpuRouteTarget, LibcintRsError> {
    route(CpuRouteKey::from(request))
}

pub fn route(key: CpuRouteKey) -> Result<CpuRouteTarget, LibcintRsError> {
    let direct_symbol = match (key.family, key.operator, key.representation) {
        (IntegralFamily::OneElectron, OperatorKind::Overlap, Representation::Cartesian) => {
            Some(CpuKernelSymbol::Int1eOvlpCart)
        }
        (IntegralFamily::OneElectron, OperatorKind::Overlap, Representation::Spherical) => {
            Some(CpuKernelSymbol::Int1eOvlpSph)
        }
        (IntegralFamily::OneElectron, OperatorKind::Overlap, Representation::Spinor) => {
            Some(CpuKernelSymbol::Int1eOvlpSpinor)
        }
        (
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ) => Some(CpuKernelSymbol::Int2eCart),
        (
            IntegralFamily::TwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ) => Some(CpuKernelSymbol::Int2eSph),
        (IntegralFamily::TwoElectron, OperatorKind::ElectronRepulsion, Representation::Spinor) => {
            Some(CpuKernelSymbol::Int2eSpinor)
        }
        (
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ) => Some(CpuKernelSymbol::Int2c2eIp1Cart),
        (
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ) => Some(CpuKernelSymbol::Int2c2eIp1Sph),
        (
            IntegralFamily::TwoCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
        ) => Some(CpuKernelSymbol::Int2c2eIp1Spinor),
        (
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Cartesian,
        ) => Some(CpuKernelSymbol::Int3c1eP2Cart),
        (
            IntegralFamily::ThreeCenterOneElectron,
            OperatorKind::Kinetic,
            Representation::Spherical,
        ) => Some(CpuKernelSymbol::Int3c1eP2Sph),
        (IntegralFamily::ThreeCenterOneElectron, OperatorKind::Kinetic, Representation::Spinor) => {
            return Ok(CpuRouteTarget::ThreeCenterOneElectronSpinor(adapter_route()));
        }
        (
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Cartesian,
        ) => Some(CpuKernelSymbol::Int3c2eIp1Cart),
        (
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spherical,
        ) => Some(CpuKernelSymbol::Int3c2eIp1Sph),
        (
            IntegralFamily::ThreeCenterTwoElectron,
            OperatorKind::ElectronRepulsion,
            Representation::Spinor,
        ) => Some(CpuKernelSymbol::Int3c2eIp1Spinor),
        _ => None,
    };

    direct_symbol
        .map(CpuRouteTarget::Direct)
        .ok_or_else(unsupported_route_error)
}

fn unsupported_route_error() -> LibcintRsError {
    LibcintRsError::UnsupportedApi {
        api: "cpu.route",
        reason: "route is outside the phase-2 stable-family envelope",
    }
}
