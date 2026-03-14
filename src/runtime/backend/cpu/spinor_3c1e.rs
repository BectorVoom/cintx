use super::ffi::CpuKernelSymbol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spinor3c1eTransform {
    SphericalKernelToSpinorLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spinor3c1eAdapter {
    pub driver_symbol: CpuKernelSymbol,
    pub transform: Spinor3c1eTransform,
}

pub const fn adapter_route() -> Spinor3c1eAdapter {
    Spinor3c1eAdapter {
        driver_symbol: CpuKernelSymbol::Int3c1eP2Sph,
        transform: Spinor3c1eTransform::SphericalKernelToSpinorLayout,
    }
}
