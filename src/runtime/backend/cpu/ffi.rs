use crate::contracts::{IntegralFamily, OperatorKind, Representation};

pub type CpuKernelFn = unsafe extern "C" fn();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuKernelSymbol {
    Int1eOvlpCart,
    Int1eOvlpSph,
    Int1eOvlpSpinor,
    Int2eCart,
    Int2eSph,
    Int2eSpinor,
    Int2c2eIp1Cart,
    Int2c2eIp1Sph,
    Int2c2eIp1Spinor,
    Int3c1eP2Cart,
    Int3c1eP2Sph,
    Int3c1eP2Spinor,
    Int3c2eIp1Cart,
    Int3c2eIp1Sph,
    Int3c2eIp1Spinor,
}

impl CpuKernelSymbol {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Int1eOvlpCart => "int1e_ovlp_cart",
            Self::Int1eOvlpSph => "int1e_ovlp_sph",
            Self::Int1eOvlpSpinor => "int1e_ovlp_spinor",
            Self::Int2eCart => "int2e_cart",
            Self::Int2eSph => "int2e_sph",
            Self::Int2eSpinor => "int2e_spinor",
            Self::Int2c2eIp1Cart => "int2c2e_ip1_cart",
            Self::Int2c2eIp1Sph => "int2c2e_ip1_sph",
            Self::Int2c2eIp1Spinor => "int2c2e_ip1_spinor",
            Self::Int3c1eP2Cart => "int3c1e_p2_cart",
            Self::Int3c1eP2Sph => "int3c1e_p2_sph",
            Self::Int3c1eP2Spinor => "int3c1e_p2_spinor",
            Self::Int3c2eIp1Cart => "int3c2e_ip1_cart",
            Self::Int3c2eIp1Sph => "int3c2e_ip1_sph",
            Self::Int3c2eIp1Spinor => "int3c2e_ip1_spinor",
        }
    }

    pub const fn family(self) -> IntegralFamily {
        match self {
            Self::Int1eOvlpCart | Self::Int1eOvlpSph | Self::Int1eOvlpSpinor => {
                IntegralFamily::OneElectron
            }
            Self::Int2eCart | Self::Int2eSph | Self::Int2eSpinor => IntegralFamily::TwoElectron,
            Self::Int2c2eIp1Cart | Self::Int2c2eIp1Sph | Self::Int2c2eIp1Spinor => {
                IntegralFamily::TwoCenterTwoElectron
            }
            Self::Int3c1eP2Cart | Self::Int3c1eP2Sph | Self::Int3c1eP2Spinor => {
                IntegralFamily::ThreeCenterOneElectron
            }
            Self::Int3c2eIp1Cart | Self::Int3c2eIp1Sph | Self::Int3c2eIp1Spinor => {
                IntegralFamily::ThreeCenterTwoElectron
            }
        }
    }

    pub const fn operator(self) -> OperatorKind {
        match self {
            Self::Int1eOvlpCart | Self::Int1eOvlpSph | Self::Int1eOvlpSpinor => {
                OperatorKind::Overlap
            }
            Self::Int2eCart | Self::Int2eSph | Self::Int2eSpinor => OperatorKind::ElectronRepulsion,
            Self::Int2c2eIp1Cart | Self::Int2c2eIp1Sph | Self::Int2c2eIp1Spinor => {
                OperatorKind::ElectronRepulsion
            }
            Self::Int3c1eP2Cart | Self::Int3c1eP2Sph | Self::Int3c1eP2Spinor => {
                OperatorKind::Kinetic
            }
            Self::Int3c2eIp1Cart | Self::Int3c2eIp1Sph | Self::Int3c2eIp1Spinor => {
                OperatorKind::ElectronRepulsion
            }
        }
    }

    pub const fn representation(self) -> Representation {
        match self {
            Self::Int1eOvlpCart
            | Self::Int2eCart
            | Self::Int2c2eIp1Cart
            | Self::Int3c1eP2Cart
            | Self::Int3c2eIp1Cart => Representation::Cartesian,
            Self::Int1eOvlpSph
            | Self::Int2eSph
            | Self::Int2c2eIp1Sph
            | Self::Int3c1eP2Sph
            | Self::Int3c2eIp1Sph => Representation::Spherical,
            Self::Int1eOvlpSpinor
            | Self::Int2eSpinor
            | Self::Int2c2eIp1Spinor
            | Self::Int3c1eP2Spinor
            | Self::Int3c2eIp1Spinor => Representation::Spinor,
        }
    }

    pub fn function(self) -> CpuKernelFn {
        match self {
            Self::Int1eOvlpCart => int1e_ovlp_cart,
            Self::Int1eOvlpSph => int1e_ovlp_sph,
            Self::Int1eOvlpSpinor => int1e_ovlp_spinor,
            Self::Int2eCart => int2e_cart,
            Self::Int2eSph => int2e_sph,
            Self::Int2eSpinor => int2e_spinor,
            Self::Int2c2eIp1Cart => int2c2e_ip1_cart,
            Self::Int2c2eIp1Sph => int2c2e_ip1_sph,
            Self::Int2c2eIp1Spinor => int2c2e_ip1_spinor,
            Self::Int3c1eP2Cart => int3c1e_p2_cart,
            Self::Int3c1eP2Sph => int3c1e_p2_sph,
            Self::Int3c1eP2Spinor => int3c1e_p2_spinor,
            Self::Int3c2eIp1Cart => int3c2e_ip1_cart,
            Self::Int3c2eIp1Sph => int3c2e_ip1_sph,
            Self::Int3c2eIp1Spinor => int3c2e_ip1_spinor,
        }
    }

    pub fn as_ptr(self) -> *const () {
        self.function() as *const ()
    }
}

pub const ALL_BOUND_SYMBOLS: &[CpuKernelSymbol] = &[
    CpuKernelSymbol::Int1eOvlpCart,
    CpuKernelSymbol::Int1eOvlpSph,
    CpuKernelSymbol::Int1eOvlpSpinor,
    CpuKernelSymbol::Int2eCart,
    CpuKernelSymbol::Int2eSph,
    CpuKernelSymbol::Int2eSpinor,
    CpuKernelSymbol::Int2c2eIp1Cart,
    CpuKernelSymbol::Int2c2eIp1Sph,
    CpuKernelSymbol::Int2c2eIp1Spinor,
    CpuKernelSymbol::Int3c1eP2Cart,
    CpuKernelSymbol::Int3c1eP2Sph,
    CpuKernelSymbol::Int3c1eP2Spinor,
    CpuKernelSymbol::Int3c2eIp1Cart,
    CpuKernelSymbol::Int3c2eIp1Sph,
    CpuKernelSymbol::Int3c2eIp1Spinor,
];

#[link(name = "cint_phase2_cpu", kind = "static")]
#[link(name = "m")]
unsafe extern "C" {
    fn int1e_ovlp_cart();
    fn int1e_ovlp_sph();
    fn int1e_ovlp_spinor();
    fn int2e_cart();
    fn int2e_sph();
    fn int2e_spinor();
    fn int2c2e_ip1_cart();
    fn int2c2e_ip1_sph();
    fn int2c2e_ip1_spinor();
    fn int3c1e_p2_cart();
    fn int3c1e_p2_sph();
    fn int3c1e_p2_spinor();
    fn int3c2e_ip1_cart();
    fn int3c2e_ip1_sph();
    fn int3c2e_ip1_spinor();
}
