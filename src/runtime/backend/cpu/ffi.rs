use core::ffi::c_void;

use crate::contracts::{IntegralFamily, OperatorKind, Representation};

pub type CpuKernelFn = unsafe extern "C" fn();
type CintInt = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuKernelSymbol {
    Int1eOvlpCart,
    Int1eKinCart,
    Int1eNucCart,
    Int1eOvlpSph,
    Int1eKinSph,
    Int1eNucSph,
    Int1eOvlpSpinor,
    Int1eKinSpinor,
    Int1eNucSpinor,
    Int2eCart,
    Int2eSph,
    Int2eSpinor,
    Int2c2eCart,
    Int2c2eSph,
    Int2c2eSpinor,
    Int3c1eP2Cart,
    Int3c1eP2Sph,
    Int3c1eP2Spinor,
    Int3c2eIp1Cart,
    Int3c2eIp1Sph,
    Int3c2eIp1Spinor,
    Int4c1eCart,
    Int4c1eSph,
}

impl CpuKernelSymbol {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Int1eOvlpCart => "int1e_ovlp_cart",
            Self::Int1eKinCart => "int1e_kin_cart",
            Self::Int1eNucCart => "int1e_nuc_cart",
            Self::Int1eOvlpSph => "int1e_ovlp_sph",
            Self::Int1eKinSph => "int1e_kin_sph",
            Self::Int1eNucSph => "int1e_nuc_sph",
            Self::Int1eOvlpSpinor => "int1e_ovlp_spinor",
            Self::Int1eKinSpinor => "int1e_kin_spinor",
            Self::Int1eNucSpinor => "int1e_nuc_spinor",
            Self::Int2eCart => "int2e_cart",
            Self::Int2eSph => "int2e_sph",
            Self::Int2eSpinor => "int2e_spinor",
            Self::Int2c2eCart => "int2c2e_cart",
            Self::Int2c2eSph => "int2c2e_sph",
            Self::Int2c2eSpinor => "int2c2e_spinor",
            Self::Int3c1eP2Cart => "int3c1e_p2_cart",
            Self::Int3c1eP2Sph => "int3c1e_p2_sph",
            Self::Int3c1eP2Spinor => "int3c1e_p2_spinor",
            Self::Int3c2eIp1Cart => "int3c2e_ip1_cart",
            Self::Int3c2eIp1Sph => "int3c2e_ip1_sph",
            Self::Int3c2eIp1Spinor => "int3c2e_ip1_spinor",
            Self::Int4c1eCart => "int4c1e_cart",
            Self::Int4c1eSph => "int4c1e_sph",
        }
    }

    pub const fn family(self) -> IntegralFamily {
        match self {
            Self::Int1eOvlpCart
            | Self::Int1eKinCart
            | Self::Int1eNucCart
            | Self::Int1eOvlpSph
            | Self::Int1eKinSph
            | Self::Int1eNucSph
            | Self::Int1eOvlpSpinor
            | Self::Int1eKinSpinor
            | Self::Int1eNucSpinor => IntegralFamily::OneElectron,
            Self::Int2eCart | Self::Int2eSph | Self::Int2eSpinor => IntegralFamily::TwoElectron,
            Self::Int2c2eCart | Self::Int2c2eSph | Self::Int2c2eSpinor => {
                IntegralFamily::TwoCenterTwoElectron
            }
            Self::Int3c1eP2Cart | Self::Int3c1eP2Sph | Self::Int3c1eP2Spinor => {
                IntegralFamily::ThreeCenterOneElectron
            }
            Self::Int3c2eIp1Cart | Self::Int3c2eIp1Sph | Self::Int3c2eIp1Spinor => {
                IntegralFamily::ThreeCenterTwoElectron
            }
            Self::Int4c1eCart | Self::Int4c1eSph => IntegralFamily::FourCenterOneElectron,
        }
    }

    pub const fn operator(self) -> OperatorKind {
        match self {
            Self::Int1eOvlpCart | Self::Int1eOvlpSph | Self::Int1eOvlpSpinor => {
                OperatorKind::Overlap
            }
            Self::Int1eKinCart | Self::Int1eKinSph | Self::Int1eKinSpinor => OperatorKind::Kinetic,
            Self::Int1eNucCart | Self::Int1eNucSph | Self::Int1eNucSpinor => {
                OperatorKind::NuclearAttraction
            }
            Self::Int2eCart | Self::Int2eSph | Self::Int2eSpinor => OperatorKind::ElectronRepulsion,
            Self::Int2c2eCart | Self::Int2c2eSph | Self::Int2c2eSpinor => {
                OperatorKind::ElectronRepulsion
            }
            Self::Int3c1eP2Cart | Self::Int3c1eP2Sph | Self::Int3c1eP2Spinor => {
                OperatorKind::Kinetic
            }
            Self::Int3c2eIp1Cart | Self::Int3c2eIp1Sph | Self::Int3c2eIp1Spinor => {
                OperatorKind::ElectronRepulsion
            }
            Self::Int4c1eCart | Self::Int4c1eSph => OperatorKind::ElectronRepulsion,
        }
    }

    pub const fn representation(self) -> Representation {
        match self {
            Self::Int1eOvlpCart
            | Self::Int1eKinCart
            | Self::Int1eNucCart
            | Self::Int2eCart
            | Self::Int2c2eCart
            | Self::Int3c1eP2Cart
            | Self::Int3c2eIp1Cart
            | Self::Int4c1eCart => Representation::Cartesian,
            Self::Int1eOvlpSph
            | Self::Int1eKinSph
            | Self::Int1eNucSph
            | Self::Int2eSph
            | Self::Int2c2eSph
            | Self::Int3c1eP2Sph
            | Self::Int3c2eIp1Sph
            | Self::Int4c1eSph => Representation::Spherical,
            Self::Int1eOvlpSpinor
            | Self::Int1eKinSpinor
            | Self::Int1eNucSpinor
            | Self::Int2eSpinor
            | Self::Int2c2eSpinor
            | Self::Int3c1eP2Spinor
            | Self::Int3c2eIp1Spinor => Representation::Spinor,
        }
    }

    pub fn function(self) -> CpuKernelFn {
        match self {
            Self::Int1eOvlpCart => int1e_ovlp_cart,
            Self::Int1eKinCart => int1e_kin_cart,
            // 1e promoted routes are executed through Rust-native specialized paths.
            // Keep non-null linked symbols for routing diagnostics and ABI contracts.
            Self::Int1eNucCart => int1e_ovlp_cart,
            Self::Int1eOvlpSph => int1e_ovlp_sph,
            Self::Int1eKinSph => int1e_ovlp_sph,
            Self::Int1eNucSph => int1e_ovlp_sph,
            Self::Int1eOvlpSpinor => int1e_ovlp_spinor,
            Self::Int1eKinSpinor => int1e_ovlp_spinor,
            Self::Int1eNucSpinor => int1e_ovlp_spinor,
            Self::Int2eCart => int2e_cart,
            Self::Int2eSph => int2e_sph,
            Self::Int2eSpinor => int2e_spinor,
            Self::Int2c2eCart => int2c2e_cart,
            Self::Int2c2eSph => int2c2e_sph,
            Self::Int2c2eSpinor => int2c2e_spinor,
            Self::Int3c1eP2Cart => int3c1e_p2_cart,
            Self::Int3c1eP2Sph => int3c1e_p2_sph,
            Self::Int3c1eP2Spinor => int3c1e_p2_spinor,
            Self::Int3c2eIp1Cart => int3c2e_ip1_cart,
            Self::Int3c2eIp1Sph => int3c2e_ip1_sph,
            Self::Int3c2eIp1Spinor => int3c2e_ip1_spinor,
            Self::Int4c1eCart => int4c1e_cart,
            Self::Int4c1eSph => int4c1e_sph,
        }
    }

    pub fn as_ptr(self) -> *const () {
        self.function() as *const ()
    }
}

pub const ALL_BOUND_SYMBOLS: &[CpuKernelSymbol] = &[
    CpuKernelSymbol::Int1eOvlpCart,
    CpuKernelSymbol::Int1eKinCart,
    CpuKernelSymbol::Int1eNucCart,
    CpuKernelSymbol::Int1eOvlpSph,
    CpuKernelSymbol::Int1eKinSph,
    CpuKernelSymbol::Int1eNucSph,
    CpuKernelSymbol::Int1eOvlpSpinor,
    CpuKernelSymbol::Int1eKinSpinor,
    CpuKernelSymbol::Int1eNucSpinor,
    CpuKernelSymbol::Int2eCart,
    CpuKernelSymbol::Int2eSph,
    CpuKernelSymbol::Int2eSpinor,
    CpuKernelSymbol::Int2c2eCart,
    CpuKernelSymbol::Int2c2eSph,
    CpuKernelSymbol::Int2c2eSpinor,
    CpuKernelSymbol::Int3c1eP2Cart,
    CpuKernelSymbol::Int3c1eP2Sph,
    CpuKernelSymbol::Int3c1eP2Spinor,
    CpuKernelSymbol::Int3c2eIp1Cart,
    CpuKernelSymbol::Int3c2eIp1Sph,
    CpuKernelSymbol::Int3c2eIp1Spinor,
    CpuKernelSymbol::Int4c1eCart,
    CpuKernelSymbol::Int4c1eSph,
];

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_two_e_real_kernel(
    symbol: CpuKernelSymbol,
    out: *mut f64,
    dims: *mut CintInt,
    shls: *mut CintInt,
    atm: *mut CintInt,
    natm: CintInt,
    bas: *mut CintInt,
    nbas: CintInt,
    env: *mut f64,
    opt: *mut c_void,
    cache: *mut f64,
) -> Option<CintInt> {
    match symbol {
        CpuKernelSymbol::Int2eCart => {
            Some(unsafe { int2e_cart_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache) })
        }
        CpuKernelSymbol::Int2eSph => {
            Some(unsafe { int2e_sph_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache) })
        }
        CpuKernelSymbol::Int2c2eCart => Some(unsafe {
            int2c2e_cart_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int2c2eSph => Some(unsafe {
            int2c2e_sph_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_two_e_spinor_kernel(
    symbol: CpuKernelSymbol,
    out: *mut c_void,
    dims: *mut CintInt,
    shls: *mut CintInt,
    atm: *mut CintInt,
    natm: CintInt,
    bas: *mut CintInt,
    nbas: CintInt,
    env: *mut f64,
    opt: *mut c_void,
    cache: *mut f64,
) -> Option<CintInt> {
    match symbol {
        CpuKernelSymbol::Int2eSpinor => Some(unsafe {
            int2e_spinor_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int2c2eSpinor => Some(unsafe {
            int2c2e_spinor_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_three_center_real_kernel(
    symbol: CpuKernelSymbol,
    out: *mut f64,
    dims: *mut CintInt,
    shls: *mut CintInt,
    atm: *mut CintInt,
    natm: CintInt,
    bas: *mut CintInt,
    nbas: CintInt,
    env: *mut f64,
    opt: *mut c_void,
    cache: *mut f64,
) -> Option<CintInt> {
    match symbol {
        CpuKernelSymbol::Int3c1eP2Cart => Some(unsafe {
            int3c1e_p2_cart_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int3c1eP2Sph => Some(unsafe {
            int3c1e_p2_sph_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int3c2eIp1Cart => Some(unsafe {
            int3c2e_ip1_cart_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int3c2eIp1Sph => Some(unsafe {
            int3c2e_ip1_sph_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_three_center_spinor_kernel(
    symbol: CpuKernelSymbol,
    out: *mut c_void,
    dims: *mut CintInt,
    shls: *mut CintInt,
    atm: *mut CintInt,
    natm: CintInt,
    bas: *mut CintInt,
    nbas: CintInt,
    env: *mut f64,
    opt: *mut c_void,
    cache: *mut f64,
) -> Option<CintInt> {
    match symbol {
        CpuKernelSymbol::Int3c1eP2Spinor => Some(unsafe {
            int3c1e_p2_spinor_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int3c2eIp1Spinor => Some(unsafe {
            int3c2e_ip1_spinor_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_four_center_real_kernel(
    symbol: CpuKernelSymbol,
    out: *mut f64,
    dims: *mut CintInt,
    shls: *mut CintInt,
    atm: *mut CintInt,
    natm: CintInt,
    bas: *mut CintInt,
    nbas: CintInt,
    env: *mut f64,
    opt: *mut c_void,
    cache: *mut f64,
) -> Option<CintInt> {
    match symbol {
        CpuKernelSymbol::Int4c1eCart => Some(unsafe {
            int4c1e_cart_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        CpuKernelSymbol::Int4c1eSph => Some(unsafe {
            int4c1e_sph_call(out, dims, shls, atm, natm, bas, nbas, env, opt, cache)
        }),
        _ => None,
    }
}

#[allow(clashing_extern_declarations)]
#[link(name = "cint_phase2_cpu", kind = "static")]
#[link(name = "m")]
unsafe extern "C" {
    fn int1e_ovlp_cart();
    fn int1e_kin_cart();
    fn int1e_ovlp_sph();
    fn int1e_ovlp_spinor();
    fn int2e_cart();
    fn int2e_sph();
    fn int2e_spinor();
    fn int2c2e_cart();
    fn int2c2e_sph();
    fn int2c2e_spinor();
    fn int3c1e_p2_cart();
    fn int3c1e_p2_sph();
    fn int3c1e_p2_spinor();
    fn int3c2e_ip1_cart();
    fn int3c2e_ip1_sph();
    fn int3c2e_ip1_spinor();
    fn int4c1e_cart();
    fn int4c1e_sph();

    #[link_name = "int2e_cart"]
    fn int2e_cart_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int2e_sph"]
    fn int2e_sph_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int2e_spinor"]
    fn int2e_spinor_call(
        out: *mut c_void,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int2c2e_cart"]
    fn int2c2e_cart_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int2c2e_sph"]
    fn int2c2e_sph_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int2c2e_spinor"]
    fn int2c2e_spinor_call(
        out: *mut c_void,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;

    #[link_name = "int3c1e_p2_cart"]
    fn int3c1e_p2_cart_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int3c1e_p2_sph"]
    fn int3c1e_p2_sph_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int3c2e_ip1_cart"]
    fn int3c2e_ip1_cart_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int3c2e_ip1_sph"]
    fn int3c2e_ip1_sph_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int3c1e_p2_spinor"]
    fn int3c1e_p2_spinor_call(
        out: *mut c_void,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int3c2e_ip1_spinor"]
    fn int3c2e_ip1_spinor_call(
        out: *mut c_void,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int4c1e_cart"]
    fn int4c1e_cart_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
    #[link_name = "int4c1e_sph"]
    fn int4c1e_sph_call(
        out: *mut f64,
        dims: *mut CintInt,
        shls: *mut CintInt,
        atm: *mut CintInt,
        natm: CintInt,
        bas: *mut CintInt,
        nbas: CintInt,
        env: *mut f64,
        opt: *mut c_void,
        cache: *mut f64,
    ) -> CintInt;
}
