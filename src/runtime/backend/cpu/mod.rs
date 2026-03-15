pub mod ffi;
pub(crate) mod overlap_cartesian;
pub mod router;
pub mod spinor_3c1e;

use core::ffi::c_void;
use core::ptr;

use crate::contracts::{BasisSet, Representation};
use crate::errors::LibcintRsError;
use crate::runtime::raw::{RawAtmView, RawBasView};

pub use ffi::{CpuKernelFn, CpuKernelSymbol, ALL_BOUND_SYMBOLS};
pub(crate) use overlap_cartesian::{
    fill_raw_one_e_kinetic_cartesian, fill_raw_one_e_kinetic_spherical,
    fill_raw_one_e_kinetic_spinor, fill_raw_one_e_nuclear_cartesian,
    fill_raw_one_e_nuclear_spherical, fill_raw_one_e_nuclear_spinor,
    fill_raw_one_e_overlap_cartesian, fill_raw_one_e_overlap_spherical,
    fill_raw_one_e_overlap_spinor, fill_safe_one_e_kinetic_cartesian,
    fill_safe_one_e_kinetic_spherical, fill_safe_one_e_kinetic_spinor,
    fill_safe_one_e_nuclear_cartesian, fill_safe_one_e_nuclear_spherical,
    fill_safe_one_e_nuclear_spinor, fill_safe_one_e_overlap_cartesian,
    fill_safe_one_e_overlap_spherical, fill_safe_one_e_overlap_spinor,
};
pub use router::{
    resolve_capi_route, resolve_raw_route, resolve_route, resolve_route_request,
    resolve_safe_route, route, route_manifest_entries, route_manifest_lock_json, route_request,
    CpuRouteKey, CpuRouteManifestEntry, CpuRouteTarget, ResolvedCpuRoute, RouteEntryKernel,
    RouteKind, RouteOptimizerMode, RouteStability, RouteStatus, RouteSurface, RouteSurfaceGroup,
};
pub use spinor_3c1e::{adapter_route, Spinor3c1eAdapter, Spinor3c1eTransform};

const ATM_SLOTS: usize = 6;
const BAS_SLOTS: usize = 8;
const ENV_DATA_START: usize = 20;

struct SafeRawLayout {
    atm: Vec<i32>,
    bas: Vec<i32>,
    env: Vec<f64>,
}

pub(crate) fn execute_safe_specialized_route(
    route: ResolvedCpuRoute,
    basis: &BasisSet,
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [f64],
) -> Result<bool, LibcintRsError> {
    match route.entry_kernel {
        RouteEntryKernel::OneElectronOverlapCartesian => {
            fill_safe_one_e_overlap_cartesian(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eOvlpSph) => {
            fill_safe_one_e_overlap_spherical(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::OneElectronKineticCartesian => {
            fill_safe_one_e_kinetic_cartesian(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eKinSph) => {
            fill_safe_one_e_kinetic_spherical(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucCart) => {
            fill_safe_one_e_nuclear_cartesian(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucSph) => {
            fill_safe_one_e_nuclear_spherical(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eCart) => {
            execute_safe_two_e_real_route(
                CpuKernelSymbol::Int2eCart,
                basis,
                shell_tuple,
                dims,
                output,
            )?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eSph) => {
            execute_safe_two_e_real_route(
                CpuKernelSymbol::Int2eSph,
                basis,
                shell_tuple,
                dims,
                output,
            )?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Cart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Sph)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Cart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Sph) => {
            execute_safe_three_center_real_route(
                route.entry_kernel,
                basis,
                shell_tuple,
                dims,
                output,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(crate) fn execute_safe_specialized_spinor_route(
    route: ResolvedCpuRoute,
    basis: &BasisSet,
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [[f64; 2]],
) -> Result<bool, LibcintRsError> {
    match route.entry_kernel {
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eOvlpSpinor) => {
            fill_safe_one_e_overlap_spinor(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eKinSpinor) => {
            fill_safe_one_e_kinetic_spinor(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucSpinor) => {
            fill_safe_one_e_nuclear_spinor(basis, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eSpinor) => {
            execute_safe_two_e_spinor_route(basis, shell_tuple, dims, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Spinor)
        | RouteEntryKernel::ThreeCenterOneElectronSpinorAdapter => {
            execute_safe_three_center_spinor_route(
                route.entry_kernel,
                basis,
                shell_tuple,
                dims,
                output,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(crate) fn execute_raw_specialized_route(
    route: ResolvedCpuRoute,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [f64],
) -> Result<bool, LibcintRsError> {
    match route.entry_kernel {
        RouteEntryKernel::OneElectronOverlapCartesian => {
            fill_raw_one_e_overlap_cartesian(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eOvlpSph) => {
            fill_raw_one_e_overlap_spherical(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eKinSph) => {
            fill_raw_one_e_kinetic_spherical(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucCart) => {
            fill_raw_one_e_nuclear_cartesian(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucSph) => {
            fill_raw_one_e_nuclear_spherical(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eOvlpSpinor) => {
            fill_raw_one_e_overlap_spinor(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eKinSpinor) => {
            fill_raw_one_e_kinetic_spinor(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int1eNucSpinor) => {
            fill_raw_one_e_nuclear_spinor(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::OneElectronKineticCartesian => {
            fill_raw_one_e_kinetic_cartesian(atm, bas, env, shell_tuple, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int2eCart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int2eSph)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int2eSpinor) => {
            execute_raw_two_e_route(route.entry_kernel, atm, bas, env, shell_tuple, dims, output)?;
            Ok(true)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Cart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Sph)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Cart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Sph)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Spinor)
        | RouteEntryKernel::ThreeCenterOneElectronSpinorAdapter => {
            execute_raw_three_center_route(
                route.entry_kernel,
                atm,
                bas,
                env,
                shell_tuple,
                dims,
                output,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn execute_safe_two_e_real_route(
    symbol: CpuKernelSymbol,
    basis: &BasisSet,
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let raw = safe_basis_to_raw_layout(basis)?;
    run_two_e_real_kernel(
        symbol,
        dims,
        shell_tuple,
        raw.atm.as_slice(),
        raw.bas.as_slice(),
        raw.env.as_slice(),
        output,
    )
}

fn execute_safe_two_e_spinor_route(
    basis: &BasisSet,
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let raw = safe_basis_to_raw_layout(basis)?;
    run_two_e_spinor_pairs_kernel(
        dims,
        shell_tuple,
        raw.atm.as_slice(),
        raw.bas.as_slice(),
        raw.env.as_slice(),
        output,
    )
}

fn execute_raw_two_e_route(
    entry_kernel: RouteEntryKernel,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let symbol = match entry_kernel {
        RouteEntryKernel::Direct(symbol) => symbol,
        _ => {
            return Err(LibcintRsError::BackendFailure {
                backend: "cpu.two_e",
                detail: "raw two-electron execution requires a direct kernel symbol".to_string(),
            });
        }
    };

    let representation = Representation::from(symbol);
    match representation {
        Representation::Cartesian | Representation::Spherical => {
            run_two_e_real_kernel(symbol, dims, shell_tuple, atm, bas, env, output)
        }
        Representation::Spinor => {
            run_two_e_spinor_scalars_kernel(dims, shell_tuple, atm, bas, env, output)
        }
    }
}

fn execute_safe_three_center_real_route(
    entry_kernel: RouteEntryKernel,
    basis: &BasisSet,
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let symbol = three_center_real_symbol(entry_kernel)?;
    let raw = safe_basis_to_raw_layout(basis)?;
    run_three_center_real_kernel(
        symbol,
        dims,
        shell_tuple,
        raw.atm.as_slice(),
        raw.bas.as_slice(),
        raw.env.as_slice(),
        output,
    )
}

fn execute_safe_three_center_spinor_route(
    entry_kernel: RouteEntryKernel,
    basis: &BasisSet,
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let symbol = three_center_spinor_symbol(entry_kernel)?;
    let raw = safe_basis_to_raw_layout(basis)?;
    run_three_center_spinor_pairs_kernel(
        symbol,
        dims,
        shell_tuple,
        raw.atm.as_slice(),
        raw.bas.as_slice(),
        raw.env.as_slice(),
        output,
    )
}

fn execute_raw_three_center_route(
    entry_kernel: RouteEntryKernel,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    shell_tuple: &[usize],
    dims: &[usize],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let symbol = match entry_kernel {
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Cart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Sph)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Cart)
        | RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Sph) => {
            three_center_real_symbol(entry_kernel)?
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Spinor)
        | RouteEntryKernel::ThreeCenterOneElectronSpinorAdapter => {
            three_center_spinor_symbol(entry_kernel)?
        }
        _ => {
            return Err(LibcintRsError::BackendFailure {
                backend: "cpu.three_center",
                detail: "raw three-center execution requires a supported route entry".to_string(),
            });
        }
    };

    let representation = Representation::from(symbol);
    match representation {
        Representation::Cartesian | Representation::Spherical => {
            run_three_center_real_kernel(symbol, dims, shell_tuple, atm, bas, env, output)
        }
        Representation::Spinor => {
            run_three_center_spinor_scalars_kernel(symbol, dims, shell_tuple, atm, bas, env, output)
        }
    }
}

fn three_center_real_symbol(
    entry_kernel: RouteEntryKernel,
) -> Result<CpuKernelSymbol, LibcintRsError> {
    match entry_kernel {
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Cart) => {
            Ok(CpuKernelSymbol::Int3c1eP2Cart)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c1eP2Sph) => {
            Ok(CpuKernelSymbol::Int3c1eP2Sph)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Cart) => {
            Ok(CpuKernelSymbol::Int3c2eIp1Cart)
        }
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Sph) => {
            Ok(CpuKernelSymbol::Int3c2eIp1Sph)
        }
        _ => Err(LibcintRsError::BackendFailure {
            backend: "cpu.three_center",
            detail: format!(
                "three-center real execution requires a direct real kernel, got `{}`",
                entry_kernel.as_str()
            ),
        }),
    }
}

fn three_center_spinor_symbol(
    entry_kernel: RouteEntryKernel,
) -> Result<CpuKernelSymbol, LibcintRsError> {
    match entry_kernel {
        RouteEntryKernel::Direct(CpuKernelSymbol::Int3c2eIp1Spinor) => {
            Ok(CpuKernelSymbol::Int3c2eIp1Spinor)
        }
        RouteEntryKernel::ThreeCenterOneElectronSpinorAdapter => {
            Ok(CpuKernelSymbol::Int3c1eP2Spinor)
        }
        _ => Err(LibcintRsError::BackendFailure {
            backend: "cpu.three_center",
            detail: format!(
                "three-center spinor execution requires a spinor-capable route, got `{}`",
                entry_kernel.as_str()
            ),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_two_e_real_kernel(
    symbol: CpuKernelSymbol,
    dims: &[usize],
    shell_tuple: &[usize],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let expected_len = checked_product(dims, "dims")?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let mut dims_i32 = usize_slice_to_i32(dims, "dims")?;
    let mut shls_i32 = usize_slice_to_i32(shell_tuple, "shls")?;
    let natm = checked_natm(atm)?;
    let nbas = checked_nbas(bas)?;
    let mut atm_owned = atm.to_vec();
    let mut bas_owned = bas.to_vec();
    let mut env_owned = env.to_vec();

    let result = unsafe {
        ffi::call_two_e_real_kernel(
            symbol,
            output.as_mut_ptr(),
            dims_i32.as_mut_ptr(),
            shls_i32.as_mut_ptr(),
            atm_owned.as_mut_ptr(),
            natm,
            bas_owned.as_mut_ptr(),
            nbas,
            env_owned.as_mut_ptr(),
            ptr::null_mut::<c_void>(),
            ptr::null_mut::<f64>(),
        )
    };
    let not0 = result.ok_or_else(|| LibcintRsError::BackendFailure {
        backend: "cpu.two_e",
        detail: format!(
            "route selected non-two-electron kernel symbol `{}` for real execution",
            symbol.name()
        ),
    })?;

    if not0 < 0 {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu.two_e",
            detail: format!("kernel `{}` returned invalid status {not0}", symbol.name()),
        });
    }

    Ok(())
}

fn run_two_e_spinor_pairs_kernel(
    dims: &[usize],
    shell_tuple: &[usize],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let expected_len = checked_product(dims, "dims")?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let mut dims_i32 = usize_slice_to_i32(dims, "dims")?;
    let mut shls_i32 = usize_slice_to_i32(shell_tuple, "shls")?;
    let natm = checked_natm(atm)?;
    let nbas = checked_nbas(bas)?;
    let mut atm_owned = atm.to_vec();
    let mut bas_owned = bas.to_vec();
    let mut env_owned = env.to_vec();

    let result = unsafe {
        ffi::call_two_e_spinor_kernel(
            CpuKernelSymbol::Int2eSpinor,
            output.as_mut_ptr().cast::<c_void>(),
            dims_i32.as_mut_ptr(),
            shls_i32.as_mut_ptr(),
            atm_owned.as_mut_ptr(),
            natm,
            bas_owned.as_mut_ptr(),
            nbas,
            env_owned.as_mut_ptr(),
            ptr::null_mut::<c_void>(),
            ptr::null_mut::<f64>(),
        )
    };
    let not0 = result.ok_or_else(|| LibcintRsError::BackendFailure {
        backend: "cpu.two_e",
        detail: "route selected non-two-electron spinor kernel".to_string(),
    })?;

    if not0 < 0 {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu.two_e",
            detail: format!("kernel `int2e_spinor` returned invalid status {not0}"),
        });
    }

    Ok(())
}

fn run_two_e_spinor_scalars_kernel(
    dims: &[usize],
    shell_tuple: &[usize],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let element_count = checked_product(dims, "dims")?;
    let expected_scalars =
        element_count
            .checked_mul(2)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "dims",
                reason: "spinor scalar count overflows usize".to_string(),
            })?;
    if output.len() != expected_scalars {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_scalars",
            expected: expected_scalars,
            got: output.len(),
        });
    }

    let mut dims_i32 = usize_slice_to_i32(dims, "dims")?;
    let mut shls_i32 = usize_slice_to_i32(shell_tuple, "shls")?;
    let natm = checked_natm(atm)?;
    let nbas = checked_nbas(bas)?;
    let mut atm_owned = atm.to_vec();
    let mut bas_owned = bas.to_vec();
    let mut env_owned = env.to_vec();

    let result = unsafe {
        ffi::call_two_e_spinor_kernel(
            CpuKernelSymbol::Int2eSpinor,
            output.as_mut_ptr().cast::<c_void>(),
            dims_i32.as_mut_ptr(),
            shls_i32.as_mut_ptr(),
            atm_owned.as_mut_ptr(),
            natm,
            bas_owned.as_mut_ptr(),
            nbas,
            env_owned.as_mut_ptr(),
            ptr::null_mut::<c_void>(),
            ptr::null_mut::<f64>(),
        )
    };
    let not0 = result.ok_or_else(|| LibcintRsError::BackendFailure {
        backend: "cpu.two_e",
        detail: "route selected non-two-electron spinor kernel".to_string(),
    })?;

    if not0 < 0 {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu.two_e",
            detail: format!("kernel `int2e_spinor` returned invalid status {not0}"),
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_three_center_real_kernel(
    symbol: CpuKernelSymbol,
    dims: &[usize],
    shell_tuple: &[usize],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let expected_len = checked_product(dims, "dims")?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let mut dims_i32 = usize_slice_to_i32(dims, "dims")?;
    let mut shls_i32 = usize_slice_to_i32(shell_tuple, "shls")?;
    let natm = checked_natm(atm)?;
    let nbas = checked_nbas(bas)?;
    let mut atm_owned = atm.to_vec();
    let mut bas_owned = bas.to_vec();
    let mut env_owned = env.to_vec();

    let result = unsafe {
        ffi::call_three_center_real_kernel(
            symbol,
            output.as_mut_ptr(),
            dims_i32.as_mut_ptr(),
            shls_i32.as_mut_ptr(),
            atm_owned.as_mut_ptr(),
            natm,
            bas_owned.as_mut_ptr(),
            nbas,
            env_owned.as_mut_ptr(),
            ptr::null_mut::<c_void>(),
            ptr::null_mut::<f64>(),
        )
    };
    let not0 = result.ok_or_else(|| LibcintRsError::BackendFailure {
        backend: "cpu.three_center",
        detail: format!(
            "route selected non-three-center kernel symbol `{}` for real execution",
            symbol.name()
        ),
    })?;

    if not0 < 0 {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu.three_center",
            detail: format!("kernel `{}` returned invalid status {not0}", symbol.name()),
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_three_center_spinor_pairs_kernel(
    symbol: CpuKernelSymbol,
    dims: &[usize],
    shell_tuple: &[usize],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    output: &mut [[f64; 2]],
) -> Result<(), LibcintRsError> {
    let expected_len = checked_product(dims, "dims")?;
    if output.len() != expected_len {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_elements",
            expected: expected_len,
            got: output.len(),
        });
    }

    let mut dims_i32 = usize_slice_to_i32(dims, "dims")?;
    // Wrapper-backed parity for int3c2e spinor requires libcint to infer dims.
    // Passing explicit dims here produces a different tensor contract.
    let dims_ptr = if matches!(symbol, CpuKernelSymbol::Int3c2eIp1Spinor) {
        ptr::null_mut()
    } else {
        dims_i32.as_mut_ptr()
    };
    let mut shls_i32 = usize_slice_to_i32(shell_tuple, "shls")?;
    let natm = checked_natm(atm)?;
    let nbas = checked_nbas(bas)?;
    let mut atm_owned = atm.to_vec();
    let mut bas_owned = bas.to_vec();
    let mut env_owned = env.to_vec();

    let result = unsafe {
        ffi::call_three_center_spinor_kernel(
            symbol,
            output.as_mut_ptr().cast::<c_void>(),
            dims_ptr,
            shls_i32.as_mut_ptr(),
            atm_owned.as_mut_ptr(),
            natm,
            bas_owned.as_mut_ptr(),
            nbas,
            env_owned.as_mut_ptr(),
            ptr::null_mut::<c_void>(),
            ptr::null_mut::<f64>(),
        )
    };
    let not0 = result.ok_or_else(|| LibcintRsError::BackendFailure {
        backend: "cpu.three_center",
        detail: format!(
            "route selected non-three-center spinor kernel symbol `{}`",
            symbol.name()
        ),
    })?;

    if not0 < 0 {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu.three_center",
            detail: format!("kernel `{}` returned invalid status {not0}", symbol.name()),
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_three_center_spinor_scalars_kernel(
    symbol: CpuKernelSymbol,
    dims: &[usize],
    shell_tuple: &[usize],
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    output: &mut [f64],
) -> Result<(), LibcintRsError> {
    let element_count = checked_product(dims, "dims")?;
    let expected_scalars =
        element_count
            .checked_mul(2)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field: "dims",
                reason: "spinor scalar count overflows usize".to_string(),
            })?;
    if output.len() != expected_scalars {
        return Err(LibcintRsError::InvalidLayout {
            item: "output_scalars",
            expected: expected_scalars,
            got: output.len(),
        });
    }

    let mut dims_i32 = usize_slice_to_i32(dims, "dims")?;
    // Wrapper-backed parity for int3c2e spinor requires libcint to infer dims.
    // Passing explicit dims here produces a different tensor contract.
    let dims_ptr = if matches!(symbol, CpuKernelSymbol::Int3c2eIp1Spinor) {
        ptr::null_mut()
    } else {
        dims_i32.as_mut_ptr()
    };
    let mut shls_i32 = usize_slice_to_i32(shell_tuple, "shls")?;
    let natm = checked_natm(atm)?;
    let nbas = checked_nbas(bas)?;
    let mut atm_owned = atm.to_vec();
    let mut bas_owned = bas.to_vec();
    let mut env_owned = env.to_vec();

    let result = unsafe {
        ffi::call_three_center_spinor_kernel(
            symbol,
            output.as_mut_ptr().cast::<c_void>(),
            dims_ptr,
            shls_i32.as_mut_ptr(),
            atm_owned.as_mut_ptr(),
            natm,
            bas_owned.as_mut_ptr(),
            nbas,
            env_owned.as_mut_ptr(),
            ptr::null_mut::<c_void>(),
            ptr::null_mut::<f64>(),
        )
    };
    let not0 = result.ok_or_else(|| LibcintRsError::BackendFailure {
        backend: "cpu.three_center",
        detail: format!(
            "route selected non-three-center spinor kernel symbol `{}`",
            symbol.name()
        ),
    })?;

    if not0 < 0 {
        return Err(LibcintRsError::BackendFailure {
            backend: "cpu.three_center",
            detail: format!("kernel `{}` returned invalid status {not0}", symbol.name()),
        });
    }

    Ok(())
}

fn safe_basis_to_raw_layout(basis: &BasisSet) -> Result<SafeRawLayout, LibcintRsError> {
    let mut env = vec![0.0f64; ENV_DATA_START];
    let mut atm =
        Vec::with_capacity(basis.atoms().len().checked_mul(ATM_SLOTS).ok_or_else(|| {
            LibcintRsError::InvalidInput {
                field: "basis",
                reason: "atom table length overflows usize".to_string(),
            }
        })?);

    for atom in basis.atoms() {
        let coord_offset = i32::try_from(env.len()).map_err(|_| LibcintRsError::InvalidInput {
            field: "env",
            reason: "coordinate offset exceeds i32 range".to_string(),
        })?;
        let atomic_number = i32::from(atom.atomic_number());
        atm.extend_from_slice(&[atomic_number, coord_offset, 1, 0, 0, 0]);
        env.extend_from_slice(&atom.coordinates());
    }

    let mut bas =
        Vec::with_capacity(basis.shells().len().checked_mul(BAS_SLOTS).ok_or_else(|| {
            LibcintRsError::InvalidInput {
                field: "basis",
                reason: "shell table length overflows usize".to_string(),
            }
        })?);
    for shell in basis.shells() {
        let center =
            i32::try_from(shell.center_index()).map_err(|_| LibcintRsError::InvalidInput {
                field: "shell.center_index",
                reason: "center index exceeds i32 range".to_string(),
            })?;
        let nprim_usize = shell.primitives().len();
        let nprim = i32::try_from(nprim_usize).map_err(|_| LibcintRsError::InvalidInput {
            field: "shell.primitives",
            reason: "primitive count exceeds i32 range".to_string(),
        })?;
        let ptr_exp = i32::try_from(env.len()).map_err(|_| LibcintRsError::InvalidInput {
            field: "env",
            reason: "exponent offset exceeds i32 range".to_string(),
        })?;
        for primitive in shell.primitives() {
            env.push(primitive.exponent());
        }
        let ptr_coeff = i32::try_from(env.len()).map_err(|_| LibcintRsError::InvalidInput {
            field: "env",
            reason: "coefficient offset exceeds i32 range".to_string(),
        })?;
        for primitive in shell.primitives() {
            env.push(primitive.coefficient());
        }

        bas.extend_from_slice(&[
            center,
            i32::from(shell.angular_momentum()),
            nprim,
            1,
            0,
            ptr_exp,
            ptr_coeff,
            0,
        ]);
    }

    Ok(SafeRawLayout { atm, bas, env })
}

fn usize_slice_to_i32(values: &[usize], field: &'static str) -> Result<Vec<i32>, LibcintRsError> {
    values
        .iter()
        .map(|value| {
            i32::try_from(*value).map_err(|_| LibcintRsError::InvalidInput {
                field,
                reason: format!("value {value} exceeds i32 range"),
            })
        })
        .collect()
}

fn checked_natm(atm: &[i32]) -> Result<i32, LibcintRsError> {
    let natm = RawAtmView::new(atm)?.natm();
    i32::try_from(natm).map_err(|_| LibcintRsError::InvalidInput {
        field: "atm",
        reason: format!("natm {natm} exceeds i32 range"),
    })
}

fn checked_nbas(bas: &[i32]) -> Result<i32, LibcintRsError> {
    let nbas = RawBasView::new(bas)?.nbas();
    i32::try_from(nbas).map_err(|_| LibcintRsError::InvalidInput {
        field: "bas",
        reason: format!("nbas {nbas} exceeds i32 range"),
    })
}

fn checked_product(dims: &[usize], field: &'static str) -> Result<usize, LibcintRsError> {
    let mut product = 1usize;
    for dim in dims {
        if *dim == 0 {
            return Err(LibcintRsError::InvalidInput {
                field,
                reason: "dimension values must be greater than zero".to_string(),
            });
        }
        product = product
            .checked_mul(*dim)
            .ok_or_else(|| LibcintRsError::InvalidInput {
                field,
                reason: "dimension product overflows usize".to_string(),
            })?;
    }
    Ok(product)
}

impl From<CpuKernelSymbol> for Representation {
    fn from(symbol: CpuKernelSymbol) -> Self {
        symbol.representation()
    }
}
