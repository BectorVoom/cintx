//! CPU backend client bootstrap for `ResolvedBackend`.
//!
//! This entire module is gated behind `#[cfg(feature = "cpu")]` because
//! `cubecl::cpu::CpuRuntime` (and `CpuDevice`) only exist when the `cpu`
//! feature of the `cubecl` crate is enabled.

#![cfg(feature = "cpu")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::cpu::{CpuDevice, CpuRuntime};

/// Resolve a CPU `ComputeClient` using the default `CpuDevice`.
pub fn resolve_cpu_client() -> Result<ComputeClient<CpuRuntime>, cintxRsError> {
    Ok(CpuRuntime::client(&CpuDevice::default()))
}
