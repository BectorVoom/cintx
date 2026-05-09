//! ROCm backend client bootstrap for `ResolvedBackend`.
//!
//! Gated behind `#![cfg(feature = "rocm")]`. Note the feature is named `rocm`
//! while the upstream dep crate is `cubecl-hip`. Runtime-verifiable on the
//! dev host (Linux + AMD ROCm); see `xtask rocm-oracle` for the opt-in
//! oracle base-family suite (Wave 3).

#![cfg(feature = "rocm")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_hip::{AmdDevice, HipRuntime};

/// Resolve a ROCm `ComputeClient` using the default `AmdDevice`.
pub fn resolve_rocm_client() -> Result<ComputeClient<HipRuntime>, cintxRsError> {
    Ok(HipRuntime::client(&AmdDevice::default()))
}
