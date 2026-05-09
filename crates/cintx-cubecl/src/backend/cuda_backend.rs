//! CUDA backend client bootstrap for `ResolvedBackend`.
//!
//! Gated behind `#![cfg(feature = "cuda")]`. Compile-only on this dev host —
//! see `.planning/notes/cuda-metal-verification-gap.md` for the verification
//! risk-accept that applies to this module. Runtime dispatch is delegated to
//! upstream `cubecl-cuda 0.10.0`; no oracle parity gate is added in Phase 16
//! for the cuda path.

#![cfg(feature = "cuda")]

use cintx_core::cintxRsError;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_cuda::{CudaDevice, CudaRuntime};

/// Resolve a CUDA `ComputeClient` using the default `CudaDevice`.
///
/// This phase ships cuda as compile-only — see the verification gap note above.
pub fn resolve_cuda_client() -> Result<ComputeClient<CudaRuntime>, cintxRsError> {
    Ok(CudaRuntime::client(&CudaDevice::default()))
}
