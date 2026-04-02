//! wgpu backend client bootstrap for `ResolvedBackend`.

use cintx_core::cintxRsError;
use cintx_runtime::BackendIntent;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

/// Resolve a live wgpu `ComputeClient` from a `BackendIntent`.
///
/// Parses the `selector` string to a `WgpuDevice`, then calls
/// `WgpuRuntime::client` to obtain (or reuse) the cached client.
pub fn resolve_wgpu_client(
    intent: &BackendIntent,
) -> Result<ComputeClient<WgpuRuntime>, cintxRsError> {
    let device = selector_to_device(&intent.selector)?;
    Ok(WgpuRuntime::client(&device))
}

/// Map a selector string to a `WgpuDevice`.
///
/// Supported formats:
/// - `"auto"` or `""` -> `WgpuDevice::default()`
/// - `"discrete:N"` -> `WgpuDevice::DiscreteGpu(N)`
/// - `"integrated:N"` -> `WgpuDevice::IntegratedGpu(N)`
fn selector_to_device(selector: &str) -> Result<WgpuDevice, cintxRsError> {
    if selector.is_empty() || selector == "auto" {
        return Ok(WgpuDevice::default());
    }
    if let Some(rest) = selector.strip_prefix("discrete:") {
        let idx: usize = rest.parse().map_err(|_| cintxRsError::UnsupportedApi {
            requested: format!("wgpu selector parse error: {selector}"),
        })?;
        return Ok(WgpuDevice::DiscreteGpu(idx));
    }
    if let Some(rest) = selector.strip_prefix("integrated:") {
        let idx: usize = rest.parse().map_err(|_| cintxRsError::UnsupportedApi {
            requested: format!("wgpu selector parse error: {selector}"),
        })?;
        return Ok(WgpuDevice::IntegratedGpu(idx));
    }
    Err(cintxRsError::UnsupportedApi {
        requested: format!("wgpu selector unknown format: {selector}"),
    })
}
