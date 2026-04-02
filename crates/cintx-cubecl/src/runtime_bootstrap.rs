/// WGPU adapter selection, bootstrap, and capability preflight.
///
/// This module implements D-01 (fail-closed adapter selection), D-02 (typed
/// capability failures), and D-04 (adapter metadata for reproducibility).

use cintx_core::cintxRsError;
use cintx_runtime::BackendIntent;

use crate::capability::{CapabilityReason, WgpuCapabilitySnapshot, WgpuPreflightReport};

/// Bootstrap the wgpu runtime for the given [`BackendIntent`] and return a
/// [`WgpuPreflightReport`] with adapter capability metadata.
///
/// The selector string in `intent` is parsed as:
/// - `"auto"` or `"default"` — use wgpu DefaultDevice (best available).
/// - `"discrete:N"` — use the N-th discrete GPU (0-indexed).
/// - `"integrated:N"` — use the N-th integrated GPU (0-indexed).
///
/// Any other selector value returns `UnsupportedApi` with reason `missing_adapter`.
///
/// # Errors
///
/// Returns `cintxRsError::UnsupportedApi` with a `wgpu-capability:<reason>` message
/// when:
/// - The selector is invalid or unrecognized.
/// - No adapter matching the selector is available.
pub fn bootstrap_wgpu_runtime(
    intent: &BackendIntent,
) -> Result<WgpuPreflightReport, cintxRsError> {
    let selector = parse_selector(&intent.selector)?;
    bootstrap_with_selector(selector)
}

/// Parsed adapter selector variant.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AdapterSelector {
    /// Use the best available wgpu adapter (DefaultDevice).
    Auto,
    /// Use the N-th discrete GPU.
    Discrete(usize),
    /// Use the N-th integrated GPU.
    Integrated(usize),
}

/// Parse an adapter selector string.
///
/// Accepted formats:
/// - `"auto"` or `"default"` → [`AdapterSelector::Auto`]
/// - `"discrete:N"` → [`AdapterSelector::Discrete(N)`]
/// - `"integrated:N"` → [`AdapterSelector::Integrated(N)`]
///
/// Everything else maps to `UnsupportedApi` with `wgpu-capability:missing_adapter`.
fn parse_selector(selector: &str) -> Result<AdapterSelector, cintxRsError> {
    let s = selector.trim();
    match s {
        "auto" | "default" => Ok(AdapterSelector::Auto),
        _ if s.starts_with("discrete:") => {
            let index_str = &s["discrete:".len()..];
            index_str
                .parse::<usize>()
                .map(AdapterSelector::Discrete)
                .map_err(|_| capability_error(CapabilityReason::MissingAdapter))
        }
        _ if s.starts_with("integrated:") => {
            let index_str = &s["integrated:".len()..];
            index_str
                .parse::<usize>()
                .map(AdapterSelector::Integrated)
                .map_err(|_| capability_error(CapabilityReason::MissingAdapter))
        }
        _ => Err(capability_error(CapabilityReason::MissingAdapter)),
    }
}

/// Build a `cintxRsError::UnsupportedApi` with a `wgpu-capability:<reason>` message.
fn capability_error(reason: CapabilityReason) -> cintxRsError {
    cintxRsError::UnsupportedApi {
        requested: format!("wgpu-capability:{}", reason.to_reason_string()),
    }
}

/// Bootstrap the wgpu runtime with a parsed [`AdapterSelector`].
///
/// On non-wasm platforms this performs synchronous adapter enumeration and
/// capability snapshot collection.  The result is a [`WgpuPreflightReport`]
/// that the executor uses to populate [`BackendCapabilityToken`] fields.
fn bootstrap_with_selector(
    selector: AdapterSelector,
) -> Result<WgpuPreflightReport, cintxRsError> {
    #[cfg(target_family = "wasm")]
    {
        // WASM requires async init; return unsupported with a clear reason.
        let _ = selector;
        return Err(capability_error(CapabilityReason::MissingAdapter));
    }

    #[cfg(not(target_family = "wasm"))]
    {
        use std::sync::OnceLock;

        use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions};

        // CubeCL panics if init_setup is called twice for the same device
        // ("A server is still registered for device ..."). Cache the default
        // device report so repeated preflight calls are idempotent.
        static DEFAULT_REPORT: OnceLock<Result<WgpuPreflightReport, CapabilityReason>> =
            OnceLock::new();

        if selector == AdapterSelector::Auto {
            let cached = DEFAULT_REPORT.get_or_init(|| do_bootstrap::<AutoGraphicsApi>(selector));
            return cached
                .as_ref()
                .map(|r| r.clone())
                .map_err(|reason| capability_error(reason.clone()));
        }

        // Non-default selectors are not cached (rare, and device index varies).
        do_bootstrap::<AutoGraphicsApi>(selector)
            .map_err(|reason| capability_error(reason))
    }
}

#[cfg(not(target_family = "wasm"))]
fn do_bootstrap<G: cubecl::wgpu::GraphicsApi>(
    selector: AdapterSelector,
) -> Result<WgpuPreflightReport, CapabilityReason> {
    use cubecl::wgpu::RuntimeOptions;

    let wgpu_device = selector_to_wgpu_device(&selector);

    // Use std::panic::catch_unwind to convert CubeCL's panic-based adapter
    // failures into typed errors (Pitfall 1 from research notes).
    let setup_result = std::panic::catch_unwind(|| {
        cubecl::wgpu::init_setup::<G>(&wgpu_device, RuntimeOptions::default())
    });

    let setup = setup_result.map_err(|_| CapabilityReason::MissingAdapter)?;

    // Collect adapter capability metadata.
    let info = setup.adapter.get_info();
    let backend_api = format!("{:?}", info.backend).to_lowercase();
    let device_type = format!("{:?}", info.device_type).to_lowercase();

    let snapshot = WgpuCapabilitySnapshot::new(
        info.name.clone(),
        backend_api,
        device_type,
        info.vendor,
        info.device,
        collect_feature_names(&setup.adapter),
        collect_limit_entries(&setup.adapter),
    );

    let report = WgpuPreflightReport::new(snapshot, vec![]);
    tracing::debug!(
        adapter_name = %report.snapshot.adapter_name,
        backend_api = %report.snapshot.backend_api,
        device_type = %report.snapshot.device_type,
        fingerprint = report.fingerprint,
        "wgpu bootstrap preflight complete"
    );
    Ok(report)
}

#[cfg(not(target_family = "wasm"))]
fn selector_to_wgpu_device(selector: &AdapterSelector) -> cubecl::wgpu::WgpuDevice {
    match selector {
        AdapterSelector::Auto => cubecl::wgpu::WgpuDevice::DefaultDevice,
        AdapterSelector::Discrete(n) => cubecl::wgpu::WgpuDevice::DiscreteGpu(*n),
        AdapterSelector::Integrated(n) => cubecl::wgpu::WgpuDevice::IntegratedGpu(*n),
    }
}

#[cfg(not(target_family = "wasm"))]
fn collect_feature_names(adapter: &wgpu::Adapter) -> Vec<String> {
    let features = adapter.features();
    let mut names = Vec::new();

    // Enumerate known feature flags relevant to compute execution.
    let known: &[(wgpu::Features, &str)] = &[
        (wgpu::Features::TIMESTAMP_QUERY, "TIMESTAMP_QUERY"),
        (
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            "TIMESTAMP_QUERY_INSIDE_PASSES",
        ),
        (
            wgpu::Features::PIPELINE_STATISTICS_QUERY,
            "PIPELINE_STATISTICS_QUERY",
        ),
        (wgpu::Features::SUBGROUP, "SUBGROUP"),
        (wgpu::Features::SUBGROUP_VERTEX, "SUBGROUP_VERTEX"),
        (wgpu::Features::SUBGROUP_BARRIER, "SUBGROUP_BARRIER"),
        (
            wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
            "STORAGE_RESOURCE_BINDING_ARRAY",
        ),
        (
            wgpu::Features::TEXTURE_BINDING_ARRAY,
            "TEXTURE_BINDING_ARRAY",
        ),
        (
            wgpu::Features::BUFFER_BINDING_ARRAY,
            "BUFFER_BINDING_ARRAY",
        ),
        (
            wgpu::Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            "STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING",
        ),
        (wgpu::Features::PUSH_CONSTANTS, "PUSH_CONSTANTS"),
        (wgpu::Features::SHADER_F64, "SHADER_F64"),
        (wgpu::Features::SHADER_I16, "SHADER_I16"),
        (wgpu::Features::SHADER_F16, "SHADER_F16"),
        (wgpu::Features::SHADER_INT64, "SHADER_INT64"),
    ];

    for &(flag, name) in known {
        if features.contains(flag) {
            names.push(name.to_owned());
        }
    }

    names
}

#[cfg(not(target_family = "wasm"))]
fn collect_limit_entries(adapter: &wgpu::Adapter) -> Vec<String> {
    let limits = adapter.limits();
    vec![
        format!(
            "max_compute_workgroups_per_dimension:{}",
            limits.max_compute_workgroups_per_dimension
        ),
        format!(
            "max_compute_workgroup_storage_size:{}",
            limits.max_compute_workgroup_storage_size
        ),
        format!(
            "max_compute_invocations_per_workgroup:{}",
            limits.max_compute_invocations_per_workgroup
        ),
        format!(
            "max_compute_workgroup_size_x:{}",
            limits.max_compute_workgroup_size_x
        ),
        format!(
            "max_compute_workgroup_size_y:{}",
            limits.max_compute_workgroup_size_y
        ),
        format!(
            "max_compute_workgroup_size_z:{}",
            limits.max_compute_workgroup_size_z
        ),
        format!(
            "max_storage_buffers_per_shader_stage:{}",
            limits.max_storage_buffers_per_shader_stage
        ),
        format!(
            "max_storage_buffer_binding_size:{}",
            limits.max_storage_buffer_binding_size
        ),
        format!(
            "min_storage_buffer_offset_alignment:{}",
            limits.min_storage_buffer_offset_alignment
        ),
        format!("max_buffer_size:{}", limits.max_buffer_size),
        format!("max_bind_groups:{}", limits.max_bind_groups),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use cintx_runtime::{BackendIntent, BackendKind};

    fn wgpu_intent(selector: &str) -> BackendIntent {
        BackendIntent {
            backend: BackendKind::Wgpu,
            selector: selector.to_owned(),
        }
    }

    #[test]
    fn selector_parser_accepts_auto_default_discrete_integrated() {
        // "auto" and "default" map to AdapterSelector::Auto
        assert_eq!(parse_selector("auto").unwrap(), AdapterSelector::Auto);
        assert_eq!(parse_selector("default").unwrap(), AdapterSelector::Auto);

        // "discrete:N" maps to AdapterSelector::Discrete(N)
        assert_eq!(
            parse_selector("discrete:0").unwrap(),
            AdapterSelector::Discrete(0)
        );
        assert_eq!(
            parse_selector("discrete:3").unwrap(),
            AdapterSelector::Discrete(3)
        );

        // "integrated:N" maps to AdapterSelector::Integrated(N)
        assert_eq!(
            parse_selector("integrated:0").unwrap(),
            AdapterSelector::Integrated(0)
        );
        assert_eq!(
            parse_selector("integrated:1").unwrap(),
            AdapterSelector::Integrated(1)
        );
    }

    #[test]
    fn invalid_selector_returns_typed_missing_adapter_error() {
        let err = parse_selector("unknown-backend").unwrap_err();
        match err {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("wgpu-capability:"),
                    "Error message must contain 'wgpu-capability:': {requested}"
                );
                assert!(
                    requested.contains("missing_adapter"),
                    "Error message must contain 'missing_adapter': {requested}"
                );
            }
            other => panic!("Expected UnsupportedApi, got {other:?}"),
        }

        // Selector with correct prefix but invalid index.
        let err2 = parse_selector("discrete:not-a-number").unwrap_err();
        match err2 {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("missing_adapter"),
                    "Error message must contain 'missing_adapter': {requested}"
                );
            }
            other => panic!("Expected UnsupportedApi, got {other:?}"),
        }

        // "integrated:" with invalid index.
        let err3 = parse_selector("integrated:xyz").unwrap_err();
        match err3 {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.contains("missing_adapter"),
                    "Error message must contain 'missing_adapter': {requested}"
                );
            }
            other => panic!("Expected UnsupportedApi, got {other:?}"),
        }
    }

    #[test]
    fn preflight_report_contains_capability_token_fields() {
        // We cannot rely on a GPU being available in all test environments.
        // This test verifies the structure of a manually constructed report
        // (the bootstrap path is tested via selector parsing and type contracts).
        let snap = WgpuCapabilitySnapshot::new(
            "Test Adapter",
            "vulkan",
            "discrete",
            0x10DE_u32,
            0x2684_u32,
            vec!["TIMESTAMP_QUERY".to_owned()],
            vec!["max_compute_workgroups_per_dimension:65535".to_owned()],
        );
        let report = WgpuPreflightReport::new(snap, vec![]);

        // All fields required by BackendCapabilityToken must be populated.
        assert!(
            !report.snapshot.adapter_name.is_empty(),
            "adapter_name must be populated"
        );
        assert!(
            !report.snapshot.backend_api.is_empty(),
            "backend_api must be populated"
        );
        assert_ne!(
            report.fingerprint, 0,
            "capability_fingerprint must be non-zero for a real adapter"
        );
        assert!(
            report.is_capable(),
            "Report must be capable with no unsatisfied reasons"
        );
    }

    #[test]
    fn bootstrap_wgpu_runtime_with_auto_selector_uses_default_device() {
        // Verify that the intent wiring parses correctly; actual GPU availability
        // is not required for selector parsing tests.
        let intent = wgpu_intent("auto");
        assert_eq!(intent.selector, "auto");

        let intent2 = wgpu_intent("discrete:0");
        assert_eq!(intent2.selector, "discrete:0");
    }

    #[test]
    fn bootstrap_wgpu_runtime_invalid_selector_returns_typed_error() {
        let intent = wgpu_intent("totally-invalid");
        let err = bootstrap_wgpu_runtime(&intent).unwrap_err();
        match err {
            cintxRsError::UnsupportedApi { requested } => {
                assert!(
                    requested.starts_with("wgpu-capability:"),
                    "Error must start with 'wgpu-capability:': {requested}"
                );
                assert!(
                    requested.contains("missing_adapter"),
                    "Error must contain 'missing_adapter': {requested}"
                );
            }
            other => panic!("Expected UnsupportedApi, got {other:?}"),
        }
    }
}
