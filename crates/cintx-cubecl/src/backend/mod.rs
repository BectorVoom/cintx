//! CubeCL backend resolution: wraps ComputeClient handles behind a single enum.
//!
//! `ResolvedBackend` holds the live client for whichever compute backend the
//! executor was configured to use. This avoids generics on `CubeClExecutor`
//! (required because `BackendExecutor` is used as `&dyn BackendExecutor`).

pub mod cpu_backend;
pub mod wgpu_backend;

use cintx_core::cintxRsError;
use cintx_runtime::{BackendIntent, BackendKind};

/// A resolved, live compute-client handle for one of the supported backends.
///
/// The `Cpu` arm is behind `#[cfg(feature = "cpu")]` because
/// `cubecl::cpu::CpuRuntime` only exists when cubecl's `cpu` feature is
/// enabled. Since `cpu` is a default feature, both arms compile in all
/// standard builds.
pub enum ResolvedBackend {
    /// wgpu GPU backend client.
    Wgpu(cubecl::client::ComputeClient<cubecl_wgpu::WgpuRuntime>),
    /// CPU backend client (requires `cpu` feature, which is enabled by default).
    #[cfg(feature = "cpu")]
    Cpu(cubecl::client::ComputeClient<cubecl::cpu::CpuRuntime>),
}

impl ResolvedBackend {
    /// Construct a `ResolvedBackend` from a `BackendIntent`.
    ///
    /// Calls the appropriate backend helper to bootstrap and validate the
    /// compute client. Returns `UnsupportedApi` when the requested backend
    /// cannot be initialised or is not compiled.
    pub fn from_intent(intent: &BackendIntent) -> Result<Self, cintxRsError> {
        match &intent.backend {
            BackendKind::Wgpu => {
                let client = wgpu_backend::resolve_wgpu_client(intent)?;
                Ok(ResolvedBackend::Wgpu(client))
            }
            BackendKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let client = cpu_backend::resolve_cpu_client()?;
                    Ok(ResolvedBackend::Cpu(client))
                }
                #[cfg(not(feature = "cpu"))]
                Err(cintxRsError::UnsupportedApi {
                    requested: "cpu-backend:feature-not-enabled".to_owned(),
                })
            }
        }
    }
}

/// Read the `CINTX_BACKEND` environment variable and return the corresponding
/// `BackendKind`. Defaults to `Wgpu` when the variable is absent or empty.
///
/// Supported values (case-sensitive): `"wgpu"`, `"cpu"`.
/// Unknown values are logged as a warning and fall back to `Wgpu`.
pub fn resolve_backend_kind() -> BackendKind {
    match std::env::var("CINTX_BACKEND").as_deref() {
        Ok("cpu") => BackendKind::Cpu,
        Ok("wgpu") | Err(_) => BackendKind::Wgpu,
        Ok(other) => {
            tracing::warn!(
                "Unknown CINTX_BACKEND value {:?}; falling back to wgpu",
                other
            );
            BackendKind::Wgpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn resolved_backend_from_intent_selects_cpu_arm() {
        let intent = BackendIntent {
            backend: BackendKind::Cpu,
            selector: "auto".to_owned(),
        };
        let backend = ResolvedBackend::from_intent(&intent);
        assert!(backend.is_ok(), "CPU backend should initialise successfully");
        assert!(matches!(backend.unwrap(), ResolvedBackend::Cpu(_)));
    }

    #[test]
    fn backend_env_var_cpu_selection() {
        // Run this test with CINTX_BACKEND=cpu in the environment to verify
        // CPU selection. In parallel test runs, env var mutation is racy, so
        // we skip mutation here and rely on external env verification.
        // If CINTX_BACKEND=cpu is set, assert Cpu; otherwise assert Wgpu default.
        let kind = resolve_backend_kind();
        match std::env::var("CINTX_BACKEND").as_deref() {
            Ok("cpu") => assert_eq!(kind, BackendKind::Cpu),
            Ok("wgpu") | Err(_) => assert_eq!(kind, BackendKind::Wgpu),
            Ok(_) => assert_eq!(kind, BackendKind::Wgpu), // unknown => falls back to wgpu
        }
    }

    #[test]
    fn backend_env_var_wgpu_default_when_unset() {
        // When CINTX_BACKEND is unset, resolve_backend_kind() must return Wgpu.
        // Env var mutation is racy in parallel test runs, so we read without mutation.
        if std::env::var("CINTX_BACKEND").is_err() {
            assert_eq!(resolve_backend_kind(), BackendKind::Wgpu);
        }
        // If the var is set, we can't safely remove it here; rely on the live value.
    }
}
