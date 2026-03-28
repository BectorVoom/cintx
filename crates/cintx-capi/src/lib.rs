//! C ABI shim exports for migration-focused compat callers.
//!
//! Phase 3 keeps this crate stable-only: no unstable source-family C symbols are exported.

pub mod errors;
pub mod shim;

pub use errors::{
    cintrs_clear_last_error, cintrs_copy_last_error_api, cintrs_copy_last_error_family,
    cintrs_copy_last_error_message, cintrs_copy_last_error_representation, cintrs_last_error_code,
    CintxErrorReport, CintxStatus,
};
pub use shim::{
    cintrs_eval, cintrs_query_workspace, CintxEvalSummary, CintxRawApi, CintxWorkspaceQuery,
};

/// Phase 3 keeps the C ABI boundary stable-only.
pub const CAPI_EXPOSES_UNSTABLE_SOURCE_API: bool = false;

/// Stable C ABI success code used by all shim entry points.
pub const CAPI_STATUS_SUCCESS: i32 = CintxStatus::Success as i32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capi_boundary_remains_stable_only() {
        assert!(!CAPI_EXPOSES_UNSTABLE_SOURCE_API);
        assert_eq!(CAPI_STATUS_SUCCESS, 0);
    }
}
