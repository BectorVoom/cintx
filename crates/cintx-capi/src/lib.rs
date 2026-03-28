//! C API exports (stub).

pub mod errors;
pub mod shim;

/// Phase 3 keeps the C ABI boundary stable-only.
pub const CAPI_EXPOSES_UNSTABLE_SOURCE_API: bool = false;
