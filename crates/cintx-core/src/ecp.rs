//! Effective Core Potential typed surface.
//!
//! Stub introduced by Phase 19 Plan 01. Plan 03 fills the field set,
//! `try_new` validation, and the `EcpChannel` enum body per
//! `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-PATTERNS.md`
//! §"crates/cintx-core/src/ecp.rs".

use std::sync::Arc;

/// Distinguishes the local Type-1 channel from semi-local Type-2 projector
/// channels.
///
/// Plan 03 promotes this from a stub. PySCF `nr_ecp.h` uses an
/// `ang_momentum = -1` sentinel for the local channel; this enum will be the
/// typed surface that callers see, with the sentinel value packed back into
/// the raw `ecpbas` row at the `cintx-compat` boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcpChannel {
    /// Local Type-1 (Coulomb-like) channel — PySCF convention `ANG_OF = -1`.
    Local,
    /// Semi-local Type-2 projector for angular momentum `l`.
    Projected(u8),
}

/// Effective Core Potential shell — typed analog of one `ecpbas` row.
///
/// Plan 03 promotes from this empty placeholder by adding `atom_index`,
/// `radial_power`, `so_type`, `nprim`, `nctr`, `exponents`, `coefficients`.
/// The two fields kept here (`channel`, `exponents`, `coefficients`) are the
/// smallest viable surface for Wave 1 downstream plans to depend on; their
/// field-set will not shrink, only grow.
#[derive(Clone, Debug, PartialEq)]
pub struct EcpShell {
    // Placeholder: Plan 03 fills with the field set from 19-PATTERNS.md
    // §"crates/cintx-core/src/ecp.rs" (atom_index, channel, radial_power,
    // nprim, nctr, so_type, exponents, coefficients).
    pub channel: EcpChannel,
    pub exponents: Arc<[f64]>,
    pub coefficients: Arc<[f64]>,
}
