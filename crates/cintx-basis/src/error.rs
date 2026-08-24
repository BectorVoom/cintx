//! Typed errors for basis-set parsing and construction.
//!
//! Public library surface, so `thiserror` per the workspace error policy.

use cintx_core::CoreError;

#[derive(Debug, thiserror::Error)]
pub enum BasisError {
    #[error("unknown element symbol `{symbol}`")]
    UnknownElement { symbol: String },

    #[error("unknown angular-momentum label `{label}`")]
    UnknownAngularLabel { label: String },

    #[error("malformed number `{token}`")]
    MalformedNumber { token: String },

    #[error("malformed basis block: {detail}")]
    MalformedBlock { detail: String },

    #[error("basis `{basis}` has no entry for element Z={atomic_number}")]
    MissingElement {
        basis: &'static str,
        atomic_number: u16,
    },

    #[error("unknown basis-set name `{name}`")]
    UnknownBasis { name: String },

    #[error("core construction failed: {0}")]
    Core(#[from] CoreError),
}
