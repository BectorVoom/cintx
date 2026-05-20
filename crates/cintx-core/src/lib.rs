//! Core domain primitives for cintx per docs/design/cintx_detailed_design.md §6-s.
//!
//! This crate exposes typed, `Arc`-backed atoms, shells, basis sets, and tensor metadata
//! so the rest of the workspace can operate without touching raw libcint arrays.

pub mod atom;
pub mod basis;
pub mod ecp;
pub mod env;
pub mod error;
pub mod operator;
pub mod precision;
pub mod shell;
pub mod tensor;

pub use atom::{Atom, NuclearModel};
pub use basis::{BasisMeta, BasisSet};
pub use ecp::{EcpShell, EcpChannel};
pub use env::{EnvBoundsError, EnvParams, EnvUnits};
pub use error::{CoreError, cintxRsError};
pub use operator::{AoSymmetry, OperatorId, Representation};
pub use precision::{CintFloat, PrecisionKind};
pub use shell::{Shell, ShellTuple, ShellTupleArityError};
pub use tensor::{TensorLayout, TensorShape};
