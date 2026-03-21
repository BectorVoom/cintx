//! Core domain primitives for cintx per docs/design/cintx_detailed_design.md §6-s.
//!
//! This crate exposes typed, `Arc`-backed atoms, shells, basis sets, and tensor metadata
//! so the rest of the workspace can operate without touching raw libcint arrays.

pub mod atom;
pub mod basis;
pub mod env;
pub mod error;
pub mod operator;
pub mod shell;
pub mod tensor;

pub use atom::{Atom, NuclearModel};
pub use basis::{BasisMeta, BasisSet};
pub use env::{EnvBoundsError, EnvParams, EnvUnits};
pub use operator::{OperatorId, Representation};
pub use shell::{Shell, ShellTuple, ShellTupleArityError};
pub use tensor::{TensorLayout, TensorShape};
