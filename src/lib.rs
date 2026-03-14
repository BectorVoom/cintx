#![deny(unsafe_op_in_unsafe_fn)]

pub mod contracts;

pub use contracts::{
    Atom, BasisSet, ContractError, IntegralFamily, Operator, OperatorKind, Representation, Shell,
    ShellPrimitive,
};
