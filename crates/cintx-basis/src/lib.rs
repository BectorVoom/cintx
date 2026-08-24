//! Standard Gaussian basis-set catalog for cintx.
//!
//! Provides def2-SVP and def2-TZVP (with def2-ECP for Z >= 37) as embedded,
//! parsed tables, and builds either a typed [`cintx_core::BasisSet`] or the raw
//! `atm`/`bas`/`env` arrays libcint-compatible entry points consume.
//!
//! ```
//! use cintx_basis::{AtomSpec, Molecule, StandardBasis, to_raw_arrays};
//!
//! let molecule = Molecule::new(
//!     vec![
//!         AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0])?,
//!         AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587])?,
//!         AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587])?,
//!     ],
//!     StandardBasis::Def2Svp,
//! );
//! let raw = to_raw_arrays(&molecule)?;
//! assert_eq!(raw.nbas(), 12);
//! # Ok::<(), cintx_basis::BasisError>(())
//! ```
//!
//! Normalization follows libcint/PySCF exactly; see [`normalize`] for the two
//! stages and why both are required.

pub mod build;
pub mod catalog;
pub mod element;
pub mod error;
pub mod format;
pub mod normalize;
pub mod raw;

pub use build::{AtomSpec, BOHR_PER_ANGSTROM, Molecule};
pub use catalog::{StandardBasis, def2_ecp_table};
pub use error::BasisError;
pub use format::{ContractionBlock, EcpBlock, EcpRecord, parse_basis, parse_ecp};
pub use normalize::{gaussian_int, gto_norm, normalize_block};
pub use raw::{RawArrays, to_raw_arrays};
