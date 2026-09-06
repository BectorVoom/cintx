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
//!
//! The `gth` Cargo feature (off by default) additionally exposes
//! `catalog::GthBasis`: two vendored GTH-MOLOPT orbital basis sets
//! (`DZVP-MOLOPT-SR-GTH`, `TZVP-MOLOPT-GTH`) as parsed tables. Unlike
//! [`StandardBasis`], it is **not** wired into [`build::Molecule`]: cintx has
//! no GTH pseudopotential integral support, so there is no automatic
//! core-electron or ECP-shell handling for it. The feature is opt-in because
//! its vendored data is GPLv2-licensed, unlike the rest of this crate — see
//! `data/gth/README.md`.

pub mod build;
pub mod catalog;
pub mod element;
pub mod error;
pub mod format;
pub mod normalize;
pub mod raw;

pub use build::{AtomSpec, BOHR_PER_ANGSTROM, Molecule};
#[cfg(feature = "gth")]
pub use catalog::GthBasis;
pub use catalog::{StandardBasis, def2_ecp_table};
pub use error::BasisError;
pub use format::{ContractionBlock, EcpBlock, EcpRecord, parse_basis, parse_ecp};
pub use normalize::{gaussian_int, gto_norm, normalize_block};
#[cfg(feature = "gth")]
pub use raw::to_raw_arrays_gth;
pub use raw::{RawArrays, to_raw_arrays, to_raw_arrays_with_auxiliary};
