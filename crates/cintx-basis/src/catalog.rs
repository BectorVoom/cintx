//! Embedded standard basis-set catalog.
//!
//! Data is vendored verbatim from the Basis Set Exchange (v0.12, Turbomole 7.3
//! source data) under `crates/cintx-basis/data/`. It is parsed once on first
//! use and cached for the process lifetime — parsing all of def2-TZVP is a few
//! milliseconds, but it should not happen per molecule.

use crate::error::BasisError;
use crate::format::{BasisTable, EcpTable, parse_basis, parse_ecp};
use std::sync::OnceLock;

const DEF2_SVP_TEXT: &str = include_str!("../data/def2-svp.nwchem");
const DEF2_TZVP_TEXT: &str = include_str!("../data/def2-tzvp.nwchem");
const DEF2_ECP_TEXT: &str = include_str!("../data/def2-ecp.nwchem");

/// A standard basis set available from the embedded catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardBasis {
    Def2Svp,
    Def2Tzvp,
}

impl StandardBasis {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Def2Svp => "def2-SVP",
            Self::Def2Tzvp => "def2-TZVP",
        }
    }

    /// Resolve a basis-set name, case- and separator-insensitive
    /// (`def2-SVP`, `def2_svp`, `DEF2SVP` all resolve).
    ///
    /// # Errors
    /// Returns [`BasisError::UnknownBasis`] when the name is not in the catalog.
    pub fn from_name(name: &str) -> Result<Self, BasisError> {
        let key: String = name
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .map(|ch| ch.to_ascii_lowercase())
            .collect();
        match key.as_str() {
            "def2svp" => Ok(Self::Def2Svp),
            "def2tzvp" => Ok(Self::Def2Tzvp),
            _ => Err(BasisError::UnknownBasis {
                name: name.to_owned(),
            }),
        }
    }

    /// The parsed orbital-basis table for this basis set.
    ///
    /// # Panics
    /// Panics if the vendored data file fails to parse. That is a build-time
    /// data-integrity failure, not a runtime condition, and is covered by
    /// `catalog_parses_every_embedded_table`.
    #[must_use]
    pub fn table(self) -> &'static BasisTable {
        match self {
            Self::Def2Svp => {
                static CACHE: OnceLock<BasisTable> = OnceLock::new();
                CACHE.get_or_init(|| {
                    parse_basis(DEF2_SVP_TEXT).expect("embedded def2-SVP data must parse")
                })
            }
            Self::Def2Tzvp => {
                static CACHE: OnceLock<BasisTable> = OnceLock::new();
                CACHE.get_or_init(|| {
                    parse_basis(DEF2_TZVP_TEXT).expect("embedded def2-TZVP data must parse")
                })
            }
        }
    }

    /// The lowest atomic number that uses def2-ECP rather than an all-electron
    /// treatment. def2 replaces the core for Z >= 37 (Rb onward).
    pub const ECP_THRESHOLD: u16 = 37;
}

/// The parsed def2-ECP table, shared by both def2 basis sets.
///
/// # Panics
/// Panics if the vendored ECP data fails to parse (see [`StandardBasis::table`]).
#[must_use]
pub fn def2_ecp_table() -> &'static EcpTable {
    static CACHE: OnceLock<EcpTable> = OnceLock::new();
    CACHE.get_or_init(|| parse_ecp(DEF2_ECP_TEXT).expect("embedded def2-ECP data must parse"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_parses_every_embedded_table() {
        for basis in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
            let table = basis.table();
            assert!(
                table.len() > 80,
                "{} should cover most of the periodic table, got {}",
                basis.name(),
                table.len()
            );
            for (z, blocks) in table {
                assert!(
                    !blocks.is_empty(),
                    "{} Z={z} parsed to zero blocks",
                    basis.name()
                );
                for block in blocks {
                    assert_eq!(
                        block.coefficients.len(),
                        block.nprim() * block.nctr,
                        "{} Z={z} l={} coefficient length mismatch",
                        basis.name(),
                        block.ang_momentum
                    );
                    assert!(block.exponents.iter().all(|e| e.is_finite() && *e > 0.0));
                }
            }
        }
        assert!(def2_ecp_table().len() > 40, "def2-ECP should cover Z>=37");
    }

    /// The published def2-SVP composition for hydrogen is (4s,1p) -> [2s,1p]:
    /// two s shells (3 + 1 primitives) and one p shell.
    #[test]
    fn def2_svp_hydrogen_matches_published_composition() {
        let blocks = &StandardBasis::Def2Svp.table()[&1];
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].ang_momentum, 0);
        assert_eq!(blocks[0].nprim(), 3);
        assert_eq!(blocks[1].ang_momentum, 0);
        assert_eq!(blocks[1].nprim(), 1);
        assert_eq!(blocks[2].ang_momentum, 1);
    }

    /// def2-TZVP oxygen is (11s,6p,2d,1f) -> [5s,3p,2d,1f]; the f function is
    /// what pushes a (ff|ff) quartet to Rys order 7 and out of the device
    /// nroots<=5 envelope.
    #[test]
    fn def2_tzvp_oxygen_reaches_f_functions() {
        let blocks = &StandardBasis::Def2Tzvp.table()[&8];
        let max_l = blocks.iter().map(|b| b.ang_momentum).max().unwrap();
        assert_eq!(max_l, 3, "def2-TZVP oxygen must carry an f function");
        let total_contractions: usize = blocks.iter().map(|b| b.nctr).sum();
        assert_eq!(total_contractions, 11, "[5s,3p,2d,1f] = 11 contractions");
    }

    #[test]
    fn resolves_basis_names_loosely() {
        assert_eq!(
            StandardBasis::from_name("def2-SVP").unwrap(),
            StandardBasis::Def2Svp
        );
        assert_eq!(
            StandardBasis::from_name("DEF2_tzvp").unwrap(),
            StandardBasis::Def2Tzvp
        );
        assert!(StandardBasis::from_name("cc-pVDZ").is_err());
    }
}
