//! Embedded standard basis-set catalog.
//!
//! Data is vendored verbatim from the Basis Set Exchange (v0.12, Turbomole 7.3
//! source data) under `crates/cintx-basis/data/`. It is parsed once on first
//! use and cached for the process lifetime — parsing all of def2-TZVP is a few
//! milliseconds, but it should not happen per molecule.

use crate::error::BasisError;
#[cfg(feature = "gth")]
use crate::format::parse_cp2k_basis;
use crate::format::{BasisTable, EcpTable, parse_basis, parse_ecp};
use std::sync::OnceLock;

const DEF2_SVP_TEXT: &str = include_str!("../data/def2-svp.nwchem");
const DEF2_TZVP_TEXT: &str = include_str!("../data/def2-tzvp.nwchem");
const DEF2_ECP_TEXT: &str = include_str!("../data/def2-ecp.nwchem");
const DEF2_JFIT_TEXT: &str = include_str!("../data/def2-universal-jfit.nwchem");
const DEF2_JKFIT_TEXT: &str = include_str!("../data/def2-universal-jkfit.nwchem");
// GPLv2-licensed (see data/gth/README.md), unlike the rest of this crate —
// embedded only when the `gth` feature is enabled.
#[cfg(feature = "gth")]
const GTH_MOLOPT_TEXT: &str = include_str!("../data/gth/BASIS_MOLOPT");

/// A standard basis set available from the embedded catalog.
///
/// The two `Def2*Fit` entries are **auxiliary** bases: they are not orbital
/// bases and are never used to build a wavefunction. They carry the fitting
/// functions that density fitting expands a product density in — `def2/J` for
/// Coulomb-only (RI-J) and `def2/JK` where exchange is fitted too. Both are
/// *universal*: one table serves every def2 orbital basis, which is why they
/// are catalog entries in their own right rather than a property of
/// [`Self::Def2Svp`] or [`Self::Def2Tzvp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardBasis {
    Def2Svp,
    Def2Tzvp,
    /// `def2-universal-JFIT`, published as **def2/J**.
    Def2JFit,
    /// `def2-universal-JKFIT`, published as **def2/JK**.
    Def2JkFit,
}

impl StandardBasis {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Def2Svp => "def2-SVP",
            Self::Def2Tzvp => "def2-TZVP",
            Self::Def2JFit => "def2/J",
            Self::Def2JkFit => "def2/JK",
        }
    }

    /// Is this an auxiliary (density-fitting) basis rather than an orbital one?
    ///
    /// Callers that build a molecule's AO shells want the orbital bases; callers
    /// assembling an RI-J or RI-JK work list want the auxiliary ones. Mixing them
    /// up produces a plausible-looking calculation of the wrong thing, so the
    /// distinction is on the type rather than left to the name.
    #[must_use]
    pub fn is_auxiliary(self) -> bool {
        matches!(self, Self::Def2JFit | Self::Def2JkFit)
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
            // BSE calls these `def2-universal-jfit`/`-jkfit`; the literature and
            // every program input file call them `def2/J` and `def2/JK`. The
            // separator-stripping key above maps both spellings onto one entry.
            "def2j" | "def2jfit" | "def2universaljfit" => Ok(Self::Def2JFit),
            "def2jk" | "def2jkfit" | "def2universaljkfit" => Ok(Self::Def2JkFit),
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
            Self::Def2JFit => {
                static CACHE: OnceLock<BasisTable> = OnceLock::new();
                CACHE.get_or_init(|| {
                    parse_basis(DEF2_JFIT_TEXT).expect("embedded def2/J data must parse")
                })
            }
            Self::Def2JkFit => {
                static CACHE: OnceLock<BasisTable> = OnceLock::new();
                CACHE.get_or_init(|| {
                    parse_basis(DEF2_JKFIT_TEXT).expect("embedded def2/JK data must parse")
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

/// A GTH-MOLOPT orbital basis set from the embedded CP2K catalog.
///
/// Gated behind the `gth` Cargo feature (off by default): the vendored data
/// backing it is GPLv2-licensed, unlike the rest of this crate — see
/// `data/gth/README.md`. Enabling `gth` opts a build into carrying that data.
///
/// **Basis data only — no pseudopotential support.** GTH-MOLOPT basis sets
/// are designed to pair with GTH-type pseudopotentials, a separable
/// local+nonlocal form unrelated to the semi-local ECP formalism
/// `cintx-core::ecp` implements for def2-ECP. cintx does not implement GTH
/// pseudopotential integrals, so — unlike [`StandardBasis`] — `GthBasis` is
/// deliberately **not** wired into [`crate::build::Molecule`]: there is no
/// `core_electrons`/ECP-shell handling for it, and using these shells for
/// anything beyond overlap/kinetic-type integrals on light elements requires
/// supplying and applying the matching GTH pseudopotential yourself. See
/// `data/gth/README.md` for the full scoping rationale and provenance.
#[cfg(feature = "gth")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GthBasis {
    /// `DZVP-MOLOPT-SR-GTH`: short-range double-zeta-valence-plus-polarization,
    /// covering 71 elements. There is no published short-range triple-zeta
    /// (`TZVP-MOLOPT-SR-GTH`) variant upstream.
    DzvpMoloptSr,
    /// `TZVP-MOLOPT-GTH`: full-range triple-zeta-valence-plus-polarization,
    /// covering the 9 elements (H, C, N, O, F, Si, P, S, Cl) from the
    /// original VandeVondele & Hutter paper.
    TzvpMolopt,
}

#[cfg(feature = "gth")]
impl GthBasis {
    /// The canonical name as CP2K's own `BASIS_MOLOPT` file writes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::DzvpMoloptSr => "DZVP-MOLOPT-SR-GTH",
            Self::TzvpMolopt => "TZVP-MOLOPT-GTH",
        }
    }

    /// Resolve a basis-set name, case- and separator-insensitive.
    ///
    /// Accepts both the literature/common spelling (`gth-dzvp-molopt-sr`)
    /// and CP2K's own file spelling (`dzvp-molopt-sr-gth`); the
    /// separator-stripping key below maps both onto one entry.
    ///
    /// # Errors
    /// Returns [`BasisError::UnknownBasis`] when the name is not in the
    /// catalog — including `gth-tzvp-molopt-sr`, which does not exist
    /// upstream.
    pub fn from_name(name: &str) -> Result<Self, BasisError> {
        let key: String = name
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .map(|ch| ch.to_ascii_lowercase())
            .collect();
        match key.as_str() {
            "gthdzvpmoloptsr" | "dzvpmoloptsrgth" => Ok(Self::DzvpMoloptSr),
            "gthtzvpmolopt" | "tzvpmoloptgth" => Ok(Self::TzvpMolopt),
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
    /// `catalog_parses_every_embedded_gth_table`.
    #[must_use]
    pub fn table(self) -> &'static BasisTable {
        match self {
            Self::DzvpMoloptSr => {
                static CACHE: OnceLock<BasisTable> = OnceLock::new();
                CACHE.get_or_init(|| {
                    parse_cp2k_basis(GTH_MOLOPT_TEXT, "DZVP-MOLOPT-SR-GTH")
                        .expect("embedded DZVP-MOLOPT-SR-GTH data must parse")
                })
            }
            Self::TzvpMolopt => {
                static CACHE: OnceLock<BasisTable> = OnceLock::new();
                CACHE.get_or_init(|| {
                    parse_cp2k_basis(GTH_MOLOPT_TEXT, "TZVP-MOLOPT-GTH")
                        .expect("embedded TZVP-MOLOPT-GTH data must parse")
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_parses_every_embedded_table() {
        for basis in [
            StandardBasis::Def2Svp,
            StandardBasis::Def2Tzvp,
            StandardBasis::Def2JFit,
            StandardBasis::Def2JkFit,
        ] {
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

    /// Both spellings of each auxiliary basis resolve to the same entry: the
    /// literature name (`def2/J`) and the BSE export name
    /// (`def2-universal-jfit`) name one table, and a caller that learned the
    /// name from either source must land on it.
    #[test]
    fn resolves_auxiliary_basis_names_from_either_spelling() {
        for name in ["def2/J", "def2-J", "def2-jfit", "def2-universal-JFIT"] {
            assert_eq!(
                StandardBasis::from_name(name).unwrap(),
                StandardBasis::Def2JFit,
                "{name}"
            );
        }
        for name in ["def2/JK", "def2-JK", "def2-jkfit", "def2-universal-JKFIT"] {
            assert_eq!(
                StandardBasis::from_name(name).unwrap(),
                StandardBasis::Def2JkFit,
                "{name}"
            );
        }
    }

    #[test]
    #[cfg(feature = "gth")]
    fn catalog_parses_every_embedded_gth_table() {
        for basis in [GthBasis::DzvpMoloptSr, GthBasis::TzvpMolopt] {
            let table = basis.table();
            assert!(
                !table.is_empty(),
                "{} parsed to zero elements",
                basis.name()
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
        assert_eq!(
            GthBasis::DzvpMoloptSr.table().len(),
            71,
            "DZVP-MOLOPT-SR-GTH covers 71 elements upstream"
        );
        assert_eq!(
            GthBasis::TzvpMolopt.table().len(),
            9,
            "TZVP-MOLOPT-GTH covers the 9 elements from the original paper"
        );
    }

    /// The published `DZVP-MOLOPT-SR-GTH` composition for hydrogen is 5
    /// primitives shared by [2s,1p] — one exponent set feeding both angular
    /// momenta, unlike def2's per-`l` exponent lists.
    #[test]
    #[cfg(feature = "gth")]
    fn gth_dzvp_molopt_sr_hydrogen_matches_published_composition() {
        let blocks = &GthBasis::DzvpMoloptSr.table()[&1];
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].ang_momentum, 0);
        assert_eq!(blocks[0].nctr, 2);
        assert_eq!(blocks[0].nprim(), 5);
        assert_eq!(blocks[1].ang_momentum, 1);
        assert_eq!(blocks[1].nctr, 1);
        assert_eq!(blocks[1].exponents, blocks[0].exponents);
    }

    #[test]
    #[cfg(feature = "gth")]
    fn resolves_gth_basis_names_loosely() {
        for name in [
            "gth-dzvp-molopt-sr",
            "GTH_DZVP_MOLOPT_SR",
            "dzvp-molopt-sr-gth",
        ] {
            assert_eq!(
                GthBasis::from_name(name).unwrap(),
                GthBasis::DzvpMoloptSr,
                "{name}"
            );
        }
        for name in ["gth-tzvp-molopt", "tzvp-molopt-gth"] {
            assert_eq!(
                GthBasis::from_name(name).unwrap(),
                GthBasis::TzvpMolopt,
                "{name}"
            );
        }
    }

    /// `gth-tzvp-molopt-sr` does not exist upstream — CP2K's MOLOPT library
    /// only ships short-range variants at SZV and DZVP quality — so it must
    /// not silently resolve to anything.
    #[test]
    #[cfg(feature = "gth")]
    fn gth_tzvp_molopt_sr_is_not_a_real_basis() {
        assert!(GthBasis::from_name("gth-tzvp-molopt-sr").is_err());
        assert!(GthBasis::from_name("tzvp-molopt-sr-gth").is_err());
    }

    /// The auxiliary tables are flagged as auxiliary and the orbital ones are
    /// not. This is what stops an RI-J work list being built from AO shells (or
    /// a wavefunction from fitting functions) — a mix-up that produces numbers
    /// rather than an error.
    #[test]
    fn auxiliary_flag_separates_fitting_from_orbital_bases() {
        assert!(StandardBasis::Def2JFit.is_auxiliary());
        assert!(StandardBasis::Def2JkFit.is_auxiliary());
        assert!(!StandardBasis::Def2Svp.is_auxiliary());
        assert!(!StandardBasis::Def2Tzvp.is_auxiliary());
    }

    /// def2/J hydrogen is the published (5s,2p,1d) -> [3s,1p,1d] fitting set,
    /// and def2/JK hydrogen is (4s,2p,2d) -> [2s,2p,2d]. Both carry angular
    /// momenta the *orbital* def2-SVP hydrogen does not (d functions), which is
    /// exactly why an RI-J work list reaches launch classes the AO-only
    /// fixtures never did.
    #[test]
    fn auxiliary_hydrogen_matches_published_composition() {
        let jfit = &StandardBasis::Def2JFit.table()[&1];
        let l_counts: Vec<u8> = jfit.iter().map(|b| b.ang_momentum).collect();
        assert_eq!(l_counts, vec![0, 0, 0, 1, 2], "def2/J H composition");
        assert_eq!(jfit[0].nprim(), 3);

        let jkfit = &StandardBasis::Def2JkFit.table()[&1];
        let l_counts: Vec<u8> = jkfit.iter().map(|b| b.ang_momentum).collect();
        assert_eq!(l_counts, vec![0, 0, 1, 1, 2, 2], "def2/JK H composition");
        assert_eq!(jkfit[0].nprim(), 3);
    }
}
