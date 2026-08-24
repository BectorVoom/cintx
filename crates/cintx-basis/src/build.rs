//! Molecule -> `BasisSet` / raw `atm`/`bas`/`env` construction.

use crate::catalog::{StandardBasis, def2_ecp_table};
use crate::element::atomic_number;
use crate::error::BasisError;
use crate::normalize::normalize_block;
use cintx_core::{Atom, BasisSet, EcpChannel, EcpShell, NuclearModel, Representation, Shell};
use std::sync::Arc;

/// Bohr per Angstrom (CODATA 2018), matching PySCF's `param.BOHR`.
pub const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.529_177_210_903;

/// One atom of an input geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomSpec {
    pub atomic_number: u16,
    /// Coordinates in Bohr.
    pub coord_bohr: [f64; 3],
}

impl AtomSpec {
    /// Build from an element symbol and Angstrom coordinates.
    ///
    /// # Errors
    /// Returns [`BasisError::UnknownElement`] for an unrecognized symbol.
    pub fn from_angstrom(symbol: &str, coord_angstrom: [f64; 3]) -> Result<Self, BasisError> {
        let z = atomic_number(symbol).ok_or_else(|| BasisError::UnknownElement {
            symbol: symbol.to_owned(),
        })?;
        Ok(Self {
            atomic_number: z,
            coord_bohr: coord_angstrom.map(|value| value * BOHR_PER_ANGSTROM),
        })
    }

    /// Build from an element symbol and Bohr coordinates.
    ///
    /// # Errors
    /// Returns [`BasisError::UnknownElement`] for an unrecognized symbol.
    pub fn from_bohr(symbol: &str, coord_bohr: [f64; 3]) -> Result<Self, BasisError> {
        let z = atomic_number(symbol).ok_or_else(|| BasisError::UnknownElement {
            symbol: symbol.to_owned(),
        })?;
        Ok(Self {
            atomic_number: z,
            coord_bohr,
        })
    }
}

/// A geometry plus the basis set assigned to it.
#[derive(Clone, Debug)]
pub struct Molecule {
    pub atoms: Vec<AtomSpec>,
    pub basis: StandardBasis,
    pub representation: Representation,
}

impl Molecule {
    #[must_use]
    pub fn new(atoms: Vec<AtomSpec>, basis: StandardBasis) -> Self {
        Self {
            atoms,
            basis,
            // def2 sets are defined and published for a spherical AO basis.
            representation: Representation::Spheric,
        }
    }

    #[must_use]
    pub fn with_representation(mut self, representation: Representation) -> Self {
        self.representation = representation;
        self
    }

    /// Effective nuclear charge seen by the valence electrons: `Z` minus the
    /// core electrons removed by the ECP, if any.
    ///
    /// libcint expects `atm[CHARGE_OF]` to carry this reduced charge whenever
    /// an ECP is present; using the bare `Z` silently double-counts the core.
    #[must_use]
    pub fn effective_charge(&self, atom: &AtomSpec) -> i32 {
        i32::from(atom.atomic_number) - i32::from(self.core_electrons(atom.atomic_number))
    }

    fn core_electrons(&self, z: u16) -> u16 {
        if z < StandardBasis::ECP_THRESHOLD {
            return 0;
        }
        def2_ecp_table()
            .get(&z)
            .map_or(0, |record| record.core_electrons)
    }

    /// Normalized contraction blocks for one atom, in catalog order.
    fn normalized_blocks(
        &self,
        z: u16,
    ) -> Result<Vec<crate::format::ContractionBlock>, BasisError> {
        let blocks = self
            .basis
            .table()
            .get(&z)
            .ok_or(BasisError::MissingElement {
                basis: self.basis.name(),
                atomic_number: z,
            })?;

        Ok(blocks
            .iter()
            .map(|block| {
                let mut normalized = block.clone();
                normalize_block(
                    i32::from(block.ang_momentum),
                    &normalized.exponents,
                    &mut normalized.coefficients,
                    normalized.nctr,
                );
                normalized
            })
            .collect())
    }

    /// Build the typed [`BasisSet`], including ECP shells for Z >= 37.
    ///
    /// Shells are emitted atom-major and, within an atom, in catalog order —
    /// the same ordering PySCF produces, so AO indices line up.
    ///
    /// # Errors
    /// Returns [`BasisError`] if an element is absent from the basis table or
    /// if core validation rejects a shell.
    pub fn to_basis_set(&self) -> Result<BasisSet, BasisError> {
        let mut atoms = Vec::with_capacity(self.atoms.len());
        let mut shells: Vec<Arc<Shell>> = Vec::new();
        let mut ecp_shells: Vec<Arc<EcpShell>> = Vec::new();

        for (atom_index, spec) in self.atoms.iter().enumerate() {
            atoms.push(Atom::try_new(
                spec.atomic_number,
                spec.coord_bohr,
                NuclearModel::Point,
                None,
                None,
            )?);

            for block in self.normalized_blocks(spec.atomic_number)? {
                shells.push(Arc::new(Shell::try_new(
                    atom_index as u32,
                    block.ang_momentum,
                    u16::try_from(block.nprim()).map_err(|_| BasisError::MalformedBlock {
                        detail: format!("nprim {} exceeds u16", block.nprim()),
                    })?,
                    u16::try_from(block.nctr).map_err(|_| BasisError::MalformedBlock {
                        detail: format!("nctr {} exceeds u16", block.nctr),
                    })?,
                    0,
                    self.representation,
                    Arc::from(block.exponents.into_boxed_slice()),
                    Arc::from(block.coefficients.into_boxed_slice()),
                )?));
            }

            if let Some(record) = def2_ecp_table().get(&spec.atomic_number) {
                for ecp_block in &record.blocks {
                    // cintx_core stores one radial power per shell, so a block
                    // mixing r-powers is split into one shell per power. def2
                    // uses a single power per channel in practice.
                    let mut powers: Vec<i16> = ecp_block.radial_powers.clone();
                    powers.sort_unstable();
                    powers.dedup();
                    for power in powers {
                        let (exponents, coefficients): (Vec<f64>, Vec<f64>) = ecp_block
                            .radial_powers
                            .iter()
                            .zip(&ecp_block.exponents)
                            .zip(&ecp_block.coefficients)
                            .filter(|&((&p, _), _)| p == power)
                            .map(|((_, &e), &c)| (e, c))
                            .unzip();
                        if exponents.is_empty() {
                            continue;
                        }
                        let channel = match ecp_block.projector {
                            None => EcpChannel::Local,
                            Some(l) => EcpChannel::Projected(l),
                        };
                        ecp_shells.push(Arc::new(EcpShell::try_new(
                            atom_index as u32,
                            channel,
                            power,
                            u16::try_from(exponents.len()).map_err(|_| {
                                BasisError::MalformedBlock {
                                    detail: "ECP nprim exceeds u16".to_owned(),
                                }
                            })?,
                            1,
                            0,
                            Arc::from(exponents.into_boxed_slice()),
                            Arc::from(coefficients.into_boxed_slice()),
                        )?));
                    }
                }
            }
        }

        Ok(BasisSet::try_new_with_ecp(
            Arc::from(atoms.into_boxed_slice()),
            Arc::from(shells.into_boxed_slice()),
            Arc::from(ecp_shells.into_boxed_slice()),
        )?)
    }
}
