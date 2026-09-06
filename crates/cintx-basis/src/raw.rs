//! Raw `atm` / `bas` / `env` emission in libcint ABI layout.
//!
//! Slot indices are the libcint 6.1.3 ABI and are duplicated here rather than
//! imported so `cintx-basis` stays a leaf crate (depending on `cintx-compat`
//! would pull the whole CubeCL backend into a data crate). They are pinned
//! against `cintx_compat::raw` by
//! `crates/cintx-oracle/tests/def2_raw_slot_parity.rs`.

use crate::build::Molecule;
use crate::error::BasisError;
use crate::normalize::normalize_block;

pub const CHARGE_OF: usize = 0;
pub const PTR_COORD: usize = 1;
pub const NUC_MOD_OF: usize = 2;
pub const PTR_ZETA: usize = 3;
pub const ATM_SLOTS: usize = 6;

pub const ATOM_OF: usize = 0;
pub const ANG_OF: usize = 1;
pub const NPRIM_OF: usize = 2;
pub const NCTR_OF: usize = 3;
pub const KAPPA_OF: usize = 4;
pub const PTR_EXP: usize = 5;
pub const PTR_COEFF: usize = 6;
pub const BAS_SLOTS: usize = 8;

pub const PTR_ENV_START: usize = 20;
pub const POINT_NUC: i32 = 1;

/// Flat libcint call arrays for a molecule.
#[derive(Clone, Debug, PartialEq)]
pub struct RawArrays {
    pub atm: Vec<i32>,
    pub bas: Vec<i32>,
    pub env: Vec<f64>,
    /// Number of leading `bas` rows that are **orbital** shells.
    ///
    /// [`to_raw_arrays`] leaves this equal to [`Self::nbas`] — every shell is an
    /// orbital shell. [`to_raw_arrays_with_auxiliary`] appends the fitting
    /// shells after the orbital ones and sets this to the boundary, so a
    /// three-centre `(mu nu | P)` list can take `mu`, `nu` from
    /// `0..n_orbital_shells` and `P` from `n_orbital_shells..nbas`.
    pub n_orbital_shells: usize,
}

impl RawArrays {
    #[must_use]
    pub fn natm(&self) -> usize {
        self.atm.len() / ATM_SLOTS
    }

    #[must_use]
    pub fn nbas(&self) -> usize {
        self.bas.len() / BAS_SLOTS
    }

    /// Shell indices of the orbital shells.
    #[must_use]
    pub fn orbital_shells(&self) -> std::ops::Range<usize> {
        0..self.n_orbital_shells
    }

    /// Shell indices of the auxiliary (fitting) shells; empty unless the arrays
    /// came from [`to_raw_arrays_with_auxiliary`].
    #[must_use]
    pub fn auxiliary_shells(&self) -> std::ops::Range<usize> {
        self.n_orbital_shells..self.nbas()
    }
}

/// Emit `atm`/`bas`/`env` for a molecule.
///
/// Layout choices that must match PySCF for oracle parity:
/// - all atom coordinates are written first, then per-shell exponent and
///   coefficient arrays in shell order;
/// - coefficients are contraction-major (`env[PTR_COEFF + ic * nprim + ip]`),
///   which is what libcint's `CINTOpt` and every `int*` entry point assume;
/// - `atm[CHARGE_OF]` carries the ECP-reduced charge, not the bare `Z`.
///
/// # Errors
/// Returns [`BasisError`] if an element is absent from the basis table.
pub fn to_raw_arrays(molecule: &Molecule) -> Result<RawArrays, BasisError> {
    let (atm, mut env) = emit_atoms(&molecule.atoms, |spec| molecule.effective_charge(spec));
    let mut bas: Vec<i32> = Vec::new();
    append_table_shells(
        &molecule.atoms,
        molecule.basis.table(),
        molecule.basis.name(),
        &mut bas,
        &mut env,
    )?;
    let n_orbital_shells = bas.len() / BAS_SLOTS;

    Ok(RawArrays {
        atm,
        bas,
        env,
        n_orbital_shells,
    })
}

/// Emit `atm`/`bas`/`env` for `atoms` in a GTH-MOLOPT orbital basis.
///
/// The same layout as [`to_raw_arrays`], with two deliberate differences that
/// follow from what `GthBasis` is (see `data/gth/README.md`):
///
/// - `atm[CHARGE_OF]` carries the **bare** `Z`. A GTH basis pairs with a GTH
///   pseudopotential cintx does not implement, so there is no core-electron
///   reduction to apply and none is invented; the operators these arrays are
///   meant for (`int2e`, overlap, kinetic) do not read the charge at all.
/// - There are no ECP shells.
///
/// Coefficients are normalized by the same libcint/PySCF rule as every other
/// basis, so a raw call and a vendored-libcint call over these arrays agree.
///
/// # Errors
/// Returns [`BasisError::MissingElement`] if an atom's element is absent from
/// the chosen table — `TZVP-MOLOPT-GTH` covers nine elements only.
#[cfg(feature = "gth")]
pub fn to_raw_arrays_gth(
    atoms: &[crate::build::AtomSpec],
    basis: crate::catalog::GthBasis,
) -> Result<RawArrays, BasisError> {
    let (atm, mut env) = emit_atoms(atoms, |spec| i32::from(spec.atomic_number));
    let mut bas: Vec<i32> = Vec::new();
    append_table_shells(atoms, basis.table(), basis.name(), &mut bas, &mut env)?;
    let n_orbital_shells = bas.len() / BAS_SLOTS;
    Ok(RawArrays {
        atm,
        bas,
        env,
        n_orbital_shells,
    })
}

/// Write the atom block: `PTR_ENV_START` reserved slots, one shared zero
/// `zeta`, then every coordinate triple, with `charge_of` deciding what
/// `atm[CHARGE_OF]` carries.
fn emit_atoms(
    atoms: &[crate::build::AtomSpec],
    charge_of: impl Fn(&crate::build::AtomSpec) -> i32,
) -> (Vec<i32>, Vec<f64>) {
    let natm = atoms.len();
    let mut env = vec![0.0_f64; PTR_ENV_START];
    let mut atm = vec![0_i32; natm * ATM_SLOTS];

    // A single shared zero `zeta` slot: every atom uses the point-nucleus
    // model, so the value is never read, but the pointer must still be valid.
    let zeta_ptr = env.len() as i32;
    env.push(0.0);

    for (index, spec) in atoms.iter().enumerate() {
        let coord_ptr = env.len() as i32;
        env.extend_from_slice(&spec.coord_bohr);

        let slot = index * ATM_SLOTS;
        atm[slot + CHARGE_OF] = charge_of(spec);
        atm[slot + PTR_COORD] = coord_ptr;
        atm[slot + NUC_MOD_OF] = POINT_NUC;
        atm[slot + PTR_ZETA] = zeta_ptr;
    }
    (atm, env)
}

/// Emit `atm`/`bas`/`env` for a molecule with a density-fitting auxiliary basis
/// appended after the orbital shells.
///
/// This is the layout a three-centre `(mu nu | P)` list needs: one `bas` array
/// holding both shell sets, with `mu`/`nu` indexing the orbital block and `P`
/// the auxiliary block. [`RawArrays::orbital_shells`] and
/// [`RawArrays::auxiliary_shells`] name the two ranges.
///
/// # Errors
/// Returns [`BasisError`] if an element is absent from either table, or
/// [`BasisError::UnknownBasis`] if `auxiliary` is not actually an auxiliary
/// basis — fitting an AO product density against orbital functions is a
/// mistake that would otherwise produce numbers rather than a failure.
pub fn to_raw_arrays_with_auxiliary(
    molecule: &Molecule,
    auxiliary: crate::catalog::StandardBasis,
) -> Result<RawArrays, BasisError> {
    if !auxiliary.is_auxiliary() {
        return Err(BasisError::UnknownBasis {
            name: format!(
                "{} is an orbital basis, not an auxiliary one",
                auxiliary.name()
            ),
        });
    }

    let mut arrays = to_raw_arrays(molecule)?;
    append_table_shells(
        &molecule.atoms,
        auxiliary.table(),
        auxiliary.name(),
        &mut arrays.bas,
        &mut arrays.env,
    )?;
    Ok(arrays)
}

/// Append every atom's shells from `table`, writing exponents and coefficients
/// into `env` and one `bas` row per contracted shell. `name` labels the table
/// in a [`BasisError::MissingElement`].
fn append_table_shells(
    atoms: &[crate::build::AtomSpec],
    table: &crate::format::BasisTable,
    name: &'static str,
    bas: &mut Vec<i32>,
    env: &mut Vec<f64>,
) -> Result<(), BasisError> {
    for (atom_index, spec) in atoms.iter().enumerate() {
        let blocks = table
            .get(&spec.atomic_number)
            .ok_or(BasisError::MissingElement {
                basis: name,
                atomic_number: spec.atomic_number,
            })?;

        for block in blocks {
            let mut coefficients = block.coefficients.clone();
            normalize_block(
                i32::from(block.ang_momentum),
                &block.exponents,
                &mut coefficients,
                block.nctr,
            );

            let exp_ptr = env.len() as i32;
            env.extend_from_slice(&block.exponents);
            let coeff_ptr = env.len() as i32;
            env.extend_from_slice(&coefficients);

            let mut row = vec![0_i32; BAS_SLOTS];
            row[ATOM_OF] = atom_index as i32;
            row[ANG_OF] = i32::from(block.ang_momentum);
            row[NPRIM_OF] = block.nprim() as i32;
            row[NCTR_OF] = block.nctr as i32;
            row[KAPPA_OF] = 0;
            row[PTR_EXP] = exp_ptr;
            row[PTR_COEFF] = coeff_ptr;
            bas.extend_from_slice(&row);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::AtomSpec;
    use crate::catalog::StandardBasis;

    fn water(basis: StandardBasis) -> Molecule {
        Molecule::new(
            vec![
                AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
                AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
                AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
            ],
            basis,
        )
    }

    /// H2O/def2-SVP is O[3s,2p,1d] + 2 x H[2s,1p] = 12 shells and, spherically,
    /// (3 + 6 + 5) + 2 x (2 + 3) = 24 AOs. These are the batching workload
    /// sizes the throughput plan is built around, so they are pinned.
    #[test]
    fn water_def2_svp_has_expected_shell_and_ao_counts() {
        let raw = to_raw_arrays(&water(StandardBasis::Def2Svp)).unwrap();
        assert_eq!(raw.natm(), 3);
        assert_eq!(raw.nbas(), 12);

        let total_ao: i32 = (0..raw.nbas())
            .map(|s| {
                let l = raw.bas[s * BAS_SLOTS + ANG_OF];
                let nctr = raw.bas[s * BAS_SLOTS + NCTR_OF];
                (2 * l + 1) * nctr
            })
            .sum();
        assert_eq!(total_ao, 24);
    }

    /// H2O/def2-TZVP is O[5s,3p,2d,1f] + 2 x H[3s,1p] = 19 shells and
    /// (5 + 9 + 10 + 7) + 2 x (3 + 3) = 43 spherical AOs.
    #[test]
    fn water_def2_tzvp_has_expected_shell_and_ao_counts() {
        let raw = to_raw_arrays(&water(StandardBasis::Def2Tzvp)).unwrap();
        assert_eq!(raw.nbas(), 19);

        let total_ao: i32 = (0..raw.nbas())
            .map(|s| {
                let l = raw.bas[s * BAS_SLOTS + ANG_OF];
                let nctr = raw.bas[s * BAS_SLOTS + NCTR_OF];
                (2 * l + 1) * nctr
            })
            .sum();
        assert_eq!(total_ao, 43);
    }

    /// Every `env` pointer must land inside `env`, and no shell may overlap
    /// the reserved header.
    #[test]
    fn env_pointers_are_in_bounds() {
        let raw = to_raw_arrays(&water(StandardBasis::Def2Tzvp)).unwrap();
        for s in 0..raw.nbas() {
            let nprim = raw.bas[s * BAS_SLOTS + NPRIM_OF] as usize;
            let nctr = raw.bas[s * BAS_SLOTS + NCTR_OF] as usize;
            let exp_ptr = raw.bas[s * BAS_SLOTS + PTR_EXP] as usize;
            let coeff_ptr = raw.bas[s * BAS_SLOTS + PTR_COEFF] as usize;
            assert!(exp_ptr >= PTR_ENV_START);
            assert!(exp_ptr + nprim <= raw.env.len());
            assert!(coeff_ptr + nprim * nctr <= raw.env.len());
        }
        for a in 0..raw.natm() {
            let ptr = raw.atm[a * ATM_SLOTS + PTR_COORD] as usize;
            assert!(ptr >= PTR_ENV_START && ptr + 3 <= raw.env.len());
        }
    }

    /// An ECP element must carry the reduced charge, not the bare Z.
    #[test]
    fn ecp_element_uses_reduced_charge() {
        let molecule = Molecule::new(
            vec![AtomSpec::from_bohr("Au", [0.0, 0.0, 0.0]).unwrap()],
            StandardBasis::Def2Svp,
        );
        // Au is Z=79 with a 60-electron def2 core.
        let raw = to_raw_arrays(&molecule).unwrap();
        assert_eq!(raw.atm[CHARGE_OF], 19);
    }

    /// Light elements have no ECP, so the charge is the bare Z.
    #[test]
    fn all_electron_element_uses_bare_charge() {
        let raw = to_raw_arrays(&water(StandardBasis::Def2Svp)).unwrap();
        assert_eq!(raw.atm[CHARGE_OF], 8);
        assert_eq!(raw.atm[ATM_SLOTS + CHARGE_OF], 1);
    }
}
