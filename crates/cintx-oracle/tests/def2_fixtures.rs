//! Shared def2 benchmark/coverage fixtures (`def2_speed_precision_plan.md` D0.1).
//!
//! Included with `#[path]` by `def2_device_coverage` and
//! `def2_throughput_benchmark` so the coverage census and the timing runs are
//! describing the *same* molecules. A second copy of a geometry is a second
//! workload wearing the first one's name, and every speed number in the plan is
//! quoted per workload.
//!
//! Not every consumer uses every helper, hence the module-level `dead_code`
//! allowance: an integration-test module is compiled once per including test
//! binary, and warning about the halves each one does not need would be noise.

#![allow(dead_code)]

use cintx_basis::raw::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, CHARGE_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD,
    PTR_EXP,
};
use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis};
use cintx_cubecl::{BatchAtom, BatchShell};

/// The def2 fixture geometries as bare atom lists, so the GTH-MOLOPT arms of
/// the benchmark run the *same* molecules — a second geometry under the same
/// name would be a second workload, and every number is quoted per workload.
pub fn water_atoms() -> Vec<AtomSpec> {
    vec![
        AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
        AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
        AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
    ]
}

pub fn methane_atoms() -> Vec<AtomSpec> {
    let d = 0.629;
    vec![
        AtomSpec::from_angstrom("C", [0.0, 0.0, 0.0]).unwrap(),
        AtomSpec::from_angstrom("H", [d, d, d]).unwrap(),
        AtomSpec::from_angstrom("H", [-d, -d, d]).unwrap(),
        AtomSpec::from_angstrom("H", [-d, d, -d]).unwrap(),
        AtomSpec::from_angstrom("H", [d, -d, -d]).unwrap(),
    ]
}

pub fn sulfur_dioxide_atoms() -> Vec<AtomSpec> {
    let (r, half) = (1.4308_f64, 59.665_f64.to_radians());
    let (y, z) = (r * half.sin(), r * half.cos());
    vec![
        AtomSpec::from_angstrom("S", [0.0, 0.0, 0.0]).unwrap(),
        AtomSpec::from_angstrom("O", [0.0, y, z]).unwrap(),
        AtomSpec::from_angstrom("O", [0.0, -y, z]).unwrap(),
    ]
}

/// Benzene, D6h, in the xy plane: r(C-C) = 1.397 A, r(C-H) = 1.084 A.
///
/// The many-shell GTH fixture. A GTH-MOLOPT atom carries two or three
/// shells, so H2O is 7 shells and 406 quartets — too few to load a device
/// grid or to time. Benzene is 30 shells and ~108 k canonical quartets, every
/// one of them generally contracted.
pub fn benzene_atoms() -> Vec<AtomSpec> {
    let (rc, rh) = (1.397_f64, 1.397_f64 + 1.084_f64);
    let mut atoms = Vec::with_capacity(12);
    for k in 0..6 {
        let theta = f64::from(k) * std::f64::consts::PI / 3.0;
        let (s, c) = theta.sin_cos();
        atoms.push(AtomSpec::from_angstrom("C", [rc * c, rc * s, 0.0]).unwrap());
    }
    for k in 0..6 {
        let theta = f64::from(k) * std::f64::consts::PI / 3.0;
        let (s, c) = theta.sin_cos();
        atoms.push(AtomSpec::from_angstrom("H", [rh * c, rh * s, 0.0]).unwrap());
    }
    atoms
}

/// H2O at the plan's reference geometry. 12 shells / 24 spherical AOs in
/// def2-SVP; 19 / 43 in def2-TZVP (pinned in `cintx-basis`'s `raw.rs` tests).
pub fn water(basis: StandardBasis) -> Molecule {
    Molecule::new(water_atoms(), basis)
}

/// CH4, tetrahedral. Five centres in a light basis: the fixture that loads the
/// *many-pair* end of the work list rather than the high-`l` end.
pub fn methane(basis: StandardBasis) -> Molecule {
    Molecule::new(methane_atoms(), basis)
}

/// SO2 at its experimental geometry (r(S-O) = 1.4308 A, angle 119.33 deg).
///
/// The second-row fixture D0.1 asks for. Sulfur's def2-TZVP block runs
/// `s s s s s p p p p p d d f`, so an `(f f | f f)` class here is carried by an
/// atom with five s-contractions and real contraction depth, rather than by a
/// single tight f function on oxygen. Without it the `nroots` 6-7 buckets in a
/// TZVP timing are too thin to weigh anything.
pub fn sulfur_dioxide(basis: StandardBasis) -> Molecule {
    // Half-angle 59.665 deg; y = r sin, z = r cos, S at the origin.
    Molecule::new(sulfur_dioxide_atoms(), basis)
}

/// The GTH-MOLOPT workloads, as raw arrays (`GthBasis` is not wired into
/// `Molecule`; see `cintx-basis/data/gth/README.md`).
///
/// One entry per `(molecule, basis)` the GTH benchmark and parity tests run,
/// so the two describe the same work lists. `TZVP-MOLOPT-GTH` covers the nine
/// elements of the original paper, which includes every atom here.
#[cfg(feature = "gth")]
pub fn gth_workloads() -> Vec<(String, RawArrays)> {
    use cintx_basis::{GthBasis, to_raw_arrays_gth};
    let mut out = Vec::new();
    for basis in [GthBasis::DzvpMoloptSr, GthBasis::TzvpMolopt] {
        for (name, atoms) in [
            ("H2O", water_atoms()),
            ("CH4", methane_atoms()),
            ("SO2", sulfur_dioxide_atoms()),
            ("C6H6", benzene_atoms()),
        ] {
            let arrays = to_raw_arrays_gth(&atoms, basis).expect("GTH raw arrays");
            out.push((format!("{name} / {}", basis.name()), arrays));
        }
    }
    out
}

pub fn shell_l(arrays: &RawArrays, shell: usize) -> usize {
    arrays.bas[shell * BAS_SLOTS + ANG_OF] as usize
}

/// Flatten raw `atm`/`bas`/`env` into the backend's batch shell table.
///
/// `env` holds coefficients contraction-major (`env[ptr + c*nprim + p]`);
/// `BatchShell` wants them primitive-major, matching `cintx_compat::raw`
/// (WR-03).
pub fn batch_shells(arrays: &RawArrays) -> Vec<BatchShell> {
    (0..arrays.nbas())
        .map(|shell| {
            let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
            let nprim = record[NPRIM_OF] as usize;
            let nctr = record[NCTR_OF] as usize;
            let exp_ptr = record[PTR_EXP] as usize;
            let coeff_ptr = record[PTR_COEFF] as usize;
            let atom = record[ATOM_OF] as usize;
            let coord_ptr = arrays.atm[atom * ATM_SLOTS + PTR_COORD] as usize;

            let mut coefficients = vec![0.0_f64; nprim * nctr];
            for c in 0..nctr {
                for p in 0..nprim {
                    coefficients[p * nctr + c] = arrays.env[coeff_ptr + c * nprim + p];
                }
            }
            BatchShell {
                l: record[ANG_OF] as u8,
                nprim: nprim as u32,
                nctr: nctr as u32,
                exponents: arrays.env[exp_ptr..exp_ptr + nprim].to_vec(),
                coefficients,
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect()
}

pub fn batch_atoms(arrays: &RawArrays) -> Vec<BatchAtom> {
    (0..arrays.natm())
        .map(|atom| {
            let record = &arrays.atm[atom * ATM_SLOTS..(atom + 1) * ATM_SLOTS];
            let coord_ptr = record[PTR_COORD] as usize;
            BatchAtom {
                charge: f64::from(record[CHARGE_OF]),
                center: [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
            }
        })
        .collect()
}
