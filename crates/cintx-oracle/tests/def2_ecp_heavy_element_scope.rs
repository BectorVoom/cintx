//! Post-wave-5 Task F — the def2-ECP scope question, answered with evidence
//! rather than with an opinion.
//!
//! `cintx-basis` already parses def2-ECP and builds `EcpShell`s for every
//! `Z >= StandardBasis::ECP_THRESHOLD` (37, i.e. Rb and up): `build.rs` reads
//! `def2_ecp_table()` and the atom's `CHARGE_OF` carries the ECP-reduced core
//! charge. What has never been exercised is the **integral** path for those
//! elements — every ECP parity gate in this crate runs on Cu/LANL2DZ, a
//! hand-built `Z = 29` fixture with a single ECP-bearing atom.
//!
//! The scope question ("is heavy-element ECP in scope, or does main-group come
//! first?") has been carried unanswered across three plans. This file does not
//! decide it. It does the cheap half — run `Z >= 37` elements through the same
//! safe-API-vs-vendored-PySCF comparison the Cu fixture uses, and see what
//! breaks — so that whoever decides is deciding against a measurement.
//!
//! # What it found
//!
//! Three defects, all now fixed, none of which Cu/LANL2DZ could have shown:
//!
//! 1. **A hard panic for every def2-ECP element.** `ecp.rs`'s Type-2 angular
//!    factor filled a buffer in the *Cartesian* component count of the
//!    projector channel but sized it by the *spherical* one. Those agree up to
//!    `lc = 1` and diverge from `lc = 2` (6 vs 5) — and every def2-ECP record
//!    carries an `l = 2` projector, while LANL2DZ's Cu record stops at `l = 1`.
//! 2. **The nuclear charge was the bare `Z`.** `Molecule::to_basis_set` stored
//!    `spec.atomic_number` in `Atom::atomic_number`, the field that becomes
//!    `atm[CHARGE_OF]`, while `to_raw_arrays` wrote the ECP-reduced charge.
//!    `int1e_nuc` through the typed API was therefore too large by exactly
//!    `Z / (Z - n_core)` — 4.111x for Rb.
//! 3. **The existing gate read blocks in the wrong order.**
//!    `safe_api_ecp_parity`'s collector read a pair block row-major where the
//!    safe API returns libcint's column-major. On a one-atom fixture that is
//!    invisible: a spherical ECP centred on the only atom conserves angular
//!    momentum, so every `l_i != l_j` block is identically zero and the
//!    scramble maps zeros onto zeros. Au's Cartesian `(p, f)` block is the
//!    first non-square, non-zero one — Cartesian `f` carries an `l = 1`
//!    contaminant — and it exposed the read immediately.
//!
//! With those fixed, Rb, I and Au all reproduce vendored PySCF `nr_ecp` to
//! ~1e-14 in both representations. The remaining scope question is about
//! coverage breadth — more elements, molecules with several ECP centres,
//! the gradient operators — not about a missing capability.
//!
//! # What is compared
//!
//! `ECPscalar` (`int1e_ecp_sph` / `_cart`) over the full AO matrix of a single
//! def2-SVP heavy atom, cintx safe API against vendored PySCF `nr_ecp`, at the
//! same unified `atol = 1e-12`, `rtol = 0` every other ECP gate uses. Both sides
//! are built from the *same* `cintx-basis` output, so a disagreement is in the
//! integral path and not in the basis marshaling.
//!
//! Run via:
//!     CINTX_ORACLE_BUILD_VENDOR=1 cargo test --locked \
//!         -p cintx-oracle --features cpu --test def2_ecp_heavy_element_scope

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_basis::{AtomSpec, Molecule, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{
    ANG_OF, AS_ECPBAS_OFFSET, AS_NECPBAS, ATOM_OF, BAS_SLOTS, NPRIM_OF, PTR_COEFF, PTR_EXP,
    RADI_POWER, SO_TYPE_OF,
};
use cintx_core::ecp::EcpChannel;
use cintx_core::{BasisSet, OperatorId, Representation, ShellTuple};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use std::sync::Arc;

/// The unified tolerance every ECP parity gate in this crate uses.
const ATOL: f64 = 1e-12;
const RTOL: f64 = 0.0;

/// Elements probed, spanning the def2-ECP range rather than sampling one point:
/// the first element that uses it at all, a 5p main-group one, and a 6th-row
/// one whose ECP replaces 60 electrons.
const HEAVY_ELEMENTS: [&str; 3] = ["Rb", "I", "Au"];

fn molecule(symbol: &str, rep: Representation) -> Molecule {
    Molecule::new(
        vec![AtomSpec::from_angstrom(symbol, [0.0, 0.0, 0.0]).expect("atom spec")],
        StandardBasis::Def2Svp,
    )
    .with_representation(rep)
}

/// Build the vendor-side `(ecpbas, env)` from the typed `BasisSet`'s ECP shells,
/// appending their primitives to a copy of the AO `env`.
///
/// Deliberately derived from the same `BasisSet` the cintx side evaluates, so
/// this comparison cannot be passed or failed by a marshaling difference.
fn vendor_ecp_tables(basis: &BasisSet, env: &[f64]) -> (Vec<i32>, Vec<f64>) {
    let mut env = env.to_vec();
    while env.len() <= AS_NECPBAS {
        env.push(0.0);
    }
    let mut ecpbas = Vec::new();
    for shell in basis.ecp_shells() {
        let exp_ptr = env.len() as i32;
        env.extend_from_slice(&shell.exponents);
        let coeff_ptr = env.len() as i32;
        env.extend_from_slice(&shell.coefficients);

        // PySCF's `ecpbas` reuses BAS slots 3 and 4 as `RADI_POWER` and
        // `SO_TYPE_OF`; the local channel is `ANG_OF = -1`.
        let mut row = vec![0_i32; BAS_SLOTS];
        row[ATOM_OF] = shell.atom_index as i32;
        row[ANG_OF] = match shell.channel {
            EcpChannel::Local => -1,
            EcpChannel::Projected(l) => i32::from(l),
        };
        row[NPRIM_OF] = i32::from(shell.nprim);
        row[RADI_POWER] = i32::from(shell.radial_power);
        row[SO_TYPE_OF] = i32::from(shell.so_type);
        row[PTR_EXP] = exp_ptr;
        row[PTR_COEFF] = coeff_ptr;
        ecpbas.extend_from_slice(&row);
    }
    (ecpbas, env)
}

fn collect_safe_api_ecp_matrix(op: OperatorId, rep: Representation, basis: &BasisSet) -> Vec<f64> {
    let shells: Vec<Arc<_>> = basis.shells().to_vec();
    let shell_nao: Vec<usize> = shells.iter().map(|s| s.ao_per_shell()).collect();
    let n_ao: usize = shell_nao.iter().sum();
    let mut matrix = vec![0.0_f64; n_ao * n_ao];

    let mut row_offset = 0usize;
    for si in 0..shells.len() {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..shells.len() {
            let nj = shell_nao[sj];
            let tuple = ShellTuple::try_from_iter([shells[si].clone(), shells[sj].clone()])
                .expect("ShellTuple");
            let request = SessionRequest::new(op, rep, basis, tuple, ExecutionOptions::default());
            let pair = request
                .query_workspace()
                .expect("query_workspace")
                .evaluate()
                .expect("evaluate")
                .tensor
                .owned_values;
            // The safe API returns a 1e block in libcint's own layout —
            // column-major, bra fastest — so it is read the same way the vendor
            // buffer is.
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = pair[jj * ni + ii];
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn collect_ecp_matrix_vendor(
    rep: &str,
    atm: &[i32],
    bas: &[i32],
    ecpbas: &[i32],
    env: &[f64],
) -> Vec<f64> {
    use cintx_compat::raw::ATM_SLOTS;
    use cintx_oracle::vendor_ffi;

    let nbas_ao = (bas.len() / BAS_SLOTS) as i32;
    let necpbas = (ecpbas.len() / BAS_SLOTS) as i32;

    let mut combined_bas = Vec::with_capacity(bas.len() + ecpbas.len());
    combined_bas.extend_from_slice(bas);
    combined_bas.extend_from_slice(ecpbas);

    let mut env = env.to_vec();
    env[AS_ECPBAS_OFFSET] = f64::from(nbas_ao);
    env[AS_NECPBAS] = f64::from(necpbas);

    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nfn = |l: i32| -> usize {
        if rep == "sph" {
            (2 * l + 1) as usize
        } else {
            ((l + 1) * (l + 2) / 2) as usize
        }
    };
    let shell_nao: Vec<usize> = (0..nbas_ao as usize)
        .map(|s| nfn(bas[s * BAS_SLOTS + ANG_OF]))
        .collect();
    let n_ao: usize = shell_nao.iter().sum();

    let mut matrix = vec![0.0_f64; n_ao * n_ao];
    let mut row_offset = 0usize;
    for si in 0..nbas_ao as usize {
        let ni = shell_nao[si];
        let mut col_offset = 0usize;
        for sj in 0..nbas_ao as usize {
            let nj = shell_nao[sj];
            let mut out = vec![0.0_f64; ni * nj];
            let shls = [si as i32, sj as i32];
            if rep == "sph" {
                vendor_ffi::vendor_ECPscalar_sph(
                    &mut out,
                    &shls,
                    atm,
                    natm,
                    &combined_bas,
                    nbas_ao + necpbas,
                    &env,
                );
            } else {
                vendor_ffi::vendor_ECPscalar_cart(
                    &mut out,
                    &shls,
                    atm,
                    natm,
                    &combined_bas,
                    nbas_ao + necpbas,
                    &env,
                );
            }
            // libcint convention: column-major within a pair block.
            for ii in 0..ni {
                for jj in 0..nj {
                    matrix[(row_offset + ii) * n_ao + (col_offset + jj)] = out[jj * ni + ii];
                }
            }
            col_offset += nj;
        }
        row_offset += ni;
    }
    matrix
}

fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "matrix length mismatch");
    reference
        .iter()
        .zip(observed)
        .filter(|(r, o)| (*r - *o).abs() > atol + rtol * r.abs())
        .count()
}

/// The precondition: `cintx-basis` really does produce ECP shells for these
/// elements, and really does reduce their nuclear charge. If this ever stops
/// holding, the parity result below becomes meaningless rather than wrong.
#[test]
fn def2_basis_carries_ecp_shells_for_heavy_elements() {
    for symbol in HEAVY_ELEMENTS {
        let molecule = molecule(symbol, Representation::Spheric);
        let basis = molecule.to_basis_set().expect("basis set");
        let ecp = basis.ecp_shells();
        assert!(
            !ecp.is_empty(),
            "{symbol} is at or above the def2-ECP threshold \
             ({}) and must carry ECP shells",
            StandardBasis::ECP_THRESHOLD
        );
        let channels: Vec<String> = ecp
            .iter()
            .map(|s| match s.channel {
                EcpChannel::Local => "local".to_string(),
                EcpChannel::Projected(l) => format!("l={l}"),
            })
            .collect();
        // `Atom::atomic_number` is what reaches `atm[CHARGE_OF]`, so on an ECP
        // element it is already the reduced core charge; the true Z comes from
        // the molecule's own spec.
        let true_z = i32::from(molecule.atoms[0].atomic_number);
        let effective = basis.atoms()[0].atomic_number as i32;
        let core = true_z - effective;
        println!(
            "{symbol}: Z={true_z} effective_charge={effective} core_replaced={core} \
             ao_shells={} ecp_shells={} channels={channels:?}",
            basis.shells().len(),
            ecp.len(),
        );
        assert!(
            core > 0,
            "{symbol}: an ECP element must carry the reduced core charge in \
             `Atom::atomic_number`, the field that becomes `atm[CHARGE_OF]` \
             (Z={true_z}, effective={effective})"
        );
    }
}

/// **The measurement.** `ECPscalar` for one `Z >= 37` atom, cintx safe API
/// against vendored PySCF `nr_ecp`, in both representations.
///
/// This is the test whose result answers Task F. A green run says heavy-element
/// def2-ECP already works and the scope question is about *coverage breadth*,
/// not about a missing capability; a red one names exactly what would have to be
/// built.
#[test]
#[cfg(all(has_vendor_libcint, has_vendor_pyscf_nr_ecp))]
fn def2_ecp_heavy_elements_match_vendored_pyscf() {
    let mut failures: Vec<String> = Vec::new();

    for symbol in HEAVY_ELEMENTS {
        for (rep_name, rep, op) in [
            ("cart", Representation::Cart, OperatorId::INT1E_ECP_CART),
            ("sph", Representation::Spheric, OperatorId::INT1E_ECP_SPH),
        ] {
            // The validator requires each shell's own representation to match
            // the request, so the molecule is rebuilt per representation.
            let molecule = molecule(symbol, rep);
            let raw = to_raw_arrays(&molecule).expect("raw arrays");
            let basis = molecule.to_basis_set().expect("basis set");
            let (ecpbas, env) = vendor_ecp_tables(&basis, &raw.env);

            let safe = collect_safe_api_ecp_matrix(op, rep, &basis);
            let vendor = collect_ecp_matrix_vendor(rep_name, &raw.atm, &raw.bas, &ecpbas, &env);
            let mismatches = count_mismatches(&vendor, &safe, ATOL, RTOL);
            let worst = vendor
                .iter()
                .zip(&safe)
                .map(|(v, s)| (v - s).abs())
                .fold(0.0_f64, f64::max);
            let scale = vendor.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            println!(
                "{symbol}/{rep_name}: n_ao^2={} max|vendor|={scale:.3e} \
                 max|diff|={worst:.3e} mismatches={mismatches}",
                vendor.len()
            );
            if mismatches != 0 {
                // Locate the failure by AO shell block so the report names a
                // class rather than an element index.
                let ls: Vec<i32> = basis
                    .shells()
                    .iter()
                    .map(|s| i32::from(s.ang_momentum))
                    .collect();
                let nao_of = |l: i32| -> usize {
                    if rep_name == "sph" {
                        (2 * l + 1) as usize
                    } else {
                        ((l + 1) * (l + 2) / 2) as usize
                    }
                };
                let n_ao: usize = ls.iter().map(|&l| nao_of(l)).sum();
                let mut offs = Vec::with_capacity(ls.len());
                let mut acc = 0usize;
                for &l in &ls {
                    offs.push(acc);
                    acc += nao_of(l);
                }
                let mut bad_blocks: std::collections::BTreeSet<(i32, i32)> =
                    std::collections::BTreeSet::new();
                for (idx, (v, sa)) in vendor.iter().zip(&safe).enumerate() {
                    if (v - sa).abs() > ATOL {
                        let (r, c) = (idx / n_ao, idx % n_ao);
                        let find = |x: usize| ls[offs.partition_point(|&o| o <= x) - 1];
                        bad_blocks.insert((find(r), find(c)));
                    }
                }
                println!("    mismatching (li,lj) blocks: {bad_blocks:?}");
                failures.push(format!(
                    "{symbol}/{rep_name}: {mismatches} of {} elements exceed \
                     atol={ATOL:.0e} (max|diff|={worst:.3e} against max|vendor|={scale:.3e})",
                    vendor.len()
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "def2-ECP heavy-element integral path diverges from vendored PySCF nr_ecp:\n  {}",
        failures.join("\n  ")
    );
}

/// The other half of the ECP contract: an ECP element's **nuclear charge** must
/// be the reduced core charge, not the bare `Z`.
///
/// `Molecule::to_raw_arrays` writes `Molecule::effective_charge` into
/// `atm[CHARGE_OF]`, and `raw.rs`'s own `ecp_element_uses_reduced_charge` test
/// pins that. `Molecule::to_basis_set` instead stores `spec.atomic_number` in
/// `Atom::atomic_number`, which is the bare `Z`. The two marshalings of the same
/// molecule therefore disagree for every `Z >= 37` element, and
/// `def2_basis_carries_ecp_shells_for_heavy_elements` prints the gap
/// (`core_replaced=0` where it should be 28, 46 or 60).
///
/// Whether that disagreement *matters* depends on what reads
/// `Atom::atomic_number`, and this test answers that directly: it evaluates
/// `int1e_nuc` — the one operator whose value is linear in the nuclear charge —
/// through the safe API and against vendored libcint driven by the raw arrays.
/// If the typed path used the bare `Z`, the safe answer would be larger than the
/// vendor answer by exactly `Z / (Z - n_core)`.
#[test]
#[cfg(has_vendor_libcint)]
fn heavy_element_nuclear_attraction_uses_the_ecp_reduced_charge() {
    use cintx_compat::raw::ATM_SLOTS;
    use cintx_oracle::vendor_ffi;

    // `int1e_nuc_sph` sits at manifest position 7; the ECP constants in
    // `OperatorId` stop at the four `int1e_ecp_*` rows, so the position is
    // spelled out the way the manifest defines it.
    const INT1E_NUC_SPH: OperatorId = OperatorId::new(7);

    let mut report = Vec::new();
    for symbol in HEAVY_ELEMENTS {
        let molecule = molecule(symbol, Representation::Spheric);
        let raw = to_raw_arrays(&molecule).expect("raw arrays");
        let basis = molecule.to_basis_set().expect("basis set");

        let true_z = i32::from(molecule.atoms[0].atomic_number);
        let raw_charge = raw.atm[cintx_compat::raw::CHARGE_OF];
        let typed_charge = i32::from(basis.atoms()[0].atomic_number);

        // One diagonal block is enough: `int1e_nuc` is linear in the charge, so
        // a charge error shows as a uniform ratio.
        let shell = basis.shells()[0].clone();
        let tuple = ShellTuple::try_from_iter([shell.clone(), shell]).expect("ShellTuple");
        let safe = SessionRequest::new(
            INT1E_NUC_SPH,
            Representation::Spheric,
            &basis,
            tuple,
            ExecutionOptions::default(),
        )
        .query_workspace()
        .expect("query_workspace")
        .evaluate()
        .expect("evaluate")
        .tensor
        .owned_values;

        let mut vendor = vec![0.0_f64; safe.len()];
        vendor_ffi::vendor_int1e_nuc_sph(
            &mut vendor,
            &[0, 0],
            &raw.atm,
            (raw.atm.len() / ATM_SLOTS) as i32,
            &raw.bas,
            (raw.bas.len() / BAS_SLOTS) as i32,
            &raw.env,
        );

        let ratio = safe[0] / vendor[0];
        report.push(format!(
            "{symbol}: Z={true_z} raw atm[CHARGE_OF]={raw_charge} \
             typed Atom::atomic_number={typed_charge} safe/vendor={ratio:.6}"
        ));
        assert!(
            (ratio - 1.0).abs() < 1e-12,
            "{symbol}: the safe API's int1e_nuc is {ratio:.6}x the vendor's. \
             The typed BasisSet carries Atom::atomic_number={typed_charge} while \
             the raw arrays carry atm[CHARGE_OF]={raw_charge}; a ratio of \
             {:.6} would be the bare Z double-counting the ECP core.",
            f64::from(true_z) / f64::from(raw_charge)
        );
    }
    for line in report {
        println!("{line}");
    }
}
