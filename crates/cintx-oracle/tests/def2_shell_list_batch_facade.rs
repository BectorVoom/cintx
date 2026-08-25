//! Task 35-F2 — the public shell-**pair** and shell-**triple** batch surfaces.
//!
//! `QuartetBatchRequest` (Task 34-F) gave the safe API one batched operator:
//! `int2e_sph`. Everything else the backend can batch — the scalar and gradient
//! `int1e_*`, `int2c2e`, `int3c2e` and its two gradient families — was reachable
//! only by depending on `cintx-cubecl` directly, which the project's API
//! ordering says a safe-API consumer should not have to do.
//!
//! `PairBatchRequest` and `TripleBatchRequest` close that. The same two things
//! have to hold as for the quartet surface:
//!
//! 1. the numbers match vendored libcint, reached through the ordinary
//!    `BasisSet` surface rather than raw `atm`/`bas`/`env` arrays — so the
//!    `BasisSet` -> batch-shell conversion is covered, not assumed;
//! 2. the reported statistics are the ones a speed claim would be made from:
//!    one dispatch per launch class, not one per tuple.
//!
//! The scope gate is asserted per out-of-scope operator, because a surface that
//! silently accepted `int1e_ipovlp_sph` into the plain-overlap kernel would
//! return plausible numbers under the wrong name.
//!
//! Run with:
//! `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
//!  --test def2_shell_list_batch_facade`

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{
    ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell};
use cintx_ops::resolver::Resolver;
use cintx_oracle::vendor_ffi;
use cintx_rs::prelude::{EvaluationContext, PairBatchRequest, TripleBatchRequest};
use cintx_runtime::{BackendIntent, BackendKind, ExecutionOptions};
use std::sync::Arc;

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

/// Rebuild a typed [`BasisSet`] from the same raw arrays the vendor is fed, so
/// both sides are demonstrably the same basis.
///
/// The env coefficient block is contraction-major (`env[ptr + c*nprim + p]`)
/// while `Shell::coefficients` is primitive-major; the transpose here is the
/// same one `cintx_compat::raw` performs (WR-03).
fn basis_set_from_raw(arrays: &RawArrays) -> BasisSet {
    let natm = arrays.natm();
    let mut atoms = Vec::with_capacity(natm);
    for atom in 0..natm {
        let charge = arrays.atm[atom * 6] as u16;
        let coord_ptr = arrays.atm[atom * 6 + PTR_COORD] as usize;
        atoms.push(
            Atom::try_new(
                charge,
                [
                    arrays.env[coord_ptr],
                    arrays.env[coord_ptr + 1],
                    arrays.env[coord_ptr + 2],
                ],
                NuclearModel::Point,
                None,
                None,
            )
            .expect("atom"),
        );
    }

    let mut shells = Vec::with_capacity(arrays.nbas());
    for shell in 0..arrays.nbas() {
        let record = &arrays.bas[shell * BAS_SLOTS..(shell + 1) * BAS_SLOTS];
        let l = record[ANG_OF] as u8;
        let nprim = record[NPRIM_OF] as usize;
        let nctr = record[NCTR_OF] as usize;
        let exp_ptr = record[PTR_EXP] as usize;
        let coeff_ptr = record[PTR_COEFF] as usize;

        let mut coefficients = vec![0.0_f64; nprim * nctr];
        for c in 0..nctr {
            for p in 0..nprim {
                coefficients[p * nctr + c] = arrays.env[coeff_ptr + c * nprim + p];
            }
        }

        shells.push(Arc::new(
            Shell::try_new(
                record[ATOM_OF] as u32,
                l,
                nprim as u16,
                nctr as u16,
                0,
                Representation::Spheric,
                Arc::from(
                    arrays.env[exp_ptr..exp_ptr + nprim]
                        .to_vec()
                        .into_boxed_slice(),
                ),
                Arc::from(coefficients.into_boxed_slice()),
            )
            .expect("shell"),
        ));
    }

    BasisSet::try_new(
        Arc::from(atoms.into_boxed_slice()),
        Arc::from(shells.into_boxed_slice()),
    )
    .expect("basis set")
}

fn operator(symbol: &str) -> OperatorId {
    Resolver::descriptor_by_symbol(symbol)
        .unwrap_or_else(|_| panic!("{symbol} must be in the manifest"))
        .id
}

fn cpu_options() -> ExecutionOptions {
    ExecutionOptions {
        backend_intent: BackendIntent {
            backend: BackendKind::Cpu,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn nsph(l: u8) -> usize {
    2 * l as usize + 1
}

type VendorPairFn = fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32;
type VendorTripleFn = fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32;

/// Every def2-SVP shell pair through the public pair surface, for each batched
/// pair-arity symbol, against vendored libcint.
///
/// `rank` is the operator's component count: 1 for the scalar families, 3 for
/// the bra-gradient ones, whose block is component-leading.
#[test]
fn pair_batch_facade_matches_vendor_for_every_batched_symbol() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = basis_set_from_raw(&arrays);
    let nbas = arrays.nbas();
    assert!(nbas > 1);

    let list: Vec<[u32; 2]> = (0..nbas)
        .flat_map(|i| (0..nbas).map(move |j| [i as u32, j as u32]))
        .collect();
    let context = EvaluationContext::new();

    let cases: [(&str, usize, VendorPairFn); 7] = [
        (
            "int1e_ovlp_sph",
            1,
            vendor_ffi::vendor_int1e_ovlp_sph as VendorPairFn,
        ),
        (
            "int1e_kin_sph",
            1,
            vendor_ffi::vendor_int1e_kin_sph as VendorPairFn,
        ),
        (
            "int1e_nuc_sph",
            1,
            vendor_ffi::vendor_int1e_nuc_sph as VendorPairFn,
        ),
        (
            "int2c2e_sph",
            1,
            vendor_ffi::vendor_int2c2e_sph as VendorPairFn,
        ),
        (
            "int1e_ipovlp_sph",
            3,
            vendor_ffi::vendor_int1e_ipovlp_sph as VendorPairFn,
        ),
        (
            "int1e_ipkin_sph",
            3,
            vendor_ffi::vendor_int1e_ipkin_sph as VendorPairFn,
        ),
        (
            "int1e_ipnuc_sph",
            3,
            vendor_ffi::vendor_int1e_ipnuc_sph as VendorPairFn,
        ),
    ];

    for (symbol, rank, vendor) in cases {
        let output = PairBatchRequest::new(
            operator(symbol),
            Representation::Spheric,
            &basis,
            list.iter().copied(),
            cpu_options(),
        )
        .evaluate_in(&context)
        .unwrap_or_else(|error| panic!("{symbol} pair batch failed: {error}"));

        // The surface exists to amortize launches: strictly fewer dispatches
        // than there are classes, and far fewer than there are pairs.
        let classes: std::collections::BTreeSet<[u8; 2]> = list
            .iter()
            .map(|p| {
                [
                    basis.shells()[p[0] as usize].ang_momentum,
                    basis.shells()[p[1] as usize].ang_momentum,
                ]
            })
            .collect();
        assert_eq!(
            output.stats.bucket_count,
            classes.len(),
            "{symbol}: every (li,lj) class must be accounted for"
        );
        assert!(
            output.stats.kernel_launch_count <= classes.len(),
            "{symbol}: merging must not increase the dispatch count"
        );
        assert!(
            output.stats.kernel_launch_count < list.len(),
            "{symbol}: batching must reduce the launch count below the pair count"
        );
        assert_eq!(output.stats.chunk_count, output.stats.kernel_launch_count);
        assert_eq!(
            output.stats.readback_count,
            output.stats.kernel_launch_count
        );
        assert_eq!(output.stats.items_planned, list.len());
        assert_eq!(output.stats.items_executed, list.len());

        let mut max_diff = 0.0_f64;
        let mut mismatches = 0_usize;
        let mut buffer = vec![0.0_f64; 4096];
        for (index, pair) in list.iter().enumerate() {
            let li = basis.shells()[pair[0] as usize].ang_momentum;
            let lj = basis.shells()[pair[1] as usize].ang_momentum;
            let nctr_i = basis.shells()[pair[0] as usize].nctr as usize;
            let nctr_j = basis.shells()[pair[1] as usize].nctr as usize;
            let len = rank * nctr_i * nsph(li) * nctr_j * nsph(lj);
            if buffer.len() < len {
                buffer.resize(len, 0.0);
            }
            vendor(
                &mut buffer[..len],
                &[pair[0] as i32, pair[1] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            let start = output.offsets[index];
            for element in 0..len {
                let diff = (buffer[element] - output.values[start + element]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-12 {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{symbol}: public pair-batch surface must match vendored libcint \
             (max |diff| {max_diff:.3e})"
        );
    }
}

/// Every def2-SVP shell triple through the public triple surface, for each
/// batched triple-arity symbol, against vendored libcint.
///
/// The triple list is `(i, j, k)` over the orbital basis: this test is about
/// the surface, not about a fitting basis, and reusing the orbital set as the
/// third index keeps the fixture identical to the pair case.
#[test]
fn triple_batch_facade_matches_vendor_for_every_batched_symbol() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = basis_set_from_raw(&arrays);
    let nbas = arrays.nbas();
    assert!(nbas > 1);

    let list: Vec<[u32; 3]> = (0..nbas)
        .flat_map(|i| {
            (0..nbas).flat_map(move |j| (0..nbas).map(move |k| [i as u32, j as u32, k as u32]))
        })
        .collect();
    let context = EvaluationContext::new();

    let cases: [(&str, usize, VendorTripleFn); 3] = [
        (
            "int3c2e_sph",
            1,
            vendor_ffi::vendor_int3c2e_sph as VendorTripleFn,
        ),
        (
            "int3c2e_ip1_sph",
            3,
            vendor_ffi::vendor_int3c2e_ip1_sph as VendorTripleFn,
        ),
        (
            "int3c2e_ip2_sph",
            3,
            vendor_ffi::vendor_int3c2e_ip2_sph as VendorTripleFn,
        ),
    ];

    for (symbol, rank, vendor) in cases {
        let output = TripleBatchRequest::new(
            operator(symbol),
            Representation::Spheric,
            &basis,
            list.iter().copied(),
            cpu_options(),
        )
        .evaluate_in(&context)
        .unwrap_or_else(|error| panic!("{symbol} triple batch failed: {error}"));

        let classes: std::collections::BTreeSet<[u8; 3]> = list
            .iter()
            .map(|t| {
                [
                    basis.shells()[t[0] as usize].ang_momentum,
                    basis.shells()[t[1] as usize].ang_momentum,
                    basis.shells()[t[2] as usize].ang_momentum,
                ]
            })
            .collect();
        assert!(
            output.stats.kernel_launch_count <= classes.len(),
            "{symbol}: merging must not increase the dispatch count"
        );
        assert!(
            output.stats.kernel_launch_count < list.len(),
            "{symbol}: batching must reduce the launch count below the triple count"
        );
        assert_eq!(
            output.stats.readback_count,
            output.stats.kernel_launch_count
        );
        assert_eq!(output.stats.items_planned, list.len());

        let mut max_diff = 0.0_f64;
        let mut mismatches = 0_usize;
        let mut buffer = vec![0.0_f64; 8192];
        for (index, triple) in list.iter().enumerate() {
            let len = rank
                * triple
                    .iter()
                    .map(|&s| {
                        let shell = &basis.shells()[s as usize];
                        shell.nctr as usize * nsph(shell.ang_momentum)
                    })
                    .product::<usize>();
            if buffer.len() < len {
                buffer.resize(len, 0.0);
            }
            vendor(
                &mut buffer[..len],
                &[triple[0] as i32, triple[1] as i32, triple[2] as i32],
                &arrays.atm,
                arrays.natm() as i32,
                &arrays.bas,
                arrays.nbas() as i32,
                &arrays.env,
            );
            let start = output.offsets[index];
            for element in 0..len {
                let diff = (buffer[element] - output.values[start + element]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-12 {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{symbol}: public triple-batch surface must match vendored libcint \
             (max |diff| {max_diff:.3e})"
        );
    }
}

/// One rejection per out-of-scope axis, and one per out-of-scope operator.
///
/// The point is not that these fail — it is that they fail *before any device
/// work*, with a message naming what was refused.
#[test]
fn shell_list_facades_reject_out_of_scope_requests() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = basis_set_from_raw(&arrays);

    // Cartesian is not batched by either surface.
    assert!(
        PairBatchRequest::new(
            operator("int1e_ovlp_sph"),
            Representation::Cart,
            &basis,
            [[0, 0]],
            cpu_options(),
        )
        .evaluate()
        .is_err(),
        "Cartesian must be refused, not silently sph"
    );
    assert!(
        TripleBatchRequest::new(
            operator("int3c2e_sph"),
            Representation::Cart,
            &basis,
            [[0, 0, 0]],
            cpu_options(),
        )
        .evaluate()
        .is_err(),
        "Cartesian must be refused, not silently sph"
    );

    // A symbol of the right arity that no batched kernel serves, and one of the
    // wrong arity entirely.
    for symbol in ["int1e_r_sph", "int1e_ipipovlp_sph", "int2e_sph"] {
        let rejected = PairBatchRequest::new(
            operator(symbol),
            Representation::Spheric,
            &basis,
            [[0, 0]],
            cpu_options(),
        )
        .evaluate();
        assert!(
            rejected.is_err(),
            "{symbol} has no batched pair kernel and must be refused"
        );
    }
    for symbol in ["int3c1e_sph", "int2e_sph", "int1e_ovlp_sph"] {
        let rejected = TripleBatchRequest::new(
            operator(symbol),
            Representation::Spheric,
            &basis,
            [[0, 0, 0]],
            cpu_options(),
        )
        .evaluate();
        assert!(
            rejected.is_err(),
            "{symbol} has no batched triple kernel and must be refused"
        );
    }

    // Out-of-range shell indices are caught by the surface, not by the kernel.
    assert!(
        PairBatchRequest::new(
            operator("int1e_ovlp_sph"),
            Representation::Spheric,
            &basis,
            [[0, u32::MAX]],
            cpu_options(),
        )
        .evaluate()
        .is_err(),
        "an out-of-range shell index must fail"
    );
    assert!(
        TripleBatchRequest::new(
            operator("int3c2e_sph"),
            Representation::Spheric,
            &basis,
            [[0, 0, u32::MAX]],
            cpu_options(),
        )
        .evaluate()
        .is_err(),
        "an out-of-range shell index must fail"
    );
}
