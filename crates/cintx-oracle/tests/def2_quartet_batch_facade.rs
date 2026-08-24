//! Task 34-F — the public shell-quartet batch surface.
//!
//! `QuartetBatchRequest` is the first safe-API entry point that submits a whole
//! work list instead of one shell tuple. Two things have to hold for it to be
//! worth adding:
//!
//! 1. It produces the same numbers as vendored libcint, reached through the
//!    ordinary `BasisSet` surface rather than raw `atm`/`bas`/`env` arrays — so
//!    the `BasisSet` -> batch-shell conversion is covered, not assumed.
//! 2. Its reported statistics are the ones a speed claim would be made from:
//!    one dispatch per launch class, not one per quartet.
//!
//! The scope gates (`int2e` / `Spheric` only) are asserted too, because a batch
//! path that silently accepted another operator would return plain Coulomb
//! integrals under the wrong name.
//!
//! Run with:
//! `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
//!  --test def2_quartet_batch_facade`

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, RawArrays, StandardBasis, to_raw_arrays};
use cintx_core::{Atom, BasisSet, NuclearModel, OperatorId, Representation, Shell};
use cintx_driver::{BasisView, enumerate_pairs, enumerate_quartets};
use cintx_ops::resolver::Resolver;
use cintx_oracle::vendor_ffi;
use cintx_rs::prelude::{EvaluationContext, QuartetBatchRequest};
use cintx_runtime::{BackendIntent, BackendKind, ExecutionOptions};
use std::sync::Arc;

use cintx_compat::raw::{
    ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};

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

fn int2e_sph_operator() -> OperatorId {
    Resolver::descriptor_by_symbol("int2e_sph")
        .expect("int2e_sph must be in the manifest")
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

#[test]
fn quartet_batch_facade_matches_vendor_and_reports_class_launches() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let view = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&view));
    assert!(!quartets.is_empty());

    let basis = basis_set_from_raw(&arrays);
    let list: Vec<[u32; 4]> = quartets
        .iter()
        .map(|q| [q.i as u32, q.j as u32, q.k as u32, q.l as u32])
        .collect();

    let context = EvaluationContext::new();
    let output = QuartetBatchRequest::new(
        int2e_sph_operator(),
        Representation::Spheric,
        &basis,
        list.iter().copied(),
        cpu_options(),
    )
    .evaluate_in(&context)
    .expect("quartet batch");

    // Far fewer dispatches than quartets — the reason the surface exists.
    //
    // `bucket_count` is the angular-momentum class count and
    // `kernel_launch_count` the dispatch count. They were equal until Task
    // 35-M1 merged classes sharing the kernel's comptime signature, so the
    // facade must now surface both and the launch count must be *strictly*
    // below the bucket count.
    let classes: std::collections::BTreeSet<[u8; 4]> = list
        .iter()
        .map(|q| {
            [
                basis.shells()[q[0] as usize].ang_momentum,
                basis.shells()[q[1] as usize].ang_momentum,
                basis.shells()[q[2] as usize].ang_momentum,
                basis.shells()[q[3] as usize].ang_momentum,
            ]
        })
        .collect();
    assert_eq!(output.stats.bucket_count, classes.len());
    assert!(
        output.stats.kernel_launch_count < classes.len(),
        "launch merging must dispatch fewer times than there are classes: {} vs {}",
        output.stats.kernel_launch_count,
        classes.len()
    );
    assert_eq!(output.stats.chunk_count, output.stats.kernel_launch_count);
    assert_eq!(
        output.stats.readback_count,
        output.stats.kernel_launch_count
    );
    assert_eq!(output.stats.items_planned, list.len());
    assert_eq!(output.stats.items_executed, list.len());
    assert!(output.stats.kernel_launch_count < list.len());

    // Values must match vendored libcint over the identical list.
    let mut max_diff = 0.0_f64;
    let mut mismatches = 0_usize;
    let mut buffer = vec![0.0_f64; 4096];
    for (index, quartet) in quartets.iter().enumerate() {
        let len = cintx_driver::execute::block_len(&view, *quartet);
        if buffer.len() < len {
            buffer.resize(len, 0.0);
        }
        vendor_ffi::vendor_int2e_sph(
            &mut buffer[..len],
            &quartet.shls(),
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
        "public quartet-batch surface must match vendored libcint (max |diff| {max_diff:.3e})"
    );
}

/// The batch surface is `int2e`/`Spheric` only, and says so before it touches a
/// device.
#[test]
fn quartet_batch_facade_rejects_out_of_scope_requests() {
    let molecule = water(StandardBasis::Def2Svp);
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let basis = basis_set_from_raw(&arrays);

    let cart = QuartetBatchRequest::new(
        int2e_sph_operator(),
        Representation::Cart,
        &basis,
        [[0, 0, 0, 0]],
        cpu_options(),
    )
    .evaluate();
    assert!(cart.is_err(), "Cartesian must be refused, not silently sph");

    let overlap = Resolver::descriptor_by_symbol("int1e_ovlp_sph")
        .expect("int1e_ovlp_sph must be in the manifest")
        .id;
    let wrong_operator = QuartetBatchRequest::new(
        overlap,
        Representation::Spheric,
        &basis,
        [[0, 0, 0, 0]],
        cpu_options(),
    )
    .evaluate();
    assert!(
        wrong_operator.is_err(),
        "a non-int2e operator must be refused rather than returning Coulomb integrals"
    );

    let bad_index = QuartetBatchRequest::new(
        int2e_sph_operator(),
        Representation::Spheric,
        &basis,
        [[0, 0, 0, u32::MAX]],
        cpu_options(),
    )
    .evaluate();
    assert!(bad_index.is_err(), "an out-of-range shell index must fail");
}
