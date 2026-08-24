//! Which def2-SVP 2e launch classes does cintx get wrong?
//!
//! The throughput benchmark reported 137 mismatched elements (max |diff| 11.68)
//! over a 236-quartet def2-SVP sample. A large, structural difference like that
//! is a coverage question, not a tolerance question, so this test walks **one
//! representative per angular-momentum class** and reports a per-class verdict
//! instead of failing on the first bad element.
//!
//! `PERMUTED` in the verdict column means the two engines produced the same
//! multiset of values in a different order — that would point at output
//! layout, not at the integral math.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_driver::{BasisView, bucket_quartets, enumerate_pairs, enumerate_quartets};
use cintx_oracle::vendor_ffi;

#[test]
fn def2_svp_2e_per_class_verdict() {
    let molecule = Molecule::new(
        vec![
            AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
        ],
        StandardBasis::Def2Svp,
    );
    let arrays = to_raw_arrays(&molecule).unwrap();
    let basis = BasisView::new(&arrays.atm, &arrays.bas, &arrays.env);
    let quartets = enumerate_quartets(&enumerate_pairs(&basis));
    let buckets = bucket_quartets(&basis, &quartets);

    println!("\n{:-<96}", "");
    println!(
        "{:<18} {:>7} {:>7} {:>13} {:>12} {:>10}",
        "l-class", "nroots", "block", "max|diff|", "mismatched", "verdict"
    );
    println!("{:-<96}", "");

    let mut bad_classes = Vec::new();
    let mut good_classes = 0_usize;

    for bucket in &buckets {
        let Some(&quartet) = bucket.quartets.first() else {
            continue;
        };
        let shls = quartet.shls();
        let len = cintx_driver::execute::block_len(&basis, quartet);

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2e_sph(
            &mut expected,
            &shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            arrays.nbas() as i32,
            &arrays.env,
        );

        let mut actual = vec![0.0_f64; len];
        let status = unsafe {
            eval_raw(
                RawApiId::INT2E_SPH,
                Some(&mut actual),
                None,
                &shls,
                &arrays.atm,
                &arrays.bas,
                &arrays.env,
                None,
                None,
            )
        };

        let (max_diff, mismatched) = if status.is_ok() {
            let mut max_diff = 0.0_f64;
            let mut mismatched = 0_usize;
            for (e, a) in expected.iter().zip(&actual) {
                let diff = (e - a).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-9 {
                    mismatched += 1;
                }
            }
            (max_diff, mismatched)
        } else {
            (f64::NAN, len)
        };

        let mut sorted_expected = expected.clone();
        let mut sorted_actual = actual.clone();
        sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_actual.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let same_multiset = sorted_expected
            .iter()
            .zip(&sorted_actual)
            .all(|(a, b)| (a - b).abs() < 1e-9);

        let verdict = if status.is_err() {
            "ERROR"
        } else if mismatched == 0 {
            good_classes += 1;
            "OK"
        } else if same_multiset {
            "PERMUTED"
        } else {
            "WRONG"
        };

        if mismatched > 0 || status.is_err() {
            bad_classes.push((bucket.class.angular_momenta, verdict));
        }

        println!(
            "{:<18} {:>7} {:>7} {:>13.3e} {:>12} {:>10}",
            format!("{:?}", bucket.class.angular_momenta),
            bucket.class.nroots,
            len,
            max_diff,
            mismatched,
            verdict
        );
    }

    println!("{:-<96}", "");
    println!(
        "classes OK: {} / {}   classes with mismatches: {}",
        good_classes,
        buckets.len(),
        bad_classes.len()
    );
    if !bad_classes.is_empty() {
        println!("\nFailing classes:");
        for (l, verdict) in &bad_classes {
            println!("  {l:?} -> {verdict}");
        }
    }

    assert!(
        bad_classes.is_empty(),
        "{} of {} def2-SVP 2e launch classes disagree with vendored libcint",
        bad_classes.len(),
        buckets.len()
    );
}
