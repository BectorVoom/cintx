//! Phase 32 completion gate — end-to-end def2 integral parity vs vendored libcint.
//!
//! Walks every shell pair / quartet class the def2-SVP and def2-TZVP water and
//! methane fixtures produce, evaluates through `cintx_compat::raw::eval_raw`,
//! and compares against vendored libcint. Failures are reported as a per-class
//! coverage map rather than a bare panic, because the point of this gate is to
//! establish *which* classes of the def2 envelope cintx already serves and
//! which are still blocked.

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, StandardBasis, raw as basis_raw, to_raw_arrays};
use cintx_compat::raw::{RawApiId, eval_raw};
use cintx_oracle::vendor_ffi;
use std::collections::BTreeMap;

const TOLERANCE: f64 = 1e-10;

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

fn shell_l(arrays: &cintx_basis::RawArrays, shell: usize) -> i32 {
    arrays.bas[shell * basis_raw::BAS_SLOTS + basis_raw::ANG_OF]
}

#[derive(Default, Clone, Copy)]
struct Tally {
    matched: usize,
    mismatched: usize,
    errored: usize,
    max_abs_diff: f64,
}

fn report(title: &str, tally: &BTreeMap<String, Tally>) -> (usize, usize, usize) {
    let (mut ok, mut bad, mut err) = (0, 0, 0);
    println!("\n=== {title} ===");
    for (class, t) in tally {
        ok += t.matched;
        bad += t.mismatched;
        err += t.errored;
        let status = if t.mismatched == 0 && t.errored == 0 {
            "OK"
        } else {
            "FAIL"
        };
        println!(
            "  {status:<4} l={class:<12} matched={:<6} mismatched={:<6} errored={:<6} max|d|={:.3e}",
            t.matched, t.mismatched, t.errored, t.max_abs_diff
        );
    }
    println!("  TOTAL matched={ok} mismatched={bad} errored={err}");
    (ok, bad, err)
}

/// One-electron overlap over every shell pair of both def2 basis sets.
#[test]
fn def2_int1e_ovlp_sph_matches_vendor() {
    for basis in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
        let molecule = water(basis);
        let arrays = to_raw_arrays(&molecule).unwrap();
        let natm = arrays.natm() as i32;
        let nbas = arrays.nbas() as i32;
        let mut tally: BTreeMap<String, Tally> = BTreeMap::new();

        for i in 0..arrays.nbas() {
            for j in 0..arrays.nbas() {
                let ni = vendor_ffi::vendor_cgto_spheric(i as i32, &arrays.bas) as usize;
                let nj = vendor_ffi::vendor_cgto_spheric(j as i32, &arrays.bas) as usize;
                let len = ni * nj;
                let class = format!("({},{})", shell_l(&arrays, i), shell_l(&arrays, j));
                let entry = tally.entry(class).or_default();

                let mut expected = vec![0.0_f64; len];
                vendor_ffi::vendor_int1e_ovlp_sph(
                    &mut expected,
                    &[i as i32, j as i32],
                    &arrays.atm,
                    natm,
                    &arrays.bas,
                    nbas,
                    &arrays.env,
                );

                let mut actual = vec![0.0_f64; len];
                match unsafe {
                    eval_raw(
                        RawApiId::INT1E_OVLP_SPH,
                        Some(&mut actual),
                        None,
                        &[i as i32, j as i32],
                        &arrays.atm,
                        &arrays.bas,
                        &arrays.env,
                        None,
                        None,
                    )
                } {
                    Ok(_) => {
                        let diff = expected
                            .iter()
                            .zip(&actual)
                            .map(|(e, a)| (e - a).abs())
                            .fold(0.0_f64, f64::max);
                        entry.max_abs_diff = entry.max_abs_diff.max(diff);
                        if diff <= TOLERANCE {
                            entry.matched += 1;
                        } else {
                            entry.mismatched += 1;
                        }
                    }
                    Err(_) => entry.errored += 1,
                }
            }
        }

        let (_, bad, err) = report(&format!("{} int1e_ovlp_sph", basis.name()), &tally);
        assert_eq!(bad, 0, "{}: value mismatches vs vendor", basis.name());
        assert_eq!(err, 0, "{}: evaluation errors", basis.name());
    }
}

/// Two-electron ERIs over one representative quartet per distinct l-class.
/// The full quartet list is enumerated but deduplicated by l-class so the test
/// stays fast while still touching every angular combination def2 produces —
/// including the (f f | f f) class that needs Rys order 7.
#[test]
fn def2_int2e_sph_matches_vendor_per_l_class() {
    for basis in [StandardBasis::Def2Svp, StandardBasis::Def2Tzvp] {
        let molecule = water(basis);
        let arrays = to_raw_arrays(&molecule).unwrap();
        let natm = arrays.natm() as i32;
        let nbas = arrays.nbas() as i32;
        let mut tally: BTreeMap<String, Tally> = BTreeMap::new();
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();

        for i in 0..arrays.nbas() {
            for j in 0..=i {
                for k in 0..arrays.nbas() {
                    for l in 0..=k {
                        let class = format!(
                            "({},{},{},{})",
                            shell_l(&arrays, i),
                            shell_l(&arrays, j),
                            shell_l(&arrays, k),
                            shell_l(&arrays, l)
                        );
                        // Two representatives per class: different centre
                        // combinations exercise different HRR branches.
                        let count = seen.entry(class.clone()).or_default();
                        if *count >= 2 {
                            continue;
                        }
                        *count += 1;

                        let dims: Vec<usize> = [i, j, k, l]
                            .iter()
                            .map(|&s| {
                                vendor_ffi::vendor_cgto_spheric(s as i32, &arrays.bas) as usize
                            })
                            .collect();
                        let len: usize = dims.iter().product();
                        let shls = [i as i32, j as i32, k as i32, l as i32];

                        let mut expected = vec![0.0_f64; len];
                        vendor_ffi::vendor_int2e_sph(
                            &mut expected,
                            &shls,
                            &arrays.atm,
                            natm,
                            &arrays.bas,
                            nbas,
                            &arrays.env,
                        );

                        let entry = tally.entry(class).or_default();
                        let mut actual = vec![0.0_f64; len];
                        match unsafe {
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
                        } {
                            Ok(_) => {
                                let diff = expected
                                    .iter()
                                    .zip(&actual)
                                    .map(|(e, a)| (e - a).abs())
                                    .fold(0.0_f64, f64::max);
                                entry.max_abs_diff = entry.max_abs_diff.max(diff);
                                if diff <= TOLERANCE {
                                    entry.matched += 1;
                                } else {
                                    entry.mismatched += 1;
                                }
                            }
                            Err(_) => entry.errored += 1,
                        }
                    }
                }
            }
        }

        let (_, bad, err) = report(&format!("{} int2e_sph", basis.name()), &tally);
        assert_eq!(bad, 0, "{}: ERI value mismatches vs vendor", basis.name());
        assert_eq!(err, 0, "{}: ERI evaluation errors", basis.name());
    }
}
