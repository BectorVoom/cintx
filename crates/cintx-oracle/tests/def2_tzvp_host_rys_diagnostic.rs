//! Diagnostic for the def2-TZVP `(p,f|f,f)` mismatch found by Part 0.
//!
//! `def2_integral_parity` reports exactly two failing def2-TZVP launch classes,
//! `(1,3,3,3)` and `(3,3,1,3)`, both at Rys order 6 — i.e. both on the **host**
//! `fill_g_tensor_2e` fallback taken when `nroots > MAX_DEVICE_NROOTS`. Every
//! other Rys-6 class passes, including `(3,1,3,3)` and `(3,3,3,1)`, which hold
//! the same multiset of angular momenta in different positions.
//!
//! The pattern that separates them is the adaptive HRR branch: the failures are
//! the ones with `ibase == false && li >= 1 && li < lj`, or
//! `kbase == false && lk >= 1 && lk < ll`. This test prints enough structure to
//! tell a layout/permutation fault from an arithmetic one.
//!
//! Run with:
//! `CINTX_ORACLE_BUILD_VENDOR=1 cargo test --release -p cintx-oracle --features cpu
//!  --test def2_tzvp_host_rys_diagnostic -- --nocapture`

#![cfg(all(feature = "cpu", has_vendor_libcint))]

use cintx_basis::{AtomSpec, Molecule, StandardBasis, to_raw_arrays};
use cintx_compat::raw::{ANG_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, RawApiId, eval_raw};
use cintx_oracle::vendor_ffi;

fn water_tzvp() -> Molecule {
    Molecule::new(
        vec![
            AtomSpec::from_angstrom("O", [0.0, 0.0, 0.0]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, 0.757, 0.587]).unwrap(),
            AtomSpec::from_angstrom("H", [0.0, -0.757, 0.587]).unwrap(),
        ],
        StandardBasis::Def2Tzvp,
    )
}

#[test]
#[ignore = "diagnostic; run explicitly with --nocapture"]
fn tzvp_rys6_class_structure() {
    let molecule = water_tzvp();
    let arrays = to_raw_arrays(&molecule).expect("raw arrays");
    let nbas = arrays.nbas();
    let l_of = |s: usize| arrays.bas[s * BAS_SLOTS + ANG_OF] as usize;
    let nprim_of = |s: usize| arrays.bas[s * BAS_SLOTS + NPRIM_OF] as usize;
    let nctr_of = |s: usize| arrays.bas[s * BAS_SLOTS + NCTR_OF] as usize;

    println!("\nshells: {nbas}");
    for s in 0..nbas {
        println!(
            "  shell {s}: l={} nprim={} nctr={}",
            l_of(s),
            nprim_of(s),
            nctr_of(s)
        );
    }

    // Enumerate exactly as `def2_integral_parity` does (`j <= i`, `l <= k`),
    // and take EVERY representative, not the first: the O-centred
    // `(1,3,3,3)` quartet is single-centre and trivially correct, while the
    // H-centred one is the multi-centre case that actually fails.
    let targets: [[usize; 4]; 4] = [[1, 3, 3, 3], [3, 3, 1, 3], [3, 1, 3, 3], [3, 3, 3, 1]];
    let mut quartets: Vec<([usize; 4], [usize; 4])> = Vec::new();
    for i in 0..nbas {
        for j in 0..=i {
            for k in 0..nbas {
                for l in 0..=k {
                    let class = [l_of(i), l_of(j), l_of(k), l_of(l)];
                    if targets.contains(&class) {
                        quartets.push((class, [i, j, k, l]));
                    }
                }
            }
        }
    }

    for (target, q) in quartets {
        let dims: Vec<usize> = q
            .iter()
            .map(|&s| vendor_ffi::vendor_cgto_spheric(s as i32, &arrays.bas) as usize)
            .collect();
        let len: usize = dims.iter().product();
        let shls = [q[0] as i32, q[1] as i32, q[2] as i32, q[3] as i32];

        let mut expected = vec![0.0_f64; len];
        vendor_ffi::vendor_int2e_sph(
            &mut expected,
            &shls,
            &arrays.atm,
            arrays.natm() as i32,
            &arrays.bas,
            nbas as i32,
            &arrays.env,
        );
        let mut actual = vec![0.0_f64; len];
        // SAFETY: `actual` is sized from the vendor's own AO counts.
        unsafe {
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
        }
        .expect("eval_raw");

        let ibase = target[0] > target[1];
        let kbase = target[2] > target[3];
        let nroots = (target.iter().sum::<usize>()) / 2 + 1;
        let mut max_diff = 0.0_f64;
        let mut bad = 0_usize;
        for (e, a) in expected.iter().zip(&actual) {
            let d = (e - a).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 1e-12 {
                bad += 1;
            }
        }

        // Is it a permutation? Compare sorted magnitudes.
        let mut se: Vec<f64> = expected.iter().map(|v| v.abs()).collect();
        let mut sa: Vec<f64> = actual.iter().map(|v| v.abs()).collect();
        se.sort_by(|x, y| x.partial_cmp(y).unwrap());
        sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let permuted = se
            .iter()
            .zip(&sa)
            .all(|(x, y)| (x - y).abs() <= 1e-12 + 1e-10 * x.abs());

        // Is it a uniform scale?
        let ratios: Vec<f64> = expected
            .iter()
            .zip(&actual)
            .filter(|(e, _)| e.abs() > 1e-8)
            .map(|(e, a)| a / e)
            .collect();
        let ratio_span = match (
            ratios.iter().cloned().fold(f64::INFINITY, f64::min),
            ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ) {
            (lo, hi) if lo.is_finite() && hi.is_finite() => format!("[{lo:.6}, {hi:.6}]"),
            _ => "n/a".to_owned(),
        };

        let centers: Vec<i32> = q.iter().map(|&s| arrays.bas[s * BAS_SLOTS]).collect();
        println!(
            "\nclass {target:?} shls={q:?} atoms={centers:?} dims={dims:?} len={len}\n  \
             nroots={nroots} ibase={ibase} kbase={kbase}\n  \
             mismatched={bad}/{len}  max|diff|={max_diff:.6e}  \
             same-multiset={permuted}  actual/expected span={ratio_span}"
        );
        if bad > 0 {
            let mut shown = 0;
            for (index, (e, a)) in expected.iter().zip(&actual).enumerate() {
                if (e - a).abs() > 1e-12 && shown < 6 {
                    println!("    [{index}] vendor={e:+.12e} cintx={a:+.12e}");
                    shown += 1;
                }
            }
        }
    }
}
