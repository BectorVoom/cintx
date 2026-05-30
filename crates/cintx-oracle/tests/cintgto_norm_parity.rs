//! Oracle parity test for the GTO normalization helper `CINTgto_norm` (HELP-02).
//!
//! libcint 6.1.3 ground truth (`libcint-master/src/misc.c`):
//!   static double _gaussian_int(FINT n, double alpha) {
//!       double n1 = (n + 1) * .5;
//!       return exp(lgamma(n1)) / (2. * pow(alpha, n1));
//!   }
//!   double CINTgto_norm(FINT n, double a) {
//!       return 1. / sqrt(_gaussian_int(n*2+2, 2*a));
//!   }
//! Documented closed form (misc.c comment, Schlegel & Frisch IJQC 54(1995) 83-87):
//!   norm = sqrt( 2^(2n+3) * (n+1)! * (2a)^(n+1.5) / ((2n+2)! * sqrt(pi)) )
//!
//! Vendor parity is double-gated: it only runs under `--features cpu` AND env
//! `CINTX_ORACLE_BUILD_VENDOR=1` (which makes build.rs set `has_vendor_libcint`).
//! Without both, the vendor body is cfg'd out and the file compiles to the
//! non-vendor smoke test only.

#![cfg(any(feature = "cpu", feature = "rocm"))]

#[allow(dead_code)]
const ATOL: f64 = 1e-12;

/// (n, a) grid: the required {0.5,1.0,2.5} set plus 0.75 and 5.0 extras.
#[allow(dead_code)]
const A_GRID: [f64; 5] = [0.5, 1.0, 2.5, 0.75, 5.0];

/// Non-vendor smoke test so the file is not a pure no-op without the vendor build.
#[cfg(feature = "cpu")]
#[test]
fn cintgto_norm_smoke() {
    let v = cintx_compat::helpers::CINTgto_norm(0, 0.5);
    assert!(v.is_finite(), "CINTgto_norm(0,0.5) must be finite, got {v}");
    assert!(
        (v - 1.502251088929885).abs() < ATOL,
        "CINTgto_norm(0,0.5) = {v} expected 1.502251088929885"
    );
}

/// Vendor parity: cintx CINTgto_norm vs libcint 6.1.3 over the full (n,a) grid.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn cintgto_norm_matches_vendor() {
    use cintx_oracle::vendor_ffi;

    let mut mismatches = 0usize;
    let mut report = String::new();
    for n in 0..5 {
        for &a in A_GRID.iter() {
            let c = cintx_compat::helpers::CINTgto_norm(n, a);
            let v = vendor_ffi::vendor_CINTgto_norm(n, a);
            let diff = (c - v).abs();
            if diff > ATOL {
                mismatches += 1;
                report.push_str(&format!(
                    "  (n={n}, a={a}): cintx={c} vendor={v} diff={diff}\n"
                ));
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "CINTgto_norm vendor parity mismatches ({mismatches}):\n{report}"
    );
}
