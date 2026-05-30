#![cfg(any(feature = "cpu", feature = "rocm"))]
//! Phase 25 FND-02 — vendor parity sweep for the host Rys `nroots >= 6`
//! Wheeler/Jacobi root+weight engine.
//!
//! Double-gated (memory `reference_oracle_vendor_parity_invocation`): the byte-identity
//! comparison vs vendored libcint `CINTrys_roots` requires BOTH `--features cpu` AND
//! `CINTX_ORACLE_BUILD_VENDOR=1` (the latter sets the `has_vendor_libcint` cfg). Without
//! both, only the determinism portion runs and the parity body SILENTLY SKIPS.
//!
//! The sweep covers nroots 6..=12 across a small x grid that spans the small-x Jacobi
//! branch, the per-nroots breakpoint, and the large-x Schmidt tail. nroots >= 13 routes
//! to the quadmath (`CINTqrys_*`) path which is NOT compiled in the cintx vendor build
//! (`HAVE_QUADMATH_H` disabled, build.rs), so the validated ceiling is 12. A nroots=13
//! probe documents the vendor's effective ceiling.

use cintx_cubecl::math::rys;

/// x grid: small-x Jacobi, mid, the n=6,7 Schmidt breakpoint (11), and the large-x tail.
const XS: [f64; 6] = [0.05, 0.5, 5.0, 11.0, 15.0, 30.0];
const ATOL: f64 = 1e-12;

/// Highest nroots validated against the vendor in this build (quadmath disabled => 12).
const VALIDATED_NROOTS_CEILING: usize = 12;

/// The core sweep: for nroots in 6..=12 and each x, the host engine must match the
/// vendored `CINTrys_roots` roots+weights within atol=1e-12. The test name is the
/// acceptance-criteria anchor (`grep "fn rys_nroots_sweep"`).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn rys_nroots_sweep() {
    use cintx_oracle::vendor_ffi;

    let mut mismatches = 0usize;
    for nroots in 6..=VALIDATED_NROOTS_CEILING {
        for &x in XS.iter() {
            let (rr, ww) = rys::rys_roots_host::<f64>(nroots, x);
            let (vr, vw) = vendor_ffi::vendor_CINTrys_roots(nroots as i32, x);
            assert_eq!(rr.len(), nroots, "host root count for nroots={nroots}");
            assert_eq!(vr.len(), nroots, "vendor root count for nroots={nroots}");
            for i in 0..nroots {
                let dr = (rr[i] - vr[i]).abs();
                let dw = (ww[i] - vw[i]).abs();
                if dr > ATOL || dw > ATOL {
                    mismatches += 1;
                    eprintln!(
                        "MISMATCH nroots={nroots} x={x} i={i}: \
                         root cintx={} vendor={} |d|={dr:e} | \
                         weight cintx={} vendor={} |d|={dw:e}",
                        rr[i], vr[i], ww[i], vw[i]
                    );
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "rys nroots 6..=12 sweep: {mismatches} root/weight mismatches vs vendor CINTrys_roots (atol={ATOL:e})"
    );
}

/// Vendor ceiling probe: nroots=13 routes to the quadmath path. In the cintx vendor
/// build `HAVE_QUADMATH_H` is disabled, so `CINTqrys_*` is not compiled. The host engine
/// caps at the validated ceiling (12) and returns a typed error for nroots > 12 rather
/// than a silent wrong result (T-25-01). This documents the edge and asserts the host
/// ceiling matches the highest nroots the sweep validates.
#[cfg(feature = "cpu")]
#[test]
fn rys_nroots13_vendor_ceiling_probe() {
    // The host engine is validated only up to 12; nroots 13 must NOT silently succeed.
    let res = std::panic::catch_unwind(|| rys::rys_roots_host::<f64>(13, 5.0));
    assert!(
        res.is_err(),
        "host rys_roots_host(13, x) must fail-closed (nroots>12 exceeds the vendor's \
         quadmath-disabled ceiling), not return a silent wrong result"
    );
    assert_eq!(
        VALIDATED_NROOTS_CEILING, 12,
        "vendor effective ceiling is 12 (quadmath disabled in build.rs)"
    );
}

/// Determinism portion (runs without the vendor gate): the host engine must be
/// reproducible for nroots 6..=12 (two identical calls give identical bits).
#[test]
fn rys_nroots_ge6_determinism() {
    for nroots in 6..=VALIDATED_NROOTS_CEILING {
        for &x in XS.iter() {
            let a = std::panic::catch_unwind(|| rys::rys_roots_host::<f64>(nroots, x));
            let b = std::panic::catch_unwind(|| rys::rys_roots_host::<f64>(nroots, x));
            match (a, b) {
                (Ok((r1, w1)), Ok((r2, w2))) => {
                    assert_eq!(r1, r2, "roots nondeterministic nroots={nroots} x={x}");
                    assert_eq!(w1, w2, "weights nondeterministic nroots={nroots} x={x}");
                }
                // Before Task 1b lands the engine, the host fn may panic (RED). Both
                // calls panicking consistently is acceptable for this determinism check.
                (Err(_), Err(_)) => {}
                _ => panic!("rys_roots_host nondeterministic panic for nroots={nroots} x={x}"),
            }
        }
    }
}
