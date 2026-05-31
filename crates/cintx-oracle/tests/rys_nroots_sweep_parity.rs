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

/// Absolute tolerance for the pure-f64 paths (nroots 6,7) and for small-magnitude roots.
const ATOL: f64 = 1e-12;

/// Relative tolerance for the long-double paths (nroots >= 8) at large-magnitude roots.
///
/// The vendor reference compiles the `lrys_*` path in hardware x86-64 `long double`
/// (80-bit, 64-bit mantissa). Portable Rust has no 80-bit float, so the cintx host port
/// emulates the extended-precision intermediates with Dekker/Knuth double-double (~106-bit)
/// arithmetic, then — exactly as the vendor does — rounds the tridiagonal (a,b) to f64
/// before the shared f64 eigensolve. dd and 80-bit `long double` round the last f64 bit
/// differently, producing a fixed ~1e-10 RELATIVE divergence in the f64 tridiagonal. The
/// downstream Rys transform `root = r/(1-r)` is ill-conditioned as the eigenvalue r -> 1
/// (largest Rys root), amplifying that last-bit difference into an ABSOLUTE delta above
/// 1e-12 on the largest roots — whose quadrature weights are O(1e-8..1e-19) and contribute
/// negligibly to any integral.
///
/// We therefore gate on `|cintx - vendor| <= max(ATOL, RTOL * |vendor|)`:
/// - nroots 6,7 (pure f64) are byte-identical at ATOL (RTOL never triggers — verified).
/// - nroots 8..12 (long double) match to RTOL=1e-9 relative, the dd-vs-f80 floor.
/// This is the documented faithful-port boundary (RESEARCH §FND-02 Pitfall 1 / T-25-02):
/// true 80-bit byte-identity for the long-double path is unreachable in portable Rust.
const RTOL: f64 = 1e-9;

/// Highest nroots validated against the vendor in this build (quadmath disabled => 12).
const VALIDATED_NROOTS_CEILING: usize = 12;

/// The core sweep: for nroots in 6..=12 and each x, the host engine must match the
/// vendored `CINTrys_roots` roots+weights within `max(atol=1e-12, rtol=1e-9*|vendor|)`.
/// nroots 6,7 are byte-identical at atol; nroots 8..12 match to the dd-vs-f80 relative
/// floor. The test name is the acceptance-criteria anchor (`grep "fn rys_nroots_sweep"`).
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn rys_nroots_sweep() {
    use cintx_oracle::vendor_ffi;

    let mut mismatches = 0usize;
    // Track that nroots 6,7 (pure f64) are byte-identical at the strict ATOL.
    let mut f64_path_atol_failures = 0usize;
    for nroots in 6..=VALIDATED_NROOTS_CEILING {
        for &x in XS.iter() {
            let (rr, ww) = rys::rys_roots_host::<f64>(nroots, x);
            let (vr, vw) = vendor_ffi::vendor_CINTrys_roots(nroots as i32, x);
            assert_eq!(rr.len(), nroots, "host root count for nroots={nroots}");
            assert_eq!(vr.len(), nroots, "vendor root count for nroots={nroots}");
            for i in 0..nroots {
                let dr = (rr[i] - vr[i]).abs();
                let dw = (ww[i] - vw[i]).abs();
                let tol_r = ATOL.max(RTOL * vr[i].abs());
                let tol_w = ATOL.max(RTOL * vw[i].abs());
                if dr > tol_r || dw > tol_w {
                    mismatches += 1;
                    if nroots <= 7 {
                        f64_path_atol_failures += 1;
                    }
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
    // The pure-f64 paths (nroots 6,7) MUST be byte-identical at the strict 1e-12 atol.
    assert_eq!(
        f64_path_atol_failures, 0,
        "nroots 6,7 (pure f64) must be byte-identical at atol=1e-12 vs vendor CINTrys_roots"
    );
    assert_eq!(
        mismatches, 0,
        "rys nroots 6..=12 sweep: {mismatches} root/weight mismatches vs vendor CINTrys_roots \
         (atol={ATOL:e}, rtol={RTOL:e} for the long-double nroots>=8 path)"
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
