#![cfg(feature = "cpu")]
//! Phase 33, task 33-04 — the accuracy gate for the **inline** extended-Rys entry.
//!
//! `math::rys_wheeler::rys_roots_ext_dev` is the whole `nroots` 6..=12 Wheeler
//! dispatch as a `#[cube]` callee, so a family kernel can get its roots without
//! leaving the device. This test is what stands between that callee and any
//! family using it: task 33-03 flips `extended-device-rys` one family at a
//! time, and the order is deliberate — the callee (33-01) and its sizing
//! (33-02) land additively with the feature off, so a failing gate here stops
//! the work with the tree still green.
//!
//! # What is compared, and why not the vendor
//!
//! The reference is `rys_roots_host_wheeler`, the production host path, which
//! `rys_nroots_sweep_parity` already pins to vendored libcint 6.1.3. Comparing
//! inline against host keeps this test measuring exactly one thing: whether
//! moving the dispatch inside a kernel changed an answer. A vendor comparison
//! would fold in the dd-vs-80-bit divergence that sweep already characterises,
//! and a regression here would be indistinguishable from it.
//!
//! # Two regimes, one gate and one record
//!
//! The sweep the plan asks for is log-spaced over 1e-8 … 1e6. That range
//! straddles a boundary the reference itself has:
//!
//! * **x >= 1e-4 — [`rys_ext_inline_matches_host_wheeler`], the gate.** Zero
//!   divergences beyond `max(atol=1e-12, rtol=1e-9·|host|)`, and in practice
//!   bit-identity almost everywhere.
//! * **x < 1e-4 — [`rys_ext_inline_below_corpus_envelope_is_bounded`], the
//!   record.** Below `SMALLX_LIMIT = 3e-7` the vendor leaves the Wheeler path
//!   entirely for the global polynomial fits of `rys_roots.c:58-78`, which
//!   neither implementation ports; and the host reference's own solver reports
//!   error 1 across roughly `[1.5e-8, 8.7e-5]` at `nroots = 12` (and at one
//!   point at `nroots = 11`), tripping its `debug_assert`. That region is
//!   therefore measured and bounded rather than gated — see that test for the
//!   numbers and for why `nroots` 6 and 7 are the only orders that move at all.

use cintx_cubecl::math::rys_wheeler;

/// Absolute floor, matching `rys_nroots_sweep_parity`.
const ATOL: f64 = 1e-12;

/// Relative tolerance at large magnitudes, matching `rys_nroots_sweep_parity`.
const RTOL: f64 = 1e-9;

/// Lower edge of the gated regime. Below this the host reference is itself
/// outside the corpus envelope; see the module note.
const GATE_X_MIN_EXP: i32 = -4;

/// Log-spaced grid over `10^lo_exp ..= 10^hi_exp` at `per_decade` points.
fn log_grid(lo_exp: i32, hi_exp: i32, per_decade: u32) -> Vec<f64> {
    let steps = (hi_exp - lo_exp) as u32 * per_decade;
    (0..=steps)
        .map(|s| 10.0_f64.powf(f64::from(lo_exp) + f64::from(s) / f64::from(per_decade)))
        .collect()
}

/// The `x` breakpoint separating the two solvers for `nroots`
/// (`rys_roots.c:97-114`), spelled out here rather than imported so that a
/// change to the dispatch has to be made twice to go unnoticed.
fn breakpoint(nroots: usize) -> f64 {
    match nroots {
        6..=8 => 11.0,
        9 => 10.0,
        10 | 11 => 18.0,
        _ => 22.0,
    }
}

/// **The gate.** The inline `#[cube]` entry reproduces `rys_roots_host_wheeler`
/// across `nroots` 6..=12 crossed with a log-spaced `x` sweep from 1e-4 to 1e6,
/// plus each arm's exact breakpoint and a point either side of it.
///
/// The test name is the acceptance anchor
/// (`grep "fn rys_ext_inline_matches_host_wheeler"`).
#[test]
fn rys_ext_inline_matches_host_wheeler() {
    let mut mismatches = 0usize;
    let mut compared = 0usize;
    let mut bit_identical = 0usize;

    for nroots in 6..=12usize {
        let mut xs = log_grid(GATE_X_MIN_EXP, 6, 8);
        // The breakpoint neighbourhood is the one place where a mis-transcribed
        // dispatch shows up as a whole wrong solver rather than as a last-bit
        // difference, so it is sampled exactly and on both sides.
        let bp = breakpoint(nroots);
        xs.extend_from_slice(&[bp - 1e-9, bp, bp + 1e-9]);

        for x in xs {
            let (hr, hw) = rys_wheeler::rys_roots_host_wheeler(nroots, x);
            let (dr, dw) = rys_wheeler::rys_roots_ext_host(nroots, x);
            assert_eq!(dr.len(), nroots, "inline root count for nroots={nroots}");
            assert_eq!(dw.len(), nroots, "inline weight count for nroots={nroots}");

            for i in 0..nroots {
                compared += 2;
                bit_identical += usize::from(dr[i].to_bits() == hr[i].to_bits());
                bit_identical += usize::from(dw[i].to_bits() == hw[i].to_bits());

                let er = (dr[i] - hr[i]).abs();
                let ew = (dw[i] - hw[i]).abs();
                let tol_r = ATOL.max(RTOL * hr[i].abs());
                let tol_w = ATOL.max(RTOL * hw[i].abs());
                if er > tol_r || ew > tol_w {
                    mismatches += 1;
                    eprintln!(
                        "MISMATCH nroots={nroots} x={x:e} i={i}: \
                         root inline={} host={} |d|={er:e} tol={tol_r:e} | \
                         weight inline={} host={} |d|={ew:e} tol={tol_w:e}",
                        dr[i], hr[i], dw[i], hw[i]
                    );
                }
            }
        }
    }

    eprintln!("rys_ext_inline: {bit_identical}/{compared} values bit-identical to the host path");
    assert_eq!(
        mismatches, 0,
        "{mismatches} inline-vs-host divergences beyond max(atol={ATOL:e}, rtol={RTOL:e})"
    );
}

/// **The record**, for `x` below the corpus envelope (1e-8 … 1e-4).
///
/// Two findings, both asserted so they cannot drift silently:
///
/// 1. **`nroots` 8..=12 are bit-identical**, everywhere the host reference is
///    defined. That is not luck: for those orders `rys_roots_host_wheeler`
///    already dispatches to the `#[cube]` device solvers (`rys_jacobi_device`,
///    `lrys_*_device`), so the inline entry and the host entry run the *same*
///    kernel bodies, once inline and once behind a launch.
/// 2. **`nroots` 6 and 7 move, by at most ~1.3e-8 relative.** Those two orders
///    are the parity-honest escape hatch: the host routes them through the
///    pure-host `rys_jacobi` / `rys_schmidt` rather than through the device
///    kernels, so here — and only here — two different transcriptions of the
///    same algorithm are being compared. Below `SMALLX_LIMIT = 3e-7` the Flocke
///    moment recursion is ill-conditioned and that shows; above it the gate
///    above finds no divergence at all.
///
/// The host reference reports solver error 1 (and `debug_assert`s on it) for
/// part of this range at `nroots` 11 and 12, so calls are caught rather than
/// assumed to return. The count of caught points is asserted to stay confined
/// to those two orders — if a lower order ever starts failing there, that is a
/// regression in the reference, and this test is where it surfaces.
#[test]
fn rys_ext_inline_below_corpus_envelope_is_bounded() {
    /// Bound on the relative inline-vs-host divergence for the two pure-host
    /// orders, measured at 1.2e-8 and rounded up by an order of magnitude.
    const SUB_ENVELOPE_RTOL: f64 = 1e-7;

    // The host reference `debug_assert`s on its own solver error in this range;
    // the hook is silenced so the expected unwinds do not bury the report.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut worst = [0.0f64; 13];
    let mut undefined = [0usize; 13];
    for nroots in 6..=12usize {
        for x in log_grid(-8, GATE_X_MIN_EXP, 32) {
            let Ok((hr, hw)) =
                std::panic::catch_unwind(|| rys_wheeler::rys_roots_host_wheeler(nroots, x))
            else {
                undefined[nroots] += 1;
                continue;
            };
            let (dr, dw) = rys_wheeler::rys_roots_ext_host(nroots, x);
            for i in 0..nroots {
                for (inline, host) in [(dr[i], hr[i]), (dw[i], hw[i])] {
                    let rel = (inline - host).abs() / host.abs().max(f64::MIN_POSITIVE);
                    if rel > worst[nroots] {
                        worst[nroots] = rel;
                    }
                }
            }
        }
    }
    std::panic::set_hook(previous);

    for nroots in 6..=12usize {
        eprintln!(
            "rys_ext_inline sub-envelope: nroots={nroots} worst_rel={:e} \
             host-undefined points={}",
            worst[nroots], undefined[nroots]
        );
    }

    for nroots in 8..=12usize {
        assert_eq!(
            worst[nroots], 0.0,
            "nroots={nroots} shares the device solvers with the host path, so it \
             must stay bit-identical below the corpus envelope; worst relative \
             divergence was {:e}",
            worst[nroots]
        );
    }
    for nroots in [6usize, 7] {
        assert!(
            worst[nroots] <= SUB_ENVELOPE_RTOL,
            "nroots={nroots} diverged by {:e} relative below the corpus envelope, \
             over the {SUB_ENVELOPE_RTOL:e} bound this test records",
            worst[nroots]
        );
    }
    for nroots in 6..=10usize {
        assert_eq!(
            undefined[nroots], 0,
            "the host reference newly fails to converge at nroots={nroots} below \
             the corpus envelope; only nroots 11 and 12 did so when this bound \
             was recorded"
        );
    }
}
