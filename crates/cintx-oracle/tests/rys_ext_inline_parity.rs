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
//! # Two regimes, two gates
//!
//! The sweep the plan asks for is log-spaced over 1e-8 … 1e6, and it straddles
//! `SMALLX_LIMIT = 3e-7`, where the vendor leaves the Wheeler dispatch entirely
//! for the global affine fits of `rys_roots.c:58-78`:
//!
//! * **x >= 1e-4 — [`rys_ext_inline_matches_host_wheeler`].** Zero divergences
//!   beyond `max(atol=1e-12, rtol=1e-9·|host|)`, and in practice bit-identity
//!   everywhere.
//! * **x < 1e-4 — [`rys_ext_inline_below_corpus_envelope_is_bit_identical`].**
//!   Both paths take the vendor's affine table below the limit and the same
//!   Wheeler arms above it, so this is bit-identity too.
//!
//! The second one used to be a *record* rather than a gate, bounding a
//! divergence instead of forbidding it, because only the host had the small-x
//! branch and the inline entry fell through to the moment recursion. That
//! fall-through was 1.5e-10 relative at `nroots = 6` and 3.6 — 360% — at 12,
//! and it was reachable from ordinary work: a single-centre quartet has
//! `rr = 0`, hence `x_rys = 0` exactly, and the def2-TZVP `(f f | f f)` block on
//! oxygen missed vendored libcint by 6.5e-11 absolute. Giving
//! `rys_roots_ext_dev` the vendor's branch closed it, and the test is a gate now
//! because there is nothing left to bound.

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

/// **The gate for `x` below the corpus envelope (1e-8 … 1e-4).**
///
/// Bit-identity, every order, every grid point. Two mechanisms produce it, and
/// the point of the test is that both hold:
///
/// 1. **Below `SMALLX_LIMIT = 3e-7`** both paths read the vendor's global affine
///    table (`rys_roots.c:58-78`) at the same triangular offset, so they agree
///    exactly and neither runs a solver.
/// 2. **Between 3e-7 and 1e-4** the inline entry and the host dispatch run the
///    same `#[cube]` solver bodies, once inline and once behind a launch.
///
/// `undefined` counts points where the host reference's own solver reported an
/// error and tripped its `debug_assert`. That used to happen across roughly
/// `[1.5e-8, 8.7e-5]` at `nroots = 12` and at one point at 11 — precisely the
/// ill-conditioned region the small-x branch now short-circuits — so the count
/// is asserted to be zero for every order. If it comes back, the reference has
/// started reaching the recurrence where it should not, and this is where that
/// surfaces.
#[test]
fn rys_ext_inline_below_corpus_envelope_is_bit_identical() {
    // The host reference `debug_assert`s on its own solver error; the hook is
    // silenced so an unexpected unwind is counted rather than burying the
    // report.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut worst = [0.0f64; 13];
    let mut undefined = [0usize; 13];
    let mut compared = 0usize;
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
                    compared += 1;
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
    eprintln!("rys_ext_inline sub-envelope: {compared} values compared");

    for nroots in 6..=12usize {
        assert_eq!(
            undefined[nroots], 0,
            "the host reference failed to converge at nroots={nroots} below the \
             corpus envelope; the small-x branch is supposed to keep it out of \
             the ill-conditioned recurrence entirely"
        );
        assert_eq!(
            worst[nroots], 0.0,
            "nroots={nroots}: below the corpus envelope the inline entry and the \
             host dispatch read the same vendor table and run the same solver \
             bodies, so they must be bit-identical; worst relative divergence \
             was {:e}",
            worst[nroots]
        );
    }
}
