//! Shared helpers for the Phase 24 position / multipole-moment vendor parity
//! scaffolds (`moment_{r,low,high,nontensor}_parity.rs`).
//!
//! This module is `#[path = "moment_common.rs"] mod moment_common;`-included by
//! each parity test file. It is ALSO compiled by cargo as its own (empty) test
//! binary — that is harmless; it carries no `#[test]` of its own.
//!
//! What it provides:
//!   (a) `vendor_parity(rank, ...)` — a RANK-PARAMETERIZED clone of
//!       `one_electron_grad_both_parity.rs::vendor_parity<FS,FC>` (which hardcoded
//!       NCOMP=9). Phase 24 ranks span 1/3/9/27/81, so every out/collect buffer is
//!       sized `rank * ni * nj`. Builds the NON-ZERO gauge-origin fixture, calls
//!       cintx `eval_raw` (sph then cart), calls the vendor fn, asserts both are
//!       non-zero, counts mismatches at ATOL=1e-12 / RTOL=0, asserts mm == 0.
//!   (b) `env_with_rinv_origin(env, origin)` — copied verbatim from
//!       `one_electron_nuc_grad_parity.rs:150-155`. Sets `env[PTR_RINV_ORIG..+3]`.
//!       The common-orig fixture is PTR_ENV_START-aligned (user data at env[20..]),
//!       so writing env[4..6] never collides with atom coordinates.
//!   (c) `non_square_shell_pair()` — returns a NON-SQUARE bra×ket shell-index pair
//!       (O-s shell 0 × O-p shell 2) on the H2O/STO-3G corpus. D-07 transpose gate:
//!       a square p×p pair is transpose-symmetric and would hide a permuted-gout bug.
//!
//! ORDERING NOTE (for plan 02): the four `moment_*_parity.rs` files reference
//! `RawApiId::INT1E_R_CART`, `INT1E_RR_SPH`, … consts that DO NOT EXIST until each
//! family's registration lands in plans 02-05 (each plan's Task 0 adds its consts to
//! `cintx-compat/src/raw.rs`). Until then the parity-test CRATE does not fully
//! compile. THIS module (the non-RawApiId helpers) compiles in isolation today.
//! Plan 02's first task is responsible for verifying `cargo test --no-run` once the
//! Cluster-A consts come online.

#![cfg(any(feature = "cpu", feature = "rocm"))]
#![allow(dead_code)]

use cintx_compat::raw::{ANG_OF, BAS_SLOTS, PTR_RINV_ORIG, RawApiId, eval_raw};
#[cfg(has_vendor_libcint)]
use cintx_compat::raw::ATM_SLOTS;

/// Unified byte-identity tolerance (Phase 24 D-10): atol = 1e-12, rtol = 0.
pub const ATOL: f64 = 1e-12;
pub const RTOL: f64 = 0.0;

/// H2O/STO-3G has 5 shells: O-1s, O-2s, O-2p, H1-1s, H2-1s.
pub const N_SHELLS: usize = 5;

/// Cartesian component count for angular momentum `l`.
pub fn ncart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

/// Spherical component count for angular momentum `l`.
pub fn nsph(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// Set the rinv origin into `env[PTR_RINV_ORIG..PTR_RINV_ORIG+3]` (returns a fresh env).
///
/// Verbatim from `one_electron_nuc_grad_parity.rs:150-155`. REQUIRED for the
/// `rinv`/`drinv` parity tests: the common-orig fixture leaves `rinv_orig=[0,0,0]`,
/// which is trivially-passing and disallowed (Phase 24 D-04 / OQ-1). The fixture is
/// PTR_ENV_START-aligned, so env[4..6] is in the reserved global block — no collision
/// with atom coordinates (which start at env[20..]).
pub fn env_with_rinv_origin(env: &[f64], origin: [f64; 3]) -> Vec<f64> {
    let mut e = env.to_vec();
    e[PTR_RINV_ORIG] = origin[0];
    e[PTR_RINV_ORIG + 1] = origin[1];
    e[PTR_RINV_ORIG + 2] = origin[2];
    e
}

/// A NON-SQUARE bra×ket shell-index pair on the H2O/STO-3G corpus.
///
/// Returns `(0, 2)` = (O-1s, O-2p): a 1×3 (cart) / 1×3 (sph) block — bra l=0, ket l=1.
/// D-07 transpose gate: a square p×p block is transpose-symmetric and would hide a
/// permuted-gout / transposed-layout bug. Every Phase 24 parity test MUST use this
/// non-square pair (or another documented non-square pair).
pub fn non_square_shell_pair() -> (usize, usize) {
    // bra = O-1s (shell 0, l=0), ket = O-2p (shell 2, l=1) — strictly non-square.
    (0, 2)
}

/// Collect cintx `eval_raw` output for ONE non-square shell pair into a
/// `rank * ni * nj` component-leading buffer.
fn collect_cintx_block(
    rank: usize,
    api_id: RawApiId,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64> {
    let (si, sj) = non_square_shell_pair();
    let li = bas[si * BAS_SLOTS + ANG_OF];
    let lj = bas[sj * BAS_SLOTS + ANG_OF];
    let ni = nf(li);
    let nj = nf(lj);
    let shls = [si as i32, sj as i32];
    let mut out = vec![0.0_f64; rank * ni * nj];

    // SAFETY: atm/bas/env are well-formed by construction; shls indices are valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for shells ({si},{sj}): {e:?}"));
    }
    out
}

/// Collect vendor libcint output for ONE non-square shell pair into a
/// `rank * ni * nj` component-leading buffer.
#[cfg(has_vendor_libcint)]
fn collect_vendor_block<F>(
    rank: usize,
    vendor_fn: F,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    nf: impl Fn(i32) -> usize,
) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    let (si, sj) = non_square_shell_pair();
    let li = bas[si * BAS_SLOTS + ANG_OF];
    let lj = bas[sj * BAS_SLOTS + ANG_OF];
    let ni = nf(li);
    let nj = nf(lj);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let shls: [i32; 2] = [si as i32, sj as i32];
    let mut out = vec![0.0_f64; rank * ni * nj];
    vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);
    out
}

/// Count element-wise mismatches at `atol + rtol*|ref|`; logs each one.
pub fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(reference.len(), observed.len(), "length mismatch");
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            eprintln!(
                "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                 diff={diff:.3e}, threshold={threshold:.3e}"
            );
        }
    }
    mismatches
}

/// Assert at least one element is meaningfully non-zero (zero-fill regression guard).
pub fn assert_any_nonzero(block: &[f64], label: &str) {
    let any_nonzero = block.iter().any(|v| v.abs() > 1e-14);
    assert!(any_nonzero, "{label}: block is all-zero (zero-fill regression)");
}

/// RANK-PARAMETERIZED vendor parity gate (Phase 24 generalization of the hardcoded
/// NCOMP=9 helper in `one_electron_grad_both_parity.rs:307`).
///
/// `rank` is the family's exact `component_rank` (r=3, rr=9, rrr=27, rrrr=81,
/// r2/r4/z/zz/p4/rinv=1, drinv=3, irp=9). Every out/collect buffer is sized
/// `rank * ni * nj`, so no trailing component is truncated (D-08).
///
/// Caller passes the (already-prepared) fixture so Cluster-B tests can inject a
/// NON-ZERO rinv origin via `env_with_rinv_origin` before calling.
#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[allow(clippy::too_many_arguments)]
pub fn vendor_parity<FS, FC>(
    rank: usize,
    api_sph: RawApiId,
    api_cart: RawApiId,
    vendor_sph: FS,
    vendor_cart: FC,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    label: &str,
) where
    FS: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
    FC: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    // ── sph ──────────────────────────────────────────────────────────────────
    let vendor_s = collect_vendor_block(rank, &vendor_sph, atm, bas, env, nsph);
    let cintx_s = collect_cintx_block(rank, api_sph, atm, bas, env, nsph);
    assert_any_nonzero(&cintx_s, &format!("{label}_sph cintx"));
    assert_any_nonzero(&vendor_s, &format!("{label}_sph vendor"));
    let mm = count_mismatches(&vendor_s, &cintx_s, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_sph: {mm} mismatches vs vendored libcint at atol={ATOL}");

    // ── cart ─────────────────────────────────────────────────────────────────
    let vendor_c = collect_vendor_block(rank, &vendor_cart, atm, bas, env, ncart);
    let cintx_c = collect_cintx_block(rank, api_cart, atm, bas, env, ncart);
    assert_any_nonzero(&cintx_c, &format!("{label}_cart cintx"));
    assert_any_nonzero(&vendor_c, &format!("{label}_cart vendor"));
    let mm = count_mismatches(&vendor_c, &cintx_c, ATOL, RTOL);
    assert_eq!(mm, 0, "{label}_cart: {mm} mismatches vs vendored libcint at atol={ATOL}");
}
