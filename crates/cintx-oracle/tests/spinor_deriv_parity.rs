//! Phase 27 — spinor DERIVATIVE parity scaffold (D-08 / D-09 / D-10).
//!
//! The RED Nyquist target for Plans 03/04/05. Drives the D-08 adversarial spinor
//! fixture (`build_adversarial_spinor_fixture`) — NON-SQUARE (p × d) bra/ket with a
//! general-contracted (nctr=2) bra, kappa=0 on every shell (both GT+LT spinor blocks,
//! di=4l+2), and a NON-ZERO rinv origin — against vendored libcint 6.1.3 for every
//! spinor derivative family beyond the four rank-3 1e ip-spinor operators that
//! already pass:
//!
//!   - int1e_ipovlp_spinor        (rank  3, pair  shls) — already supported (sanity)
//!   - int1e_ipovlpip_spinor      (rank  9, pair  shls) — RED until Plan 03
//!   - int1e_ipipipiprinv_spinor  (rank 81, pair  shls) — RED until Plan 03
//!   - int3c2e_ip1_spinor         (rank  3, triple shls) — RED until Plan 03
//!   - int2c2e_ip1_spinor         (rank  3, pair  shls) — RED until Plan 03
//!   - int3c1e_ip1_spinor         (rank  3, triple shls, D-02 arity-3) — RED until Plan 04
//!   - int3c1e_iprinv_spinor      (rank  3, triple shls, D-02 arity-3) — RED until Plan 04
//!
//! Each parity test asserts `assert_any_nonzero` on BOTH cintx and vendor, then
//! `count_mismatches(&vendor, &cintx, 1e-12, 1e-12) == 0`. Until the launchers are
//! rewired (Plans 03/04), cintx returns `UnsupportedApi` for the non-rank-3 families
//! and for the two int3c1e arity-3 families, so the cintx collector PANICS / mismatches
//! — that RED state is the intended Nyquist target handed downstream. The
//! `--no-run` COMPILE gate (this file building) is what Plan 01 Task 2 must satisfy.
//!
//! Orientation negative control (`test_orientation_negative_control`): a j-fastest
//! reindex of the cintx buffer MUST diverge from vendor on the non-square block,
//! proving i-fastest orientation (the square-block H2O test was orientation-blind).
//!
//! No-silent-skip (`test_no_silent_skip`, D-10 / T-27-01): the vendor arm MUST
//! actually execute and produce nonzero output for a SUPPORTED family — fail (not
//! skip) if the vendor side compiled out or the fixture was skipped.
//!
//! Aux-k spinor sizing (27-SPIKE-FINDINGS D2/D3): the arity-3 families size the
//! auxiliary-k axis with `CINTcgto_spinor(k) = 4l+2 = 2` (kappa=0), NOT nsph(lk).
//!
//! Double gate: vendor parity requires `--features cpu` AND `CINTX_ORACLE_BUILD_VENDOR=1`
//! (the `has_vendor_libcint` cfg). Without both, the vendor bodies compile out and the
//! banner shows `running 0 tests` for the vendor functions.

#![cfg(any(feature = "cpu", feature = "rocm"))]

use cintx_compat::raw::{ANG_OF, BAS_SLOTS, NCTR_OF, RawApiId, eval_raw};
use cintx_oracle::fixtures::build_adversarial_spinor_fixture;

#[allow(dead_code)]
const ATOL: f64 = 1e-12;
#[allow(dead_code)]
const RTOL: f64 = 1e-12;

// The single shell triple under test from build_adversarial_spinor_fixture():
//   shell 0: bra i = p (l=1, nctr=2), shell 1: ket j = d (l=2), shell 2: aux k = s (l=0).
const SI: usize = 0;
const SJ: usize = 1;
const SK: usize = 2;

/// Spinor block length for ang `l` at kappa==0: 4*l + 2.
fn spinor_len_kappa0(l: i32) -> usize {
    (4 * l + 2) as usize
}

fn shell_ang(bas: &[i32], s: usize) -> i32 {
    bas[s * BAS_SLOTS + ANG_OF]
}
fn shell_nctr(bas: &[i32], s: usize) -> usize {
    bas[s * BAS_SLOTS + NCTR_OF] as usize
}

/// Full spinor axis length for shell `s`: `nctr * (4l+2)`.
fn shell_nsp_full(bas: &[i32], s: usize) -> usize {
    spinor_len_kappa0(shell_ang(bas, s)) * shell_nctr(bas, s)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parity helpers (cloned from one_electron_grad_spinor_parity.rs, generalized to
// per-family `ncomp` and arity-2 / arity-3 shell tuples).
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn count_mismatches(reference: &[f64], observed: &[f64], atol: f64, rtol: f64) -> usize {
    assert_eq!(
        reference.len(),
        observed.len(),
        "length mismatch: {} vs {}",
        reference.len(),
        observed.len()
    );
    let mut mismatches = 0usize;
    for (i, (&ref_val, &obs_val)) in reference.iter().zip(observed.iter()).enumerate() {
        let diff = (obs_val - ref_val).abs();
        let threshold = atol + rtol * ref_val.abs();
        if diff > threshold {
            mismatches += 1;
            if mismatches <= 20 {
                eprintln!(
                    "  MISMATCH at index {i}: reference={ref_val:.15e}, observed={obs_val:.15e}, \
                     diff={diff:.3e}, threshold={threshold:.3e}"
                );
            }
        }
    }
    mismatches
}

#[allow(dead_code)]
fn assert_any_nonzero(matrix: &[f64], label: &str) {
    let any_nonzero = matrix.iter().any(|v| v.abs() > 1e-14);
    assert!(
        any_nonzero,
        "{label}: derivative matrix is all-zero (zero-fill / silent-skip regression)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// cintx collector for a SINGLE arity-2 spinor-derivative block (i, j).
//
// Component-leading, complex-interleaved, ket-major / bra-fastest within a comp:
//   out[comp * ni_sp * nj_sp * 2 + (j*ni_sp + i)*2 + {0:re,1:im}]
// where ni_sp / nj_sp are the FULL spinor axis lengths (nctr * (4l+2)).
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn collect_cintx_2c(api_id: RawApiId, ncomp: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let ni = shell_nsp_full(bas, SI);
    let nj = shell_nsp_full(bas, SJ);
    let shls = [SI as i32, SJ as i32];
    let mut out = vec![0.0_f64; ncomp * ni * nj * 2];
    // SAFETY: atm/bas/env are well-formed by construction; shls are valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for {api_id:?} ({SI},{SJ}): {e:?}"));
    }
    out
}

/// cintx collector for a SINGLE arity-3 spinor-derivative block (i, j, k).
/// The aux-k axis is SPINOR-sized (CINTcgto_spinor(k)=4l+2), per 27-SPIKE-FINDINGS D2/D3.
#[allow(dead_code)]
fn collect_cintx_3c(api_id: RawApiId, ncomp: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64> {
    let ni = shell_nsp_full(bas, SI);
    let nj = shell_nsp_full(bas, SJ);
    let nk = shell_nsp_full(bas, SK);
    let shls = [SI as i32, SJ as i32, SK as i32];
    let mut out = vec![0.0_f64; ncomp * ni * nj * nk * 2];
    // SAFETY: atm/bas/env are well-formed by construction; shls are valid.
    unsafe {
        eval_raw(api_id, Some(&mut out), None, &shls, atm, bas, env, None, None)
            .unwrap_or_else(|e| panic!("eval_raw failed for {api_id:?} ({SI},{SJ},{SK}): {e:?}"));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor collectors (only available with has_vendor_libcint).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
fn collect_vendor_2c<F>(vendor_fn: F, ncomp: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 2], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    use cintx_compat::raw::ATM_SLOTS;
    let ni = shell_nsp_full(bas, SI);
    let nj = shell_nsp_full(bas, SJ);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let shls = [SI as i32, SJ as i32];
    let mut out = vec![0.0_f64; ncomp * ni * nj * 2];
    vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);
    out
}

#[cfg(has_vendor_libcint)]
fn collect_vendor_3c<F>(vendor_fn: F, ncomp: usize, atm: &[i32], bas: &[i32], env: &[f64]) -> Vec<f64>
where
    F: Fn(&mut [f64], &[i32; 3], &[i32], i32, &[i32], i32, &[f64]) -> i32,
{
    use cintx_compat::raw::ATM_SLOTS;
    let ni = shell_nsp_full(bas, SI);
    let nj = shell_nsp_full(bas, SJ);
    let nk = shell_nsp_full(bas, SK);
    let natm = (atm.len() / ATM_SLOTS) as i32;
    let nbas = (bas.len() / BAS_SLOTS) as i32;
    let shls = [SI as i32, SJ as i32, SK as i32];
    let mut out = vec![0.0_f64; ncomp * ni * nj * nk * 2];
    vendor_fn(&mut out, &shls, atm, natm, bas, nbas, env);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Vendor parity tests (require has_vendor_libcint + cpu feature).
//
// EXPECTED RED for the non-rank-3 1e families AND the two int3c1e arity-3 families:
// cintx still returns UnsupportedApi (Plans 03/04 wire the launchers). The compile
// gate (--no-run) is what Plan 01 Task 2 satisfies.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlp_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_2c(vendor_ffi::vendor_int1e_ipovlp_spinor, 3, &atm, &bas, &env);
    let cintx = collect_cintx_2c(RawApiId::INT1E_IPOVLP_SPINOR, 3, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipovlp_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ipovlp_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_ipovlp_spinor adversarial parity vs vendored libcint"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipovlpip_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_2c(vendor_ffi::vendor_int1e_ipovlpip_spinor, 9, &atm, &bas, &env);
    let cintx = collect_cintx_2c(RawApiId::INT1E_IPOVLPIP_SPINOR, 9, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipovlpip_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ipovlpip_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_ipovlpip_spinor (rank 9) adversarial parity vs vendored libcint"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int1e_ipipipiprinv_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    // The fixture sets a NON-ZERO rinv origin (env[PTR_RINV_ORIG..+3]); the rinv
    // center path is exercised, not a zero-origin shortcut (Phase 25-03 landmine).
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_2c(vendor_ffi::vendor_int1e_ipipipiprinv_spinor, 81, &atm, &bas, &env);
    let cintx = collect_cintx_2c(RawApiId::INT1E_IPIPIPIPRINV_SPINOR, 81, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int1e_ipipipiprinv_spinor cintx");
    assert_any_nonzero(&vendor, "int1e_ipipipiprinv_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int1e_ipipipiprinv_spinor (rank 81, nonzero rinv) adversarial parity vs vendored libcint"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c2e_ip1_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_3c(vendor_ffi::vendor_int3c2e_ip1_spinor, 3, &atm, &bas, &env);
    let cintx = collect_cintx_3c(RawApiId::INT3C2E_IP1_SPINOR, 3, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int3c2e_ip1_spinor cintx");
    assert_any_nonzero(&vendor, "int3c2e_ip1_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int3c2e_ip1_spinor (arity-3, aux-k spinor-sized) adversarial parity vs vendored libcint"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int2c2e_ip1_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_2c(vendor_ffi::vendor_int2c2e_ip1_spinor, 3, &atm, &bas, &env);
    let cintx = collect_cintx_2c(RawApiId::INT2C2E_IP1_SPINOR, 3, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int2c2e_ip1_spinor cintx");
    assert_any_nonzero(&vendor, "int2c2e_ip1_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int2c2e_ip1_spinor adversarial parity vs vendored libcint"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_ip1_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    // D-02 arity-3 family. The int3c1e launchers are host-side (center_3c1e.rs);
    // Plan 04 wires the spinor fold. RED until then.
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_3c(vendor_ffi::vendor_int3c1e_ip1_spinor, 3, &atm, &bas, &env);
    let cintx = collect_cintx_3c(RawApiId::INT3C1E_IP1_SPINOR, 3, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int3c1e_ip1_spinor cintx");
    assert_any_nonzero(&vendor, "int3c1e_ip1_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int3c1e_ip1_spinor (D-02 arity-3) adversarial parity vs vendored libcint"
    );
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_int3c1e_iprinv_spinor_adversarial_parity() {
    use cintx_oracle::vendor_ffi;
    // D-02 arity-3 family, Rys-driven. MUST use a NON-ZERO rinv origin — the
    // fixture sets env[PTR_RINV_ORIG..+3] non-zero so the rinv-center path is
    // actually exercised, NOT a zero-origin shortcut (T-27-04).
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let vendor = collect_vendor_3c(vendor_ffi::vendor_int3c1e_iprinv_spinor, 3, &atm, &bas, &env);
    let cintx = collect_cintx_3c(RawApiId::INT3C1E_IPRINV_SPINOR, 3, &atm, &bas, &env);

    assert_any_nonzero(&cintx, "int3c1e_iprinv_spinor cintx");
    assert_any_nonzero(&vendor, "int3c1e_iprinv_spinor vendor");
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "int3c1e_iprinv_spinor (D-02 arity-3, nonzero rinv) adversarial parity vs vendored libcint"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Orientation negative control (T-27-02): a j-fastest reindex of the cintx buffer
// MUST diverge from the (i-fastest) vendor buffer on the non-square block. Uses the
// already-supported rank-3 int1e_ipovlp_spinor family so the control is meaningful
// even before Plans 03/04 land.
// ─────────────────────────────────────────────────────────────────────────────

/// Reindex a component-leading, ket-major / bra-fastest block to bra-major /
/// ket-fastest (j-fastest) — the WRONG orientation. On a non-square block this
/// produces a different element ordering that MUST diverge from the vendor.
#[allow(dead_code)]
fn to_j_fastest(out: &[f64], ncomp: usize, ni: usize, nj: usize) -> Vec<f64> {
    let mut swapped = vec![0.0_f64; out.len()];
    for comp in 0..ncomp {
        let comp_off = comp * ni * nj * 2;
        for j in 0..nj {
            for i in 0..ni {
                let src = comp_off + (j * ni + i) * 2; // i-fastest (correct)
                let dst = comp_off + (i * nj + j) * 2; // j-fastest (wrong)
                swapped[dst] = out[src];
                swapped[dst + 1] = out[src + 1];
            }
        }
    }
    swapped
}

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_orientation_negative_control() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    let ni = shell_nsp_full(&bas, SI);
    let nj = shell_nsp_full(&bas, SJ);
    // ni_full=12 (p nctr=2 → 2*6), nj_full=10 (d → 10): genuinely non-square so the
    // i↔j transpose cannot coincide.
    assert_ne!(ni, nj, "orientation control requires a NON-SQUARE block");

    let vendor = collect_vendor_2c(vendor_ffi::vendor_int1e_ipovlp_spinor, 3, &atm, &bas, &env);
    let cintx = collect_cintx_2c(RawApiId::INT1E_IPOVLP_SPINOR, 3, &atm, &bas, &env);

    // Sanity: correct orientation matches.
    assert_eq!(
        count_mismatches(&vendor, &cintx, ATOL, RTOL),
        0,
        "negative control precondition: correct (i-fastest) orientation must match vendor"
    );

    // The j-fastest misread MUST diverge on the non-square block.
    let cintx_jfastest = to_j_fastest(&cintx, 3, ni, nj);
    let mismatches = count_mismatches(&vendor, &cintx_jfastest, ATOL, RTOL);
    assert!(
        mismatches > 0,
        "orientation negative control: j-fastest reindex MUST diverge on the non-square block \
         (got {mismatches} mismatches; i-fastest orientation not actually pinned)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// No-silent-skip (D-10 / T-27-01): the vendor arm MUST actually execute and produce
// nonzero output for a SUPPORTED family. Fails (not skips) if the vendor side
// compiled out or the fixture produced an all-zero (skipped) result. The
// oracle_covered=true assertion is wired in Plan 05 once the lock flips; here we
// assert the fixture is NOT skipped and the real vendor link is live.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(has_vendor_libcint)]
#[cfg(feature = "cpu")]
#[test]
fn test_no_silent_skip() {
    use cintx_oracle::vendor_ffi;
    let (atm, bas, env) = build_adversarial_spinor_fixture();

    // A supported rank-3 family proves the vendor arm is live and the fixture drives
    // a REAL comparison (not the determinism-only silent-skip path).
    let vendor = collect_vendor_2c(vendor_ffi::vendor_int1e_ipovlp_spinor, 3, &atm, &bas, &env);
    assert_any_nonzero(
        &vendor,
        "no-silent-skip: vendor arm produced all-zero output (fixture skipped / vendor compiled out)",
    );

    // The cintx side of the SAME supported family must also run (no UnsupportedApi).
    let cintx = collect_cintx_2c(RawApiId::INT1E_IPOVLP_SPINOR, 3, &atm, &bas, &env);
    assert_any_nonzero(
        &cintx,
        "no-silent-skip: cintx arm produced all-zero output (fixture skipped)",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-vendor smoke: when has_vendor_libcint is NOT set, this keeps the test binary
// non-empty under `--features cpu` alone (build/run without vendor produces a no-op
// pass; the parity bodies above compile out).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "cpu", not(has_vendor_libcint)))]
#[test]
fn test_fixture_builds_without_vendor() {
    let (atm, bas, env) = build_adversarial_spinor_fixture();
    assert_eq!(bas.len() % BAS_SLOTS, 0, "bas rows well-formed");
    assert!(!atm.is_empty() && !env.is_empty(), "fixture populated");
    // Spinor axis lengths at kappa=0: p(nctr=2)=12, d=10, s=2.
    assert_eq!(shell_nsp_full(&bas, SI), 12);
    assert_eq!(shell_nsp_full(&bas, SJ), 10);
    assert_eq!(shell_nsp_full(&bas, SK), 2);
}
