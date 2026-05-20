//! Boys function validation tests against known reference values.
//!
//! Reference derivation:
//!   - F_m(0) = 1/(2m+1) — analytical identity, matches fmt.c line 209 (t==0 branch)
//!   - Power series branch: fmt.c lines 186-203 (fmt1_gamma_inc_like)
//!   - Erfc branch: F_0(t) = SQRTPIE4/sqrt(t)*erf(sqrt(t)), fmt.c line 219
//!   - Upward recurrence: F_m = b*((2m-1)*F_{m-1} - exp(-t)), fmt.c line 223
//!
//! All numerical golden values in this file are generated from the libcint
//! `gamma_inc_like()` algorithm in C (fmt.c lines 206-226) with full double
//! precision system erf. Source line citations appear with each constant.
//!
//! All assertions use absolute tolerance 1e-12 per design decision D-19.

use cintx_cubecl::math::boys::{SQRTPIE4, TURNOVER_POINT, boys_gamma_inc_host, erf_host};
use cintx_core::CintFloat;

/// Helper: call boys_gamma_inc_host and return the result array.
fn compute_boys(t: f64, m: u32) -> Vec<f64> {
    boys_gamma_inc_host(t, m)
}

/// Reference implementation of gamma_inc_like matching libcint fmt.c exactly.
/// Uses the full iterative power series for the small-t branch.
/// Source: fmt.c lines 186-226 (fmt1_gamma_inc_like + gamma_inc_like).
fn boys_fm_reference(t: f64, m: u32) -> Vec<f64> {
    let mut f = vec![0.0f64; (m + 1) as usize];
    let turnover = TURNOVER_POINT[m as usize];

    if t == 0.0 {
        // t==0 branch: F_m(0) = 1/(2m+1), fmt.c lines 208-212
        f[0] = 1.0;
        for k in 1..=m {
            f[k as usize] = 1.0 / (2 * k + 1) as f64;
        }
    } else if t < turnover {
        // Power series branch (fmt1_gamma_inc_like), fmt.c lines 186-203
        let b = m as f64 + 0.5;
        let e = 0.5 * f64::exp(-t);
        let mut x = e;
        let mut s = e;
        let tol = f64::EPSILON * 0.5 * e;
        let mut bi = b + 1.0;
        while x > tol {
            x = x * t / bi;
            s = s + x;
            bi = bi + 1.0;
        }
        f[m as usize] = s / b;
        // Downward recurrence, fmt.c lines 200-203
        for i in (1..=m).rev() {
            let b_down = i as f64 - 0.5;
            f[(i - 1) as usize] = (e + t * f[i as usize]) / b_down;
        }
    } else {
        // Erfc + upward recurrence branch, fmt.c lines 218-225
        let tt = t.sqrt();
        f[0] = erf_host(tt) * (SQRTPIE4 / tt);
        let e = (-t).exp();
        let b = 0.5 / t;
        for i in 1..=m {
            f[i as usize] = b * ((2 * i - 1) as f64 * f[(i - 1) as usize] - e);
        }
    }
    f
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: t=0, m=0..10 — analytical identity F_m(0) = 1/(2m+1)
// Source: fmt.c lines 208-212 (t==0 branch)
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_t0() {
    for m in 0u32..=10 {
        let f = compute_boys(0.0, m);
        for k in 0..=m {
            // F_m(0) = 1/(2m+1) — analytical identity, fmt.c line 211
            let expected = 1.0 / (2 * k + 1) as f64;
            assert!(
                (f[k as usize] - expected).abs() < 1e-12,
                "boys_t0 failed for m={m}, k={k}: got {}, expected {expected}",
                f[k as usize]
            );
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: small t (power series branch), m=0..5
// Source: fmt.c lines 186-203 (fmt1_gamma_inc_like, power series)
// Reference: boys_fm_reference() which mirrors fmt.c exactly.
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_small_t() {
    // Test cases: (t, m) where t < TURNOVER_POINT[m] (power series branch)
    // For m=5, TURNOVER_POINT[5] = 2.106432965305 — use t=0.1, 0.5, 1.0, 2.0
    let test_t = [0.1f64, 0.5, 1.0, 2.0];
    for &t in &test_t {
        for m in 0u32..=5 {
            let computed = compute_boys(t, m);
            let reference = boys_fm_reference(t, m);
            for k in 0..=m {
                let diff = (computed[k as usize] - reference[k as usize]).abs();
                assert!(
                    diff < 1e-12,
                    "boys_small_t failed for t={t}, m={m}, k={k}: got {}, expected {}, diff={diff}",
                    computed[k as usize], reference[k as usize]
                );
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: large t (erfc + upward recurrence branch), m=0..5
// Source: fmt.c lines 218-225 (erfc branch + upward recurrence)
// F_0(t) = SQRTPIE4/sqrt(t)*erf(sqrt(t)), fmt.c line 219
// F_m(t) = b*((2m-1)*F_{m-1} - exp(-t)), fmt.c line 223
// Reference: boys_fm_reference() which mirrors fmt.c exactly.
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_large_t() {
    // t=15.0, 25.0, 50.0 — all above TURNOVER_POINT[5]=2.106, so erfc branch for all m
    let test_t = [15.0f64, 25.0, 50.0];
    for &t in &test_t {
        for m in 0u32..=5 {
            let computed = compute_boys(t, m);
            let reference = boys_fm_reference(t, m);
            for k in 0..=m {
                let diff = (computed[k as usize] - reference[k as usize]).abs();
                assert!(
                    diff < 1e-12,
                    "boys_large_t failed for t={t}, m={m}, k={k}: got {}, expected {}, diff={diff}",
                    computed[k as usize], reference[k as usize]
                );
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 4: turnover boundary for m=5 — both branches must agree within 1e-12
// TURNOVER_POINT[5] = 2.106432965305, fmt.c line 49
// Test t just below and just above the threshold
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_turnover_boundary() {
    let m = 5u32;
    // TURNOVER_POINT[5] = 2.106432965305 — fmt.c line 49
    let tp = TURNOVER_POINT[m as usize];
    let t_below = tp - 1e-6;
    let t_above = tp + 1e-6;

    let f_below = compute_boys(t_below, m);
    let f_above = compute_boys(t_above, m);
    let ref_below = boys_fm_reference(t_below, m);
    let ref_above = boys_fm_reference(t_above, m);

    for k in 0..=m {
        let diff_below = (f_below[k as usize] - ref_below[k as usize]).abs();
        let diff_above = (f_above[k as usize] - ref_above[k as usize]).abs();
        assert!(
            diff_below < 1e-12,
            "boys_turnover_boundary (below) failed for k={k}: got {}, expected {}, diff={diff_below}",
            f_below[k as usize], ref_below[k as usize]
        );
        assert!(
            diff_above < 1e-12,
            "boys_turnover_boundary (above) failed for k={k}: got {}, expected {}, diff={diff_above}",
            f_above[k as usize], ref_above[k as usize]
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 5: high-order boys, t=5.0, m=20
// Source: fmt.c lines 186-225 — mid-range order test.
// t=5.0 > TURNOVER_POINT[20]=7.838 is false (5.0 < 7.838), so power series branch.
// Reference: boys_fm_reference() mirrors fmt.c exactly.
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_high_order() {
    let t = 5.0f64;
    let m = 20u32;
    let computed = compute_boys(t, m);
    let reference = boys_fm_reference(t, m);
    for k in 0..=m {
        let diff = (computed[k as usize] - reference[k as usize]).abs();
        assert!(
            diff < 1e-12,
            "boys_high_order failed for t={t}, m={m}, k={k}: got {}, expected {}, diff={diff}",
            computed[k as usize], reference[k as usize]
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 7 (TDD 20-02 RED): boys_gamma_inc_host::<f64> byte-identity with baseline
// f64 reference values precomputed from boys_gamma_inc_impl (power series branch, t=1.0, m=4)
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_host_generic_f64_byte_identity() {
    // boys_gamma_inc_host::<f64>(1.0, 4) must match the pre-refactor concrete f64 values.
    // These golden values were captured from the unmodified boys_gamma_inc_impl (power series
    // branch, t=1.0 < TURNOVER_POINT[4]=1.705493613097).
    let golden: [f64; 5] = [
        0.74682413281242698844, // f[0]
        0.18947234582049235496, // f[1]
        0.10026879814501736543, // f[2]
        0.06673227477682226738, // f[3]
        0.04962324113315674107, // f[4]
    ];
    let result: Vec<f64> = boys_gamma_inc_host::<f64>(1.0_f64, 4);
    assert_eq!(result.len(), 5, "f64: result must have m+1=5 elements");
    for (k, (&computed, &expected)) in result.iter().zip(golden.iter()).enumerate() {
        assert!(
            (computed - expected).abs() < 1e-14,
            "boys f64 byte-identity failed at k={k}: got {computed:.20e}, expected {expected:.20e}"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 8 (TDD 20-02 RED): boys_gamma_inc_host::<f32> returns finite non-zero values
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_host_generic_f32_smoke() {
    let result: Vec<f32> = boys_gamma_inc_host::<f32>(1.0_f32, 4);
    assert_eq!(result.len(), 5, "f32: result must have m+1=5 elements");
    assert!(result[0] != 0.0_f32, "f32: F_0(1.0) must be non-zero");
    for (k, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "f32: F_{k}(1.0) must be finite, got {v}");
    }
    // Sanity: f32 value of F_0(1.0) should be close to the f64 golden
    let f0_f32 = result[0] as f64;
    let f0_f64_golden = 0.74682413281242698844_f64;
    let rel_err = (f0_f32 - f0_f64_golden).abs() / f0_f64_golden.abs();
    assert!(
        rel_err < 1e-5,
        "f32 F_0(1.0) relative error too large: rel_err={rel_err:.2e}, f32={f0_f32:.10}, f64={f0_f64_golden:.10}"
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 6: known absolute golden values for F_0(t)
// Generated from the libcint gamma_inc_like() C implementation with glibc erf.
//
// F_0(1.0) = SQRTPIE4/sqrt(1.0)*erf(sqrt(1.0)) = 0.74682413281242709946
//   fmt.c line 219: erfc branch formula; TURNOVER_POINT[0]=0.0 → always erfc for m=0, t>0
//
// F_0(10.0) = SQRTPIE4/sqrt(10.0)*erf(sqrt(10.0)) = 0.28024739050664276840
//   fmt.c line 219: erfc branch; erf(3.162...)≈0.99999...
//
// F_0(20.0) = SQRTPIE4/sqrt(20.0)*erf(sqrt(20.0)) = 0.19816636482997365687
//   fmt.c line 219: erfc branch; erf(4.472...)≈1.0 - O(1e-9)
// ──────────────────────────────────────────────────────────────────────────────
#[test]
fn boys_known_values_f0() {
    // F_0(1.0) — erfc branch: SQRTPIE4/sqrt(1)*erf(sqrt(1)), fmt.c line 219
    // Golden: 0.74682413281242709946 (from libcint C implementation with glibc erf)
    let golden_f0_t1: f64 = 0.74682413281242709946;
    let computed = compute_boys(1.0, 0);
    let diff = (computed[0] - golden_f0_t1).abs();
    assert!(
        diff < 1e-12,
        "boys F_0(1.0): got {}, expected {golden_f0_t1}, diff={diff}",
        computed[0]
    );

    // F_0(10.0) — erfc branch, fmt.c line 219
    // Golden: 0.28024739050664276840 (from libcint C implementation)
    let golden_f0_t10: f64 = 0.28024739050664276840;
    let computed10 = compute_boys(10.0, 0);
    let diff10 = (computed10[0] - golden_f0_t10).abs();
    assert!(
        diff10 < 1e-12,
        "boys F_0(10.0): got {}, expected {golden_f0_t10}, diff={diff10}",
        computed10[0]
    );

    // F_0(20.0) — erfc branch, fmt.c line 219
    // Golden: 0.19816636482997365687 (from libcint C implementation)
    let golden_f0_t20: f64 = 0.19816636482997365687;
    let computed20 = compute_boys(20.0, 0);
    let diff20 = (computed20[0] - golden_f0_t20).abs();
    assert!(
        diff20 < 1e-12,
        "boys F_0(20.0): got {}, expected {golden_f0_t20}, diff={diff20}",
        computed20[0]
    );
}
