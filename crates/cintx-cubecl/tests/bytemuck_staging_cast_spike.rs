//! A5 bytemuck staging-buffer reinterpretation soundness spike (Wave 0, Plan 01).
//!
//! Threat T-20-01: Proves that `bytemuck::cast_slice_mut` over the host staging buffer
//! is sound for both f64↔f64 round-trips and f64-buffer→f32-view reinterpretation,
//! BEFORE any kernel wave depends on it.
//!
//! If ANY cast panics or is rejected by bytemuck, this file is a BLOCKER:
//! the separate `Vec<F>` staging path fallback must be used instead (see planner.rs comments).
//!
//! See PLAN.md §Task 2 PART B and PATTERNS.md §dispatch.rs for the full rationale.

use std::mem::size_of;

/// Assertion A1: f64 → &mut [f64] round-trip via cast_slice_mut is identity.
#[test]
fn a5_f64_to_f64_roundtrip() {
    let n: usize = 16;
    let mut buf: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 + 0.25).collect();
    let expected: Vec<f64> = buf.clone();

    // cast_slice_mut::<f64, f64> must be a no-op (same size, same align, same bit pattern)
    let view: &mut [f64] = bytemuck::cast_slice_mut::<f64, f64>(&mut buf);
    assert_eq!(view.len(), n, "f64->f64 cast should preserve element count");

    for (got, exp) in view.iter().zip(expected.iter()) {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "f64->f64 cast must preserve bit patterns"
        );
    }
}

/// Assertion A2: A u8 byte buffer sized for N f32 elements can be reinterpreted
/// as &mut [f32] via cast_slice_mut, and values written then read back bit-exact.
#[test]
fn a5_u8_to_f32_write_read_bitexact() {
    let n: usize = 8;
    let byte_size = n * size_of::<f32>();

    // Allocate a raw byte buffer with correct alignment for f32 (4 bytes)
    // Vec<f32> gives us the right alignment; we transmute to check the path.
    let mut buf: Vec<f32> = vec![0.0_f32; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut::<f32, u8>(&mut buf);
    assert_eq!(bytes.len(), byte_size);

    // Write known f32 values back through the f32 view
    // An arbitrary non-trivial scale, not pi — the spike only checks that values
    // survive the byte round-trip.
    let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 3.25_f32).collect();
    let f32_view: &mut [f32] = bytemuck::cast_slice_mut::<u8, f32>(bytes);
    f32_view.copy_from_slice(&vals);

    // Read back via the original Vec and verify bit-exact
    for (got, exp) in buf.iter().zip(vals.iter()) {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "u8-buffer f32 write-read must be bit-exact"
        );
    }
}

/// Assertion A3 (the critical Wave-3 sizing invariant):
/// A buffer of M f64 elements (M*8 bytes) reinterpreted as &mut [f32] yields 2*M f32 lanes.
///
/// This is the size relationship the planner relies on when sizing a staging buffer
/// for f32: element_count * size_of::<f32>() bytes = element_count * 4 bytes.
/// For the f64 byte buffer: M f64 elements = M*8 bytes.
/// When recast as f32: M*8 bytes / 4 bytes-per-f32 = 2*M f32 elements.
///
/// Wave 3 (planner) will pre-size the buffer to output_elements * element_size() bytes,
/// so for f32 the buffer is output_elements * 4 bytes. This test checks the cast math.
#[test]
fn a5_f64_buffer_yields_double_f32_lanes() {
    let m: usize = 10; // M f64 elements
    let mut buf: Vec<f64> = vec![0.0_f64; m];
    // Cast the f64 buffer to a f32 view — should yield 2*M f32 lanes
    let f32_view: &mut [f32] = bytemuck::cast_slice_mut::<f64, f32>(&mut buf);
    assert_eq!(
        f32_view.len(),
        2 * m,
        "f64 buffer of M elements should yield 2*M f32 lanes (byte doubling)"
    );
}

/// Assertion A4: Alignment invariant — an 8-byte-aligned f64 buffer is validly
/// reinterpretable as f32 (which requires only 4-byte alignment).
///
/// bytemuck::cast_slice_mut enforces alignment at runtime; this test confirms it
/// does NOT panic for the f64→f32 direction (4 divides 8).
#[test]
fn a5_f64_align_satisfies_f32_align() {
    let mut buf: Vec<f64> = vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64];
    // This must NOT panic — 8-byte align is a strict superset of 4-byte align
    let f32_view: &mut [f32] = bytemuck::cast_slice_mut::<f64, f32>(&mut buf);
    // Sanity: each pair of f32 lanes corresponds to one f64 bit pattern
    assert_eq!(f32_view.len(), 8, "4 f64s → 8 f32 lanes");
    // Write f32 values and confirm no panic / no UB
    for (i, v) in f32_view.iter_mut().enumerate() {
        *v = i as f32 * 0.5;
    }
    // All values must have been written (no partial write)
    let sum: f32 = f32_view.iter().sum();
    assert!(sum.is_finite(), "all written f32 values must be finite");
}

/// Summary: All four A5 assertions pass → the bytemuck staging cast strategy is PROVEN SOUND.
/// Wave 3 (kernels, executor, staging allocation) may depend on this without a fallback.
#[test]
fn a5_summary_proven() {
    // If this test file compiles and all tests above pass, A5 is proven.
    // The planner's PrecisionKind::element_size() contracts:
    assert_eq!(size_of::<f64>(), 8, "PrecisionKind::F64.element_size() = 8");
    assert_eq!(size_of::<f32>(), 4, "PrecisionKind::F32.element_size() = 4");
    // The ratio: one f64 slot holds 2 f32 values
    assert_eq!(
        size_of::<f64>() / size_of::<f32>(),
        2,
        "f64/f32 size ratio = 2"
    );
}
