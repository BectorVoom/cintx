//! Precision type vocabulary for the f64/f32 precision-generic refactor (Phase 20).
//!
//! This module provides the sealed [`CintFloat`] trait and the [`PrecisionKind`] enum,
//! which together form the shared foundation for the whole compute-path generic refactor.
//!
//! # Usage boundary (IMPORTANT)
//!
//! - **Device `#[cube]` kernels** bound `F: cubecl::prelude::Float` (CubeCL expansion
//!   requires its own trait; do NOT add `+ num_traits::Float` to device function bounds —
//!   that leaks a host-only crate into device code).
//! - **Host wrappers** (`*_host::<F>` helpers) and the public `evaluate::<F>()` method
//!   bound `F: CintFloat` (sealed to exactly `f64 | f32`; includes `num_traits::Float`
//!   for host-side math).
//!
//! # Const tables
//!
//! All const tables (e.g. `SQRTPIE4`, `TURNOVER_POINT: [f64; 40]`) stay `[f64; N]`
//! with FROZEN values. They are cast to `F` at the host/launcher boundary via
//! [`CintFloat::from_f64_lossy`] — never via `const TABLE: [F; N]` (Rust forbids
//! generic const arrays) and never via `F::new(f64_literal)` for precision-critical
//! constants.

mod sealed {
    /// Private supertrait that seals [`super::CintFloat`] to exactly `f64` and `f32`.
    pub trait Sealed {}
}

/// A float type supported by the cintx compute path.
///
/// Sealed to exactly `f64` (default, byte-identity with libcint) and `f32`
/// (opt-in, ~1e-4 rtol, unlocks non-`SHADER_F64` wgpu adapters).
///
/// # Implementors
///
/// Only `f64` and `f32` implement this trait. The private `sealed::Sealed` supertrait
/// prevents external implementations.
pub trait CintFloat:
    Copy
    + Send
    + Sync
    + 'static
    + num_traits::Float
    + num_traits::FromPrimitive
    + sealed::Sealed
{
    /// The `PrecisionKind` runtime tag for this float type.
    ///
    /// `f64::PRECISION == PrecisionKind::F64` (default, byte-identity with libcint).
    /// `f32::PRECISION == PrecisionKind::F32` (opt-in, ~1e-4 rtol).
    ///
    /// This const enables zero-cost `F → PrecisionKind` mapping in generic
    /// host-side code (e.g. `evaluate::<F>()`) without runtime branching.
    const PRECISION: PrecisionKind;

    /// Convert an `f64` const-table value to `Self` at the host/staging boundary.
    ///
    /// For `f64`: returns `x` unchanged (lossless).
    /// For `f32`: truncates via `x as f32` (documented lossy at the threshold boundary
    /// only; no NaN/Inf is introduced by the conversion for physically valid inputs).
    fn from_f64_lossy(x: f64) -> Self;
}

impl sealed::Sealed for f64 {}
impl CintFloat for f64 {
    const PRECISION: PrecisionKind = PrecisionKind::F64;

    #[inline(always)]
    fn from_f64_lossy(x: f64) -> Self {
        x
    }
}

impl sealed::Sealed for f32 {}
impl CintFloat for f32 {
    const PRECISION: PrecisionKind = PrecisionKind::F32;

    #[inline(always)]
    fn from_f64_lossy(x: f64) -> Self {
        x as f32
    }
}

/// Runtime precision tag for enum dispatch.
///
/// Keeps `BackendExecutor` and related traits object-safe: instead of making
/// `BackendExecutor::execute` generic over `F`, the precision is carried as a plain
/// `PrecisionKind` field on [`cintx_runtime::ExecutionPlan`] and matched internally.
///
/// `F64` is the default and produces byte-identical results to upstream libcint 6.1.3.
/// `F32` is the opt-in path that unlocks wgpu adapters lacking `SHADER_F64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionKind {
    /// 64-bit double precision — default, byte-identical to libcint (D-08).
    #[default]
    F64,
    /// 32-bit single precision — opt-in, ~1e-4 rtol oracle gate (D-09).
    F32,
}

impl PrecisionKind {
    /// Returns the size in bytes of one element for this precision.
    ///
    /// `F64` → 8, `F32` → 4.
    pub const fn element_size(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_implements_cint_float_lossless() {
        // sqrt(pi/4) from boys.rs — exact f64 value must round-trip unchanged
        let x = 0.886226925452758_f64;
        assert_eq!(<f64 as CintFloat>::from_f64_lossy(x), x);
    }

    #[test]
    fn f32_implements_cint_float_lossy() {
        // f32 must truncate via `x as f32`; NOT via f64::from(x as f32) or any other path
        let x = 0.886226925452758_f64;
        let expected = x as f32;
        assert_eq!(<f32 as CintFloat>::from_f64_lossy(x), expected);
    }

    #[test]
    fn precision_kind_default_is_f64() {
        assert_eq!(PrecisionKind::default(), PrecisionKind::F64);
    }

    #[test]
    fn precision_kind_element_sizes() {
        assert_eq!(PrecisionKind::F64.element_size(), 8);
        assert_eq!(PrecisionKind::F32.element_size(), 4);
    }

    // Compile-negative property (not a runtime test):
    // No type other than f64/f32 can implement CintFloat because `sealed::Sealed`
    // is private to this module. Attempting:
    //   struct MyFloat; impl sealed::Sealed for MyFloat {} impl CintFloat for MyFloat {...}
    // in external code would fail to compile.
}
