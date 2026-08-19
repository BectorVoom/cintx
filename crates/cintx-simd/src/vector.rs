use std::ops::{Add, Div, Mul, Neg, Sub};
use wide::{f32x4, f32x8, f64x2, f64x4};

/// Unified SIMD float abstraction trait supporting both scalar and SIMD vector types.
pub trait SimdFloat:
    Copy
    + Clone
    + std::fmt::Debug
    + PartialEq
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Scalar underlying element type (`f64` or `f32`).
    type Scalar: Copy + Clone + std::fmt::Debug + PartialEq + Default + Into<f64>;

    /// Number of SIMD lanes in this vector type.
    const LANES: usize;

    /// Broadcast a single scalar value to all SIMD lanes.
    fn splat(val: Self::Scalar) -> Self;

    /// Load up to LANES elements from a slice of `f64`. If slice has fewer than LANES elements,
    /// pad with `pad_val`.
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self;

    /// Store elements into a mutable slice of `f64`.
    fn store_to_f64_slice(self, dst: &mut [f64]);

    /// Element-wise square root.
    fn sqrt(self) -> Self;

    /// Element-wise exponential.
    fn exp(self) -> Self;

    /// Element-wise natural logarithm.
    fn ln(self) -> Self;

    /// Element-wise error function.
    fn erf(self) -> Self;

    /// Element-wise complementary error function.
    fn erfc(self) -> Self;

    /// Element-wise sine.
    fn sin(self) -> Self;

    /// Element-wise cosine.
    fn cos(self) -> Self;

    /// Element-wise log-gamma function.
    fn lgamma(self) -> Self;

    /// Element-wise power ($x^y$).
    fn pow(self, y: Self) -> Self;

    /// Element-wise reciprocal (1 / x).
    fn recip(self) -> Self;

    /// Element-wise absolute value.
    fn abs(self) -> Self;

    /// Horizontal sum across all SIMD lanes.
    fn reduce_add(self) -> Self::Scalar;

    /// Helper to convert integer $n$ into a splatted vector.
    fn from_usize(n: usize) -> Self;

    /// Helper to convert f64 constant into a splatted vector.
    fn from_f64(v: f64) -> Self;
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementations for `wide` SIMD vector types (f64)
// ─────────────────────────────────────────────────────────────────────────────

impl SimdFloat for f64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: f64) -> Self {
        f64x4::splat(val)
    }

    #[inline(always)]
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self {
        let mut arr = [pad_val; 4];
        let count = slice.len().min(4);
        arr[..count].copy_from_slice(&slice[..count]);
        f64x4::new(arr)
    }

    #[inline(always)]
    fn store_to_f64_slice(self, dst: &mut [f64]) {
        let arr = self.to_array();
        let count = dst.len().min(4);
        dst[..count].copy_from_slice(&arr[..count]);
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        rmath::sqrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        rmath::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        rmath::ln(self)
    }

    #[inline(always)]
    fn erf(self) -> Self {
        rmath::erf(self)
    }

    #[inline(always)]
    fn erfc(self) -> Self {
        rmath::erfc(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        rmath::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        rmath::cos(self)
    }

    #[inline(always)]
    fn lgamma(self) -> Self {
        rmath::lgamma(self)
    }

    #[inline(always)]
    fn pow(self, y: Self) -> Self {
        rmath::pow(self, y)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f64x4::splat(1.0) / self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let arr = self.to_array();
        f64x4::new([arr[0].abs(), arr[1].abs(), arr[2].abs(), arr[3].abs()])
    }

    #[inline(always)]
    fn reduce_add(self) -> f64 {
        let arr = self.to_array();
        arr[0] + arr[1] + arr[2] + arr[3]
    }

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        f64x4::splat(n as f64)
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        f64x4::splat(v)
    }
}

impl SimdFloat for f64x2 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline(always)]
    fn splat(val: f64) -> Self {
        f64x2::splat(val)
    }

    #[inline(always)]
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self {
        let mut arr = [pad_val; 2];
        let count = slice.len().min(2);
        arr[..count].copy_from_slice(&slice[..count]);
        f64x2::new(arr)
    }

    #[inline(always)]
    fn store_to_f64_slice(self, dst: &mut [f64]) {
        let arr = self.to_array();
        let count = dst.len().min(2);
        dst[..count].copy_from_slice(&arr[..count]);
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        rmath::sqrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        rmath::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        rmath::ln(self)
    }

    #[inline(always)]
    fn erf(self) -> Self {
        rmath::erf(self)
    }

    #[inline(always)]
    fn erfc(self) -> Self {
        rmath::erfc(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        rmath::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        rmath::cos(self)
    }

    #[inline(always)]
    fn lgamma(self) -> Self {
        rmath::lgamma(self)
    }

    #[inline(always)]
    fn pow(self, y: Self) -> Self {
        rmath::pow(self, y)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f64x2::splat(1.0) / self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let arr = self.to_array();
        f64x2::new([arr[0].abs(), arr[1].abs()])
    }

    #[inline(always)]
    fn reduce_add(self) -> f64 {
        let arr = self.to_array();
        arr[0] + arr[1]
    }

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        f64x2::splat(n as f64)
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        f64x2::splat(v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementations for `wide` SIMD vector types (f32)
// ─────────────────────────────────────────────────────────────────────────────

impl SimdFloat for f32x4 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        f32x4::splat(val)
    }

    #[inline(always)]
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self {
        let mut arr = [pad_val as f32; 4];
        let count = slice.len().min(4);
        for i in 0..count {
            arr[i] = slice[i] as f32;
        }
        f32x4::new(arr)
    }

    #[inline(always)]
    fn store_to_f64_slice(self, dst: &mut [f64]) {
        let arr = self.to_array();
        let count = dst.len().min(4);
        for i in 0..count {
            dst[i] = arr[i] as f64;
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        rmath::sqrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        rmath::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        rmath::ln(self)
    }

    #[inline(always)]
    fn erf(self) -> Self {
        rmath::erf(self)
    }

    #[inline(always)]
    fn erfc(self) -> Self {
        rmath::erfc(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        rmath::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        rmath::cos(self)
    }

    #[inline(always)]
    fn lgamma(self) -> Self {
        rmath::lgamma(self)
    }

    #[inline(always)]
    fn pow(self, y: Self) -> Self {
        rmath::pow(self, y)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f32x4::recip(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f32x4::abs(self)
    }

    #[inline(always)]
    fn reduce_add(self) -> f32 {
        f32x4::reduce_add(self)
    }

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        f32x4::splat(n as f32)
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        f32x4::splat(v as f32)
    }
}

impl SimdFloat for f32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        f32x8::splat(val)
    }

    #[inline(always)]
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self {
        let mut arr = [pad_val as f32; 8];
        let count = slice.len().min(8);
        for i in 0..count {
            arr[i] = slice[i] as f32;
        }
        f32x8::new(arr)
    }

    #[inline(always)]
    fn store_to_f64_slice(self, dst: &mut [f64]) {
        let arr = self.to_array();
        let count = dst.len().min(8);
        for i in 0..count {
            dst[i] = arr[i] as f64;
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        rmath::sqrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        rmath::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        rmath::ln(self)
    }

    #[inline(always)]
    fn erf(self) -> Self {
        rmath::erf(self)
    }

    #[inline(always)]
    fn erfc(self) -> Self {
        rmath::erfc(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        rmath::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        rmath::cos(self)
    }

    #[inline(always)]
    fn lgamma(self) -> Self {
        rmath::lgamma(self)
    }

    #[inline(always)]
    fn pow(self, y: Self) -> Self {
        rmath::pow(self, y)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f32x8::splat(1.0) / self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let arr = self.to_array();
        f32x8::new([
            arr[0].abs(),
            arr[1].abs(),
            arr[2].abs(),
            arr[3].abs(),
            arr[4].abs(),
            arr[5].abs(),
            arr[6].abs(),
            arr[7].abs(),
        ])
    }

    #[inline(always)]
    fn reduce_add(self) -> f32 {
        f32x8::reduce_add(self)
    }

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        f32x8::splat(n as f32)
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        f32x8::splat(v as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar implementations (LANES = 1) for baseline reference & unaligned tails
// ─────────────────────────────────────────────────────────────────────────────

impl SimdFloat for f64 {
    type Scalar = f64;
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(val: f64) -> Self {
        val
    }

    #[inline(always)]
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self {
        if !slice.is_empty() {
            slice[0]
        } else {
            pad_val
        }
    }

    #[inline(always)]
    fn store_to_f64_slice(self, dst: &mut [f64]) {
        if !dst.is_empty() {
            dst[0] = self;
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        rmath::sqrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        rmath::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        rmath::ln(self)
    }

    #[inline(always)]
    fn erf(self) -> Self {
        rmath::erf(self)
    }

    #[inline(always)]
    fn erfc(self) -> Self {
        rmath::erfc(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        rmath::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        rmath::cos(self)
    }

    #[inline(always)]
    fn lgamma(self) -> Self {
        rmath::lgamma(self)
    }

    #[inline(always)]
    fn pow(self, y: Self) -> Self {
        rmath::pow(self, y)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn reduce_add(self) -> f64 {
        self
    }

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f64
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl SimdFloat for f32 {
    type Scalar = f32;
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(val: f32) -> Self {
        val
    }

    #[inline(always)]
    fn from_f64_slice(slice: &[f64], pad_val: f64) -> Self {
        if !slice.is_empty() {
            slice[0] as f32
        } else {
            pad_val as f32
        }
    }

    #[inline(always)]
    fn store_to_f64_slice(self, dst: &mut [f64]) {
        if !dst.is_empty() {
            dst[0] = self as f64;
        }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        rmath::sqrt(self)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        rmath::exp(self)
    }

    #[inline(always)]
    fn ln(self) -> Self {
        rmath::ln(self)
    }

    #[inline(always)]
    fn erf(self) -> Self {
        rmath::erf(self)
    }

    #[inline(always)]
    fn erfc(self) -> Self {
        rmath::erfc(self)
    }

    #[inline(always)]
    fn sin(self) -> Self {
        rmath::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        rmath::cos(self)
    }

    #[inline(always)]
    fn lgamma(self) -> Self {
        rmath::lgamma(self)
    }

    #[inline(always)]
    fn pow(self, y: Self) -> Self {
        rmath::pow(self, y)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn reduce_add(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f32
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
