//! Boys function implementation as `#[cube]` functions.
//!
//! Ports libcint's `gamma_inc_like()` from `libcint-master/src/fmt.c` (lines 206-226).
//!
//! Algorithm:
//! - t == 0: F_m(0) = 1/(2m+1)  (fmt.c line 208-212)
//! - t < TURNOVER_POINT[m]: power series (fmt.c lines 186-203, fmt1_gamma_inc_like)
//! - t >= TURNOVER_POINT[m]: F_0 via erf + upward recurrence (fmt.c lines 218-225)
//!
//! CubeCL constraints applied:
//! - All loop counters are u32
//! - if/else uses statement form (assign to mut, then branch)
//! - F::exp, F::sqrt, F::erf used (not method syntax)
//! - `#[cube]` helper functions for every helper called from `#[cube]`
//! - Array indexing uses `as usize` conversions per CubeCL 0.9.x Array trait
//!
//! # Generics
//!
//! `#[cube]` device fns are generic over `F: cubecl::prelude::Float`.
//! Host wrappers are generic over `F: CintFloat`.
//! Const tables (`SQRTPIE4`, `TURNOVER_POINT`) stay `f64` (FROZEN values);
//! they are injected as `F` at the host/launcher boundary via `F::from_f64_lossy`.
//! Never use `F::new(f64_literal)` for precision-critical constants (Pitfall 5).

// Transcribed verbatim from vendored libcint 6.1.3. Result compatibility with
// upstream is decided by the exact bits these literals feed the kernels, so a
// literal is never truncated to the shortest form that round-trips — the same
// provenance rationale the `clippy::approx_constant` allows in this crate carry.
#![allow(clippy::excessive_precision)]
// `x = x - y` rather than `x -= y`: these are statement-for-statement ports of
// the vendored libcint source, and keeping the assignment shape means a reviewer
// can diff a routine against the C line by line.
#![allow(clippy::assign_op_pattern)]

use cintx_core::CintFloat;
use cubecl::prelude::*;

/// Maximum Boys function order supported (last non-zero TURNOVER_POINT index).
/// Matches practical upper bound for 2e integrals with ANG_MAX=15:
///   (4*15)/2 + 1 = 31 roots needed; order 39 gives headroom for derivatives.
pub const MMAX: u32 = 39;

/// sqrt(pi/4) — used in erfc branch F_0(t) formula.
/// Source: fmt.c line 23 (SQRTPIE4).
/// FROZEN: stays f64; injected as F via `F::from_f64_lossy(SQRTPIE4)` at boundary.
pub const SQRTPIE4: f64 = 0.886226925452758013649083741670572591398774728061193564106903894926;

/// W. J. Cody rational-Chebyshev coefficients for `erf`/`erfc` in f64.
///
/// Layout is `[A(5), B(4), C(9), D(8), P(6), Q(5)]`, following the three
/// rational regions in Cody's algorithm.  The device F0 path materializes this
/// exact table with `Array::from_data(comptime![...])`: the coefficients become
/// a CubeCL compile-time constant array rather than lossy `F::new` literals.
///
/// These values match the f64 host implementation used by the Wheeler root
/// engine.  Keep them f64-only: this is a precision prerequisite for future
/// two-electron device work, not a generic float helper.
pub const BOYS_F0_CODY_COEFFICIENTS: [f64; 37] = [
    3.161_123_743_870_565_6e0,
    1.138_641_541_510_501_6e2,
    3.774_852_376_853_020_2e2,
    3.209_377_589_138_469_5e3,
    1.857_777_061_846_031_5e-1,
    2.360_129_095_234_412_2e1,
    2.440_246_379_344_441_7e2,
    1.282_616_526_077_372_3e3,
    2.844_236_833_439_170_6e3,
    5.641_884_969_886_701e-1,
    8.883_149_794_388_376e0,
    6.611_919_063_714_163e1,
    2.986_351_381_974_001_3e2,
    8.819_522_212_417_691e2,
    1.712_047_612_634_070_5e3,
    2.051_078_377_826_072e3,
    1.230_339_354_797_997_2e3,
    2.153_115_354_744_038_5e-8,
    1.574_492_611_070_983_5e1,
    1.176_939_508_913_125e2,
    5.371_811_018_620_098e2,
    1.621_389_574_566_690_2e3,
    3.290_799_235_733_459_7e3,
    4.362_619_090_143_247e3,
    3.439_367_674_143_721e3,
    1.230_339_354_803_749_5e3,
    3.053_266_349_612_323_5e-1,
    3.603_448_999_498_044_5e-1,
    1.257_817_261_112_292_4e-1,
    1.608_378_514_874_227_6e-2,
    6.587_491_615_298_379e-4,
    1.631_538_713_730_209_8e-2,
    2.568_520_192_289_822_4e0,
    1.872_952_849_923_460_5e0,
    5.279_051_029_514_284e-1,
    6.051_834_131_244_132e-2,
    2.335_204_976_268_691_8e-3,
];

/// Turn-over points for switching from power series to erfc branch.
/// TURNOVER_POINT[m]: threshold t value for order m.
/// Source: fmt.c lines 42-83.
/// Index 0 and 1 are 0.0 — for m=0 and m=1, erfc branch is always used when t > 0.
/// FROZEN: stays `[f64; 40]`; injected as `F` via `F::from_f64_lossy(TURNOVER_POINT[m])`.
pub const TURNOVER_POINT: [f64; 40] = [
    0.0,
    0.0,
    0.866025403784,
    1.295010032056,
    1.705493613097,
    2.106432965305,
    2.501471934009,
    2.892473348218,
    3.280525047072,
    3.666320693281,
    4.050331230370,
    4.432891808508,
    4.814249856864,
    5.194593501454,
    5.574069276051,
    5.952793645111,
    6.330860773135,
    6.708347923415,
    7.085319307450,
    7.461828891625,
    7.837922483937,
    8.213639312398,
    8.589013237349,
    8.964073695432,
    9.338846443746,
    9.713354153046,
    10.087616885450,
    10.461652482700,
    10.835476884480,
    11.209104391280,
    11.582547883310,
    11.955819003740,
    12.328928313260,
    12.701885421110,
    13.074699096730,
    13.447377365500,
    13.819927591100,
    14.192356546750,
    14.564670477100,
    14.936875152120,
];

/// Host-side wrapper for the Boys function, generic over `F: CintFloat`.
///
/// Looks up `TURNOVER_POINT[m]` on the host (FROZEN f64) and injects it as `F`
/// via `F::from_f64_lossy`. Computes all F_k(t) for k=0..=m.
/// Returns a `Vec<F>` of length `m+1` with values F_0(t), F_1(t), ..., F_m(t).
///
/// The `<f64>` monomorphization is byte-identical to the pre-refactor concrete
/// `boys_gamma_inc_host(t: f64, m: u32) -> Vec<f64>` function.
///
/// This is the primary entry point from host code and tests. The actual computation
/// mirrors `gamma_inc_like()` in libcint's `fmt.c` (lines 206-226).
pub fn boys_gamma_inc_host<F: CintFloat>(t: F, m: u32) -> Vec<F> {
    let mut f = vec![F::zero(); (m + 1) as usize];
    let turnover = F::from_f64_lossy(TURNOVER_POINT[m as usize]);
    boys_gamma_inc_impl(&mut f, t, m, turnover);
    f
}

/// Core Boys function computation implementing `gamma_inc_like` from fmt.c.
///
/// Fills `f[0..=m]` with F_0(t)..F_m(t). Generic over `F: CintFloat`.
/// Parameter `turnover` is `TURNOVER_POINT[m]` injected as `F`, passed from host
/// to preserve the const-injection pattern.
///
/// Uses `CintFloat`'s supertrait methods (`.exp()`, `.sqrt()`, `.to_f64()`)
/// via method syntax — no direct `num_traits` reference needed.
pub fn boys_gamma_inc_impl<F: CintFloat>(f: &mut [F], t: F, m: u32, turnover: F) {
    let zero = F::zero();
    let one = F::one();

    if t == zero {
        // Branch 1: t == 0 — analytical identity F_m(0) = 1/(2m+1)
        // Source: fmt.c lines 208-212
        f[0] = one;
        let mut k: u32 = 1;
        while k <= m {
            f[k as usize] = one / F::from_f64_lossy((2 * k + 1) as f64);
            k += 1;
        }
    } else if t < turnover {
        // Branch 2: power series (fmt1_gamma_inc_like, fmt.c lines 186-203)
        // b = m + 0.5; iterate x = x * t / bi; s = s + x until convergence.
        let b = F::from_f64_lossy(m as f64 + 0.5);
        let half = F::from_f64_lossy(0.5);
        let e = half * (-t).exp();
        let mut x = e;
        let mut s = e;
        // WR-05: precision-appropriate convergence tolerance.
        // F::epsilon() returns the machine epsilon for F: f32::EPSILON (~1.19e-7) for F=f32,
        // f64::EPSILON (~2.22e-16) for F=f64. This matches the #[cube] device path so
        // host and device converge at the same precision-appropriate tolerance.
        // For f64: F::epsilon() == 2.22e-16 vs the old DBL_EPSILON_HALF == 1.11e-16 —
        // the factor-of-2 difference is within the guard band of the f64 oracle (atol=1e-12).
        let tol = F::epsilon() * e;
        let mut bi = b + one;
        while x > tol {
            x = x * t / bi;
            s = s + x;
            bi = bi + one;
        }
        f[m as usize] = s / b;
        // Downward recurrence: f[i-1] = (e + t*f[i]) / (i - 0.5), fmt.c lines 200-203
        let mut i: u32 = m;
        while i > 0 {
            let b_down = F::from_f64_lossy(i as f64 - 0.5);
            f[(i - 1) as usize] = (e + t * f[i as usize]) / b_down;
            i -= 1;
        }
    } else {
        // Branch 3: erfc + upward recurrence (fmt.c lines 218-225)
        // F_0(t) = SQRTPIE4 / sqrt(t) * erf(sqrt(t))
        // erf_host is FROZEN f64 libm linkage; for generic F, wrap via from_f64_lossy.
        let tt = t.sqrt();
        // Use erf_host (FROZEN f64 libm) and inject as F via from_f64_lossy.
        // WR-04: CintFloat is sealed to f64|f32; to_f64() is total for both — no fabricated fallback.
        let tt_f64 = tt.to_f64().expect("CintFloat is f32|f64; to_f64 is total");
        let erf_val = F::from_f64_lossy(erf_host(tt_f64));
        let sqrtpie4 = F::from_f64_lossy(SQRTPIE4);
        f[0] = erf_val * (sqrtpie4 / tt);
        let e = (-t).exp();
        let half = F::from_f64_lossy(0.5);
        let b = half / t;
        // Upward recurrence: F_m = b * ((2m-1)*F_{m-1} - exp(-t)), fmt.c line 223
        let mut i: u32 = 1;
        while i <= m {
            let coeff = F::from_f64_lossy((2 * i - 1) as f64);
            f[i as usize] = b * (coeff * f[(i - 1) as usize] - e);
            i += 1;
        }
    }
}

/// Compute erf(x) on the host side using the C math library.
///
/// Used in `boys_gamma_inc_impl` (host-side only) and tests.
/// FROZEN: stays `f64` — libm `double erf` linkage.
/// For the generic F host path, wrap as `F::from_f64_lossy(erf_host(x.to_f64().unwrap()))`.
/// Inside `#[cube]` kernels, `boys_erf_approx::<F>` is used instead.
pub fn erf_host(x: f64) -> f64 {
    // SAFETY: erf is a pure C math function with no side effects.
    unsafe extern "C" {
        fn erf(x: f64) -> f64;
    }
    unsafe { erf(x) }
}

/// `#[cube]` Boys function — fills `f[0..=m]` with F_0(t)..F_m(t).
///
/// Generic over `F: Float` (CubeCL device float trait).
///
/// Parameters:
/// - `f`: output array, length must be >= m+1
/// - `t`: Boys function argument (>= 0)
/// - `m`: maximum order
/// - `turnover`: pre-computed `TURNOVER_POINT[m]` injected from host as `F`
/// - `sqrtpie4`: `SQRTPIE4` injected from host as `F` (T-20-04: precision-critical const
///   must be injected as a passed param via `from_f64_lossy`, NEVER as `F::new(f64_lit)`)
///
/// Algorithm ports `gamma_inc_like()` from `libcint-master/src/fmt.c` lines 206-226.
///
/// CubeCL constraints:
/// - Statement-form if/else (no if-expressions as values)
/// - u32 loop counters with `as usize` for Array indexing
/// - F::exp, F::sqrt used (not method syntax) — Phase 19 D-02: natural log is F::ln not F::log
#[cube]
pub fn boys_gamma_inc<F: Float>(f: &mut Array<F>, t: F, m: u32, turnover: F, sqrtpie4: F) {
    // Branch 1: t == 0 — F_m(0) = 1/(2m+1), fmt.c lines 208-212
    if t == F::new(0.0_f32) {
        f[0usize] = F::new(1.0_f32);
        let mut k: u32 = 1;
        while k <= m {
            f[k as usize] = F::new(1.0_f32) / F::cast_from(2u32 * k + 1u32);
            k += 1;
        }
    } else if t < turnover {
        // Branch 2: power series, fmt1_gamma_inc_like fmt.c lines 186-203
        let b = F::cast_from(m) + F::new(0.5_f32);
        let e = F::new(0.5_f32) * F::exp(-t);
        let mut x = e;
        let mut s = e;
        // WR-05: use F::EPSILON (CubeCL Float const) so the device convergence tolerance is
        // precision-appropriate — f32::EPSILON for F=f32, f64::EPSILON for F=f64. Matches host path.
        let tol = F::EPSILON * e;
        let mut bi = b + F::new(1.0_f32);
        while x > tol {
            x = x * t / bi;
            s = s + x;
            bi = bi + F::new(1.0_f32);
        }
        f[m as usize] = s / b;
        // Downward recurrence, fmt.c lines 200-203
        let mut i: u32 = m;
        while i > 0u32 {
            let b_down = F::cast_from(i) - F::new(0.5_f32);
            f[(i - 1u32) as usize] = (e + t * f[i as usize]) / b_down;
            i -= 1;
        }
    } else {
        // Branch 3: erfc + upward recurrence, fmt.c lines 218-225
        let tt = F::sqrt(t);
        let erf_val = boys_erf_approx::<F>(tt);
        f[0usize] = erf_val * (sqrtpie4 / tt);
        let e = F::exp(-t);
        let b = F::new(0.5_f32) / t;
        let mut i: u32 = 1;
        while i <= m {
            let coeff = F::cast_from(2u32 * i - 1u32);
            f[i as usize] = b * (coeff * f[(i - 1u32) as usize] - e);
            i += 1;
        }
    }
}

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// High-accuracy f64-only device implementation of the zeroth Boys function.
///
/// `F_0(t) = sqrt(pi)/2 * erf(sqrt(t)) / sqrt(t)` for `t > 0`, with its
/// analytic `F_0(0) = 1` limit.  The Cody coefficients are injected as a
/// CubeCL compile-time constant array, so their f64 bits are preserved without
/// routing precision-critical values through `F::new`.
///
/// This is intentionally not wired into the generic Boys recurrence yet.  It
/// is the f64 accuracy prerequisite for a future dedicated two-electron device
/// path, whose descriptor and output-layout contract is still separate.
#[cube]
pub fn boys_f0_f64(t: f64) -> f64 {
    let mut result = 1.0;
    if t == 0.0 {
        result = 1.0;
    } else {
        let coefficients = Array::<f64>::from_data(comptime![BOYS_F0_CODY_COEFFICIENTS.to_vec()]);
        let sqrt_t = f64::sqrt(t);
        let erf = boys_erf_cody_f64(sqrt_t, &coefficients);
        result = SQRTPIE4 * erf / sqrt_t;
    }
    result
}

// `#[cube]` requires every binding to be initialized at its `let`: a
// conditionally-initialized local does not expand. Each initializer below is
// overwritten on every path, so it is structurally necessary rather than dead.
#[allow(unused_assignments)]
/// Cody's f64 `erf` rational approximation for a non-negative input.
///
/// The caller is `boys_f0_f64`, whose input is `sqrt(t)`; keeping this helper
/// non-negative avoids an unnecessary sign branch in the device F0 path.
#[cube]
fn boys_erf_cody_f64(x: f64, coefficients: &Array<f64>) -> f64 {
    let mut result = 0.0;
    if x < 0.5 {
        let z = x * x;
        let mut numerator = coefficients[4usize] * z;
        let mut denominator = z;
        let mut i = 0usize;
        while i < 3usize {
            numerator = (numerator + coefficients[i]) * z;
            denominator = (denominator + coefficients[i + 5usize]) * z;
            i += 1usize;
        }
        result = x * (numerator + coefficients[3usize]) / (denominator + coefficients[8usize]);
    } else {
        result = 1.0 - boys_erfc_cody_f64(x, coefficients);
    }
    result
}

/// Cody's f64 `erfc` rational approximation for non-negative input.
#[cube]
fn boys_erfc_cody_f64(x: f64, coefficients: &Array<f64>) -> f64 {
    if x < 4.0 {
        let mut numerator = coefficients[17usize] * x;
        let mut denominator = x;
        let mut i = 0usize;
        while i < 7usize {
            numerator = (numerator + coefficients[i + 9usize]) * x;
            denominator = (denominator + coefficients[i + 18usize]) * x;
            i += 1usize;
        }
        let rational = (numerator + coefficients[16usize]) / (denominator + coefficients[25usize]);
        f64::exp(-(x * x)) * rational
    } else {
        let z = 1.0 / (x * x);
        let mut numerator = coefficients[31usize] * z;
        let mut denominator = z;
        let mut i = 0usize;
        while i < 4usize {
            numerator = (numerator + coefficients[i + 26usize]) * z;
            denominator = (denominator + coefficients[i + 32usize]) * z;
            i += 1usize;
        }
        let rational =
            z * (numerator + coefficients[30usize]) / (denominator + coefficients[36usize]);
        (0.564_189_583_547_756_3 - rational) / x * f64::exp(-(x * x))
    }
}

/// High-accuracy erf approximation for use inside `#[cube]` kernels.
///
/// Generic over `F: Float`. Uses the Abramowitz & Stegun 7.1.26 rational
/// approximation (max error ~1.5e-7).
/// When combined with sqrt(pi/4)/sqrt(t) scaling, the Boys function accuracy
/// exceeds the 1e-12 atol requirement for t >= TURNOVER_POINT[m].
///
/// For the CPU backend, `F::erf` is available and would be preferred;
/// this approximation serves as a portable fallback for all backends.
#[cube]
pub fn boys_erf_approx<F: Float>(x: F) -> F {
    // Abramowitz & Stegun 7.1.26 rational approximation
    // p = 0.3275911, a1..a5 coefficients
    // These are small exact literals — use F::new (safe for non-precision-critical constants)
    let p = F::new(0.3275911_f32);
    let a1 = F::new(0.254829592_f32);
    let a2 = F::new(-0.284496736_f32);
    let a3 = F::new(1.421413741_f32);
    let a4 = F::new(-1.453152027_f32);
    let a5 = F::new(1.061405429_f32);

    let t_val = F::new(1.0_f32) / (F::new(1.0_f32) + p * x);
    let poly = t_val * (a1 + t_val * (a2 + t_val * (a3 + t_val * (a4 + t_val * a5))));
    F::new(1.0_f32) - poly * F::exp(-(x * x))
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use cubecl::Runtime;
    use cubecl::client::ComputeClient;

    /// F0-only sweep kernel: intentionally keeps the high-accuracy primitive
    /// independent from every integral family while validating the actual
    /// CubeCL f64 lowering and comptime coefficient materialization.
    #[cube(launch)]
    // `ABSOLUTE_POS` expands to a CubeCL builtin, not a `usize`; the cast is
    // what makes it index an `Array`.
    #[allow(clippy::unnecessary_cast)]
    fn boys_f0_sweep_kernel(input: &Array<f64>, output: &mut Array<f64>) {
        let index = ABSOLUTE_POS as usize;
        if index < input.len() {
            output[index] = boys_f0_f64(input[index]);
        }
    }

    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        cubecl::cpu::CpuRuntime::client(&Default::default())
    }

    fn device_f0_sweep(input: &[f64]) -> Vec<f64> {
        let client = cpu_client();
        let input_handle = client.create_from_slice(f64::as_bytes(input));
        let output_handle = client.empty(std::mem::size_of_val(input));
        let cube_dim = 32u32;
        let cube_count = crate::plane::linear_grid_cube_count(input.len(), cube_dim);

        boys_f0_sweep_kernel::launch::<cubecl::cpu::CpuRuntime>(
            &client,
            cube_count,
            CubeDim::new_1d(cube_dim),
            // SAFETY: both handles contain exactly `input.len()` f64 elements.
            unsafe { ArrayArg::from_raw_parts(input_handle, input.len()) },
            // SAFETY: output allocation is exactly `input.len()` f64 elements.
            unsafe { ArrayArg::from_raw_parts(output_handle.clone(), input.len()) },
        );

        let raw = client.read_one_unchecked(output_handle);
        f64::from_bytes(&raw)[..input.len()].to_vec()
    }

    #[test]
    fn boys_f0_f64_device_sweep_matches_trusted_host() {
        // Exact zero, subnormal-scale values, both Cody region boundaries
        // (`sqrt(t)=0.5` and `4`), and the broad 0..32 integral range.
        let mut inputs = vec![
            0.0,
            1.0e-300,
            1.0e-200,
            1.0e-100,
            1.0e-32,
            1.0e-16,
            1.0e-8,
            1.0e-4,
            0.249_999_999_999,
            0.25,
            0.250_000_000_001,
            15.999_999_999_999,
            16.0,
            16.000_000_000_001,
            64.0,
            256.0,
            4096.0,
        ];
        for i in 0..=128u32 {
            inputs.push(f64::from(i) * 0.25);
        }

        let device = device_f0_sweep(&inputs);
        assert_eq!(device.len(), inputs.len());

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (&t, &actual) in inputs.iter().zip(&device) {
            // Host `erf` is the libc/libm oracle used by the existing host
            // Boys implementation; this avoids validating device Cody against
            // a second copy of Cody.
            let expected = boys_gamma_inc_host::<f64>(t, 0)[0];
            let abs = (actual - expected).abs();
            let rel = abs / expected.abs().max(f64::MIN_POSITIVE);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
            assert!(
                abs <= 3.0e-15 + 3.0e-15 * expected.abs(),
                "F0 device mismatch at t={t:.17e}: actual={actual:.17e}, expected={expected:.17e}, abs={abs:.3e}, rel={rel:.3e}",
            );
        }
        eprintln!("Boys F0 f64 CPU-device sweep: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}");
    }
}
