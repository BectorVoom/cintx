//! Gaussian pair data computation as `#[cube]` functions.
//!
//! Ports pair data setup from `libcint-master/src/g1e.c` lines 125-183.
//!
//! The `PairData` struct holds all quantities needed for two-center integrals:
//! - `zeta_ab`: sum of exponents (ai + aj), g1e.c line 130
//! - `center_p_{x,y,z}`: weighted center (ai*Ri + aj*Rj) / zeta_ab, g1e.c lines 131-133
//! - `rirj_{x,y,z}`: displacement Ri - Rj, g1e.c line 135
//! - `fac`: pre-exponential factor exp(-ai*aj/zeta_ab * |rirj|^2) * norm_i * norm_j, g1e.c line 134
//! - `aij2`: 0.5 / zeta_ab (used in recurrence), g1e.c line 168
//!
//! Separate x/y/z scalar fields avoid CubeType array field limitations (Research Pitfall 5).
//!
//! # Generics
//!
//! `PairData` stays a concrete `f64` struct for `#[cube]` device use — CubeCL 0.10.0
//! does not support generic CubeType structs in `#[cube]` return positions when the
//! type parameter differs from the concrete default (Wave 2 limitation).
//!
//! The `#[cube]` `compute_pdata<F: Float>` device kernel is generic over `F: Float`,
//! using `PairData` (= `PairData<f64>`) as the return type with `F::cast_from` at the
//! output boundary.
//!
//! `compute_pdata_host<F: CintFloat>` accepts generic `F` inputs, performs the
//! computation in `f64` (via `CintFloat::to_f64()`) and returns the concrete `PairData`
//! (f64). This keeps all kernel callers working unchanged and allows both f64 and f32
//! inputs — the f32 path is tested to confirm it compiles and produces finite values.
//!
//! Wave 2 (kernel launchers) will genericize the full pipeline once `PairData<F>` is
//! needed end-to-end.

use cubecl::prelude::*;
use cintx_core::CintFloat;

/// Gaussian pair data — all quantities for a two-center shell pair.
///
/// Concrete f64 struct for `#[cube]` device kernels. All existing kernel callers
/// use `PairData` (without type args) — these are unchanged.
///
/// Field layout uses separate x/y/z scalars to avoid `#[derive(CubeType)]`
/// limitations with fixed-size array fields.
#[derive(CubeType, Clone, Copy)]
pub struct PairData {
    /// Sum of exponents: ai + aj. Source: g1e.c line 130.
    pub zeta_ab: f64,
    /// Weighted center x: (ai*Ri_x + aj*Rj_x) / zeta_ab. Source: g1e.c line 131.
    pub center_p_x: f64,
    /// Weighted center y: (ai*Ri_y + aj*Rj_y) / zeta_ab. Source: g1e.c line 132.
    pub center_p_y: f64,
    /// Weighted center z: (ai*Ri_z + aj*Rj_z) / zeta_ab. Source: g1e.c line 133.
    pub center_p_z: f64,
    /// Displacement x: Ri_x - Rj_x. Source: g1e.c line 135.
    pub rirj_x: f64,
    /// Displacement y: Ri_y - Rj_y. Source: g1e.c line 135.
    pub rirj_y: f64,
    /// Displacement z: Ri_z - Rj_z. Source: g1e.c line 135.
    pub rirj_z: f64,
    /// Pre-exponential factor: exp(-ai*aj/zeta_ab * |rirj|^2) * norm_i * norm_j.
    /// Source: g1e.c line 134.
    pub fac: f64,
    /// Half inverse sum: 0.5 / zeta_ab. Used in Obara-Saika recurrence.
    /// Source: g1e.c line 168.
    pub aij2: f64,
}

/// Compute pair data for two Gaussian centers inside a `#[cube]` kernel.
///
/// Generic over `F: Float` (CubeCL device float trait).
/// Uses `PairData` (concrete f64) as the return type with `F::cast_from` at the
/// output boundary. The `<f64>` monomorphization is byte-identical to the
/// pre-refactor version.
///
/// Mirrors the setup in `libcint-master/src/g1e.c` lines 125-183.
///
/// Parameters:
/// - `ai`, `aj`: exponent coefficients for shells i and j
/// - `ri_{x,y,z}`: center of shell i
/// - `rj_{x,y,z}`: center of shell j
/// - `norm_i`, `norm_j`: normalization factors
#[cube]
pub fn compute_pdata<F: Float>(
    ai: F,
    aj: F,
    ri_x: F,
    ri_y: F,
    ri_z: F,
    rj_x: F,
    rj_y: F,
    rj_z: F,
    norm_i: F,
    norm_j: F,
) -> PairData {
    // zeta_ab = ai + aj, g1e.c line 130
    let zeta_ab = ai + aj;

    // center_p = (ai*Ri + aj*Rj) / zeta_ab, g1e.c lines 131-133
    let center_p_x = (ai * ri_x + aj * rj_x) / zeta_ab;
    let center_p_y = (ai * ri_y + aj * rj_y) / zeta_ab;
    let center_p_z = (ai * ri_z + aj * rj_z) / zeta_ab;

    // rirj = Ri - Rj, g1e.c line 135
    let rirj_x = ri_x - rj_x;
    let rirj_y = ri_y - rj_y;
    let rirj_z = ri_z - rj_z;

    // rr = |rirj|^2
    let rr = rirj_x * rirj_x + rirj_y * rirj_y + rirj_z * rirj_z;

    // fac = exp(-ai*aj/zeta_ab * rr) * norm_i * norm_j, g1e.c line 134
    let fac = F::exp(-ai * aj / zeta_ab * rr) * norm_i * norm_j;

    // aij2 = 0.5 / zeta_ab, g1e.c line 168
    let aij2 = F::new(0.5) / zeta_ab;

    // Return PairData (concrete f64) via cast_from: for the f64 path this is a no-op.
    PairData {
        zeta_ab: f64::cast_from(zeta_ab),
        center_p_x: f64::cast_from(center_p_x),
        center_p_y: f64::cast_from(center_p_y),
        center_p_z: f64::cast_from(center_p_z),
        rirj_x: f64::cast_from(rirj_x),
        rirj_y: f64::cast_from(rirj_y),
        rirj_z: f64::cast_from(rirj_z),
        fac: f64::cast_from(fac),
        aij2: f64::cast_from(aij2),
    }
}

/// Host-side wrapper for `compute_pdata` — returns `PairData` (concrete f64).
///
/// Generic over `F: CintFloat`. Inputs of type `F` are converted to `f64`
/// via `CintFloat::to_f64()`, the computation runs in f64, and the result is
/// a concrete `PairData`. This keeps all kernel callers working unchanged.
///
/// The `<f64>` monomorphization is byte-identical to the pre-refactor concrete version.
/// The `<f32>` monomorphization accepts f32 inputs, computes in f32 precision,
/// and returns f64-fielded `PairData` — tested to compile and produce finite values.
///
/// Used by tests and host-side integral planning code.
pub fn compute_pdata_host<F: CintFloat>(
    ai: F,
    aj: F,
    ri_x: F,
    ri_y: F,
    ri_z: F,
    rj_x: F,
    rj_y: F,
    rj_z: F,
    norm_i: F,
    norm_j: F,
) -> PairData {
    // Compute in F precision using CintFloat's num_traits::Float methods.
    let zeta_ab = ai + aj;
    let center_p_x = (ai * ri_x + aj * rj_x) / zeta_ab;
    let center_p_y = (ai * ri_y + aj * rj_y) / zeta_ab;
    let center_p_z = (ai * ri_z + aj * rj_z) / zeta_ab;
    let rirj_x = ri_x - rj_x;
    let rirj_y = ri_y - rj_y;
    let rirj_z = ri_z - rj_z;
    let rr = rirj_x * rirj_x + rirj_y * rirj_y + rirj_z * rirj_z;
    let fac = (-ai * aj / zeta_ab * rr).exp() * norm_i * norm_j;
    let half = F::from_f64_lossy(0.5);
    let aij2 = half / zeta_ab;

    // Convert to concrete f64 PairData at the boundary.
    PairData {
        zeta_ab: zeta_ab.to_f64().unwrap_or(0.0),
        center_p_x: center_p_x.to_f64().unwrap_or(0.0),
        center_p_y: center_p_y.to_f64().unwrap_or(0.0),
        center_p_z: center_p_z.to_f64().unwrap_or(0.0),
        rirj_x: rirj_x.to_f64().unwrap_or(0.0),
        rirj_y: rirj_y.to_f64().unwrap_or(0.0),
        rirj_z: rirj_z.to_f64().unwrap_or(0.0),
        fac: fac.to_f64().unwrap_or(0.0),
        aij2: aij2.to_f64().unwrap_or(0.0),
    }
}
