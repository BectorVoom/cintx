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

use cubecl::prelude::*;

/// Gaussian pair data — all quantities for a two-center shell pair.
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
/// Mirrors the setup in `libcint-master/src/g1e.c` lines 125-183.
///
/// Parameters:
/// - `ai`, `aj`: exponent coefficients for shells i and j
/// - `ri_{x,y,z}`: center of shell i
/// - `rj_{x,y,z}`: center of shell j
/// - `norm_i`, `norm_j`: normalization factors
#[cube]
pub fn compute_pdata(
    ai: f64,
    aj: f64,
    ri_x: f64,
    ri_y: f64,
    ri_z: f64,
    rj_x: f64,
    rj_y: f64,
    rj_z: f64,
    norm_i: f64,
    norm_j: f64,
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
    let fac = f64::exp(-ai * aj / zeta_ab * rr) * norm_i * norm_j;

    // aij2 = 0.5 / zeta_ab, g1e.c line 168
    let aij2 = 0.5f64 / zeta_ab;

    PairData {
        zeta_ab,
        center_p_x,
        center_p_y,
        center_p_z,
        rirj_x,
        rirj_y,
        rirj_z,
        fac,
        aij2,
    }
}

/// Host-side wrapper for `compute_pdata` — returns `PairData` directly.
///
/// Used by tests and host-side integral planning code.
pub fn compute_pdata_host(
    ai: f64,
    aj: f64,
    ri_x: f64,
    ri_y: f64,
    ri_z: f64,
    rj_x: f64,
    rj_y: f64,
    rj_z: f64,
    norm_i: f64,
    norm_j: f64,
) -> PairData {
    let zeta_ab = ai + aj;
    let center_p_x = (ai * ri_x + aj * rj_x) / zeta_ab;
    let center_p_y = (ai * ri_y + aj * rj_y) / zeta_ab;
    let center_p_z = (ai * ri_z + aj * rj_z) / zeta_ab;
    let rirj_x = ri_x - rj_x;
    let rirj_y = ri_y - rj_y;
    let rirj_z = ri_z - rj_z;
    let rr = rirj_x * rirj_x + rirj_y * rirj_y + rirj_z * rirj_z;
    let fac = f64::exp(-ai * aj / zeta_ab * rr) * norm_i * norm_j;
    let aij2 = 0.5 / zeta_ab;

    PairData {
        zeta_ab,
        center_p_x,
        center_p_y,
        center_p_z,
        rirj_x,
        rirj_y,
        rirj_z,
        fac,
        aij2,
    }
}
