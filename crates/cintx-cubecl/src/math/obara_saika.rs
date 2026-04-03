//! Obara-Saika recurrence relations as `#[cube]` functions.
//!
//! Ports the VRR (vertical recurrence relation) and HRR (horizontal recurrence relation)
//! from `libcint-master/src/g1e.c` (CINTg1e_ovlp, lines 125-184) and
//! `libcint-master/src/g2e.c` (CINTg0_2e_2d, lines 272-410).
//!
//! ## VRR (1e overlap, g1e.c lines 164-172)
//!
//! ```text
//! g[1*di] = rijrx * g[0]
//! g[(n+1)*di] = n * aij2 * g[(n-1)*di] + rijrx * g[n*di],  n = 1..nmax-1
//! ```
//!
//! ## HRR (g1e.c lines 175-182)
//!
//! ```text
//! g[j*dj + i*di] = g[(j-1)*dj + (i+1)*di] + rirj * g[(j-1)*dj + i*di]
//! ```
//!
//! ## VRR 2e (g2e.c lines 306-322)
//!
//! ```text
//! g[1*stride] = c00 * g[0]
//! g[(n+1)*stride] = n * b10 * g[(n-1)*stride] + c00 * g[n*stride]
//! ```
//!
//! CubeCL constraints applied:
//! - All loop counters are u32
//! - Statement-form if/else (no if-expressions as values)
//! - Array indexing uses `as usize` conversions per CubeCL 0.9.x
//! - No recursion; all iterative

use cubecl::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
//  vrr_step — 1e vertical recurrence for one Cartesian dimension
// ─────────────────────────────────────────────────────────────────────────────

/// Vertical recurrence relation for one Cartesian dimension (1e overlap integral).
///
/// Fills `g[di], g[2*di], ..., g[nmax*di]` using:
/// ```text
/// g[1*stride]       = rijrx * g[0]
/// g[(n+1)*stride]   = n * aij2 * g[(n-1)*stride] + rijrx * g[n*stride],  n = 1..nmax-1
/// ```
///
/// The caller must set `g[0]` before calling this function (base case).
///
/// Source: `libcint-master/src/g1e.c` lines 164-172 (CINTg1e_ovlp VRR).
///
/// Parameters:
/// - `g`: G-array, indexed as `g[n * stride]`
/// - `rijrx`: displacement `rij[d] - rx[d]` for this Cartesian dimension
/// - `aij2`: `0.5 / (ai + aj)` — recurrence denominator
/// - `nmax`: maximum angular momentum on this center
/// - `stride`: stride between consecutive n-indices in `g`
#[cube]
pub fn vrr_step(g: &mut Array<f64>, rijrx: f64, aij2: f64, nmax: u32, stride: u32) {
    if nmax >= 1u32 {
        // g[1*stride] = rijrx * g[0], g1e.c line 164
        g[stride as usize] = rijrx * g[0usize];
        // n = 1..nmax-1: g[(n+1)*stride] = n*aij2*g[(n-1)*stride] + rijrx*g[n*stride]
        // g1e.c lines 169-172
        let mut n: u32 = 1;
        while n < nmax {
            g[((n + 1u32) * stride) as usize] =
                n as f64 * aij2 * g[((n - 1u32) * stride) as usize]
                    + rijrx * g[(n * stride) as usize];
            n += 1;
        }
    }
}

/// Host-side wrapper for `vrr_step` — operates on a plain `&mut [f64]` slice.
///
/// Used by tests and host-side integral planning code.
///
/// See `vrr_step` for algorithm description.
pub fn vrr_step_host(g: &mut [f64], rijrx: f64, aij2: f64, nmax: u32, stride: u32) {
    if nmax >= 1 {
        g[stride as usize] = rijrx * g[0];
        let mut n: u32 = 1;
        while n < nmax {
            g[((n + 1) * stride) as usize] =
                n as f64 * aij2 * g[((n - 1) * stride) as usize]
                    + rijrx * g[(n * stride) as usize];
            n += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  hrr_step — horizontal recurrence for one Cartesian dimension
// ─────────────────────────────────────────────────────────────────────────────

/// Horizontal recurrence relation for one Cartesian dimension.
///
/// Applies the HRR transfer:
/// ```text
/// g[j*dj + i*di] = g[(j-1)*dj + (i+1)*di] + rirj * g[(j-1)*dj + i*di]
/// ```
/// for j = 1..=lj, i = 0..=(li_max - j).
///
/// Source: `libcint-master/src/g1e.c` lines 175-182 (CINTg1e_ovlp HRR).
///
/// Parameters:
/// - `g`: G-array with both i and j indices packed linearly
/// - `rirj`: displacement `Ri[d] - Rj[d]` for this Cartesian dimension
/// - `di`: stride along i-index (must match VRR stride)
/// - `dj`: stride along j-index
/// - `li_max`: total angular momentum on both centers (nmax in VRR sense)
/// - `lj`: target j angular momentum to build
#[cube]
pub fn hrr_step(g: &mut Array<f64>, rirj: f64, di: u32, dj: u32, li_max: u32, lj: u32) {
    let mut j: u32 = 1;
    while j <= lj {
        // i runs from 0 to li_max - j (inclusive)
        let i_max = li_max - j;
        let mut i: u32 = 0;
        while i <= i_max {
            // g[j*dj + i*di] = g[(j-1)*dj + (i+1)*di] + rirj * g[(j-1)*dj + i*di]
            let idx_out = (j * dj + i * di) as usize;
            let idx_in_hi = ((j - 1u32) * dj + (i + 1u32) * di) as usize;
            let idx_in_lo = ((j - 1u32) * dj + i * di) as usize;
            g[idx_out] = g[idx_in_hi] + rirj * g[idx_in_lo];
            i += 1;
        }
        j += 1;
    }
}

/// Host-side wrapper for `hrr_step` — operates on a plain `&mut [f64]` slice.
///
/// Used by tests and host-side integral planning code.
///
/// See `hrr_step` for algorithm description.
pub fn hrr_step_host(g: &mut [f64], rirj: f64, di: u32, dj: u32, li_max: u32, lj: u32) {
    let mut j: u32 = 1;
    while j <= lj {
        let i_max = li_max - j;
        let mut i: u32 = 0;
        while i <= i_max {
            let idx_out = (j * dj + i * di) as usize;
            let idx_in_hi = ((j - 1) * dj + (i + 1) * di) as usize;
            let idx_in_lo = ((j - 1) * dj + i * di) as usize;
            g[idx_out] = g[idx_in_hi] + rirj * g[idx_in_lo];
            i += 1;
        }
        j += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  vrr_2e_step — 2e vertical recurrence with Rys root-dependent coefficients
// ─────────────────────────────────────────────────────────────────────────────

/// Vertical recurrence relation for 2e integrals using Rys root-specific coefficients.
///
/// Fills `g[stride], g[2*stride], ..., g[nmax*stride]` using:
/// ```text
/// g[1*stride]       = c00 * g[0]
/// g[(n+1)*stride]   = n * b10 * g[(n-1)*stride] + c00 * g[n*stride],  n = 1..nmax-1
/// ```
///
/// Source: `libcint-master/src/g2e.c` lines 306-322 (CINTg0_2e_2d bra VRR).
///
/// Parameters:
/// - `g`: G-array for one Rys root, stride-indexed
/// - `c00`: Rys root-weighted center displacement for bra (c00x/y/z in g2e.c)
/// - `b10`: Rys root recurrence coefficient bra-direction (b10 in g2e.c)
/// - `nmax`: max angular momentum bra
/// - `stride`: stride between consecutive n-indices
#[cube]
pub fn vrr_2e_step(g: &mut Array<f64>, c00: f64, b10: f64, nmax: u32, stride: u32) {
    if nmax >= 1u32 {
        // g[1*stride] = c00 * g[0], g2e.c line 312
        g[stride as usize] = c00 * g[0usize];
        // n = 1..nmax-1: g[(n+1)*stride] = n*b10*g[(n-1)*stride] + c00*g[n*stride]
        // g2e.c lines 319-321
        let mut n: u32 = 1;
        while n < nmax {
            g[((n + 1u32) * stride) as usize] =
                n as f64 * b10 * g[((n - 1u32) * stride) as usize]
                    + c00 * g[(n * stride) as usize];
            n += 1;
        }
    }
}

/// Host-side wrapper for `vrr_2e_step` — operates on a plain `&mut [f64]` slice.
///
/// Used by tests and host-side integral planning code.
///
/// See `vrr_2e_step` for algorithm description.
pub fn vrr_2e_step_host(g: &mut [f64], c00: f64, b10: f64, nmax: u32, stride: u32) {
    if nmax >= 1 {
        g[stride as usize] = c00 * g[0];
        let mut n: u32 = 1;
        while n < nmax {
            g[((n + 1) * stride) as usize] =
                n as f64 * b10 * g[((n - 1) * stride) as usize]
                    + c00 * g[(n * stride) as usize];
            n += 1;
        }
    }
}
