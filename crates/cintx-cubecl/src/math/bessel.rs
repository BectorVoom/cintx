//! Modified spherical Bessel functions for Type-2 ECP angular projection.
//!
//! Stub introduced by Phase 19 Plan 01 (Wave 0 scaffold). Algorithm bodies
//! land in Plan 02 (math infrastructure).
//!
//! Ports PySCF `nr_ecp.h` K_TAB tables (`K_TAYLOR_MAX = 7`,
//! `K_TAB_ENTRIES = 400`, `K_TAB_INTERVAL = 0.04`, `K_TAB_COL = 24`).
//!
//! CubeCL constraints (mirrors `boys.rs` Phase 8 P02 incident):
//! - Loop counters in `#[cube]`: `u32` only.
//! - if/else in `#[cube]`: statement-form (mutate, then branch); no
//!   if-expressions as values.
//! - Function calls: `f64::exp(x)` form, not `x.exp()`.
//! - Array indexing: `arr[idx as usize]` with explicit cast.

use cubecl::prelude::*;

/// Maximum angular momentum supported by the ECP code path.
/// Matches PySCF `nr_ecp.h` `ECP_LMAX = 5`.
pub const ECP_LMAX: u32 = 5;

/// Taylor-series term count for the small-argument branch of $i_l(x)$.
/// Matches PySCF `nr_ecp.h` `K_TAYLOR_MAX = 7`.
pub const K_TAYLOR_MAX: u32 = 7;

/// Tabulated-branch row count for the modified-spherical-Bessel lookup.
/// Matches PySCF `nr_ecp.h` `K_TAB_ENTRIES = 400`.
pub const K_TAB_ENTRIES: u32 = 400;

/// Tabulated-branch column count (covers $l = 0..23$ with Taylor headroom).
/// Matches PySCF `nr_ecp.h` `K_TAB_COL = 24`.
pub const K_TAB_COL: u32 = 24;

/// Step size between tabulated argument samples ($[0, 16]$ / 400 = 0.04).
/// Matches PySCF `nr_ecp.h` `K_TAB_INTERVAL = 0.04`.
pub const K_TAB_INTERVAL: f64 = 16.0 / (K_TAB_ENTRIES as f64);

/// Host-side wrapper — primary entry point for unit tests.
///
/// Returns the modified spherical Bessel functions of the first kind
/// $i_l(x)$ for $l = 0..=l\_max$. Plan 02 fills the body using the
/// PySCF table-then-recurrence hybrid strategy documented in
/// `.planning/phases/19-int1e-ecp-type1-type2-evaluator/19-RESEARCH.md`
/// §"Bessel function evaluation strategy".
pub fn modified_spherical_bessel_in_host(_l_max: u32, _x: f64) -> Vec<f64> {
    unimplemented!("Phase 19 Plan 02: modified_spherical_bessel_in_host")
}

/// `#[cube]` core — used by launchers.
/// Plan 02 fills the body.
#[cube]
pub fn modified_spherical_bessel_in(_out: &mut Array<f64>, _l_max: u32, _x: f64) {
    // Plan 02 fills.
}
