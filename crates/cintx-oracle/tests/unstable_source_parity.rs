//! Oracle parity tests for unstable-source API families (Phase 14).
//!
//! Per D-09: single file with per-family modules, all gated behind
//! `#[cfg(feature = "unstable-source-api")]`.
//! Per D-10: reuse H2O/STO-3G fixture molecule for all unstable families.
//! Grids family adds grid point coordinates to env but uses the same molecule.
//!
//! Gate summary:
//!   Family  | Symbols | Status
//!   --------|---------|-------
//!   origi   | 4       | Wave 2 (Phase 14 Plan 02)
//!   grids   | 5       | Wave 2 (Phase 14 Plan 02)
//!   breit   | 2       | Wave 2 (Phase 14 Plan 03)
//!   origk   | 6       | Wave 2 (Phase 14 Plan 03)
//!   ssc     | 1       | Wave 2 (Phase 14 Plan 04)
//!
//! Requirements: #[cfg(feature = "cpu")] + #[cfg(feature = "unstable-source-api")]
//! Run: CINTX_ORACLE_BUILD_VENDOR=1 cargo test --features cpu,unstable-source-api -p cintx-oracle -- unstable_source_parity

#![cfg(feature = "cpu")]
#![cfg(feature = "unstable-source-api")]

/// origi family parity tests.
/// 4 symbols: int1e_r2_origi_sph, int1e_r4_origi_sph,
///            int1e_r2_origi_ip2_sph, int1e_r4_origi_ip2_sph.
/// Implementation added in Phase 14 Plan 02.
mod origi_parity {}

/// grids family parity tests.
/// 5 symbols: int1e_grids_sph, int1e_grids_ip_sph, int1e_grids_ipvip_sph,
///            int1e_grids_spvsp_sph, int1e_grids_ipip_sph.
/// Uses H2O/STO-3G fixture with grid point coordinates in env.
/// Implementation added in Phase 14 Plan 02.
mod grids_parity {}

/// breit family parity tests.
/// 2 spinor-only symbols: int2e_breit_r1p2_spinor, int2e_breit_r2p2_spinor.
/// Implementation added in Phase 14 Plan 03.
mod breit_parity {}

/// origk family parity tests.
/// 6 symbols: int3c1e_r2/r4/r6_origk_sph, int3c1e_ip1_r2/r4/r6_origk_sph.
/// Implementation added in Phase 14 Plan 03.
mod origk_parity {}

/// ssc family parity tests.
/// 1 symbol: int3c2e_sph_ssc.
/// Implementation added in Phase 14 Plan 04.
mod ssc_parity {}
