//! Embedded PySCF K-Taylor tables for `ECPsph_ine_opt` (table-interpolation
//! modified-spherical-Bessel evaluation).
//!
//! Two precomputed tables ship as little-endian IEEE-754 binary64 blobs,
//! mirroring `crates/cintx-cubecl/src/math/roots_xw_data.rs` exactly
//! (`AlignedBytes<{N*8}>` + `include_bytes!` + `bytemuck::cast_slice`):
//!
//! - `sph_ine_tab()`        — `_sph_ine_tab` (nr_ecp.c:30), `K_TAB_ENTRIES * K_TAB_COL
//!                            = 400 * 24 = 9600` f64. The order>7 downward-recurrence
//!   table.
//! - `sph_ine_tab_order7()` — `_sph_ine_tab_order7` (nr_ecp.c:434),
//!   `K_TAB_ENTRIES * ORDER7OFFSET * (K_TAYLOR_MAX+1)
//!                            = 400 * 8 * 8 = 25600` f64. The per-order (0..=7)
//!   Taylor-coefficient table (`ORDER7OFFSET = 8`).
//!
//! Source: `vendor/pyscf-nr-ecp/src/nr_ecp.c` lines 30 and 434.
//!
//! These tables encode PySCF's *exact* literal values (D-14): the K-Taylor
//! interpolation in `ecp_k_taylor::ecpsph_ine_opt_host` Taylor-expands them
//! around `z0 = entry*K_TAB_INTERVAL + K_TAB_INTERVAL/2`; recomputing the table
//! from cintx's own Bessel series would diverge and break byte-identity vs PySCF.
//!
//! Do NOT edit the `.bin` files manually — regenerate via
//! `cargo run --manifest-path xtask/Cargo.toml -- gen-ecp-tables` (byte-exact
//! drift-checked by `--check`, enforced in the `manifest_drift_gate` CI job per
//! D-14/D-15).

/// Alignment-safe wrapper so `include_bytes!` aligns to 8 bytes for f64.
#[repr(C, align(8))]
struct AlignedBytes<const N: usize>([u8; N]);

/// `_sph_ine_tab` raw bytes (little-endian f64; 9600 elements = 76800 bytes).
static SPH_INE_TAB_BYTES: &AlignedBytes<{ 9600 * 8 }> =
    &AlignedBytes(*include_bytes!("ecp_k_taylor_in.bin"));

/// `_sph_ine_tab_order7` raw bytes (little-endian f64; 25600 elements = 204800 bytes).
static SPH_INE_TAB_ORDER7_BYTES: &AlignedBytes<{ 25600 * 8 }> =
    &AlignedBytes(*include_bytes!("ecp_k_taylor_order7.bin"));

/// `_sph_ine_tab` (400x24): the order>7 downward-recurrence table.
///
/// Indexed as `sph_ine_tab()[entry * K_TAB_COL + col]` per `ECPsph_ine_opt`
/// (nr_ecp.c:4814).
pub fn sph_ine_tab() -> &'static [f64] {
    bytemuck::cast_slice(&SPH_INE_TAB_BYTES.0)
}

/// `_sph_ine_tab_order7` (400x8x8): per-order Taylor coefficients for order 0..=7.
///
/// Indexed as `sph_ine_tab_order7()[entry * ORDER7OFFSET * (K_TAYLOR_MAX+1)
/// + j * ORDER7OFFSET + order]` per `ECPsph_ine_opt` (nr_ecp.c:4703).
pub fn sph_ine_tab_order7() -> &'static [f64] {
    bytemuck::cast_slice(&SPH_INE_TAB_ORDER7_BYTES.0)
}
