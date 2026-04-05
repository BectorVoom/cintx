//! Static table data for CINTstg_roots (F12/STG/YP quadrature).
//!
//! DATA_X and DATA_W are the roots and weights tables from roots_xw.dat,
//! stored as little-endian IEEE 754 binary64 values.
//!
//! Source: libcint-master/src/roots_xw.dat
//! Layout: DATA_X base offset = (nroots-1)*nroots/2 * 19600 elements.
//! Generation: Python extraction preserving exact float values from C source.
//!
//! Do NOT edit the .bin files manually. Re-generate by running the extraction script.

/// Alignment-safe wrapper so include_bytes aligns to 8 bytes for f64.
#[repr(C, align(8))]
struct AlignedBytes<const N: usize>([u8; N]);

/// Roots table raw bytes (little-endian f64 values from DATA_X in roots_xw.dat).
static DATA_X_BYTES: &AlignedBytes<{ 1783600 * 8 }> =
    &AlignedBytes(*include_bytes!("roots_xw_x.bin"));

/// Weights table raw bytes (little-endian f64 values from DATA_W in roots_xw.dat).
static DATA_W_BYTES: &AlignedBytes<{ 1783600 * 8 }> =
    &AlignedBytes(*include_bytes!("roots_xw_w.bin"));

/// Roots table (Chebyshev coefficients for quadrature root values).
///
/// Indexed as DATA_X\[(nroots-1)*nroots/2 * 19600 + ...\] per CINTstg_roots.
pub fn data_x() -> &'static [f64] {
    bytemuck::cast_slice(&DATA_X_BYTES.0)
}

/// Weights table (Chebyshev coefficients for quadrature weight values).
///
/// Indexed as DATA_W\[(nroots-1)*nroots/2 * 19600 + ...\] per CINTstg_roots.
pub fn data_w() -> &'static [f64] {
    bytemuck::cast_slice(&DATA_W_BYTES.0)
}
