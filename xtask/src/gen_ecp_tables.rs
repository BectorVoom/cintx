//! `gen-ecp-tables` — extract PySCF's K-Taylor modified-spherical-Bessel tables
//! from the vendored `nr_ecp.c` and emit them as little-endian f64 `.bin` blobs.
//!
//! Background (Phase 19 D-14 / D-15)
//! ---------------------------------
//! PySCF's `ECPsph_ine_opt` (vendor/pyscf-nr-ecp/src/nr_ecp.c) interpolates two
//! precomputed tables to evaluate the modified spherical Bessel function used by
//! the ECP radial machinery:
//!
//! - `_sph_ine_tab`        (nr_ecp.c:30)  — `K_TAB_ENTRIES * K_TAB_COL = 400 * 24 = 9600` f64
//! - `_sph_ine_tab_order7` (nr_ecp.c:434) — `K_TAB_ENTRIES * ORDER7OFFSET * (K_TAYLOR_MAX+1)
//!                                            = 400 * 8 * 8 = 25600` f64
//!
//! Byte-identity vs PySCF requires embedding PySCF's *exact* literal values
//! (D-14): the embedded blobs are consumed by
//! `crates/cintx-cubecl/src/math/ecp_k_taylor_data.rs` via `include_bytes!` +
//! `bytemuck::cast_slice`, mirroring `roots_xw_data.rs`.
//!
//! This subcommand parses the `static double NAME[] = { ... };` literal arrays
//! directly out of the vendored C source and writes the `.bin` files (generate
//! path) or byte-compares the freshly-extracted bytes against the committed
//! blobs and fails closed on any diff (the `--check` drift gate). The drift gate
//! is wired into the `manifest_drift_gate` job of
//! `.github/workflows/compat-governance-pr.yml`, mirroring the manifest-lock
//! discipline (D-15), so a stale or hand-edited blob fails the PR.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

// --- Table dimensions (cite vendor/pyscf-nr-ecp/include/gto/nr_ecp.h:19-22) ---
// K_TAB_ENTRIES = 400, K_TAB_COL = 24, K_TAYLOR_MAX = 7; ORDER7OFFSET = 8 is
// defined in nr_ecp.c:433 (not the header).
const K_TAB_ENTRIES: usize = 400;
const K_TAB_COL: usize = 24;
const K_TAYLOR_MAX: usize = 7;
const ORDER7OFFSET: usize = 8;

/// `_sph_ine_tab` element count = 400 * 24 = 9600.
const SPH_INE_TAB_COUNT: usize = K_TAB_ENTRIES * K_TAB_COL;
/// `_sph_ine_tab_order7` element count = 400 * 8 * 8 = 25600.
const SPH_INE_TAB_ORDER7_COUNT: usize = K_TAB_ENTRIES * ORDER7OFFSET * (K_TAYLOR_MAX + 1);

/// Vendored C source path (relative to workspace root).
const NR_ECP_C_REL: &str = "vendor/pyscf-nr-ecp/src/nr_ecp.c";
/// Target blob from `_sph_ine_tab` (relative to workspace root).
const SPH_INE_TAB_BIN_REL: &str = "crates/cintx-cubecl/src/math/ecp_k_taylor_in.bin";
/// Target blob from `_sph_ine_tab_order7` (relative to workspace root).
const SPH_INE_TAB_ORDER7_BIN_REL: &str = "crates/cintx-cubecl/src/math/ecp_k_taylor_order7.bin";

/// Resolve the workspace root from this xtask crate's manifest dir.
///
/// `CARGO_MANIFEST_DIR` is the `xtask/` directory; the workspace root is its
/// parent (matching the `../{...}` join the other xtask subcommands use).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest dir has a parent (workspace root)")
        .to_path_buf()
}

/// Parse the `static double NAME[] = { ... };` literal array out of the C source.
///
/// Finds the line whose trimmed text starts with `static double NAME[]`, then
/// collects every comma-separated f64 literal (which may span many lines,
/// skipping `//` comments and whitespace) up to the terminating `};`. Each token
/// is parsed with `f64::from_str`, exact for the printed 16-significant-digit
/// literals.
fn parse_static_double_array(source: &str, name: &str) -> Result<Vec<f64>> {
    let needle = format!("static double {name}[]");
    let start_line_offset = source
        .find(&needle)
        .with_context(|| format!("could not find `{needle}` in {NR_ECP_C_REL}"))?;

    // Locate the opening brace after the declaration.
    let after_decl = &source[start_line_offset..];
    let brace_rel = after_decl
        .find('{')
        .with_context(|| format!("no opening brace after `{needle}`"))?;
    let body_start = start_line_offset + brace_rel + 1;

    // Locate the closing `};` that terminates this array.
    let close_rel = source[body_start..]
        .find("};")
        .with_context(|| format!("no closing `}};` for `{needle}`"))?;
    let body = &source[body_start..body_start + close_rel];

    let mut values: Vec<f64> = Vec::new();
    for raw_line in body.lines() {
        // Strip a trailing `// ...` comment if present.
        let line = match raw_line.find("//") {
            Some(idx) => &raw_line[..idx],
            None => raw_line,
        };
        for token in line.split(',') {
            let tok = token.trim();
            if tok.is_empty() {
                continue;
            }
            let value: f64 = tok
                .parse::<f64>()
                .with_context(|| format!("failed to parse f64 literal `{tok}` in `{name}`"))?;
            values.push(value);
        }
    }

    Ok(values)
}

/// Extract a named table and validate its element count.
fn extract_table(source: &str, name: &str, expected: usize) -> Result<Vec<f64>> {
    let values = parse_static_double_array(source, name)?;
    if values.len() != expected {
        bail!(
            "table `{name}` parsed {} values, expected {} (a vendored-source edit \
             may have added or removed table rows); refusing to write a wrong-shaped blob",
            values.len(),
            expected
        );
    }
    Ok(values)
}

/// Write a freshly-extracted table to its `.bin` path as little-endian f64 bytes.
fn write_blob(path: &Path, values: &[f64]) -> Result<()> {
    let bytes: &[u8] = bytemuck::cast_slice(values);
    fs::write(path, bytes).with_context(|| format!("write blob to {}", path.display()))?;
    println!(
        "gen-ecp-tables: wrote {} ({} f64 = {} bytes)",
        path.display(),
        values.len(),
        bytes.len()
    );
    Ok(())
}

/// Byte-compare the committed `.bin` against the freshly-extracted bytes.
///
/// Whole-blob byte-compare is the chosen comparison granularity (D-15 names it as
/// the simplest sufficient form): a stale or hand-edited committed blob fails
/// closed naming the diverging file.
fn check_blob(path: &Path, values: &[f64]) -> Result<()> {
    let fresh: &[u8] = bytemuck::cast_slice(values);
    let committed = fs::read(path)
        .with_context(|| format!("read committed blob {} for drift-check", path.display()))?;
    if committed.len() != fresh.len() {
        bail!(
            "ECP K-Taylor drift detected: committed blob {} is {} bytes but the \
             vendored source re-extracts to {} bytes — regenerate with \
             `cargo run -p xtask -- gen-ecp-tables`",
            path.display(),
            committed.len(),
            fresh.len()
        );
    }
    if committed != fresh {
        bail!(
            "ECP K-Taylor drift detected: committed blob {} no longer matches the \
             vendored {} source — regenerate with `cargo run -p xtask -- gen-ecp-tables`",
            path.display(),
            NR_ECP_C_REL
        );
    }
    println!(
        "gen-ecp-tables --check: {} matches vendored source ({} bytes)",
        path.display(),
        fresh.len()
    );
    Ok(())
}

/// Entry point for the `gen-ecp-tables [--check]` subcommand.
///
/// - `check == false`: extract both tables and write the two `.bin` blobs.
/// - `check == true`:  extract both tables and byte-compare against the
///   committed blobs, failing closed (non-zero exit) on any divergence.
pub fn run_gen_ecp_tables(check: bool) -> Result<()> {
    let root = workspace_root();
    let nr_ecp_c = root.join(NR_ECP_C_REL);
    let source = fs::read_to_string(&nr_ecp_c)
        .with_context(|| format!("read vendored C source {}", nr_ecp_c.display()))?;

    let sph_ine_tab = extract_table(&source, "_sph_ine_tab", SPH_INE_TAB_COUNT)?;
    let sph_ine_tab_order7 =
        extract_table(&source, "_sph_ine_tab_order7", SPH_INE_TAB_ORDER7_COUNT)?;

    let tab_path = root.join(SPH_INE_TAB_BIN_REL);
    let order7_path = root.join(SPH_INE_TAB_ORDER7_BIN_REL);

    if check {
        check_blob(&tab_path, &sph_ine_tab)?;
        check_blob(&order7_path, &sph_ine_tab_order7)?;
        println!("gen-ecp-tables --check: both K-Taylor blobs match the vendored source");
    } else {
        write_blob(&tab_path, &sph_ine_tab)?;
        write_blob(&order7_path, &sph_ine_tab_order7)?;
        println!("gen-ecp-tables: emitted both K-Taylor blobs from the vendored source");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_static_double_array() {
        let src = "static double _demo[] = { // 2x1\n1.5, 2.5,\n3.0, 4.0,\n};\n";
        let values = parse_static_double_array(src, "_demo").expect("parse");
        assert_eq!(values, vec![1.5, 2.5, 3.0, 4.0]);
    }

    #[test]
    fn extract_table_rejects_wrong_count() {
        let src = "static double _demo[] = {\n1.0, 2.0,\n};\n";
        let err = extract_table(src, "_demo", 3).expect_err("count mismatch should bail");
        assert!(err.to_string().contains("parsed 2 values, expected 3"));
    }

    #[test]
    fn table_dimension_constants_match_header() {
        assert_eq!(SPH_INE_TAB_COUNT, 9600);
        assert_eq!(SPH_INE_TAB_ORDER7_COUNT, 25600);
    }
}
