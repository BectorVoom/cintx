//! `gen-c2spinor-table` — extract libcint 6.1.3's Cartesian-to-spinor
//! Clebsch-Gordan coupling tables from the vendored C source and emit them as a
//! Rust data module.
//!
//! Why this exists
//! ---------------
//! `crates/cintx-cubecl/src/transform/c2spinor_coeffs.rs` carries
//! hand-transcribed tables for `l = 0..=4` only. Above that the accessors had
//! two behaviours, both wrong: `bra_coeff_refs` **panicked** ("l=5 > 4 not
//! supported"), reachable from `eval_raw` with an ordinary `h` shell, and
//! `gt_coeff_rows`/`lt_coeff_rows` returned **empty row lists**, so the
//! single-block transforms wrote nothing and handed back zeros with an `Ok`.
//!
//! Neither is a table ceiling libcint has. Its `g_c2s[]` (`cart2sph.c`) points
//! `cart2j_lt_*` / `cart2j_gt_*` into `g_trans_cart2jR[]` / `g_trans_cart2jI[]`
//! for `l = 0..=12` (rows 13..=15 carry `NULL` spinor pointers), so that is the
//! envelope cintx now matches. 69 160 coefficients are not something to
//! transcribe by hand — the whole point of the Rys and c2s generators was to
//! stop doing that — so this parses them out of the vendored source instead,
//! and `--check` re-derives and diffs to keep the committed module from
//! drifting.
//!
//! Layout, from `g_c2s[]`'s pointer arithmetic
//! -------------------------------------------
//! For each `l`, the LT block (`j = l - 1/2`, `2l` rows) is immediately followed
//! by the GT block (`j = l + 1/2`, `2l + 2` rows); every row holds `2 * ncart(l)`
//! values — `ncart(l)` alpha coefficients then `ncart(l)` beta. The flat offsets
//! that reproduces (`lt` 0, 4, 40, 160, 440, ...; `gt` 0, 16, 88, 280, 680, ...)
//! are exactly the constants `g_c2s[]` is written with, and
//! [`tests::block_offsets_match_libcint_g_c2s`] pins them. The adjacency is
//! load-bearing: libcint's `kappa == 0` path reads the LT pointer and
//! over-runs into GT, and `c2spinor.rs` reproduces that by construction.
//!
//! Unlike `g_trans_cart2sph[]` there is no `#ifdef PYPZPX` inside these arrays.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Vendored libcint source file holding the spinor coupling tables.
const LIBCINT_C2S_REL: &str = "libcint-master/src/cart2sph.c";
/// Target generated Rust data module (relative to workspace root).
const TARGET_REL: &str = "crates/cintx-cubecl/src/transform/c2spinor_data.rs";
/// Highest `l` libcint's `g_c2s` table carries spinor coefficients for.
const LMAX: usize = 12;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest dir has a parent (workspace root)")
        .to_path_buf()
}

fn ncart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// Values in `l`'s LT block: `2l` rows of `2 * ncart(l)`.
fn lt_len(l: usize) -> usize {
    2 * l * 2 * ncart(l)
}

/// Values in `l`'s GT block: `2l + 2` rows of `2 * ncart(l)`.
fn gt_len(l: usize) -> usize {
    (2 * l + 2) * 2 * ncart(l)
}

/// Flat offset of `l`'s LT block; its GT block follows at `+ lt_len(l)`.
fn block_offset(l: usize) -> usize {
    (0..l).map(|k| lt_len(k) + gt_len(k)).sum()
}

/// Parse `static double <name>[] = { ... };` into its values.
fn parse_flat_table(source: &str, name: &str) -> Result<Vec<f64>> {
    let needle = format!("{name}[]");
    let decl = source
        .find(&needle)
        .with_context(|| format!("could not find `{needle}` in cart2sph.c"))?;
    let body_start = decl
        + source[decl..]
            .find('{')
            .with_context(|| format!("no opening brace after {needle}"))?
        + 1;
    let close = source[body_start..]
        .find("};")
        .with_context(|| format!("no closing `}};` for {needle}"))?;
    let mut body = source[body_start..body_start + close].to_owned();

    if body.contains("#if") {
        bail!("unexpected preprocessor conditional inside {needle}");
    }
    while let Some(s) = body.find("/*") {
        match body[s..].find("*/") {
            Some(e) => body.replace_range(s..s + e + 2, " "),
            None => bail!("unterminated block comment in {needle}"),
        }
    }
    let mut values = Vec::new();
    for raw in body.lines() {
        let line = match raw.find("//") {
            Some(idx) => &raw[..idx],
            None => raw,
        };
        for token in line.split(',') {
            let tok = token.trim();
            if tok.is_empty() {
                continue;
            }
            let tok = tok.trim_end_matches(['l', 'L', 'f', 'F']);
            values.push(
                tok.parse::<f64>()
                    .with_context(|| format!("failed to parse literal `{tok}` in {needle}"))?,
            );
        }
    }

    let expected = block_offset(LMAX + 1);
    if values.len() != expected {
        bail!(
            "{needle} parsed {} values, expected {expected} for l = 0..={LMAX} — the \
             vendored table's shape changed",
            values.len()
        );
    }
    Ok(values)
}

fn render_table(out: &mut String, name: &str, doc: &str, values: &[f64]) {
    out.push_str(&format!(
        "/// {doc} ({} values).\n#[rustfmt::skip]\npub static {name}: [f64; {}] = [\n",
        values.len(),
        values.len()
    ));
    for l in 0..=LMAX {
        let width = 2 * ncart(l);
        let base = block_offset(l);
        for (block, rows, start) in [("LT", 2 * l, base), ("GT", 2 * l + 2, base + lt_len(l))] {
            if rows == 0 {
                out.push_str(&format!("    // l = {l} {block}: no rows (j = l - 1/2 is empty at l = 0).\n"));
                continue;
            }
            out.push_str(&format!(
                "    // l = {l} {block}: {rows} spinor rows x {width} (alpha then beta) columns.\n"
            ));
            for r in 0..rows {
                let row = &values[start + r * width..start + (r + 1) * width];
                let mut line = String::from("    ");
                for v in row {
                    let tok = format!("{v:?}, ");
                    if line.len() + tok.len() > 100 {
                        out.push_str(line.trim_end());
                        out.push('\n');
                        line = String::from("    ");
                    }
                    line.push_str(&tok);
                }
                if !line.trim().is_empty() {
                    out.push_str(line.trim_end());
                    out.push('\n');
                }
            }
        }
    }
    out.push_str("];\n\n");
}

/// Render `c2spinor_data.rs` from the vendored source.
fn render_module(root: &Path) -> Result<String> {
    let src_path = root.join(LIBCINT_C2S_REL);
    let source = fs::read_to_string(&src_path)
        .with_context(|| format!("read vendored C source {}", src_path.display()))?;
    let real = parse_flat_table(&source, "g_trans_cart2jR")?;
    let imag = parse_flat_table(&source, "g_trans_cart2jI")?;

    let mut out = String::new();
    out.push_str(&format!(
        "//! AUTO-GENERATED Cartesian-to-spinor coupling coefficients, `l = 0..={LMAX}`.\n\
         //!\n\
         //! Extracted verbatim from the vendored libcint 6.1.3 source\n\
         //! (`libcint-master/src/cart2sph.c`, `g_trans_cart2jR[]` and\n\
         //! `g_trans_cart2jI[]`, which `g_c2s[l].cart2j_lt_*` / `.cart2j_gt_*` index).\n\
         //!\n\
         //! DO NOT hand-edit. Regenerate via `cargo run -p xtask -- gen-c2spinor-table`;\n\
         //! `--check` re-derives and fails closed if this file has drifted from the\n\
         //! vendored C source.\n\
         //!\n\
         //! # Layout\n\
         //!\n\
         //! For each `l`, block `l` is `CJ_OFFSET[l]..CJ_OFFSET[l + 1]` and holds the\n\
         //! LT block (`j = l - 1/2`, `2l` rows) followed immediately by the GT block\n\
         //! (`j = l + 1/2`, `2l + 2` rows). Every row is `2 * ncart(l)` values: the\n\
         //! `ncart(l)` alpha-spinor coefficients, then the `ncart(l)` beta ones, over\n\
         //! Cartesian components in libcint ordering. LT-then-GT adjacency is what\n\
         //! libcint's `kappa == 0` path relies on when it over-reads from the LT\n\
         //! pointer into GT, so it is preserved rather than re-derived.\n\
         //!\n\
         //! `c2spinor::tests::generated_table_matches_the_hand_transcribed_tables`\n\
         //! pins the `l <= 4` blocks against `c2spinor_coeffs.rs`, bit for bit.\n\
         \n\
         #![allow(clippy::all)]\n\
         \n\
         /// Highest `l` this table covers — libcint's own `g_c2s` spinor ceiling.\n\
         pub const C2SPINOR_LMAX: u8 = {LMAX};\n\
         \n"
    ));

    out.push_str(&format!(
        "/// Flat offset of each `l` block (LT rows first, then GT) in [`CJ_R`] and\n\
         /// [`CJ_I`]; the final entry is the table length.\n\
         #[rustfmt::skip]\n\
         pub static CJ_OFFSET: [usize; {}] = [\n",
        LMAX + 2
    ));
    let offsets: Vec<String> = (0..=LMAX + 1).map(|l| block_offset(l).to_string()).collect();
    out.push_str(&format!("    {},\n", offsets.join(", ")));
    out.push_str("];\n\n");

    render_table(&mut out, "CJ_R", "Real parts of the coupling coefficients", &real);
    render_table(&mut out, "CJ_I", "Imaginary parts of the coupling coefficients", &imag);
    // Drop the trailing blank line so the file ends with a single newline.
    while out.ends_with("\n\n") {
        out.pop();
    }
    Ok(out)
}

/// Entry point for `gen-c2spinor-table [--check]`.
///
/// - `check == false`: re-derive and write `c2spinor_data.rs`.
/// - `check == true`:  re-derive and byte-compare against the committed file,
///   failing closed on any divergence — the same drift gate `gen-c2s-table`,
///   `gen-rys-tables` and `gen-ecp-tables` use.
pub fn run_gen_c2spinor_table(check: bool) -> Result<()> {
    let root = workspace_root();
    let target = root.join(TARGET_REL);
    let rendered = render_module(&root)?;

    if check {
        let committed = fs::read_to_string(&target)
            .with_context(|| format!("read committed {} for drift-check", target.display()))?;
        if committed != rendered {
            bail!(
                "c2spinor table drift detected: committed {} no longer matches the vendored \
                 libcint source — regenerate with `cargo run -p xtask -- gen-c2spinor-table`",
                target.display()
            );
        }
        println!(
            "gen-c2spinor-table --check: {} matches the vendored libcint source (no drift)",
            target.display()
        );
    } else {
        fs::write(&target, &rendered).with_context(|| format!("write {}", target.display()))?;
        println!(
            "gen-c2spinor-table: wrote {} from the vendored libcint source",
            target.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The LT and GT offsets must reproduce libcint's own `g_c2s[]` pointer
    /// arithmetic (`g_trans_cart2jR + 4, + 16, + 40, + 88, ...`). If they did
    /// not, every block above `l = 0` would be read from the wrong place while
    /// still parsing cleanly.
    #[test]
    fn block_offsets_match_libcint_g_c2s() {
        let lt = [0, 4, 40, 160, 440, 980, 1904, 3360, 5520, 8580, 12760, 18304, 25480];
        let gt = [0, 16, 88, 280, 680, 1400, 2576, 4368, 6960, 10560, 15400, 21736, 29848];
        for l in 0..=LMAX {
            assert_eq!(block_offset(l), lt[l], "LT offset for l={l}");
            assert_eq!(block_offset(l) + lt_len(l), gt[l], "GT offset for l={l}");
        }
        assert_eq!(block_offset(LMAX + 1), 34580, "total table length");
    }

    /// A conditional inside the array would mean libcint had grown a second
    /// ordering for spinors the way it has for p orbitals, and the parser would
    /// have to choose a branch rather than concatenate both.
    #[test]
    fn a_preprocessor_conditional_is_refused() {
        let src = "static double g_trans_cart2jR[] = {\n1,\n#ifdef X\n2,\n#endif\n};\n";
        let err = parse_flat_table(src, "g_trans_cart2jR").unwrap_err().to_string();
        assert!(err.contains("preprocessor"), "{err}");
    }
}
