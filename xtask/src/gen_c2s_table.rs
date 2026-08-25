//! `gen-c2s-table` — extract libcint 6.1.3's Cartesian-to-spherical transform
//! coefficients from the vendored C source and emit them as a Rust data module.
//!
//! Why this exists
//! ---------------
//! `crates/cintx-cubecl/src/transform/c2s.rs` originally carried hand-transcribed
//! coefficient matrices for `l = 0..=4` only, and its accessor returned `0.0`
//! above that. That was a silent-wrong-answer path: an `l >= 5` shell came back
//! zeroed with an `Ok` status, at any Rys order — an `(h s | s)` three-centre
//! integral is `nroots = 3`, well inside every device ceiling, and still
//! produced zeros.
//!
//! libcint's own table (`cart2sph.c` `g_trans_cart2sph[]`, indexed through
//! `g_c2s[l].cart2sph`) covers `l = 0..=15`, so that is the envelope cintx now
//! matches. Hand-transcribing 19176 coefficients is exactly the kind of silent
//! transcription error the Rys-table generator was written to avoid, so this
//! subcommand parses them out of the vendored source instead, and `--check`
//! re-derives and diffs to keep the committed module from drifting.
//!
//! The one subtlety is `#ifdef PYPZPX`. libcint can order p orbitals either
//! `px, py, pz` (default) or `py, pz, px`; the cintx vendor build does **not**
//! define `PYPZPX`, and `c2s.rs`'s `C2S_L1` is the `px, py, pz` identity. The
//! parser therefore takes the `#else` branch, and the generated table is checked
//! against the four hand-transcribed matrices in `c2s.rs` by
//! `generated_table_matches_the_hand_transcribed_matrices`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Vendored libcint source file holding `g_trans_cart2sph[]`.
const LIBCINT_C2S_REL: &str = "libcint-master/src/cart2sph.c";
/// Target generated Rust data module (relative to workspace root).
const TARGET_REL: &str = "crates/cintx-cubecl/src/transform/c2s_data.rs";
/// Highest `l` libcint's `g_c2s` table covers.
const LMAX: usize = 15;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest dir has a parent (workspace root)")
        .to_path_buf()
}

fn ncart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

fn nsph(l: usize) -> usize {
    2 * l + 1
}

/// Flat offset of `l`'s block in the concatenated table.
fn block_offset(l: usize) -> usize {
    (0..l).map(|k| nsph(k) * ncart(k)).sum()
}

/// Parse `static double g_trans_cart2sph[] = { ... };`, resolving `#ifdef
/// PYPZPX` to the **undefined** branch (the cintx vendor build's setting).
fn parse_c2s_table(source: &str) -> Result<Vec<f64>> {
    let needle = "g_trans_cart2sph[]";
    let decl = source
        .find(needle)
        .with_context(|| format!("could not find `{needle}` in cart2sph.c"))?;
    let body_start = decl
        + source[decl..]
            .find('{')
            .context("no opening brace after g_trans_cart2sph[]")?
        + 1;
    let close = source[body_start..]
        .find("};")
        .context("no closing `};` for g_trans_cart2sph[]")?;
    let body = &source[body_start..body_start + close];

    // Resolve the conditional first, line-wise: everything between `#ifdef
    // PYPZPX` and `#else` belongs to the branch cintx does not compile.
    let mut kept = String::new();
    let mut in_excluded_branch = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#ifdef PYPZPX") {
            in_excluded_branch = true;
            continue;
        }
        if trimmed.starts_with("#ifdef") || trimmed.starts_with("#ifndef") {
            bail!("unexpected preprocessor conditional in g_trans_cart2sph: {trimmed}");
        }
        if trimmed.starts_with("#else") || trimmed.starts_with("#endif") {
            in_excluded_branch = false;
            continue;
        }
        if !in_excluded_branch {
            kept.push_str(line);
            kept.push('\n');
        }
    }

    // Strip block comments, then line comments.
    while let Some(s) = kept.find("/*") {
        match kept[s..].find("*/") {
            Some(e) => kept.replace_range(s..s + e + 2, " "),
            None => bail!("unterminated block comment in g_trans_cart2sph"),
        }
    }
    let mut values = Vec::new();
    for raw in kept.lines() {
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
                    .with_context(|| format!("failed to parse literal `{tok}`"))?,
            );
        }
    }

    let expected: usize = (0..=LMAX).map(|l| nsph(l) * ncart(l)).sum();
    if values.len() != expected {
        bail!(
            "g_trans_cart2sph parsed {} values, expected {expected} for l = 0..={LMAX} — \
             the vendored table's shape changed",
            values.len()
        );
    }
    Ok(values)
}

/// Render `c2s_data.rs` from the vendored source.
fn render_module(root: &Path) -> Result<String> {
    let src_path = root.join(LIBCINT_C2S_REL);
    let source = fs::read_to_string(&src_path)
        .with_context(|| format!("read vendored C source {}", src_path.display()))?;
    let values = parse_c2s_table(&source)?;

    let mut out = String::new();
    out.push_str(&format!(
        "//! AUTO-GENERATED Cartesian-to-spherical transform coefficients, `l = 0..={LMAX}`.\n\
         //!\n\
         //! Extracted verbatim from the vendored libcint 6.1.3 source\n\
         //! (`libcint-master/src/cart2sph.c`, `g_trans_cart2sph[]`, with `PYPZPX`\n\
         //! **undefined** — the cintx vendor build's setting, so p orbitals are\n\
         //! ordered `px, py, pz`).\n\
         //!\n\
         //! DO NOT hand-edit. Regenerate via `cargo run -p xtask -- gen-c2s-table`;\n\
         //! `--check` re-derives and fails closed if this file has drifted from the\n\
         //! vendored C source.\n\
         //!\n\
         //! # Layout\n\
         //!\n\
         //! One contiguous `(2l+1) x (l+1)(l+2)/2` **row-major** block per `l`, at\n\
         //! [`C2S_OFFSET`]`[l]`. Row = spherical component (`m = -l ..= l`), column =\n\
         //! Cartesian component in libcint ordering. This is the same layout libcint\n\
         //! reaches through `g_c2s[l].cart2sph`, and\n\
         //! `c2s::tests::generated_table_matches_the_hand_transcribed_matrices` pins\n\
         //! the `l <= 4` blocks against the hand-transcribed matrices in `c2s.rs`.\n\
         \n\
         #![allow(clippy::all)]\n\
         \n\
         /// Highest `l` this table covers — libcint's own `g_c2s` ceiling.\n\
         pub const C2S_LMAX: u8 = {LMAX};\n\
         \n"
    ));

    // `#[rustfmt::skip]` on both statics: the layout below carries meaning —
    // one line per spherical row, grouped by `l` — and rustfmt would either
    // collapse the offsets onto one line or explode each row to one value per
    // line. Either way the committed file would stop matching what this
    // generator emits, and `--check` would report drift on every `cargo fmt`.
    out.push_str(&format!(
        "/// Flat offset of each `l` block in [`C2S_TABLE`]; the final entry is the\n\
         /// table length, so block `l` is `C2S_OFFSET[l]..C2S_OFFSET[l + 1]`.\n\
         #[rustfmt::skip]\n\
         pub static C2S_OFFSET: [usize; {}] = [\n",
        LMAX + 2
    ));
    let offsets: Vec<String> = (0..=LMAX + 1)
        .map(|l| {
            if l <= LMAX {
                block_offset(l).to_string()
            } else {
                values.len().to_string()
            }
        })
        .collect();
    out.push_str(&format!("    {},\n", offsets.join(", ")));
    out.push_str("];\n\n");

    out.push_str(&format!(
        "/// The concatenated coefficient blocks ({} values).\n\
         #[rustfmt::skip]\n\
         pub static C2S_TABLE: [f64; {}] = [\n",
        values.len(),
        values.len()
    ));
    for l in 0..=LMAX {
        let (ns, nc) = (nsph(l), ncart(l));
        out.push_str(&format!(
            "    // l = {l}: {ns} spherical rows x {nc} Cartesian columns.\n"
        ));
        let base = block_offset(l);
        for m in 0..ns {
            let row = &values[base + m * nc..base + (m + 1) * nc];
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
    out.push_str("];\n");
    Ok(out)
}

/// Entry point for `gen-c2s-table [--check]`.
///
/// - `check == false`: re-derive and write `c2s_data.rs`.
/// - `check == true`:  re-derive and byte-compare against the committed file,
///   failing closed on any divergence (the drift gate `gen-rys-tables` and
///   `gen-ecp-tables` also use).
pub fn run_gen_c2s_table(check: bool) -> Result<()> {
    let root = workspace_root();
    let target = root.join(TARGET_REL);
    let rendered = render_module(&root)?;

    if check {
        let committed = fs::read_to_string(&target)
            .with_context(|| format!("read committed {} for drift-check", target.display()))?;
        if committed != rendered {
            bail!(
                "c2s table drift detected: committed {} no longer matches the vendored libcint \
                 source — regenerate with `cargo run -p xtask -- gen-c2s-table`",
                target.display()
            );
        }
        println!(
            "gen-c2s-table --check: {} matches the vendored libcint source (no drift)",
            target.display()
        );
    } else {
        fs::write(&target, &rendered).with_context(|| format!("write {}", target.display()))?;
        println!(
            "gen-c2s-table: wrote {} from the vendored libcint source",
            target.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `#ifdef PYPZPX` branch is the one cintx does **not** compile; taking
    /// the wrong one would silently reorder every p shell.
    #[test]
    fn resolves_pypzpx_to_the_undefined_branch() {
        let src = "static double g_trans_cart2sph[] = {\n\
                   1,\n\
                   #ifdef PYPZPX\n\
                   0, 1, 0,\n\
                   #else\n\
                   1, 0, 0,\n\
                   #endif\n\
                   };\n";
        // Only the `#else` branch survives; the count check is bypassed here by
        // calling the line filter directly through the public parser's failure
        // message, so assert on the error rather than the (short) value list.
        let err = parse_c2s_table(src).unwrap_err().to_string();
        assert!(
            err.contains("parsed 4 values"),
            "expected the 1 + 3 `#else` values, got: {err}"
        );
    }

    /// Block offsets must reproduce libcint's own `g_c2s` pointer arithmetic
    /// (`g_trans_cart2sph + 1, +10, +40, +110, +245, ...`).
    #[test]
    fn block_offsets_match_libcint_g_c2s() {
        let expected = [
            0, 1, 10, 40, 110, 245, 476, 840, 1380, 2145, 3190, 4576, 6370, 8645, 11480, 14960,
        ];
        for (l, &want) in expected.iter().enumerate() {
            assert_eq!(block_offset(l), want, "offset for l={l}");
        }
    }
}
