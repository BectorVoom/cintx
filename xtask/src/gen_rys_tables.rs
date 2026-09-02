//! `gen-rys-tables` — extract libcint 6.1.3's Jacobi/Flocke Rys constant tables
//! from the vendored C source and emit them as a Rust data module.
//!
//! Background (Phase 25 FND-02 / D-01)
//! -----------------------------------
//! The host Wheeler/Jacobi `nroots >= 6` Rys engine
//! (`crates/cintx-cubecl/src/math/rys_wheeler.rs`) consumes a family of constant
//! tables that libcint emits as `static double NAME[] = { ... };` arrays:
//!
//! - `JACOBI_ALPHA`, `JACOBI_BETA`, `JACOBI_RN_PART2`, `JACOBI_SN`, `JACOBI_COEF`
//!   (f64) and `JACOBI_COEF_ORDER` (int) — `libcint-master/src/rys_wheeler.c`.
//! - the long-double siblings `lJACOBI_ALPHA/BETA/RN_PART2/SN/COEF`
//!   (stored as f64; the cintx vendor build disables `HAVE_SQRTL`/`HAVE_QUADMATH_H`).
//! - `TURNOVER_POINT` — `libcint-master/src/fmt.c` (the FMT power-series cutoff).
//!
//! Byte-identity vs the vendor requires the embedded literals to round to the
//! exact same f64 the C compiler emits (RESEARCH §FND-02 "Don't Hand-Roll":
//! transcription errors are silent). This subcommand parses those arrays directly
//! out of the vendored C source and writes `roots_jacobi_data.rs` (generate path),
//! or re-derives and byte-compares against the committed file, failing closed on
//! any diff (the `--check` drift gate, mirroring `gen-ecp-tables` / D-04).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Vendored libcint source dir (relative to workspace root).
const LIBCINT_SRC_REL: &str = "libcint-master/src";
/// Target generated Rust data module (relative to workspace root).
const TARGET_REL: &str = "crates/cintx-cubecl/src/math/roots_jacobi_data.rs";

/// One table spec: Rust name == C name (upper-cased), source file, integer flag.
struct TableSpec {
    name: &'static str,
    file: &'static str,
    is_int: bool,
}

const TABLES: &[TableSpec] = &[
    TableSpec {
        name: "JACOBI_ALPHA",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "JACOBI_BETA",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "JACOBI_RN_PART2",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "JACOBI_SN",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "JACOBI_COEF",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "JACOBI_COEF_ORDER",
        file: "rys_wheeler.c",
        is_int: true,
    },
    TableSpec {
        name: "lJACOBI_ALPHA",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "lJACOBI_BETA",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "lJACOBI_RN_PART2",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "lJACOBI_SN",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "lJACOBI_COEF",
        file: "rys_wheeler.c",
        is_int: false,
    },
    TableSpec {
        name: "TURNOVER_POINT",
        file: "fmt.c",
        is_int: false,
    },
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest dir has a parent (workspace root)")
        .to_path_buf()
}

/// A parsed table value (f64 or int), preserving the source ordering.
enum Vals {
    F(Vec<f64>),
    I(Vec<i64>),
}

/// Parse `static (double|long double|int) NAME[] = { ... };` out of the C source.
///
/// Strips `/* */` and `//` comments, splits on commas, and drops the
/// long-double/quad/float literal suffixes (`l`, `q`, `f`) before parsing.
fn parse_array(source: &str, name: &str, is_int: bool) -> Result<Vals> {
    // The C declaration may be `static double NAME[]`, `static long double NAME[]`,
    // or `static int NAME[]`. Find the first `NAME[]` token followed by `=`.
    let needle = format!("{name}[]");
    let decl_pos = source
        .find(&needle)
        .with_context(|| format!("could not find `{needle}` in source"))?;
    let after = &source[decl_pos..];
    let brace_rel = after
        .find('{')
        .with_context(|| format!("no opening brace after `{needle}`"))?;
    let body_start = decl_pos + brace_rel + 1;
    let close_rel = source[body_start..]
        .find("};")
        .with_context(|| format!("no closing `}};` for `{needle}`"))?;
    let mut body = source[body_start..body_start + close_rel].to_string();

    // Strip /* ... */ block comments.
    while let Some(s) = body.find("/*") {
        if let Some(e) = body[s..].find("*/") {
            body.replace_range(s..s + e + 2, "");
        } else {
            break;
        }
    }

    let mut f: Vec<f64> = Vec::new();
    let mut i: Vec<i64> = Vec::new();
    for raw_line in body.lines() {
        let line = match raw_line.find("//") {
            Some(idx) => &raw_line[..idx],
            None => raw_line,
        };
        for token in line.split(',') {
            let tok = token.trim();
            if tok.is_empty() {
                continue;
            }
            // Strip a single trailing long-double/quad/float suffix.
            let tok = tok.trim_end_matches(['l', 'L', 'q', 'Q', 'f', 'F']);
            if is_int {
                let v: i64 = tok
                    .parse()
                    .with_context(|| format!("failed to parse int literal `{tok}` in `{name}`"))?;
                i.push(v);
            } else {
                let v: f64 = tok
                    .parse()
                    .with_context(|| format!("failed to parse f64 literal `{tok}` in `{name}`"))?;
                f.push(v);
            }
        }
    }
    Ok(if is_int { Vals::I(i) } else { Vals::F(f) })
}

/// Render the whole `roots_jacobi_data.rs` module from the vendored source.
fn render_module(root: &Path) -> Result<String> {
    let mut out = String::new();
    out.push_str(
        "//! AUTO-GENERATED Rys Jacobi/Flocke constant tables for Phase 25 FND-02 (Task 1b/2).\n\
         //!\n\
         //! Extracted verbatim from the vendored libcint 6.1.3 source\n\
         //! (`libcint-master/src/rys_wheeler.c` JACOBI_* tables, `src/fmt.c` TURNOVER_POINT).\n\
         //! DO NOT hand-edit. Regenerate via `cargo run -p xtask -- gen-rys-tables`\n\
         //! (Task 2 adds the `--check` drift-gate that re-derives and diffs these against the C source).\n\
         //!\n\
         //! Long-double (`lJACOBI_*`) tables are stored as f64. The cintx vendor build disables\n\
         //! `HAVE_SQRTL`/`HAVE_QUADMATH_H` (build.rs), so the lrys path uses c99_sqrtl/c99_expl\n\
         //! (f64-backed); these decimal literals round to the same f64 the C compiler emits.\n\
         //!\n\
         //! Every `static` below carries `#[rustfmt::skip]`: the packed layout is what this\n\
         //! generator emits, and without the attribute `cargo fmt --all` would explode each\n\
         //! array to one value per line and `--check` would then report drift on every\n\
         //! formatting run.\n\
         \n\
         #![allow(clippy::all)]\n\
         #![allow(clippy::approx_constant)]\n\
         \n",
    );

    for spec in TABLES {
        let src_path = root.join(LIBCINT_SRC_REL).join(spec.file);
        let source = fs::read_to_string(&src_path)
            .with_context(|| format!("read vendored C source {}", src_path.display()))?;
        let vals = parse_array(&source, spec.name, spec.is_int)?;
        let rname = spec.name.to_uppercase();
        match vals {
            Vals::F(v) => {
                out.push_str(&format!(
                    "/// `{}` ({} entries) — from libcint {}.\n",
                    spec.name,
                    v.len(),
                    spec.file
                ));
                out.push_str(RUSTFMT_SKIP);
                out.push_str(&format!("pub static {rname}: [f64; {}] = [\n", v.len()));
                push_values(&mut out, v.iter().map(|x| format!("{x:?}")));
                out.push_str("];\n\n");
            }
            Vals::I(v) => {
                out.push_str(&format!(
                    "/// `{}` ({} entries) — from libcint {}.\n",
                    spec.name,
                    v.len(),
                    spec.file
                ));
                out.push_str(RUSTFMT_SKIP);
                out.push_str(&format!("pub static {rname}: [i32; {}] = [\n", v.len()));
                push_values(&mut out, v.iter().map(|x| format!("{x}")));
                out.push_str("];\n\n");
            }
        }
    }
    // The loop leaves a blank line after the last table, and rustfmt strips a
    // trailing blank line at EOF — a one-byte difference that shows up as
    // "drift". Trim it here so the rendered text is already fmt-stable.
    while out.ends_with("\n\n") {
        out.pop();
    }
    Ok(out)
}

/// Emitted before every generated `static`.
///
/// **Load-bearing, not cosmetic.** `push_values` packs several values per line
/// to keep a 2080-entry table readable; rustfmt's default for an array literal
/// is one element per line. `roots_jacobi_data.rs` is an ordinary module in
/// `cintx-cubecl`, so `cargo fmt --all` rewrites it — and the committed file
/// then stops matching what this generator emits, so `--check` reports drift
/// that is pure formatting.
///
/// That is exactly what happened: the gate was red from the first `cargo fmt`
/// run after generation until 2026-09-03, with the committed table
/// *value-identical* to the vendored source the whole time. `gen-c2s-table`
/// carries the same guard for the same reason.
const RUSTFMT_SKIP: &str = "#[rustfmt::skip]\n";

/// Append comma-separated values, wrapping at ~100 columns.
fn push_values(out: &mut String, items: impl Iterator<Item = String>) {
    let mut line = String::from("    ");
    for it in items {
        let tok = format!("{it}, ");
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

/// Entry point for `gen-rys-tables [--check]`.
///
/// - `check == false`: re-derive and write `roots_jacobi_data.rs`.
/// - `check == true`:  re-derive and byte-compare against the committed file,
///   failing closed (non-zero exit) on any divergence (the D-04 drift gate).
pub fn run_gen_rys_tables(check: bool) -> Result<()> {
    let root = workspace_root();
    let target = root.join(TARGET_REL);
    let rendered = render_module(&root)?;

    if check {
        let committed = fs::read_to_string(&target)
            .with_context(|| format!("read committed {} for drift-check", target.display()))?;
        if committed != rendered {
            bail!(
                "Rys table drift detected: committed {} no longer matches the vendored libcint \
                 source — regenerate with `cargo run -p xtask -- gen-rys-tables`",
                target.display()
            );
        }
        println!(
            "gen-rys-tables --check: {} matches the vendored libcint source (no drift)",
            target.display()
        );
    } else {
        fs::write(&target, &rendered).with_context(|| format!("write {}", target.display()))?;
        println!(
            "gen-rys-tables: wrote {} from the vendored libcint source",
            target.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The rendered module must carry `#[rustfmt::skip]` on every `static`, or
    /// the next `cargo fmt --all` silently re-breaks the drift gate. This is the
    /// regression anchor for that: the gate was red for three months because the
    /// attribute was missing.
    #[test]
    fn every_generated_static_is_fmt_skipped() {
        let module = super::render_module(&super::workspace_root())
            .expect("render the module from the vendored source");
        // Count the emitted attribute *line*, not every mention of the string:
        // the module header talks about the attribute too.
        let statics = module
            .lines()
            .filter(|l| l.starts_with("pub static "))
            .count();
        let skips = module
            .lines()
            .filter(|l| l.trim_end() == "#[rustfmt::skip]")
            .count();
        assert_eq!(statics, super::TABLES.len(), "one static per table spec");
        assert_eq!(
            skips, statics,
            "every generated static needs #[rustfmt::skip]; without it `cargo fmt` \
             reformats the committed file and `--check` reports formatting as drift"
        );
        assert!(
            !module.ends_with("\n\n"),
            "a trailing blank line is stripped by rustfmt, which `--check` then \
             reports as drift"
        );
    }

    use super::*;

    #[test]
    fn parses_double_array_with_suffix() {
        let src = "static long double _demo[] = { 1.5l, 2.5l, /* c */ 3.0l };\n";
        match parse_array(src, "_demo", false).unwrap() {
            Vals::F(v) => assert_eq!(v, vec![1.5, 2.5, 3.0]),
            _ => panic!("expected f64"),
        }
    }

    #[test]
    fn parses_int_array() {
        let src = "static int _demo[] = { 0, 1, // x\n2, 3 };\n";
        match parse_array(src, "_demo", true).unwrap() {
            Vals::I(v) => assert_eq!(v, vec![0, 1, 2, 3]),
            _ => panic!("expected int"),
        }
    }
}
