# Technology Stack

**Analysis Date:** 2026-03-21

## Languages

**Primary:**
- Rust (edition `2024`) - crate implementation in `Cargo.toml`, `src/lib.rs`, `src/runtime/`, `src/api/`, and `tests/`.
- C (vendored `libcint`) - native kernels compiled from `libcint-master/src/*.c` and `libcint-master/src/autocode/*.c` via `build.rs`.

**Secondary:**
- YAML - CI automation in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.
- JSON - governance lock artifacts in `compiled_manifest.lock.json` and `src/runtime/backend/cpu/route_coverage_manifest.lock.json`.

## Runtime

**Environment:**
- Native Rust crate runtime with both library and binaries from `src/lib.rs`, `src/main.rs`, and `src/bin/manifest_audit.rs`.
- Build-time environment injected by Cargo (`CARGO_MANIFEST_DIR`, `OUT_DIR`) and consumed in `build.rs`.

**Package Manager:**
- Cargo - dependency and build orchestration from `Cargo.toml`.
- Lockfile: present (`Cargo.lock`).
- Toolchain pinning: `rust-toolchain` and `rust-toolchain.toml` not detected at repo root; CI installs stable Rust in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.

## Frameworks

**Core:**
- `libcint` `0.2.2` with feature `with_4c1e` - primary numerical backend integration declared in `Cargo.toml` and exercised in `tests/common/oracle_runner.rs`.
- `cubecl` `0.9.0` - compute framework dependency declared in `Cargo.toml`.

**Testing:**
- Rust built-in test harness (`cargo test`) - invoked directly by workflow gates in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.

**Build/Dev:**
- `cc` `1.2.15` - compiles vendored C sources in `build.rs`.
- GitHub Actions (`actions/checkout@v4`, `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2`) - CI execution and caching in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`.

## Key Dependencies

**Critical:**
- `libcint` `0.2.2` - wrapper/oracle compatibility and symbol-level route parity (`Cargo.toml`, `tests/*_wrapper_parity.rs`, `src/runtime/backend/cpu/ffi.rs`).
- `serde` `1.0.228` + `serde_json` `1.0.145` - manifest lock serialization and parsing (`src/manifest/lock.rs`, `src/manifest/compiled.rs`).
- `thiserror` `2.0.18` - typed error taxonomy (`src/errors/libcint_error.rs`, `src/diagnostics/report.rs`).

**Infrastructure:**
- `tracing` `0.1.41` - diagnostics logging in `src/diagnostics/report.rs`.
- `anyhow` `1.0.102` - general error handling dependency declared in `Cargo.toml`.
- `cc` `1.2.15` (build dependency) - native archive build path in `build.rs`.

## Configuration

**Environment:**
- Optional route policy is controlled by feature-flag labels processed in `src/runtime/policy.rs` and route entries in `src/runtime/backend/cpu/router.rs`.
- Build configuration depends on Cargo-provided variables (`CARGO_MANIFEST_DIR`, `OUT_DIR`) in `build.rs`.
- `.env`-style files are not detected at repository root (no matches from `ls .env*` in `/home/chemtech/workspace/cintx`).

**Build:**
- Build definition files: `Cargo.toml`, `Cargo.lock`, and `build.rs`.
- Vendored source requirement: `build.rs` requires `libcint-master/` and validates template/source presence before compilation.
- Manifest/route lock inputs: `compiled_manifest.lock.json` and `src/runtime/backend/cpu/route_coverage_manifest.lock.json`.

## Platform Requirements

**Development:**
- Rust + Cargo toolchain (stable channel used by CI in `.github/workflows/compat-governance-pr.yml` and `.github/workflows/compat-governance-release.yml`).
- Native C compiler/linker for `cc` crate compilation and `libm` link (`println!("cargo:rustc-link-lib=m")` in `build.rs`).
- Linux CI baseline (`runs-on: ubuntu-latest`) in both workflow files under `.github/workflows/`.

**Production:**
- Hosted deployment target: Not detected (no deploy manifests or hosting descriptors in repository root).
- Artifact shape: Rust crate/library surfaces in `src/lib.rs` plus governance CLI in `src/bin/manifest_audit.rs`.

---

*Stack analysis: 2026-03-21*
