# Technology Stack

**Analysis Date:** 2026-03-14

## Languages

**Primary:**
- Rust (edition 2024) - active root application crate and executable entrypoint.

**Secondary:**
- C (C99) - vendored upstream numerical library implementation.
- Fortran - optional example programs in vendored upstream project.
- Python - optional vendored upstream test harness.

**Evidence:**
- `Cargo.toml` (`edition = "2024"`)
- `src/main.rs`
- `libcint-master/CMakeLists.txt` (`project (cint C)`, `set(CMAKE_C_STANDARD 99)`)
- `libcint-master/examples/CMakeLists.txt` (`fortran_call_cartesian.F90`, `fortran_call_spheric.F90`, `fortran_call_spinor.F90`)
- `libcint-master/CMakeLists.txt` (`find_package(PythonInterp)`, `add_test(...)`)
- `libcint-master/.travis.yml` (`python-numpy`)

## Runtime

**Environment:**
- Rust toolchain and Cargo for the root `cintx` crate.
- Native C/C++ build toolchain with CMake when building vendored `libcint-master`.

**Package Manager:**
- Cargo (Rust).
- Lockfile: `Cargo.lock` present, with crates sourced from crates.io index.

**Evidence:**
- `Cargo.toml`
- `Cargo.lock` (`source = "registry+https://github.com/rust-lang/crates.io-index"`)
- `libcint-master/README.rst` (CMake build/install instructions)
- `libcint-master/CMakeLists.txt`

## Frameworks

**Core:**
- No web/app framework in the root crate; currently a minimal Rust binary.
- CMake-based build system for vendored `libcint` C library.

**Testing:**
- No Rust test framework configured in the root crate today.
- Vendored `libcint` has optional Python-driven tests enabled by CMake flags.

**Build/Dev:**
- Cargo for Rust crate builds.
- CMake for vendored C library builds and options.

**Evidence:**
- `src/main.rs`
- `Cargo.toml`
- `libcint-master/CMakeLists.txt` (`option(ENABLE_TEST ...)`, `add_test(...)`)

## Key Dependencies

**Critical:**
- `anyhow 1.0.102` - error propagation convenience in Rust code.
- `thiserror 2.0.18` - typed error definitions in Rust code.

**Infrastructure:**
- BLAS and `libm` linkage for vendored C library builds.
- Optional `quadmath` linkage when detected.

**Evidence:**
- `Cargo.toml` (`anyhow = "1.0.102"`, `thiserror = "2.0.18"`)
- `Cargo.lock` (`anyhow 1.0.102`, `thiserror 2.0.18`)
- `libcint-master/README.rst` (BLAS prerequisite)
- `libcint-master/CMakeLists.txt` (`target_link_libraries(cint "-lm")`, `find_package(QUADMATH)`)

## Configuration

**Environment:**
- No runtime environment-variable configuration implemented in the root binary.
- Vendored C library configuration is compile-time via CMake options (for example `WITH_F12`, `WITH_4C1E`, `WITH_FORTRAN`, `WITH_CINT2_INTERFACE`).

**Build:**
- Rust build metadata in `Cargo.toml` / `Cargo.lock`.
- C library build metadata/options in `libcint-master/CMakeLists.txt`.

**Evidence:**
- `src/main.rs`
- `Cargo.toml`
- `Cargo.lock`
- `libcint-master/CMakeLists.txt`

## Platform Requirements

**Development:**
- Any OS with Rust/Cargo support for the root crate.
- C compiler + CMake + BLAS required to build vendored `libcint-master`.

**Production:**
- Current root output is a local CLI-style binary (`main` prints to stdout); no deployment platform config is defined yet.

**Evidence:**
- `src/main.rs`
- `Cargo.toml`
- `libcint-master/README.rst` (prerequisites and build instructions)
- `libcint-master/CMakeLists.txt`

---

*Stack analysis: 2026-03-14*
*Update after major dependency changes*
