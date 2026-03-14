# Codebase Structure

**Analysis Date:** 2026-03-14

## Directory Layout

```
cintx/
├── .planning/                    # Planning artifacts
│   └── codebase/                 # Codebase analysis docs (this folder)
├── docs/                         # Project design documents
│   └── libcint_detailed_design_resolved_en.md
├── src/                          # Root Rust crate source
│   └── main.rs
├── libcint-master/               # Vendored upstream libcint C project
│   ├── .github/workflows/        # Upstream CI definitions
│   ├── cmake/                    # CMake helper modules
│   ├── doc/                      # Upstream program reference docs
│   ├── examples/                 # C/Fortran/Python/Julia examples
│   ├── include/                  # Public C headers/templates
│   ├── src/                      # C implementation + numeric data
│   │   └── autocode/             # Generated integral family wrappers
│   └── testsuite/                # Python regression tests
├── Cargo.toml                    # Root Rust manifest
├── Cargo.lock                    # Root Rust lockfile
├── README.md                     # Root README (minimal)
├── LICENSE                       # License
└── libcint-master.zip            # Archived vendor source snapshot
```

**Evidence:** `Cargo.toml`, `src/main.rs`, `docs/libcint_detailed_design_resolved_en.md`, `libcint-master/CMakeLists.txt`, `libcint-master/src/autocode/intor1.c`.

## Directory Purposes

**`.planning/`:**
- Purpose: Workspace planning metadata and generated analysis docs.
- Contains: `.planning/codebase/*.md` documents.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.
- Subdirectories: `codebase/`.

**`docs/`:**
- Purpose: High-level design intent for Rust reimplementation work.
- Contains: Markdown design spec(s).
- Key files: `docs/libcint_detailed_design_resolved_en.md`.
- Subdirectories: None currently.

**`src/`:**
- Purpose: Root Rust crate implementation.
- Contains: Rust source files.
- Key files: `src/main.rs`.
- Subdirectories: None currently.

**`libcint-master/include/`:**
- Purpose: Public C API headers and build-configured header template.
- Contains: `cint_funcs.h` declarations and `cint.h.in` template.
- Key files: `libcint-master/include/cint_funcs.h`, `libcint-master/include/cint.h.in`.
- Subdirectories: None currently.

**`libcint-master/src/`:**
- Purpose: Core C implementation, math kernels, basis helpers, and data tables.
- Contains: `cint*.c`, `g*.c`, optimizer logic, root-finding code, large `*.dat` tables.
- Key files: `libcint-master/src/cint1e.c`, `libcint-master/src/cint2e.c`, `libcint-master/src/g2e.c`, `libcint-master/src/optimizer.c`.
- Subdirectories: `autocode/` for generated wrappers.

**`libcint-master/src/autocode/`:**
- Purpose: Generated integral-family wrapper implementations.
- Contains: Generated files such as `intor1.c`, `intor2.c`, `deriv3.c`, `hess.c`.
- Key files: `libcint-master/src/autocode/intor1.c`, `libcint-master/src/autocode/intor4.c`.
- Subdirectories: None.

**`libcint-master/examples/`:**
- Purpose: Usage examples in multiple languages.
- Contains: C, Fortran, Python, Julia example programs and benchmarks.
- Key files: `libcint-master/examples/c_call_cartesian.c`, `libcint-master/examples/python_call.py`, `libcint-master/examples/CMakeLists.txt`.
- Subdirectories: None.

**`libcint-master/testsuite/`:**
- Purpose: Regression and correctness tests (Python).
- Contains: `test_*.py` scripts by feature/family.
- Key files: `libcint-master/testsuite/test_cint.py`, `libcint-master/testsuite/test_int2e.py`, `libcint-master/testsuite/test_rys_roots.py`.
- Subdirectories: None.

**`libcint-master/doc/`:**
- Purpose: Upstream reference docs and packaging description.
- Contains: program reference in txt/tex/pdf and CPack description.
- Key files: `libcint-master/doc/program_ref.txt`, `libcint-master/doc/libcint.CPack.txt`.
- Subdirectories: None.

**Evidence:** `docs/libcint_detailed_design_resolved_en.md`, `src/main.rs`, `libcint-master/include/cint_funcs.h`, `libcint-master/src/cint2e.c`, `libcint-master/src/autocode/intor1.c`, `libcint-master/examples/c_call_cartesian.c`, `libcint-master/testsuite/test_cint.py`, `libcint-master/doc/program_ref.txt`.

## Key File Locations

**Entry Points:**
- `src/main.rs`: Root Rust executable entry point.
- `libcint-master/include/cint_funcs.h`: Public declarations for callable integral entry points.
- `libcint-master/src/autocode/intor1.c`: Generated concrete implementations for many `int1e_*` families.
- `libcint-master/CMakeLists.txt`: Native build entry for the vendored C library.

**Configuration:**
- `Cargo.toml`: Rust package metadata and dependencies.
- `.gitignore`: Root ignore policy (Rust build artifacts, tooling outputs).
- `libcint-master/CMakeLists.txt`: CMake build options/features (`WITH_F12`, `WITH_4C1E`, etc.).
- `libcint-master/.github/workflows/ci.yml`: Upstream CI build/test matrix.
- `libcint-master/.gitignore`: C/CMake build artifact ignores for vendored subtree.

**Core Logic:**
- `libcint-master/src/cint1e.c`: 1-electron integral driver loop and cache handling.
- `libcint-master/src/cint2e.c`: 2-electron integral driver loop and contraction orchestration.
- `libcint-master/src/g2e.c`: 2-electron env setup and indexing/kernel dispatch support.
- `libcint-master/src/optimizer.c`: Optional optimizer lifecycle and precomputed tables.
- `libcint-master/src/cint_bas.c`: Basis and shell helper utilities.

**Testing:**
- `libcint-master/testsuite/test_cint.py`: Broad API regression script.
- `libcint-master/testsuite/test_int1e.py`: 1e-focused tests.
- `libcint-master/testsuite/test_int2e.py`: 2e-focused tests.
- `libcint-master/testsuite/test_cart2sph.py`: representation conversion tests.

**Documentation:**
- `README.md`: Root-level project name marker.
- `docs/libcint_detailed_design_resolved_en.md`: Rust redesign design spec.
- `libcint-master/README.rst`: Upstream libcint usage/build reference.
- `libcint-master/doc/program_ref.txt`: Detailed API layout and buffer contract reference.

**Evidence:** `src/main.rs`, `Cargo.toml`, `.gitignore`, `libcint-master/CMakeLists.txt`, `libcint-master/src/cint1e.c`, `libcint-master/testsuite/test_cint.py`, `docs/libcint_detailed_design_resolved_en.md`.

## Naming Conventions

**Files:**
- `snake_case.rs` for Rust sources (example: `src/main.rs`).
- Lowercase C files with domain prefixes in vendored core (`cint*.c`, `g*.c`, `cart2sph.c`, `optimizer.c`).
- Generated wrapper files in `autocode` use family names (`intor1.c`, `deriv3.c`, `hess.c`).
- Python tests follow `test_*.py` convention.
- Top-level project metadata docs use uppercase names (`README.md`, `LICENSE`).

**Directories:**
- Lowercase directory names across root and vendored subtree (`docs`, `src`, `include`, `testsuite`, `examples`, `autocode`).
- Functional grouping inside vendored project (`include` for API, `src` for implementation, `testsuite` for tests).

**Special Patterns:**
- Representation/function suffixes are systematic in API names (`*_cart`, `*_sph`, `*_spinor`, `*_optimizer`).
- Build-time template files use `.in` suffix (`libcint-master/include/cint.h.in`, `libcint-master/src/cint_config.h.in`).
- Imported Windows metadata sidecars are present as `*:Zone.Identifier` files in vendored content.

**Evidence:** `src/main.rs`, `libcint-master/src/cint2e.c`, `libcint-master/src/autocode/intor1.c`, `libcint-master/testsuite/test_int2e.py`, `README.md`, `libcint-master/include/cint.h.in`, `libcint-master/CMakeLists.txt:Zone.Identifier`.

## Where to Add New Code

**New Rust Feature:**
- Primary code: `src/`
- Tests: Add Rust tests near module files or under a future `tests/` directory at repo root.
- Config if needed: `Cargo.toml`
- Evidence: `src/main.rs`, `Cargo.toml`.

**New Integral/Kernel Work in Vendored C Core:**
- Definition surface: `libcint-master/include/cint_funcs.h`
- Driver/kernel implementation: `libcint-master/src/`
- Generated-family updates: `libcint-master/src/autocode/` (generated outputs currently committed here)
- Build wiring: `libcint-master/CMakeLists.txt`
- Evidence: `libcint-master/include/cint_funcs.h`, `libcint-master/src/cint2e.c`, `libcint-master/src/autocode/intor1.c`, `libcint-master/CMakeLists.txt`.

**New Regression Tests:**
- Primary test scripts: `libcint-master/testsuite/`
- Example-driven smoke tests: `libcint-master/examples/`
- CI invocation: `libcint-master/.github/workflows/ci.yml`
- Evidence: `libcint-master/testsuite/test_cint.py`, `libcint-master/examples/c_call_cartesian.c`, `libcint-master/.github/workflows/ci.yml`.

**Documentation Updates:**
- Rust redesign docs: `docs/`
- Upstream API/build docs: `libcint-master/README.rst`, `libcint-master/doc/`
- Evidence: `docs/libcint_detailed_design_resolved_en.md`, `libcint-master/README.rst`, `libcint-master/doc/program_ref.txt`.

## Special Directories

**`libcint-master/src/autocode/`:**
- Purpose: Generated C source for many integral families.
- Source: Generated externally (file headers say "code generated by gen-code.cl").
- Committed: Yes (present in repository tree and included in `cintSrc` list in CMake).
- Evidence: `libcint-master/src/autocode/intor1.c`, `libcint-master/CMakeLists.txt`.

**`.planning/codebase/`:**
- Purpose: Planning/codebase analysis artifacts for this workspace.
- Source: Generated by planning/mapping workflows.
- Committed: Project-dependent; currently present as workspace docs.
- Evidence: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

**`libcint-master.zip`:**
- Purpose: Archived vendor source snapshot alongside extracted `libcint-master/`.
- Source: External archive imported into repository workspace.
- Committed: Present in root tree.
- Evidence: `libcint-master.zip`, `libcint-master/`.

**`*:Zone.Identifier` sidecar files:**
- Purpose: Windows-origin metadata attached to imported files.
- Source: Download/extraction provenance.
- Committed: Present in workspace vendored subtree and not ignored by root/vendored `.gitignore`.
- Evidence: `libcint-master/CMakeLists.txt:Zone.Identifier`, `libcint-master/.github/workflows/ci.yml:Zone.Identifier`, `.gitignore`, `libcint-master/.gitignore`.

---

*Structure analysis: 2026-03-14*
*Update when directory structure changes*
