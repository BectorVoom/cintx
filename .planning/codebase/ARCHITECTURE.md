# Architecture

**Analysis Date:** 2026-03-14

## Pattern Overview

**Overall:** Hybrid repository with a minimal Rust host crate and a vendored C computational core (`libcint`) that exposes a large generated API surface.

**Key Characteristics:**
- Rust crate exists but is currently a scaffold (`src/main.rs`) with no FFI bridge yet.
- Core domain logic is in vendored C sources built with CMake.
- Many integral entry points are generated C source files under `src/autocode/`.
- Feature flags gate optional integral families and interfaces (Fortran, F12, 4c1e, etc.).

**Evidence:** `Cargo.toml`, `src/main.rs`, `libcint-master/CMakeLists.txt`, `libcint-master/src/autocode/intor1.c`, `libcint-master/src/c2f.c`.

## Layers

**Rust Host Layer:**
- Purpose: Defines the Rust package boundary and executable entry.
- Contains: Cargo manifest and a placeholder binary entry point.
- Depends on: Rust toolchain only at present.
- Used by: Local `cargo run`/`cargo build` workflows.
- Evidence: `Cargo.toml`, `src/main.rs`.

**Public C API Layer:**
- Purpose: Declares the stable callable integral/API surface and shared structs/constants.
- Contains: Function declarations for integral families and core data/slot definitions.
- Depends on: Build-time configured header generation and C core implementation.
- Used by: Examples, tests, external C/Fortran/Python callers.
- Evidence: `libcint-master/include/cint_funcs.h`, `libcint-master/include/cint.h.in`, `libcint-master/doc/program_ref.txt`.

**Driver/Orchestration Layer:**
- Purpose: Initializes per-call environment and orchestrates primitive/contraction loops.
- Contains: Env initialization, cache sizing, loop drivers, optimizer setup/teardown.
- Depends on: Kernel/math layer and representation transforms.
- Used by: Generated API wrappers in `src/autocode`.
- Evidence: `libcint-master/src/cint1e.c`, `libcint-master/src/cint2e.c`, `libcint-master/src/optimizer.c`, `libcint-master/src/cint1e.h`, `libcint-master/src/cint2e.h`.

**Kernel/Math Layer:**
- Purpose: Performs low-level Gaussian integral math and index generation.
- Contains: Core numerical kernels, root solvers, basis helpers, and precomputed data tables.
- Depends on: Internal shared structs/macros and math libs (`libm`, optional quadmath).
- Used by: Driver layer.
- Evidence: `libcint-master/src/g1e.c`, `libcint-master/src/g2e.c`, `libcint-master/src/rys_roots.c`, `libcint-master/src/find_roots.c`, `libcint-master/src/cint_bas.c`, `libcint-master/src/roots_xw.dat`.

**Interop/Validation Layer:**
- Purpose: Provides language interop examples and regression coverage around the C API.
- Contains: Fortran bridge wrappers, C/Python/Fortran/Julia examples, Python tests, CI workflow.
- Depends on: Public API layer and built shared library.
- Used by: Upstream CI and manual verification.
- Evidence: `libcint-master/src/c2f.c`, `libcint-master/examples/c_call_cartesian.c`, `libcint-master/examples/python_call.py`, `libcint-master/testsuite/test_cint.py`, `libcint-master/.github/workflows/ci.yml`.

## Data Flow

**Integral API Call (C/Python/Fortran callers):**

1. Caller invokes an exported integral symbol (for example `int1e_*`/`int2e_*`) declared in `include/cint_funcs.h`.
2. Generated wrapper in `src/autocode/*.c` defines `ng` metadata, initializes `CINTEnvVars`, and binds `envs.f_gout`.
3. Wrapper dispatches to a driver (`CINT1e_drv`, `CINT2e_drv`, `CINT3c*e_drv`) in `cint1e.c`/`cint2e.c`.
4. Driver loop performs primitive/contraction accumulation (`CINT1e_loop`, `CINT2e_loop_nopt`) and optional optimizer-assisted screening (`optimizer.c`).
5. Kernel function pointers in `CINTEnvVars` execute low-level math (`g1e.c`, `g2e.c`) and index mapping.
6. Result buffers are transformed to cartesian/spheric/spinor layout via `cart2sph`/related converters and returned to caller.

**State Management:**
- Call-level state is passed in caller-owned arrays (`atm`, `bas`, `env`, `shls`) and optional `CINTOpt`.
- Temporary scratch uses `cache` buffers; when `out == NULL`, drivers return required cache size.
- Optimizer state is explicit and manually lifecycle-managed (`CINTinit_*_optimizer` / `CINTdel_*_optimizer`).

**Evidence:** `libcint-master/include/cint_funcs.h`, `libcint-master/src/autocode/intor1.c`, `libcint-master/src/cint1e.c`, `libcint-master/src/cint2e.c`, `libcint-master/src/optimizer.c`, `libcint-master/src/g2e.c`, `libcint-master/src/cart2sph.c`, `libcint-master/doc/program_ref.txt`.

## Key Abstractions

**`CINTEnvVars`:**
- Purpose: Canonical per-evaluation execution context (shells, angular data, strides, function pointers, coords).
- Examples: Initialized in `CINTinit_int2e_EnvVars`, consumed by drivers and kernel index routines.
- Pattern: Mutable context struct passed through layered C call chain.
- Evidence: `libcint-master/include/cint.h.in`, `libcint-master/src/g2e.c`, `libcint-master/src/cint1e.c`, `libcint-master/src/cint2e.c`.

**`CINTOpt`:**
- Purpose: Optional precomputed optimizer data for coefficient/pair screening and index caches.
- Examples: `CINTinit_2e_optimizer`, `CINTdel_2e_optimizer`, `CINTall_2e_optimizer`.
- Pattern: Explicitly allocated/free'd cache object passed as pointer.
- Evidence: `libcint-master/include/cint.h.in`, `libcint-master/src/optimizer.c`.

**Integral Family Triplets (`*_cart`, `*_sph`, `*_spinor` + `*_optimizer`):**
- Purpose: Uniform API shape across many operator families and output representations.
- Examples: `int1e_kin_cart/sph/spinor` and `int1e_kin_optimizer`.
- Pattern: Generated wrapper families with common driver internals.
- Evidence: `libcint-master/include/cint_funcs.h`, `libcint-master/src/autocode/intor1.c`.

## Entry Points

**Rust Binary Entry:**
- Location: `src/main.rs`
- Triggers: `cargo run` for the root crate.
- Responsibilities: Currently prints a placeholder message.
- Evidence: `src/main.rs`.

**C Library API Entry Surface:**
- Location: `libcint-master/include/cint_funcs.h` with implementations in `libcint-master/src/autocode/*.c` and `libcint-master/src/*.c`.
- Triggers: Direct C/Fortran calls or `ctypes`/FFI bindings.
- Responsibilities: Accept raw basis/atom/env arrays, dispatch to drivers/kernels, return integral results.
- Evidence: `libcint-master/include/cint_funcs.h`, `libcint-master/src/autocode/intor1.c`, `libcint-master/src/cint2e.c`.

**Build System Entry:**
- Location: `Cargo.toml` and `libcint-master/CMakeLists.txt`.
- Triggers: `cargo` and `cmake` build invocations.
- Responsibilities: Build Rust binary crate and C shared/static library with optional feature flags.
- Evidence: `Cargo.toml`, `libcint-master/CMakeLists.txt`.

## Error Handling

**Strategy:** Status-oriented C API with lightweight runtime guards; no centralized exception object model in C core.

**Patterns:**
- Integral functions return a status indicating whether computed integrals are non-zero.
- Driver APIs support a query mode (`out == NULL`) to return required cache size instead of writing output.
- Runtime invariants are protected with `assert` in core initialization paths.
- Build-time behavior can be modified with flags (for example `KEEP_GOING`) in CMake.

**Evidence:** `libcint-master/README.rst`, `libcint-master/doc/program_ref.txt`, `libcint-master/src/cint1e.c`, `libcint-master/src/g2e.c`, `libcint-master/CMakeLists.txt`.

## Cross-Cutting Concerns

**Logging/Diagnostics:**
- Primarily build/test console output (`message(...)` in CMake, Python test prints/assertions); no unified runtime logger in C core.
- Evidence: `libcint-master/CMakeLists.txt`, `libcint-master/testsuite/test_cint.py`.

**Validation & Regression:**
- CI runs Python regression suites and targeted test scripts across key integral families.
- Examples provide executable smoke tests for multiple language front ends.
- Evidence: `libcint-master/.github/workflows/ci.yml`, `libcint-master/testsuite/test_int2e.py`, `libcint-master/examples/CMakeLists.txt`.

**Performance/Memory Controls:**
- Optional optimizer precomputation and screening logic are integrated through `CINTOpt`.
- Optional build flags (`WITH_F12`, `WITH_4C1E`, vectorization-related options) shape capability/performance.
- Evidence: `libcint-master/src/optimizer.c`, `libcint-master/src/cint2e.c`, `libcint-master/CMakeLists.txt`.

**Authentication/Security Boundary:**
- Not applicable as a local numerical library; no network/auth subsystem is present in code.
- Evidence: `src/main.rs`, `libcint-master/src/*.c`, `libcint-master/examples/*.c`.

---

*Architecture analysis: 2026-03-14*
*Update when major patterns change*
