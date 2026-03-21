# Stack Research

**Domain:** Rust scientific-computing library reimplementation with libcint result-compatibility guarantees
**Researched:** 2026-03-14
**Confidence:** HIGH for CPU/compatibility baseline, MEDIUM for GPU path (CubeCL release churn)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Rust + Cargo workspace | `edition = "2024"`, rustc `>=1.85, <1.90` (pin exact patch in `rust-toolchain.toml`) | Primary implementation language and build system | Matches project direction, enforces safety boundaries, and supports modern lint/safety tooling (`unsafe_op_in_unsafe_fn`, clippy gates) | HIGH |
| Vendored oracle libcint | `libcint 6.1.3` (pinned vendored source) | Compatibility oracle for result and API-family validation | Required by design: compatibility is defined against upstream libcint behavior, not reimplementation internals | HIGH |
| C toolchain + CMake | C99 compiler (`clang`/`gcc`), CMake `>=3.24` | Builds vendored oracle and optional C ABI artifacts | Keeps oracle and C migration path reproducible in CI and local dev | HIGH |
| BLAS provider | OpenBLAS `0.3.x` (preferred) or system BLAS with locked CI image | Numerical backend for upstream-compatible build assumptions and optional benchmark paths | Aligns with upstream performance assumptions and avoids hidden ABI drift | MEDIUM |
| CubeCL backend (feature-gated) | `cubecl 0.9.x` stable line (`gpu` feature, default off) | Optional GPU acceleration backend sharing planner logic with CPU reference | Required by project constraints, but must remain optional with deterministic CPU fallback | MEDIUM |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Confidence |
|---------|---------|---------|-------------|------------|
| `thiserror` | `2.0.18` (`2.0.x` band) | Typed public error surface | Use in all public crates that return library errors | HIGH |
| `anyhow` | `1.0.102` (`1.0.x` band) | Context-rich errors for tools/harnesses | Use only in `xtask`, benchmark binaries, oracle tooling, CI glue | HIGH |
| `tracing` + `tracing-subscriber` | `0.1.44` + `0.3.23` | Structured observability (planner decisions, fallback reasons, OOM) | Use in runtime/planner/backend paths and test diagnostics | HIGH |
| `rayon` | `1.11.0` (`1.11.x` band) | CPU chunk-level parallel execution | Use for synchronous compute parallelism; keep API blocking | HIGH |
| `smallvec` | `1.15.x` stable line | Small fixed-size vectors for dims/shell tuples/strides | Use where hot paths benefit from stack-first storage | HIGH |
| `num-complex` | `0.4.6` (`0.4.x` band) | Complex scalar representation for spinor-related outputs | Use in safe API typing and internal adapters | HIGH |
| `approx` (test-only) | `0.5.1` (`0.5.x` stable line) | Tolerance assertions for oracle comparisons | Use in unit/integration/golden comparison tests | HIGH |
| `proptest` (test-only) | `1.10.0` (`1.10.x` band) | Property-based validation of layout, symmetry, chunk invariants | Use for validators, layout round-trips, fallback invariants | HIGH |
| `criterion` (bench-only) | `0.8.2` (`0.8.x` band) | Reproducible micro/macro benchmark baselines | Use for release-gate regression tracking and CPU/GPU crossover analysis | HIGH |

### Development Tools

| Tool | Version | Purpose | Notes | Confidence |
|------|---------|---------|-------|------------|
| `bindgen` | `0.72.1` (`0.72.x` band) | Generate oracle-side FFI bindings to vendored libcint | Prefer generated bindings over handwritten declarations | HIGH |
| `cc` | `1.2.57` (`1.2.x` band) | Hermetic C compilation from Cargo build scripts | Use in oracle crate and C-shim build integration | HIGH |
| `cbindgen` | `0.29.2` (`0.29.x` band) | Generate C headers for `capi` surface | Keep C ABI docs/header generation tied to Rust API changes | HIGH |
| `cargo-nextest` | `0.9.130` (`0.9.x` band) | Fast, sharded test execution for CI matrix | Use for feature/profile split jobs (`base`, `with-f12`, `with-4c1e`) | HIGH |
| `cargo-llvm-cov` | `0.8.4` (`0.8.x` band) | Coverage gating and trend tracking | Gate critical crates and compatibility paths | HIGH |
| `cargo-deny` + `cargo-audit` | `0.19.0` + `0.22.1` | Supply-chain and vulnerability checks | Run in CI; block releases on high-severity findings | MEDIUM |
| `rustfmt` + `clippy` | Toolchain-matched (same rustc pin) | Style, lint, and unsafe-boundary hygiene | Enforce `-D warnings` selectively for library crates | HIGH |

## Installation

```bash
# System prerequisites (Debian/Ubuntu example)
sudo apt-get update
sudo apt-get install -y build-essential clang cmake pkg-config libopenblas-dev

# Core libraries
cargo add thiserror@2.0.18 anyhow@1.0.102 tracing@0.1.44 tracing-subscriber@0.3.23
cargo add rayon@1.11.0 smallvec@1.15 num-complex@0.4.6
cargo add cubecl@0.9 --optional

# Test/benchmark dependencies
cargo add --dev approx@0.5.1 proptest@1.10.0 criterion@0.8.2

# Build dependencies for oracle + capi
cargo add --build bindgen@0.72.1 cc@1.2.57
cargo install cbindgen --version 0.29.2

# CI/dev workflow tools
cargo install cargo-nextest --version 0.9.130
cargo install cargo-llvm-cov --version 0.8.4
cargo install cargo-deny --version 0.19.0
cargo install cargo-audit --version 0.22.1
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative | Confidence |
|-------------|-------------|-------------------------|------------|
| `cubecl 0.9.x` feature-gated | CUDA-specific stack (`cust`/custom kernels) | Use only if CubeCL blocks required kernels on target hardware and project accepts vendor lock-in | MEDIUM |
| `rayon` for compute parallelism | `tokio` task runtime | Use only at application/service boundary; do not make core integral API async-first | HIGH |
| Vendored libcint + `bindgen` oracle | System-installed libcint linkage | Use only for downstream packaging; never for compatibility gate CI because reproducibility degrades | HIGH |
| Flat-buffer + explicit stride views | `ndarray`-centric tensor model | Use `ndarray` only in tests/examples if needed; avoid in compat/hot compute paths | HIGH |

## What NOT to Use

| Avoid | Why | Use Instead | Confidence |
|-------|-----|-------------|------------|
| Exposing `anyhow::Error` in public APIs | Breaks typed error contracts and weakens downstream handling | Public `thiserror` enums + structured variants | HIGH |
| `ndarray` as core compat/output representation | Fights libcint-compatible flat layout and stride-level control requirements | Custom tensor/view structs over contiguous buffers | HIGH |
| Pre-release crates in the default shipping path (`cubecl 0.10.0-pre.*`, `smallvec 2.0.0-alpha.*`, `approx 0.6.0-rc*`) | Raises upgrade risk and CI nondeterminism | Stable lines (`cubecl 0.9.x`, `smallvec 1.15.x`, `approx 0.5.x`) | HIGH |
| Direct large-path `vec![0; n]` / `Vec::with_capacity(n)` allocations in runtime kernels | Conflicts with OOM-safe-stop requirement and centralized fallible allocation policy | `WorkspaceAllocator` + fallible buffer wrappers + chunk planner | HIGH |
| Async-first public compute API | Adds scheduling complexity and weakens deterministic fallback semantics in initial scope | Blocking public API with internal parallelism/GPU queues | HIGH |
| Public GTG feature in GA scope | Explicitly out-of-scope/deprecated in current design constraints | Keep GTG roadmap-only and excluded from public manifest | HIGH |

## Stack Patterns by Variant

**If strict compatibility release candidate (`base` profile):**
- Use CPU backend as reference and require oracle comparison + helper parity gates.
- Keep `gpu` optional and non-blocking for base release acceptance.
- Confidence: HIGH.

**If GPU performance profile (`gpu` + large homogeneous batches):**
- Enable `cubecl 0.9.x` and enforce CPU fallback for unsupported families/representations.
- Track CPU/GPU consistency in CI, not benchmark-only jobs.
- Confidence: MEDIUM.

**If optional-family profile (`with-f12`, `with-4c1e`):**
- Gate F12/STG/YP as sph-only; gate 4c1e to validated bug-envelope inputs.
- Reject out-of-envelope requests as typed `UnsupportedApi` instead of silent best-effort.
- Confidence: HIGH.

**If C migration profile (`capi` enabled):**
- Generate headers with `cbindgen`; keep raw compat contracts (`atm/bas/env`, `dims`, buffer sizing) authoritative.
- Keep safe Rust API and C ABI shim separate; do not leak raw-pointer ergonomics upward.
- Confidence: HIGH.

## Version Compatibility

| Package A | Compatible With | Notes | Confidence |
|-----------|-----------------|-------|------------|
| `rustc >=1.85,<1.90` | `edition = 2024` | Pin exact patch in `rust-toolchain.toml` for CI reproducibility | HIGH |
| `thiserror 2.0.x` | `anyhow 1.0.x` | Public-vs-boundary error split keeps API stable and tooling ergonomic | HIGH |
| `tracing 0.1.x` | `tracing-subscriber 0.3.x` | Use subscriber config in binaries/tests, not core library constructors | HIGH |
| `rayon 1.11.x` | Blocking public API model | Safe fit for chunk-level CPU parallelism without async surface changes | HIGH |
| `cubecl 0.9.x` | `gpu` feature + CPU fallback path | Treat as optional acceleration; never hard requirement for correctness | MEDIUM |
| `bindgen 0.72.x` | `clang/libclang` in CI image | Keep toolchain image pinned for deterministic generated bindings | MEDIUM |
| `approx 0.5.x` | `proptest 1.10.x`, `criterion 0.8.x` | Stable tolerance/assertion stack for compat + regression testing | HIGH |

## Sources

- `/home/chemtech/workspace/cintx/.planning/PROJECT.md` — project goals, constraints, compatibility target (HIGH)
- `/home/chemtech/workspace/cintx/docs/libcint_detailed_design_resolved_en.md` — authoritative architecture, feature gates, error/memory/test policies (HIGH)
- `/home/chemtech/workspace/cintx/.planning/codebase/STACK.md` — current repository baseline and existing dependencies (HIGH)
- crates.io API (`https://crates.io/api/v1/crates/<name>`) checked on 2026-03-14 — current crate release lines for version bands (MEDIUM)

---
*Stack research for: Rust redesign/reimplementation of libcint with compatibility guarantees*
*Researched: 2026-03-14*
