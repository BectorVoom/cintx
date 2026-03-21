# Stack Research

**Domain:** Rust-native libcint-compatible integral library
**Researched:** 2026-03-21
**Confidence:** HIGH for core toolchain and foundational crates; MEDIUM for long-term CubeCL ecosystem assumptions.

## Recommended Stack

### Core Platform

| Technology | Version guidance | Purpose | Why recommended |
|------------|------------------|---------|-----------------|
| Rust toolchain | Pin `1.94.0` in `rust-toolchain.toml` | Reproducible compiler behavior across local dev and CI | Rust 1.94.0 is the current stable release as of 2026-03-05, and pinning an exact toolchain keeps oracle and manifest results reproducible. |
| Cargo lockfile | Commit `Cargo.lock`; run CI with `cargo --locked` | Deterministic dependency graph | Oracle comparisons and manifest audits are only credible if every runner uses the same resolved graph. |
| Cargo resolver | Use edition-2024 default `resolver = "3"`; if the root becomes a virtual workspace, declare it explicitly under `[workspace]` | Predictable feature resolution in a multi-crate workspace | Resolver 3 is the 2024-edition default and is the right baseline for the workspace described in the design doc. |
| Multi-crate workspace | Keep the crate split from the design (`core`, `ops`, `runtime`, `cubecl`, `compat`, `capi`, `oracle`, `xtask`) | Isolate domain types, execution, compat, verification, and tooling | The project has hard boundaries between typed API, compat contracts, backend execution, and release gating; the crate layout should reflect them. |

### Core Libraries

| Library | Version guidance | Purpose | Notes |
|---------|------------------|---------|-------|
| `cubecl` | Keep the current `0.9.x` line unless a verified backend issue forces a change | Shared GPU compute backend | `docs.rs` shows `cubecl 0.9.0` as the latest published crate. Keep the public API backend-agnostic enough that a backend swap remains possible if the ecosystem shifts. |
| `thiserror` | `2.0.18` | Public typed error surface | Fits the design requirement for library-facing error enums without leaking implementation details into the API contract. |
| `anyhow` | `1.0.102` | App-boundary, xtask, benchmark, and oracle tooling errors | Matches the design choice to keep ergonomic context-rich errors out of the public library surface. |
| `tracing` | Stay on the current stable `0.1.x` line used by the workspace | Structured spans and diagnostics | Required for planner decisions, chunking, transfers, fallback reasons, and OOM visibility. |
| `bindgen` | Current workspace is `0.71.1`; latest published line is `0.72.1` | Oracle/header binding generation | Upgrade deliberately, not automatically: header-generation changes must be validated against the manifest and oracle harness. |
| `cc` | Keep current stable `1.2.x` line | Vendored upstream libcint build integration | Needed to keep the oracle harness hermetic and reproducible. |

### Supporting Libraries from the Design

| Library | Use | Why it belongs here |
|---------|-----|---------------------|
| `rayon` | Host-side staging and chunk-preparation parallelism | Good fit for CPU-side marshaling without exposing threading complexity in the public API. |
| `smallvec` | Small fixed-ish collections (`dims`, shell tuples, strides) | Cuts heap churn in hot control-plane paths. |
| `num-complex` | Safe API complex/spinor outputs | Better than raw interleaved buffers leaking into typed callers. |
| `approx`, `proptest`, `criterion` | Verification and benchmarking | Match the design's emphasis on oracle comparison, property testing, and repeatable perf baselines. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `cargo nextest` | Faster and more controllable CI test execution | Useful once oracle, feature-matrix, and regression suites become expensive. |
| `rustfmt` + `clippy` | Baseline style and lint enforcement | Already aligned with the current `rust-toolchain.toml` components. |
| `xtask` commands | Manifest audit, oracle refresh, docs generation, bench reporting | Keeps release gates expressed as code instead of tribal knowledge. |

## Alternatives Considered

| Recommended | Alternative | When the alternative is justified |
|-------------|-------------|----------------------------------|
| `cubecl` | Another GPU backend or a CPU compute backend | Only if CubeCL blocks correctness, platform coverage, or maintainability; do not leak CubeCL-specific types into the public API. |
| `thiserror` for library errors | `anyhow` everywhere | Only for internal binaries or scripts; not for the public library contract. |
| Exact toolchain pin | Floating `stable` | Acceptable for quick local experimentation, but not for release-gated CI or oracle baselines. |

## What Not to Use

| Avoid | Why | Use instead |
|-------|-----|-------------|
| Nightly as the project baseline | Changes compiler behavior and weakens reproducibility | Stable Rust pinned in `rust-toolchain.toml` |
| Unpinned dependency resolution in CI | Makes manifest/oracle drift hard to diagnose | `Cargo.lock` plus `cargo --locked` |
| Public APIs that expose backend-specific runtime types | Makes future backend changes expensive and risky | Keep backend details behind planner/executor traits and typed output views |
| Best-effort partial writes on allocation failure | Violates the design's OOM-safe stop contract | Fallible allocation + typed failure + no partial writes |

## Sources

### Official / primary
- Rust 1.94.0 release announcement: https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/
- Cargo resolver guidance: https://doc.rust-lang.org/nightly/cargo/reference/resolver.html
- Cargo feature resolver details: https://doc.rust-lang.org/stable/cargo/reference/features.html
- Cargo nextest docs: https://nexte.st/
- CubeCL crate docs: https://docs.rs/crate/cubecl/latest
- thiserror crate docs: https://docs.rs/crate/thiserror/latest
- anyhow crate docs: https://docs.rs/crate/anyhow/latest
- bindgen crate docs: https://docs.rs/crate/bindgen/0.71.1 and https://docs.rs/crate/bindgen/latest

### Local project evidence
- `Cargo.toml`
- `Cargo.lock`
- `rust-toolchain.toml`
- `docs/design/cintx_detailed_design.md`

---
*Stack research for: cintx*
*Researched: 2026-03-21*
