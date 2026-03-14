# Project Research Summary

**Project:** libcint-rs
**Domain:** Rust-first quantum-chemistry integral engine with libcint-compatible results
**Researched:** 2026-03-14
**Confidence:** HIGH

## Executive Summary

`libcint-rs` is a Rust redesign/reimplementation of `libcint 6.1.3` where compatibility is defined by numerical-result parity and API-family behavior, not by reproducing C internals. The research converges on a layered architecture: safe Rust API, raw compat API, and optional C ABI shim all flowing through one shared runtime (validation, planning, chunking, dispatch), with CPU as the correctness reference and CubeCL as optional acceleration.

The recommended build strategy is contract-first and gate-first: pin toolchains and oracle versions, make manifest lock/audit (`compiled_manifest.lock.json`) authoritative for coverage claims, centralize `dims` and layout semantics, and enforce OOM-safe stop behavior through fallible allocation and chunk planning. This allows staged delivery of stable families while keeping optional families (`with-f12`, `with-4c1e`) and unstable-source APIs isolated behind explicit gates.

Primary delivery risk is drift: manifest drift, layout drift (especially spinor/complex), and policy drift for optional families. The mitigation pattern is consistent across files: shared runtime, centralized validators/writers, oracle CI matrix across feature profiles, strict negative tests, and release gates that block unapproved lock/symbol changes.

## Key Findings

### Recommended Stack

The stack is Rust `edition = 2024` with pinned rustc band (`>=1.85,<1.90`), vendored `libcint 6.1.3` as oracle, C/CMake toolchain for reproducible oracle and C ABI builds, and feature-gated CubeCL `0.9.x` for optional GPU acceleration. Supporting crates emphasize explicit error contracts (`thiserror` public, `anyhow` tooling-only), observability (`tracing`), deterministic CPU parallelism (`rayon`), and robust compatibility testing (`approx`, `proptest`, `criterion`).

The strongest recommendation is to treat reproducibility as part of architecture: pin toolchains, generate bindings/headers (`bindgen`, `cbindgen`) in controlled CI, and make manifest/oracle audits non-optional release criteria.

**Core technologies:**
- Rust/Cargo workspace (`edition 2024`, pinned toolchain): primary implementation and safety/lint boundary enforcement.
- Vendored `libcint 6.1.3` + `bindgen`: compatibility oracle for results and API-family verification.
- Shared runtime + CPU reference backend (`rayon`): deterministic baseline behavior and performance.
- CubeCL `0.9.x` (`gpu` feature, default off): optional acceleration with deterministic CPU fallback.
- CI/tooling (`cargo-nextest`, `cargo-llvm-cov`, manifest audit): enforce compatibility claims and regression gates.

### Expected Features

Launch viability depends on parity and migration contracts, not only kernel speed. Table-stakes are stable-family compatibility, raw `atm/bas/env` contract parity (`shls/dims/cache/opt` semantics), safe Rust API (`query_workspace`/`evaluate_into`), OOM-safe stop behavior, helper/transform parity, and automated release gates.

Differentiators are shared planner CPU/GPU dispatch with traceable fallback, manifest-driven stability governance, and strict unsafe minimization. The research is explicit that async public API, GTG exposure, and broad unstable-family promotion should be deferred.

**Must have (table stakes):**
- Stable-family result compatibility with oracle-backed CI gates.
- Raw compatibility API parity including sentinel and `dims` behavior.
- Safe Rust API with typed validation and typed errors.
- OOM-safe stop semantics via centralized fallible allocation/chunking.
- Helper/transform/optimizer baseline parity required for migration.

**Should have (competitive):**
- Shared planner across CPU/CubeCL with deterministic fallback reasons.
- Manifest-driven feature/stability matrix (`stable`, `optional`, `unstable_source`).
- Optional C ABI shim for phased migration.
- First-class tracing for planner/dispatch/fallback/OOM diagnostics.

**Defer (v2+):**
- Public async API surface.
- Broad unstable-source promotion to stable.
- GTG in GA scope (unless independently validated later).

### Architecture Approach

Architecture should remain layered and execution-centralized: API surfaces (`facade`, `compat`, optional `capi`) feed one runtime that performs validation, manifest resolution, planning, chunking, workspace control, and backend dispatch. Backends (`cpu`, optional `cubecl`) are isolated behind executor traits, while `oracle`/`xtask` own parity and release gates. This prevents semantic drift between safe and raw interfaces and makes coverage/performance claims auditable.

**Major components:**
1. `libcint-core` + `libcint-ops` — domain types/errors and generated manifest/stability metadata.
2. `libcint-runtime` — validator, planner, workspace allocator, scheduler/chunker, dispatch policy.
3. `libcint-cpu` + `libcint-cubecl` — reference correctness backend plus optional GPU backend.
4. `libcint-compat` + `libcint-rs` (+ optional `libcint-capi`) — raw/safe/C ABI surfaces over shared runtime.
5. `libcint-oracle` + `xtask` — oracle comparisons, manifest audits, feature-matrix gates, release checks.

### Critical Pitfalls

1. **Manifest/ABI drift** — prevent with compiled-symbol lock regeneration/audit across all supported feature profiles and CI blockers on unapproved diffs.
2. **`dims` contract drift and partial writes** — prevent with one canonical required-size calculation and strict rejection of truncating writes.
3. **Spinor/complex layout mismatch** — prevent with a unified tested writer, explicit representation contracts, and spinor-heavy oracle fixtures.
4. **OOM-safe-stop violations from ad hoc allocations** — prevent by routing large allocations through fallible workspace allocator + mandatory chunk planning.
5. **Optional-family leakage (`with-f12`, `with-4c1e`, GTG boundaries)** — prevent with resolver-enforced policy matrix, explicit negative tests, and symbol-absence assertions.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Contract and Runtime Foundation
**Rationale:** Every later capability depends on stable types, manifest policy, and shared runtime contracts.
**Delivers:** `core`/`ops`/`runtime` skeletons, typed errors, manifest lock pipeline, validator interfaces, fallible allocation policy.
**Addresses:** Typed API/error expectations, release-gate foundation, OOM-safe design prerequisites.
**Avoids:** Manifest drift, unbounded allocation paths.

### Phase 2: CPU Compatibility Core (Safe + Raw)
**Rationale:** Compatibility value must be proven on deterministic CPU before optional backends.
**Delivers:** Stable-family CPU execution, canonical writer/layout logic, raw `atm/bas/env` parity (`dims/shls/cache`), safe API (`query_workspace`, `evaluate_into`).
**Addresses:** Core migration table-stakes and parity baseline.
**Avoids:** `dims` drift, spinor layout regressions, silent safe/compat behavior divergence.

### Phase 3: Verification and Gate Hardening
**Rationale:** Claims of compatibility require automated evidence before expanding scope.
**Delivers:** Oracle harness, helper/transform/optimizer parity tests, OOM/resource-pressure suite, manifest-audit + feature-matrix CI gates.
**Addresses:** Release reliability and regression prevention.
**Avoids:** Non-reproducible oracle runs, optimizer on/off drift, false coverage claims.

### Phase 4: Optional Acceleration (CubeCL)
**Rationale:** Add performance path only after correctness and gates are stable.
**Delivers:** CubeCL backend integration via shared planner, conservative dispatch heuristics, deterministic CPU fallback, CPU/GPU consistency checks.
**Addresses:** Performance differentiator without changing external contracts.
**Avoids:** Transfer-dominated GPU regressions and untraceable fallback behavior.

### Phase 5: Migration Surfaces and Optional Families
**Rationale:** Expand adoption and API breadth once baseline behavior is controlled.
**Delivers:** Optional `capi` shim with TLS error semantics, `with-f12` (sph-only) and `with-4c1e` (validated envelope) rollout, support-matrix docs.
**Addresses:** Incremental migration needs and controlled optional capability growth.
**Avoids:** Optional-family leakage, C ABI error ambiguity, unstable-family confusion.

### Phase 6: Release Qualification and Stabilization
**Rationale:** Finalize reproducibility, governance, and performance envelopes before broad claims.
**Delivers:** Locked manifest/signoff process, benchmark baselines + threshold calibration, release checklist and artifact provenance controls.
**Addresses:** Production readiness and long-term maintainability.
**Avoids:** Last-mile drift, flaky compatibility gates, undocumented support boundaries.

### Phase Ordering Rationale

- Order follows hard dependencies: manifest/runtime contracts before kernels; CPU correctness before GPU optimization; gates before feature-surface expansion.
- Grouping mirrors architecture boundaries: foundation (`core/ops/runtime`), execution (`cpu/cubecl`), surfaces (`safe/compat/capi`), governance (`oracle/xtask`).
- This sequence directly neutralizes highest-risk pitfalls early (`dims`, OOM, manifest drift) before introducing optional complexity.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (CubeCL acceleration):** CubeCL release churn and workload-specific crossover heuristics need targeted validation on intended hardware.
- **Phase 5 (Optional families):** 4c1e validated envelope and F12 representation boundaries need stricter evidence mapping before scope commitments.
- **Phase 5 (C ABI shim):** Thread-local error semantics and multithreaded FFI behavior need ABI-focused test design research.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Rust workspace/toolchain pinning, manifest generation, and layered runtime setup are well-established patterns in this domain.
- **Phase 2:** Shared-runtime safe/raw API layering and CPU-first reference execution are strongly documented by project design and architecture research.
- **Phase 3:** Oracle + CI gate enforcement patterns are explicit and already defined; execution focus should be implementation quality, not discovery.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Strongly anchored to project constraints and pinned versions; only GPU crate stability is medium-risk. |
| Features | HIGH | Table stakes and anti-features are explicit in project/design docs and aligned across all research outputs. |
| Architecture | HIGH | Convergent guidance on layered shared-runtime model with clear component boundaries and build order. |
| Pitfalls | HIGH | Risks are concrete, phase-mapped, and come with directly testable prevention strategies. |

**Overall confidence:** HIGH

### Gaps to Address

- GPU dispatch thresholds are not yet calibrated against this project’s real workload distribution; define acceptance telemetry and benchmark corpus in planning.
- 4c1e validated-envelope limits need explicit operator/input boundary catalog before committing full phase scope.
- C ABI status/error mapping needs a formal contract table (status codes, TLS lifetime, thread rules) before implementation lock.
- Tolerance policy by family/representation (especially near-zero and spinor-heavy cases) should be frozen as release criteria early in Phase 3.

## Sources

### Primary (HIGH confidence)
- `/home/chemtech/workspace/cintx/.planning/PROJECT.md` — scope, constraints, in/out-of-scope boundaries.
- `/home/chemtech/workspace/cintx/docs/libcint_detailed_design_resolved_en.md` — detailed architecture and policy requirements.
- `/home/chemtech/workspace/cintx/.planning/research/STACK.md` — recommended technology/version baseline.
- `/home/chemtech/workspace/cintx/.planning/research/FEATURES.md` — feature prioritization and dependency graph.
- `/home/chemtech/workspace/cintx/.planning/research/ARCHITECTURE.md` — component model, build order, patterns.
- `/home/chemtech/workspace/cintx/.planning/research/PITFALLS.md` — risk catalog, warning signs, and phase mapping.

### Secondary (MEDIUM confidence)
- crates.io package metadata snapshots referenced in stack research for crate-line currency and version banding.

### Tertiary (LOW confidence)
- None identified.

---
*Research completed: 2026-03-14*
*Ready for roadmap: yes*
