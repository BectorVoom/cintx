# Project Research Summary

**Project:** cintx
**Domain:** Rust-native libcint-compatible integral library
**Researched:** 2026-03-21
**Confidence:** MEDIUM

## Executive Summary

cintx is a Rust-native reimplementation of libcint whose credibility depends on matching the upstream API result matrix while adding typed safety, predictable error contracts, and observable planning/allocator behavior. Experts build it by keeping a manifest-driven API catalog, distinguishing the safe Rust facade from the raw compat surface and optional C shim, and letting a single CubeCL-backed planner execute every integral family so that parity proofs stay centralized.

The recommended approach is to stabilize the foundational stack first: pin Rust 1.94.0, lock Cargo dependencies, generate the compiled manifest lock, and implement the typed domain models plus the manifest-aware planner/validator before wiring the CubeCL executor, compat writer, and safe API on top of it. Fallible allocation, tracing spans for chunking/fallbacks, and manifest/oracle audits are non-negotiable requirements.

Key risks surface when reduction order drifts, large feature families hit GPU limits, or the manifest/oracle inventory diverges; mitigate them by enforcing deterministic planner contracts, routing every allocation through fallible limits with explicit `UnsupportedApi` for envelopes outside the support matrix, and gating every change with manifest diffs plus oracle regressions before release.

## Key Findings

### Recommended Stack

The project relies on a pinned Rust 1.94.0 toolchain, Cargo with `resolver = "3"` and `cargo --locked`, CubeCL as the GPU backend, and crates that expose observable diagnostics (e.g., `tracing`, `thiserror`, `anyhow`). These choices keep verification reproducible, errors typed, and backend switching future-proof while preserving the manifest/oracle audit loops spelled out in the design.

**Core technologies:**
- `rustc 1.94.0` + `rust-toolchain.toml`: reproducible compiler output and CI parity.
- Cargo lockfile & resolver 3: deterministic dependency resolution across the multi-crate workspace.
- CubeCL (current `0.9.x`): the shared GPU executor that keeps host work focused on planning and validation.
- `tracing` + fallible allocators (`thiserror`/`anyhow` for boundaries): observable planner decisions and safe-stop semantics.

### Expected Features

The launch must deliver the three-layer surface (safe Rust API, raw compat API, optional C shim), manifest-backed symbol coverage with multi-profile CI, CubeCL-backed planner with fallible allocation, and the legacy/helper/optimizer parity that migration users expect. Optional feature families must be gated behind documented flags and issue explicit `UnsupportedApi` responses when out of scope.

**Must have (table stakes):**
- Tri-layer API surface with safe builders, raw compat validators, and the C ABI shim sharing the planner.
- Manifest-backed API inventory plus feature-matrix/oracle verification covering base, `with-f12`, `with-4c1e`, and combined profiles.
- CubeCL planner locked into fallible allocation, chunking heuristics, and typed failure diagnostics so OOM/rejection paths stay deterministic.
- Helper/optimizer/transform parity so the legacy symbol list stays intact.

**Should have (competitive):**
- Ergonomic builders & typed domain helpers for atoms, shells, environments, and tensor views beyond the strict compat layout.
- Advanced profiling/benchmarking beyond tracing spans (criterion baselines, CubeCL throughput studies).
- Extended oracle/property coverage for optional/unstable families once stable coverage is in place.
- Multi-device CubeCL consistency and transfer/fallback heuristics for polished GPU behavior.

**Defer (v2+):**
- GTG support in the public surface (roadmap-only and explicitly out of scope).
- Bitwise-identical libcint internals or Fortran wrapper telemetry.
- Asynchronous public APIs for the initial GA.
- Unbounded optional profiling transforms or partial support for optional families outside their validated envelopes.

### Architecture Approach

The architecture splits alignment, planning, execution, compat writing, safe facades, and verification into separate crates so that the manifest, planner, CubeCL executor, compat helpers, safe API, C shim, and oracle tooling can evolve with clear dependencies; verification tooling reuses the runtime and compat paths to enforce parity.

**Major components:**
1. `cintx-core` + `cintx-ops` — define typed domain models and the compiled manifest/resolver required before any planner logic can exist.
2. `cintx-runtime` + `cintx-cubecl` — implement validator/planner/scheduler/workspace contracts around the CubeCL executor and fallible allocator hooks.
3. `cintx-compat` + `cintx-rs` + `cintx-capi` — validate raw layouts, provide helper/legacy bridges, and expose the safe Rust and optional C APIs atop the runtime.

### Critical Pitfalls

1. **Non-deterministic reduction order** — lock reduction strategies into the planner contract, trace chunk plans, and require oracle fixtures whenever chunking heuristics change.
2. **GPU memory blowups on high families** — route every allocation through fallible pools, pre-check workspace needs, chunk deterministically, and treat unsupported envelopes as explicit errors.
3. **Compat layout divergence** — centralize layout validation, exercise compatible permutations via both API paths, and fail fast on mismatched `dims`/buffers.
4. **Manifest/oracle drift** — tie every API change to lock file regeneration, multi-profile CI, and manifest/oracle diffs before gating releases.
5. **Backend leakage into the public surface** — keep public contracts typed and backend-agnostic, exposing only diagnostics while hiding CubeCL handles.

## Implications for Roadmap

### Phase 1: Manifest & Planner Foundation
**Rationale:** Manifest-driven APIs, typed domain models, and planner scaffolding are prerequisites for every other layer because the planner requires a stable symbol catalog and fallible memory policies.
**Delivers:** `cintx-core`/`cintx-ops` definitions, `cintx-runtime` validator/planner hooks with mock executor stubs, manifest lock generation pipeline, and fallible workspace pools.
**Addresses:** Manifest-backed API inventory, base compatibility requirements, and deterministic chunking (table stakes).
**Avoids:** Manifest/oracle drift by locking the symbol catalog early and backend leakage by keeping CubeCL detached.

### Phase 2: Execution & Compat Stabilization
**Rationale:** Once the planner and manifest exist, bring in the CubeCL executor and raw compatibility glue to prove results, handle helper parity, chunking, and layout writes before exposing any higher-level APIs.
**Delivers:** `cintx-cubecl` executor, `cintx-compat` layout validators/writers/helpers, fallible allocator tied to CubeCL, and tracing for chunking/fallback.
**Uses:** CubeCL backend, `tracing`, fallible allocators (`thiserror`/`anyhow`), and existing manifest resolver to pick kernels.
**Implements:** The runtime-to-executor data flow (scheduler→CubeCL→layout writer) from the architecture guidance.
**Addresses:** CubeCL-backed planner, compat raw API, helper/legacy parity.
**Avoids:** Non-deterministic reduction order, compat layout divergence, high-memory family blowups.

### Phase 3: Safe Facade, C ABI & Optional Families
**Rationale:** Once the raw runtime proves stability, expose the typed Rust builders, safe tensor views, compatibility helpers, C shim, and optional feature flags so downstream users can migrate confidently.
**Delivers:** `cintx-rs` safe API, `cintx-capi` shim, feature gates (`with-f12`, `with-4c1e`, `unstable-source-api`), and optional family behavior (`UnsupportedApi` handling).
**Addresses:** Three-layer public surface, feature-gated optional families, helper APIs, and ergonomics differentiators.
**Avoids:** Backend leakage, partial optional-family support by default, unsynchronized helper parity.

### Phase 4: Verification & Release Automation
**Rationale:** With the execution stack and facades in place, close the manifest/oracle loop, run multi-profile CI, and produce the automation/benchmark artifacts needed for release confidence.
**Delivers:** `cintx-oracle` comparison harness, `xtask` manifest/oracle audit jobs, multi-profile CI workflows, OOM/benchmarks, and documentation/regression analytics.
**Addresses:** Manifest-backed verification, property tests, multi-profile oracle coverage, and release gates.
**Avoids:** Manifest/oracle drift, GPU regression surprises, insufficient verification before release.

### Phase Ordering Rationale
- Foundations (manifest, planner, memory policy) must precede real execution/compat code so that every API change can be validated against a stable lock.
- Execution & compat stabilization naturally lead into safe APIs, optional families, and C shims because the higher layers reuse the runtime/compat plumbing.
- Verification closes the loop once the whole stack exists so manifest/oracle diffs and performance regressions stop gating release.

### Research Flags
Phases likely needing deeper research during planning:
- **Phase 3:** Optional families (F12, 4c1e, unstable source APIs) involve complex feature-gating, tolerance envelopes, and multi-device CubeCL behavior that should be validated before approval.
- **Phase 4:** Multi-profile oracle and benchmark automation demand reproducible GPU CI (hardware variance, tolerance thresholds, regression tracking) requiring more investigation.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Manifest generation, resolver patterns, and planner scaffolding follow well-documented Cargo/workspace practices and the detailed design.
- **Phase 2:** CubeCL execution pipeline and compat layout writers follow the documented architecture and manifest-driven data flow.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | STACK.md directly references the pinned toolchain, Cargo policies, and CubeCL choice with explicit version guidance. |
| Features | HIGH | FEATURE.md enumerates table stakes, differentiators, and anti-goals drawn from the detailed design’s requirement sections. |
| Architecture | HIGH | ARCHITECTURE.md defines clear component boundaries, data flow, and build order; it aligns with the design doc. |
| Pitfalls | MEDIUM | PITFALLS.md distills major risks from the same design doc but mitigation estimates are less quantified. |

**Overall confidence:** MEDIUM

### Gaps to Address

- Multi-device CubeCL behavior and fallback heuristics: plan labs to validate transfer/queue selection and caching before optional families expand.
- GPU CI stability and tolerance thresholds for oracle comparisons: document hardware assumptions and stress the CI matrix early.
- Optional-family and unstable-source feature gating costs: scope how many manifest/oracle fixtures each new profile adds before locking them in.

## Sources

### Primary (HIGH confidence)
- `docs/design/cintx_detailed_design.md` — detailed scope, manifest policy, verification plan, architecture, and release gating.

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` — pinned toolchain, Cargo policy, CubeCL choice, and supporting libraries.
- `.planning/research/FEATURES.md` — table stakes, differentiators, and anti-features in the compatibility story.
- `.planning/research/ARCHITECTURE.md` — component boundaries, data flow, and build sequencing guidance.
- `.planning/PROJECT.md` — project intent, requirements, and scope constraints.

### Tertiary (LOW confidence)
- `.planning/research/PITFALLS.md` — key risks and their mitigation strategies (valuation lacks cost estimates).

---
*Research completed: 2026-03-21*
*Ready for roadmap: yes*
