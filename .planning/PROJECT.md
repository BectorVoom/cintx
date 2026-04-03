# cintx

## What This Is

cintx is a public Rust library that redesigns and reimplements libcint with result compatibility as the primary goal. It provides a Rust-native safe API, a raw compatibility API for `atm`/`bas`/`env` style callers, and an optional C ABI shim for migration and interoperability. The target users are Rust developers and systems that need libcint-compatible integral evaluation with stronger type safety, clear failure modes, and high-confidence verification.

## Core Value

Deliver libcint-compatible results through a Rust-native API surface that stays type-safe, verifiable, and safe under memory pressure.

## Requirements

### Validated

- [x] Typed domain primitives, canonical manifest generation, and manifest-aware resolver foundations are in place and verified in Phase 1: Manifest & Planner Foundation.
- [x] Runtime planner/workspace scaffolding now exposes typed query/evaluate contracts, memory-limit chunking, and explicit validation failures, verified in Phase 1 Plan 02.
- [x] The three-layer surface (safe Rust API, raw compatibility API, optional C ABI shim) is implemented with feature-gated optional/unstable families and verified in Phase 3: Safe Surface, C ABI Shim & Optional Families.
- [x] Gaussian primitive infrastructure (Boys function, pair data, Rys quadrature, Obara-Saika recurrence) implemented as validated #[cube] functions in cintx-cubecl. Validated in Phase 8: Gaussian Primitive Infrastructure and Boys Function.

### Active

- [ ] Reimplement the libcint API surface needed for the target 6.1.3 compatibility profile, including helper, optimizer, legacy wrapper, and selected source-only families.
- [x] Execute all integral-family computation through a shared planner and CubeCL backend while preserving performance, memory efficiency, and OOM-safe stop behavior. Validated in Phase 10: all five base families (1e, 2e, 2c2e, 3c1e, 3c2e) produce real kernel output through CubeCL backend with oracle parity confirmed.
- [x] Prove compatibility through compiled-manifest audits, oracle comparisons against upstream libcint, regression gates, and reproducible CI artifacts. Validated in Phase 10: oracle gate closure passes all five families against vendored libcint 6.1.3 with 0 mismatches.

### Out of Scope

- Bitwise-identical reproduction of libcint internals - the project targets result compatibility, not implementation identity.
- Public GTG support - the design explicitly excludes GTG from initial GA because upstream marks it deprecated and incorrect.
- Reproducing the upstream Fortran wrapper - not part of the Rust library's public scope.
- Public asynchronous APIs - excluded from the initial design to keep execution and compatibility contracts tighter.

## Current Milestone: v1.1 CubeCL Direct Client API & Real Kernel Compute

**Goal:** Rewrite executor internals to use CubeCL client API directly, implement real GPU integral kernels, achieve oracle parity with upstream libcint 6.1.3.

**Target features:**
- Rewrite executor internals to use CubeCL client API directly (`WgpuRuntime::client()`, `client.create()`/`client.read()`, `#[cube(launch)]` with `ArrayArg`)
- Remove RecordingExecutor — direct buffer management replaces the wrapper
- Configurable backend switching (wgpu + cpu now; cuda/rocm/metal extensible)
- Implement real GPU integral kernels replacing stubs (1e, 2e, 2c2e, 3c1e, 3c2e)
- Achieve oracle parity — numerical output matching upstream libcint 6.1.3

## Context

The project is driven by `docs/design/cintx_detailed_design.md`, which defines an implementation-ready redesign for libcint in Rust. The workspace contains the multi-crate Rust layout (`crates/`, `xtask/`, `benches/`, `ci/`) plus a vendored upstream reference in `libcint-master/`, with the design document as the source of truth for scope and release gates. v1.0 is complete (6 phases, 30 plans): typed domain primitives, manifest, planner, runtime, three-layer API surface (safe Rust, raw compat, C ABI shim), CI governance gates, CubeCL/wgpu GPU execution path with stub kernels, and staging/fingerprint plumbing. The compatibility target remains libcint 6.1.3. v1.1 replaces the executor abstraction layer with direct CubeCL client API usage and implements real integral kernels to achieve oracle parity.

## Constraints

- **Compatibility**: Target upstream libcint 6.1.3 result compatibility - the project must match upstream outputs closely enough to satisfy oracle comparison gates.
- **Architecture**: CubeCL is the primary compute backend - host CPU work stays limited to planning, validation, marshaling, and test/oracle glue.
- **API Surface**: Safe Rust API first, raw compatibility API second, optional C ABI shim third - this ordering drives module boundaries and migration strategy.
- **Error Handling**: Public library errors use `thiserror` v2, while CLI, xtask, benchmarks, and oracle harness code use `anyhow`.
- **Verification**: Full API coverage claims must be backed by the compiled manifest lock, feature-matrix CI, and helper/transform parity checks.
- **Artifacts**: Deliverables written to `/mnt/data` remain a mandatory part of the design and verification workflow.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Prioritize result compatibility over implementation compatibility | Users need libcint-equivalent outputs, not a line-by-line clone of upstream internals | Pending |
| Use a three-layer public surface (safe Rust, raw compat, optional C ABI) | This balances Rust ergonomics with migration and interoperability needs | Validated in Phase 3 |
| Use a generated compiled manifest lock as the API source of truth | Full API coverage must be mechanically auditable across feature profiles | Validated in Phase 1 |
| Standardize on a shared planner plus CubeCL executor | A single compute path simplifies optimization, memory policy, and verification | Validated through Phases 1-5; v1.1 replaces executor internals with direct CubeCL client API |
| Use CubeCL client API directly in executor internals | Direct buffer management (`client.create`/`client.read`/`ArrayArg`) removes need for RecordingExecutor wrapper; kernels use `#[cube(launch)]` | v1.1 — user-directed architectural decision |
| Configurable backend (wgpu + cpu; cuda/rocm/metal extensible) | Multi-backend support ensures testing on CPU and deployment on GPU; future backends require only runtime trait impl | v1.1 — Pending |
| Centralize fallible allocation and typed OOM errors | Safe stop on memory pressure is a non-negotiable design goal | Partially validated in Phase 1 through `WorkspaceAllocator`, `ChunkPlanner`, and typed runtime errors |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-03 after Phase 10 complete — Oracle gate closed for all five base integral families, v1.1 milestone complete*
