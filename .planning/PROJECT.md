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
- [x] All five base integral families (1e, 2e, 2c2e, 3c1e, 3c2e) produce real kernel output through CubeCL backend with oracle parity confirmed against vendored libcint 6.1.3. Validated in Phase 10: 2e, 2c2e, 3c1e, 3c2e Real Kernels and Oracle Gate Closure.
- [x] Oracle gate closure passes all five base families with 0 mismatches. Validated in Phase 10.
- [x] Every helper, transform, and wrapper symbol in the manifest is oracle-wired with unified atol=1e-12 tolerance; 4c1e stub replaced with real polynomial recurrence kernel matching vendored libcint. Validated in Phase 11: Helper/Transform Completion & 4c1e Real Kernel.

### Active

- [ ] Cover the full libcint API surface — with-f12, with-4c1e (beyond validated envelope), and unstable-source families — with unified atol=1e-12 oracle tolerance and objective CI evidence.
- [x] Implement with-f12 (F12/STG/YP) family kernels — all 10 sph symbols at oracle parity (atol=1e-12). Cart and spinor remain unsupported (sph-only enforcement). Validated in Phase 13.
- [ ] Implement with-4c1e family kernels beyond the initial validated envelope with oracle parity.
- [ ] Implement unstable-source family APIs behind feature gate with oracle parity.
- [ ] Unify oracle tolerance to atol=1e-12 for every family and extend oracle harness, fixtures, and CI gates for full API coverage.
- [ ] Resolve pending v1.1 executor infrastructure items (EXEC-06/07/08/09, VERI-06) if not already closed.

### Out of Scope

- Bitwise-identical reproduction of libcint internals - the project targets result compatibility, not implementation identity.
- Public GTG support - the design explicitly excludes GTG from initial GA because upstream marks it deprecated and incorrect.
- Reproducing the upstream Fortran wrapper - not part of the Rust library's public scope.
- Public asynchronous APIs - excluded from the initial design to keep execution and compatibility contracts tighter.

## Current Milestone: v1.2 Full API Parity & Unified Oracle Gate

**Goal:** Close all remaining libcint API surface gaps — helper, transform, wrapper, with-f12, with-4c1e, and unstable-source families — with unified atol=1e-12 oracle tolerance across every family and objective evidence from CI gates.

**Target features:**
- Implement missing helper, transform, and wrapper APIs with oracle-backed coverage
- Implement with-f12 (F12/STG/YP) family kernels including cart and spinor representations
- Implement with-4c1e family kernels beyond initial validated envelope
- Implement unstable-source family APIs behind feature gate
- Unify oracle tolerance to atol=1e-12 for ALL families (immutable unless explicitly approved spec update)
- Extend manifest lock to cover full API surface
- Extend oracle harness, fixtures, and CI gates for every supported API
- Resolve pending v1.1 executor items (EXEC-06/07/08/09, VERI-06) if not already closed

## Context

The project is driven by `docs/design/cintx_detailed_design.md`, which defines an implementation-ready redesign for libcint in Rust. The workspace contains the multi-crate Rust layout (`crates/`, `xtask/`, `benches/`, `ci/`) plus a vendored upstream reference in `libcint-master/`, with the design document as the source of truth for scope and release gates. v1.0 is complete (6 phases, 30 plans): typed domain primitives, manifest, planner, runtime, three-layer API surface (safe Rust, raw compat, C ABI shim), CI governance gates, CubeCL/wgpu GPU execution path with stub kernels, and staging/fingerprint plumbing. v1.1 is complete (Phases 7-10): real integral kernels for all five base families (1e, 2e, 2c2e, 3c1e, 3c2e) with oracle parity against vendored libcint 6.1.3, Gaussian math infrastructure (#[cube] Boys, Rys, Obara-Saika), and cart-to-sph Condon-Shortley transforms. The compatibility target remains libcint 6.1.3. v1.2 extends coverage to the full API surface with unified tolerance.

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
*Last updated: 2026-04-05 after Phase 13 complete — F12/STG/YP kernels with oracle parity for all 10 sph symbols*
