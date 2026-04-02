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

### Active

- [ ] Reimplement the libcint API surface needed for the target 6.1.3 compatibility profile, including helper, optimizer, legacy wrapper, and selected source-only families.
- [ ] Execute all integral-family computation through a shared planner and CubeCL backend while preserving performance, memory efficiency, and OOM-safe stop behavior.
- [ ] Prove compatibility through compiled-manifest audits, oracle comparisons against upstream libcint, regression gates, and reproducible CI artifacts.

### Out of Scope

- Bitwise-identical reproduction of libcint internals - the project targets result compatibility, not implementation identity.
- Public GTG support - the design explicitly excludes GTG from initial GA because upstream marks it deprecated and incorrect.
- Reproducing the upstream Fortran wrapper - not part of the Rust library's public scope.
- Public asynchronous APIs - excluded from the initial design to keep execution and compatibility contracts tighter.

## Context

The project is driven by `docs/design/cintx_detailed_design.md`, which defines an implementation-ready redesign for libcint in Rust. The workspace contains the multi-crate Rust layout (`crates/`, `xtask/`, `benches/`, `ci/`) plus a vendored upstream reference in `libcint-master/`, with the design document as the source of truth for scope and release gates. Phases 1 through 3 are complete: typed domain and manifest foundations are in place, runtime/compat execution is stabilized, and the safe Rust facade plus optional C ABI shim and optional-family gates are verified. The compatibility target remains libcint 6.1.3, and the next focus is Phase 4 verification/release automation (oracle loop closure, CI gates, benchmarks, diagnostics).

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
| Standardize on a shared planner plus CubeCL executor | A single compute path simplifies optimization, memory policy, and verification | Validated through Phases 1-3 for planner/runtime/compat/safe surfaces; full release gate evidence still depends on Phase 4 CI/oracle automation |
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
*Last updated: 2026-04-02 after Phase 5 completion — real CubeCL/wgpu GPU execution path, typed backend contracts, capability-aware CI gates*
