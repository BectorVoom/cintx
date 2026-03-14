# libcint-rs

## What This Is

`libcint-rs` is a Rust redesign and reimplementation project targeting numerical-result compatibility with upstream `libcint` 6.1.3. It will provide a safe Rust-first API, a raw compatibility API, and an optional C ABI shim so existing libcint-style integrations can migrate incrementally. The current repository already vendors upstream libcint C sources and now uses this project plan to drive phased Rust implementation.

## Core Value

Users can compute libcint-equivalent integrals through a Rust-native library with explicit safety/error guarantees and verifiable compatibility gates.

## Requirements

### Validated

- ✓ Existing `libcint` C source and API surface are present locally and buildable via CMake (`libcint-master/`) — existing
- ✓ Existing integral family declarations, helper headers, examples, and tests are available as implementation/oracle reference — existing
- ✓ Existing repository has Rust toolchain scaffolding and planning assets for phased delivery — existing

### Active

- [ ] Rust library crate implements phased parity for targeted libcint integral families with typed error handling
- [ ] Public API includes safe Rust interface plus raw compatibility interface for `atm/bas/env`, `shls`, `dims`, and buffers
- [ ] Optional C ABI shim supports migration and interop without weakening Rust-side safety boundaries
- [ ] Result compatibility gates (oracle comparisons, helper transform parity, regression checks) are automated in CI
- [ ] OOM-safe stop behavior, allocator boundaries, and memory-limit chunking contracts are enforced and testable
- [ ] Optional families are explicitly feature-gated (`with-f12`, `with-4c1e`, unstable source-only APIs) with clear support matrix
- [ ] CPU reference backend and CubeCL GPU backend share planner logic with deterministic fallback rules

### Out of Scope

- Bitwise identical results and internal implementation parity with upstream — result compatibility is the explicit target
- GTG family support in initial GA scope — upstream flags this area as deprecated/incorrect
- Public asynchronous API in initial scope — deferred until core compatibility/performance goals are met
- Fortran wrapper reproduction in initial scope — migration priority is Rust API + compatibility layers

## Context

- Primary design source: `docs/libcint_detailed_design_resolved_en.md` (version 0.4-resolved)
- Brownfield baseline analyzed in `.planning/codebase/` (stack, architecture, structure, conventions, testing, integrations, concerns)
- Upstream reference tree is vendored at `libcint-master/` and includes headers, implementation, docs, examples, and tests
- Project emphasizes measurable compatibility (compiled-symbol inventory, requirement traceability, release gates)
- Current Rust code is scaffold-level and requires phased buildout aligned to design requirements and CI validation

## Constraints

- **Compatibility Target**: Upstream `libcint` 6.1.3 behavior — required to satisfy migration and result-comparison goals
- **Language Direction**: Rust-first public library with constrained `unsafe` boundaries — required for type safety and maintainability
- **Backend Scope**: CPU reference plus CubeCL GPU extension — required by design goals and performance strategy
- **Error Contract**: Typed public errors and OOM-safe stop semantics — required by explicit non-functional requirements
- **Verification**: Oracle/regression/release gates must be automated — required before claiming full API coverage
- **Repository Baseline**: Existing brownfield vendored C code must remain a trustworthy reference during phased reimplementation

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Prioritize result compatibility over implementation parity | Design explicitly defines compatibility hierarchy; this reduces unnecessary reimplementation constraints | — Pending |
| Use phased Rust implementation against vendored upstream oracle | Brownfield repository already contains complete references/tests for iterative parity validation | — Pending |
| Keep optional/unstable API families behind explicit feature gates | Upstream itself contains conditional/buggy/deprecated regions that must not silently leak into stable surface | — Pending |
| Require release gates (manifest audit + oracle comparison) for compatibility claims | "Full coverage" must be mechanically verifiable, not narrative | — Pending |

---
*Last updated: 2026-03-14 after initialization*
