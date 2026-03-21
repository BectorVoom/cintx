# Phase 1: Contracts and Typed Foundations - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning
**Source:** PRD Express Path (/home/chemtech/workspace/cintx/docs/libcint_detailed_design_resolved_en.md)

<domain>
## Phase Boundary

This phase establishes the contract layer for `libcint-rs` so downstream execution phases can build safely on stable interfaces.

In scope for this phase:
- Typed domain inputs for atom/shell/environment and operator selection
- Deterministic workspace-introspection contract (`query_workspace`) for safe preflight sizing
- Typed public error taxonomy and diagnostics policy
- Validation boundaries and contract-level invariants that avoid raw-pointer misuse in safe API surfaces

Out of scope for this phase:
- Full numerical kernel parity implementation
- Backend acceleration implementation
- Optional family enablement (`with-f12`, `with-4c1e`) execution logic

</domain>

<decisions>
## Implementation Decisions

### Compatibility Model (Locked)
- Result compatibility is prioritized over implementation parity.
- Internal representation may differ if external contracts and numerical outcomes remain compatible.

### Public Surface Layering (Locked)
- Three-layer model is fixed: Safe Rust API, raw compatibility API, optional C ABI shim.
- Phase 1 focuses on contract definitions and typed boundaries used by these surfaces.

### Error Contract (Locked)
- Public library uses typed errors (`thiserror`-style categories) for unsupported API, layout/input failures, memory failures, and backend failures.
- CLI/tooling/harness boundaries may use broader error wrappers, but public API remains typed.

### Memory Safety Policy (Locked)
- OOM-safe stop semantics are mandatory project-wide.
- Contract layer must enforce deterministic preflight sizing and diagnosable failure output.

### Raw Contract Semantics (Locked for downstream)
- `dims`/buffer contract must reject partial-write and implicit truncation behavior.
- Null-equivalent workspace-query behavior is preserved for compatibility paths.

### Unsafe Boundary Policy (Locked)
- Unsafe is constrained to narrow implementation boundaries (FFI/layout/device/SIMD), not exposed via safe API contracts.

### Claude's Discretion
- Exact Rust module decomposition and naming within the phase directory structure
- Internal trait boundaries for validator/workspace/error diagnostics as long as locked decisions above remain satisfied
- Formatting of diagnostics metadata (`tracing` fields) as long as failure categories remain explicit

</decisions>

<specifics>
## Specific Ideas

Concrete requirements extracted from PRD and mapped to this phase focus:
- Define explicit types for atom/shell/basis/environment/operator inputs rather than raw index arithmetic
- Separate workspace query from evaluation to avoid C-style sentinel usage in safe APIs
- Establish typed error categories and conversion boundaries early to prevent ad-hoc failure handling later
- Make contract-level validation deterministic and reusable by both safe and compatibility surfaces
- Ensure design leaves room for later CPU reference execution and compatibility CI gates without reworking public contracts

</specifics>

<deferred>
## Deferred Ideas

Items explicitly deferred beyond this phase:
- Full stable-family CPU numerical implementation and oracle parity checks
- CubeCL backend acceleration and dispatch heuristics
- Optional families rollout (`with-f12`, `with-4c1e`) and envelope enforcement
- Optional public C ABI implementation details
- Async public API work
- GTG support

</deferred>

---

*Phase: 01-contracts-and-typed-foundations*
*Context gathered: 2026-03-14 via PRD Express Path*
