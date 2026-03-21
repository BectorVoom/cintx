# Phase 1 Research: Contracts and Typed Foundations

**Phase:** 01
**Date:** 2026-03-14
**Primary inputs:**
- `.planning/phases/01-contracts-and-typed-foundations/01-CONTEXT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `docs/libcint_detailed_design_resolved_en.md`

## Objective and Phase Boundary

Phase 1 should establish stable, typed contracts that all later execution work depends on. The implementation should be prescriptive and narrow:
- In scope: typed domain models, deterministic workspace introspection, typed error taxonomy, reusable validation boundaries.
- Out of scope: numerical kernel parity, optimizer parity, backend acceleration, optional-family execution (`with-f12`, `with-4c1e`), and C ABI implementation details.

This phase is successful when contract semantics are locked, testable, and reusable by both safe and raw compatibility surfaces.

## Requirement Targets (Explicit)

| Requirement | What must exist in Phase 1 |
|---|---|
| `SAFE-01` | Typed input models (`Atom`, `Shell`, basis/environment context, operator/representation enums) with constructors/builders that prevent raw-pointer arithmetic in safe API use. |
| `SAFE-02` | A deterministic `query_workspace` contract in the safe surface, separate from evaluation, with stable shape/workspace estimation semantics. |
| `SAFE-04` | Public typed error taxonomy (not `anyhow`) that distinguishes unsupported API, input/layout failures, memory failures, and backend failures. |
| `MEM-03` | Structured diagnostics payload and tracing fields so users can identify failure causes (`api`, `rep`, `shells`, `dims/bytes`, backend/fallback reason, feature flags). |

## Standard Stack

Use this stack now (do not defer):
- `thiserror` v2: public error enum and variant-level messages.
- `tracing`: structured spans/events for validator/planner/workspace paths.
- `smallvec`: compact shape/stride/dims internals without mandatory heap allocation.
- `num-complex`: type-level support for spinor-compatible output contracts.
- `anyhow`: tooling/harness boundary only, never public API.

## Architecture Patterns

1. Three-layer surface with shared runtime contracts:
- Safe API layer: typed inputs, no sentinel/null-style contracts.
- Raw compatibility layer: accepts `atm/bas/env`, `shls`, optional `dims`, strict validation.
- Optional C ABI shim later; Phase 1 should keep error/reporting contracts compatible with adding it.

2. Split query from execute in safe API:
- `query_workspace(...) -> WorkspaceQuery`
- `evaluate_into(...) -> ExecutionStats` (execution may be stubbed or deferred, but signature/contract must be fixed now)

3. Single validation core reused by both surfaces:
- Typed validator (`BasisSet`, shell tuple, representation support)
- Raw validator (`RawAtmView`, `RawBasView`, `RawEnvView`, dims contract)
- Shared error variants and diagnostic metadata

4. Explicit unsafe boundary policy:
- No `unsafe` in public safe facade for Phase 1.
- Unsafe only in raw/FFI boundary scaffolding and only with documented safety invariants.

## Contract Surface to Lock in Phase 1

Lock these contracts now to avoid Phase 2 churn:

1. Typed domain model surface (`SAFE-01`):
- `Atom` with nuclear/position fields.
- `Shell` with validated angular momentum, contraction metadata, and coefficient/exponent ownership.
- `Representation` enum (`Cart`, `Spheric`, `Spinor`).
- Shell arity types (`ShellTuple2/3/4`) to encode family arity in types.

2. Workspace contract (`SAFE-02`):
- `WorkspaceQuery { required_bytes, alignment, shape/layout metadata }`.
- Deterministic computation from `(operator, representation, basis, shells, options)`.
- Safe API does not expose `dims`; raw compat may accept validated `dims` override only.

3. Error contract (`SAFE-04`):
- Define `LibcintRsError` variants at minimum for: unsupported API/representation, input layout validation failures, buffer/dims failures, memory limit/allocation failures, backend failures.
- Keep variant naming stable so downstream tests and C ABI last-error mapping remain stable.

4. Diagnostics contract (`MEM-03`):
- Create a structured error report type (can wrap `LibcintRsError`) with machine-parseable fields.
- Attach required context fields at validation/planning boundaries before returning errors.

## Validation Architecture

Concrete strategy for this phase:

1. Validation layers and order:
- Layer A: API family/representation/arity precheck (`OperatorDescriptor`, shell tuple arity).
- Layer B: input structure validation (typed model invariants for safe; slot-width and pointer-offset checks for raw).
- Layer C: shape/dims validation (`natural_shape` vs `CompatDims` override in raw).
- Layer D: buffer/workspace feasibility validation (required elements/bytes, memory limit feasibility).

2. Canonical validator outputs:
- `ValidatedInputs` (normalized typed/raw views, resolved arity/representation).
- `ValidatedShape` (component count, logical extents, flat element count).
- `WorkspaceQuery` (deterministic bytes/alignment).
- These outputs must be pure/deterministic for identical inputs.

3. Failure typing rules:
- Structural input violations -> `InvalidAtmLayout` / `InvalidBasLayout` / `InvalidEnvOffset` / `InvalidShellTuple`.
- Shape/layout violations -> `InvalidDims` / `BufferTooSmall`.
- Support envelope violations -> `UnsupportedApi` / `UnsupportedRepresentation`.
- Resource violations -> `MemoryLimitExceeded`, allocation failures.
- Never downgrade these to opaque strings in public API.

4. Diagnostics instrumentation (`MEM-03`):
- Emit `tracing` events at validator entry/exit and error return points.
- Include fields: `api`, `representation`, `shell_tuple`, `dims`, `required_bytes`, `provided_bytes`, `memory_limit_bytes`, `backend_candidate`, `feature_flags`.
- Ensure error values and trace events carry consistent identifiers for correlation.

5. Determinism checks in tests:
- Repeated identical `query_workspace` calls return identical results.
- `dims` mismatch always fails pre-execution and never permits partial write semantics.
- Same invalid input maps to the same typed error variant and message template.

## Don’t Hand-Roll

Do not custom-build these in Phase 1:
- Ad-hoc string error systems (use `thiserror` + structured report).
- Multiple independent validators per surface (share one validation core).
- Generic tensor frameworks for core contracts (keep explicit layout/shape contracts).
- Hidden fallback allocators in call sites (enforce central workspace allocator policy).

## Common Pitfalls and Gotchas

- Exposing `dims` on the safe API leaks C-style semantics and conflicts with the design boundary.
- Casting raw offsets to `usize` before bounds checks can hide negative/overflow-invalid input.
- Allowing “best effort” output writes on size mismatch violates raw contract (`InvalidDims`/`BufferTooSmall` only).
- Mixing `anyhow` into public API blocks downstream error matching and fails `SAFE-04` intent.
- Logging without structured fields weakens `MEM-03`; messages must be machine-joinable.
- Premature kernel implementation in this phase causes contract churn; freeze contracts first.

## Implementation Guidance (Actionable)

1. Create contract modules first (no execution logic):
- `core`: domain types (`atom`, `shell`, `basis/env`, `operator`, `tensor`, `error`).
- `runtime`: `validator`, `workspace_estimator` interfaces, and diagnostics context structs.
- `compat`: raw view wrappers and `CompatDims` validator.

2. Implement validation as pure functions:
- Input -> `ValidatedInputs/ValidatedShape/WorkspaceQuery`.
- No allocation side effects in validator path beyond small metadata.

3. Introduce API signatures early, gate unimplemented execution:
- Stabilize `query_workspace` now.
- If `evaluate_into` is not implemented, return explicit `UnsupportedApi`/`BackendFailure` placeholder with clear reason instead of panics.

4. Add requirement-mapped tests now:
- `SAFE-01`: typed construction and invariant failure tests.
- `SAFE-02`: workspace determinism and shell-arity tests.
- `SAFE-04`: error category differentiation tests.
- `MEM-03`: trace/error report field completeness tests.

5. Enforce unsafe/allocation policy with linting and review gates:
- `#![deny(unsafe_op_in_unsafe_fn)]`.
- Centralized allocation entry points for workspace-related paths.

## Code Examples (Contract Sketches)

```rust
pub fn query_workspace(
    op: OperatorId,
    rep: Representation,
    basis: &BasisSet,
    shells: ShellTuple,
    opts: &ExecutionOptions,
) -> Result<WorkspaceQuery, LibcintRsError>;
```

```rust
#[derive(Debug, thiserror::Error)]
pub enum LibcintRsError {
    #[error("invalid shell tuple for {api}: expected {expected}, got {got}")]
    InvalidShellTuple { api: &'static str, expected: usize, got: usize },
    #[error("invalid dims for {api}: expected={expected:?}, got={got:?}")]
    InvalidDims { api: &'static str, expected: Vec<usize>, got: Vec<usize> },
    #[error("memory limit exceeded: required={required}, limit={limit}")]
    MemoryLimitExceeded { required: usize, limit: usize },
    #[error("backend failure in {backend}: {detail}")]
    BackendFailure { backend: &'static str, detail: String },
    #[error("unsupported api {api}: {reason}")]
    UnsupportedApi { api: &'static str, reason: &'static str },
}
```

## Plan Implications for Phase Execution

Recommended plan decomposition for Phase 1:

1. Plan A: Core contract types and enums (`SAFE-01` baseline).
2. Plan B: Shared validation core + raw view/dims validation (`SAFE-01`, `SAFE-02` prerequisites).
3. Plan C: Workspace query contract + deterministic estimator scaffolding (`SAFE-02`).
4. Plan D: Error taxonomy + diagnostics report + tracing instrumentation (`SAFE-04`, `MEM-03`).
5. Plan E: Requirement-mapped tests and contract freeze checklist.

Phase 1 exit criteria should require all four requirement targets passing in tests before any Phase 2 kernel work starts.

## Confidence and Open Risks

- Confidence: High on contract architecture and requirement mapping because this aligns across context, requirements, and detailed design.
- Remaining risk: over-designing module topology before practical usage. Mitigation: keep module split minimal but enforce contract boundaries and test coverage now.
- Remaining risk: introducing provisional execution stubs that later conflict with backend traits. Mitigation: keep execution traits in place, but avoid concrete kernel assumptions in Phase 1.
