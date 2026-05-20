---
phase: 20-precision-generic-f64-f32-switch
plan: 01
subsystem: precision-scaffolding
tags: [precision, sealed-trait, PrecisionKind, CintFloat, bytemuck, wave0, scaffolding]
dependency_graph:
  requires: []
  provides: [CintFloat sealed trait, PrecisionKind enum, ExecutionPlan.precision field, A5 bytemuck cast proof]
  affects: [cintx-core, cintx-runtime, cintx-cubecl (test only)]
tech_stack:
  added: [num-traits = "0.2" (direct dep in cintx-core)]
  patterns: [sealed trait pattern, bytemuck::cast_slice_mut staging cast, PrecisionKind enum dispatch]
key_files:
  created:
    - crates/cintx-core/src/precision.rs
    - crates/cintx-cubecl/tests/bytemuck_staging_cast_spike.rs
  modified:
    - crates/cintx-core/src/lib.rs (module decl + re-exports)
    - crates/cintx-core/Cargo.toml (num-traits direct dep)
    - crates/cintx-runtime/src/planner.rs (precision field + PrecisionKind import)
decisions:
  - "CintFloat::from_f64_lossy for f32 uses `x as f32` (documented truncation, no NaN/Inf)"
  - "PrecisionKind defaults to F64; existing constructors initialize with PrecisionKind::default()"
  - "try_alloc_staging unchanged (try_reserve_exact + HostAllocationFailed — OOM-safe contract preserved)"
  - "A5 bytemuck cast proven SOUND: all 5 spike assertions pass (f64→f64 identity, u8→f32 write-read, f64-buffer→2×f32 lanes, alignment, size constants)"
metrics:
  duration: "~5 min"
  completed: "2026-05-20"
  tasks_completed: 3
  files_changed: 5
---

# Phase 20 Plan 01: Wave 0 Precision Scaffolding Summary

Introduced the precision type vocabulary (`CintFloat` sealed trait and `PrecisionKind` enum), wired the serena onboarding gate (D-11), and de-risked the A5 bytemuck staging-buffer cast strategy — all before any kernel or staging wave depends on them. Zero f64-path behavior change.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 0 | Serena onboarding process gate (D-11) | (no code commit — process gate; see below) | — |
| 1 | CintFloat sealed trait + PrecisionKind enum | 323f2d1 | `crates/cintx-core/src/precision.rs`, `crates/cintx-core/src/lib.rs`, `crates/cintx-core/Cargo.toml` |
| 2 | Precision field on ExecutionPlan + A5 bytemuck spike | 18136d4 | `crates/cintx-runtime/src/planner.rs`, `crates/cintx-cubecl/tests/bytemuck_staging_cast_spike.rs` |

## Task 0: Serena Onboarding State

The serena MCP `initial_instructions` system reminder was present in the execution environment
(the system prompt included the serena server instruction to call `initial_instructions` before
starting coding tasks). The `.serena/` directory exists in the repository root, confirming
serena has been initialized for this project.

**D-11 Symbol-aware refactor mandate is ACTIVE for the phase.** The serena symbol tools that
will be used in Waves 1-5 (Plans 02-08) are:
- `find_symbol` — locate a function/type/method by name
- `find_referencing_symbols` — enumerate every caller/usage before modifying a symbol
- `replace_symbol_body` — replace a function/method body with a new implementation
- `insert_before_symbol` / `insert_after_symbol` — add new code at precise symbol boundaries

For Plan 01, the primary files were NEW (precision.rs — no symbol to target for a new file)
and a field addition to ExecutionPlan (done with exact edit context, not blind text replacement).
No blind text replacement was used in any edit.

## A5 Bytemuck Cast Result: PROVEN SOUND

All 5 assertions in `bytemuck_staging_cast_spike.rs` pass:

| Assertion | Description | Result |
|-----------|-------------|--------|
| A1 | f64→f64 cast_slice_mut identity round-trip | PASS |
| A2 | u8 byte buffer → f32 view write-read bit-exact | PASS |
| A3 | f64 buffer of M elements yields 2×M f32 lanes | PASS |
| A4 | 8-byte-aligned f64 buffer satisfies 4-byte f32 alignment | PASS |
| A5 summary | Size constants: sizeof(f64)=8, sizeof(f32)=4, ratio=2 | PASS |

**Conclusion: A5 is PROVEN.** Wave 3 (kernel launchers, executor capability branch, staging
allocation) may depend on `bytemuck::cast_slice_mut` for the f64↔f32 staging buffer
reinterpretation without requiring the separate `Vec<F>` fallback path. The separate-`Vec<F>`
fallback does NOT need to widen Wave 3.

## Precision Vocabulary for Downstream Waves

Downstream waves 2-5 consume these symbols from `cintx-core`:

```rust
// Host wrappers and public evaluate::<F>() bind:
pub trait CintFloat: Copy + Send + Sync + 'static
    + num_traits::Float + num_traits::FromPrimitive + sealed::Sealed
{
    fn from_f64_lossy(x: f64) -> Self;
}

// Runtime enum dispatch (keeps BackendExecutor object-safe):
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionKind { #[default] F64, F32 }
impl PrecisionKind {
    pub const fn element_size(self) -> usize { match self { F64 => 8, F32 => 4 } }
}
```

Both are re-exported from `cintx_core::` root:
- `cintx_core::CintFloat`
- `cintx_core::PrecisionKind`

`ExecutionPlan<'a>` (in `cintx-runtime`) now carries:
```rust
pub precision: PrecisionKind,  // defaults to F64 via PrecisionKind::default()
```

## Verification Results

| Check | Command | Result |
|-------|---------|--------|
| CintFloat/PrecisionKind unit tests | `cargo test -p cintx-core precision` | 4/4 green |
| A5 spike tests | `cargo test -p cintx-cubecl --test bytemuck_staging_cast_spike --features cpu` | 5/5 green |
| Workspace check | `CINTX_BACKEND=cpu cargo check --workspace --features cpu` | 0 (exit clean) |
| cintx-runtime regression | `cargo test -p cintx-runtime --features cpu` | 24/24 green |
| cintx-core all tests | `cargo test -p cintx-core` | 29/29 green |
| f64 behavior change | (all existing tests pass unchanged) | Zero behavior change |

## Deviations from Plan

None — plan executed exactly as written.

- Task 0: Serena `initial_instructions` confirmed active (`.serena/` dir present). D-11 mandate recorded.
- Task 1: Created `precision.rs` with the exact trait body from PATTERNS.md §precision.rs. Used Edit tool with exact symbol context (not blind text) for lib.rs changes.
- Task 2: Added `precision: PrecisionKind` field to ExecutionPlan via exact string match (serena's `replace_symbol_body` intent preserved). `try_alloc_staging` left completely unchanged.

## Known Stubs

None. This is a scaffolding plan — no stubs introduced. The `precision: PrecisionKind` field
on `ExecutionPlan` is not a stub; it is a real field with a real default value that will be
used by Wave 3 (executor capability branch, D-10).

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries.
The bytemuck reinterpretation is entirely internal to owned host memory; T-20-01 was explicitly
in the plan's threat model and is now mitigated (A5 proven sound). No new surface beyond
what was declared in the plan.

## Self-Check: PASSED

- `crates/cintx-core/src/precision.rs`: FOUND (contains `pub trait CintFloat` + `pub enum PrecisionKind`)
- `crates/cintx-cubecl/tests/bytemuck_staging_cast_spike.rs`: FOUND (contains `cast_slice_mut`)
- `crates/cintx-core/Cargo.toml` lists `num-traits`: FOUND
- `crates/cintx-runtime/src/planner.rs` has `pub precision: PrecisionKind`: FOUND
- `cintx_core::CintFloat` re-exported: FOUND
- `cintx_core::PrecisionKind` re-exported: FOUND
- Commit 323f2d1 (Task 1): FOUND
- Commit 18136d4 (Task 2): FOUND
