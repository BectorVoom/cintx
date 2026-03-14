---
phase: 02-cpu-compatibility-execution
status: gaps_found
score: "8/8 phase requirement IDs behavior-verified; 1 verification gap"
updated: 2026-03-14T09:35:23Z
verified: 2026-03-14T09:35:23Z
---

# Phase 2: CPU Compatibility Execution Verification Report

**Phase Goal:** Users can run supported stable-family integrals on CPU via both safe and raw interfaces while preserving explicit memory-limit behavior.
**Status:** gaps_found

## Plan Frontmatter Requirement Accounting

| Plan | Frontmatter requirement IDs | Accounted in REQUIREMENTS.md / ROADMAP.md | Result |
|---|---|---|---|
| `02-01-PLAN.md` | `EXEC-01` | Yes | ✓ |
| `02-02-PLAN.md` | `RAW-01` | Yes | ✓ |
| `02-03-PLAN.md` | `SAFE-03`, `RAW-03`, `COMP-01` | Yes | ✓ |
| `02-04-PLAN.md` | `MEM-01`, `MEM-02` | Yes | ✓ |
| `02-05-PLAN.md` | `COMP-01`, `RAW-01`, `RAW-02`, `RAW-03`, `SAFE-03`, `MEM-01`, `MEM-02`, `EXEC-01` | Yes | ✓ |
| `02-06-PLAN.md` | `EXEC-01`, `COMP-01` | Yes | ✓ |
| `02-07-PLAN.md` | `RAW-02` | Yes | ✓ |
| `02-08-PLAN.md` | `MEM-01`, `MEM-02`, `RAW-02` | Yes | ✓ |

**Accounting result:** all Phase 2 IDs are present and accounted for: `COMP-01`, `RAW-01`, `RAW-02`, `RAW-03`, `SAFE-03`, `MEM-01`, `MEM-02`, `EXEC-01`.

## Goal-Backward Verification

| Goal condition | Status | Evidence |
|---|---|---|
| Stable-family CPU execution works across `1e/2e/2c2e/3c1e/3c2e` and `cart/sph/spinor` | ✓ VERIFIED | CPU route map and adapter (`src/runtime/backend/cpu/router.rs`, `src/runtime/backend/cpu/spinor_3c1e.rs`), matrix tests (`tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_oracle_tolerance.rs`) |
| Safe + raw APIs both execute through shared contracts | ✓ VERIFIED | Shared execution request/planner/executor (`src/runtime/execution_plan.rs`, `src/runtime/planner.rs`, `src/runtime/executor.rs`), integration tests (`tests/phase2_safe_raw_equivalence.rs`, `tests/phase2_raw_query_execute.rs`) |
| Dims/buffer incompatibility yields explicit errors with no partial writes | ✓ VERIFIED | Output staging + contract checks (`src/runtime/output_writer.rs`, `src/runtime/raw/evaluate.rs`), failure tests (`tests/phase2_no_partial_write.rs`, `tests/phase2_raw_failure_semantics.rs`) |
| `memory_limit_bytes` gives chunk-or-explicit-failure and no unhandled OOM in supported paths | ✓ VERIFIED | Memory planner/fallible alloc (`src/runtime/memory/chunking.rs`, `src/runtime/memory/allocator.rs`, `src/runtime/executor.rs`), tests (`tests/phase2_memory_limits.rs`, `tests/phase2_memory_contracts.rs`, `tests/phase2_allocation_failures.rs`) |

## Requirement Coverage

| Requirement ID | Status | Verification evidence |
|---|---|---|
| `COMP-01` | ✓ SATISFIED | Stable-family route + execution + oracle checks (`tests/phase2_cpu_execution_matrix.rs`, `tests/phase2_oracle_tolerance.rs`) |
| `RAW-01` | ✓ SATISFIED | Raw layout/view/validator boundary (`src/runtime/raw/views.rs`, `src/runtime/raw/validator.rs`), tests (`tests/phase2_raw_contracts.rs`) |
| `RAW-02` | ✓ SATISFIED | Raw query-then-execute path and contract parity (`src/runtime/raw/query.rs`, `src/runtime/raw/evaluate.rs`), tests (`tests/phase2_raw_query_execute.rs`, `tests/phase2_memory_limits.rs`) |
| `RAW-03` | ✓ SATISFIED | Explicit layout mismatch failures + no partial writes (`src/runtime/output_writer.rs`, `src/runtime/raw/evaluate.rs`), tests (`tests/phase2_no_partial_write.rs`, `tests/phase2_raw_failure_semantics.rs`) |
| `SAFE-03` | ✓ SATISFIED | Safe evaluate/evaluate_into with representation-correct tensors (`src/api/safe.rs`, `src/runtime/layout.rs`, `src/runtime/planner.rs`), tests (`tests/phase2_safe_evaluate_layout.rs`, `tests/phase2_safe_raw_equivalence.rs`) |
| `MEM-01` | ✓ SATISFIED | Explicit memory-limit policy (`src/runtime/memory/chunking.rs`, `src/runtime/executor.rs`), tests (`tests/phase2_memory_limits.rs`, `tests/phase2_memory_contracts.rs`) |
| `MEM-02` | ✓ SATISFIED | Fallible allocation and typed allocation failures (`src/runtime/memory/allocator.rs`, `src/runtime/executor.rs`), tests (`tests/phase2_allocation_failures.rs`, `tests/phase2_memory_limits.rs`) |
| `EXEC-01` | ✓ SATISFIED | CPU reference backend dispatch and symbol linkage (`build.rs`, `src/runtime/backend/cpu/ffi.rs`, `src/runtime/backend/cpu/router.rs`), tests (`tests/phase2_cpu_backend_routing.rs`, `tests/phase2_cpu_execution_matrix.rs`) |

## Must-Have Cross-Check (Plan Claims vs Code)

| Plan | Must-have status | Notes |
|---|---|---|
| `02-01` | ⚠ PARTIAL | Build/linkage and routing obligations are present; one contract test (`execution_request_contract`) currently fails due feature-flag order assertion drift. |
| `02-02` | ✓ VERIFIED | Raw view/validator boundary implemented and tested. |
| `02-03` | ✓ VERIFIED | Safe evaluate + no-partial-write enforced in shared executor/writer path. |
| `02-04` | ✓ VERIFIED | Runtime chunking/fallible allocation memory core implemented and tested. |
| `02-05` | ✓ VERIFIED | Phase-close matrix/oracle/failure evidence exists and passes. |
| `02-06` | ✓ VERIFIED | CPU routing matrix complete with explicit `3c1e` spinor adapter route. |
| `02-07` | ✓ VERIFIED | Raw query/evaluate integration and query-contract checks implemented and tested. |
| `02-08` | ✓ VERIFIED | API-level memory threading/allocation-failure contracts implemented and tested. |

## Automated Checks Run

- `cargo test --workspace --test phase2_cpu_backend_routing --test phase2_raw_contracts --test phase2_raw_query_execute --test phase2_safe_evaluate_layout --test phase2_no_partial_write --test phase2_memory_limits --test phase2_allocation_failures --test phase2_cpu_execution_matrix --test phase2_safe_raw_equivalence --test phase2_oracle_tolerance --test phase2_raw_failure_semantics --test phase2_memory_contracts`  
  Result: **failed** (1 test): `tests/phase2_cpu_backend_routing.rs::execution_request_contract`
- `cargo test --workspace --test phase2_raw_contracts --test phase2_raw_query_execute --test phase2_safe_evaluate_layout --test phase2_no_partial_write --test phase2_memory_limits --test phase2_allocation_failures --test phase2_cpu_execution_matrix --test phase2_safe_raw_equivalence --test phase2_oracle_tolerance --test phase2_raw_failure_semantics --test phase2_memory_contracts`  
  Result: **passed**
- `cargo test --workspace --test phase2_cpu_backend_routing execution_request_contract`  
  Result: **failed** (asserted feature-flag order mismatch)
- `cargo test --workspace --lib --quiet`  
  Result: **passed**

## Gap Summary

1. **Failing Phase 2 regression gate in `phase2_cpu_backend_routing`**
   - **Failure:** `execution_request_contract` expects feature flags in insertion order but runtime normalizes/sorts flags in `WorkspaceQueryOptions::normalized_feature_flags()` and `ExecutionMemoryOptions::from`.
   - **Observed mismatch:** left `["phase2-contract", "trace-workspace"]` vs expected `["trace-workspace", "phase2-contract"]`.
   - **Impact:** verification suite for Phase 2 is not fully green, so phase cannot be marked `passed`.
   - **Suggested fix:** update test expectation to normalized order (or compare as set if order is intentionally non-contractual).

## Final Verdict

Phase 2 goal behavior is implemented and evidenced across safe/raw CPU execution, routing, memory policy, and failure semantics; however, one required Phase 2 regression test is currently failing, so final status is **`gaps_found`**.
