---
phase: 02
slug: cpu-compatibility-execution
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-14
updated: 2026-03-14
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` |
| **Config file** | `Cargo.toml` (workspace test targets) |
| **Quick run command** | `cargo test --workspace --test phase2_cpu_backend_routing --test phase2_raw_contracts --test phase2_raw_query_execute --test phase2_safe_evaluate_layout --test phase2_memory_limits` |
| **Full suite command** | `cargo test --workspace --all-targets` |
| **Estimated runtime** | ~12 minutes |

---

## Sampling Rate

- **After every task commit:** Run that task's exact `<automated>` command from the map below.
- **After every plan wave:**
  - Wave 1: `cargo test --workspace --test phase2_cpu_backend_routing`
  - Wave 2: `cargo test --workspace --test phase2_cpu_backend_routing --test phase2_raw_contracts`
  - Wave 3: `cargo test --workspace --test phase2_raw_query_execute --test phase2_safe_evaluate_layout --test phase2_no_partial_write`
  - Wave 4: `cargo test --workspace --test phase2_memory_limits`
  - Wave 5: `cargo test --workspace --test phase2_memory_limits --test phase2_allocation_failures`
  - Wave 6: `cargo test --workspace --test phase2_cpu_execution_matrix --test phase2_safe_raw_equivalence --test phase2_oracle_tolerance --test phase2_raw_failure_semantics --test phase2_memory_contracts`
- **Before `$gsd-verify-work`:** `cargo test --workspace --all-targets` must be green.
- **Max feedback latency:** 15 minutes.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | EXEC-01 | integration | `cargo test --workspace --test phase2_cpu_backend_routing cpu_backend_symbols_link` | ❌ planned | ⬜ pending |
| 02-01-02 | 01 | 1 | EXEC-01 | unit | `cargo test --workspace --test phase2_cpu_backend_routing execution_request_contract` | ❌ planned | ⬜ pending |
| 02-01-03 | 01 | 1 | EXEC-01 | unit | `cargo test --workspace --test phase2_cpu_backend_routing stable_family_required_matrix_contract` | ❌ planned | ⬜ pending |
| 02-06-01 | 06 | 2 | EXEC-01 | integration | `cargo test --workspace --test phase2_cpu_backend_routing backend_route_matrix` | ❌ planned | ⬜ pending |
| 02-06-02 | 06 | 2 | COMP-01 | integration | `cargo test --workspace --test phase2_cpu_backend_routing three_c_one_e_spinor_supported` | ❌ planned | ⬜ pending |
| 02-06-03 | 06 | 2 | COMP-01 | integration | `cargo test --workspace --test phase2_cpu_backend_routing stable_family_route_matrix_complete` | ❌ planned | ⬜ pending |
| 02-02-01 | 02 | 2 | RAW-01 | unit | `cargo test --workspace --test phase2_raw_contracts raw_layout_slot_and_offset_checks` | ❌ planned | ⬜ pending |
| 02-02-02 | 02 | 2 | RAW-01 | unit | `cargo test --workspace --test phase2_raw_contracts raw_validation_matrix` | ❌ planned | ⬜ pending |
| 02-02-03 | 02 | 2 | RAW-01 | integration | `cargo test --workspace --test phase2_raw_contracts` | ❌ planned | ⬜ pending |
| 02-07-01 | 07 | 3 | RAW-02 | integration | `cargo test --workspace --test phase2_raw_query_execute raw_query_null_equivalent_contract` | ❌ planned | ⬜ pending |
| 02-07-02 | 07 | 3 | RAW-02 | integration | `cargo test --workspace --test phase2_raw_query_execute raw_query_then_execute_success` | ❌ planned | ⬜ pending |
| 02-07-03 | 07 | 3 | RAW-02 | integration | `cargo test --workspace --test phase2_raw_query_execute` | ❌ planned | ⬜ pending |
| 02-03-01 | 03 | 3 | SAFE-03 | unit | `cargo test --workspace --test phase2_safe_evaluate_layout planner_representation_dims` | ❌ planned | ⬜ pending |
| 02-03-02 | 03 | 3 | SAFE-03 | integration | `cargo test --workspace --test phase2_safe_evaluate_layout safe_evaluate_representation_layout` | ❌ planned | ⬜ pending |
| 02-03-03 | 03 | 3 | RAW-03 | integration | `cargo test --workspace --test phase2_no_partial_write no_partial_write_on_contract_error` | ❌ planned | ⬜ pending |
| 02-04-01 | 04 | 4 | MEM-02 | unit | `cargo test --workspace --test phase2_memory_limits allocation_paths_use_fallible_policy` | ❌ planned | ⬜ pending |
| 02-04-02 | 04 | 4 | MEM-01 | integration | `cargo test --workspace --test phase2_memory_limits chunk_or_memory_limit_exceeded` | ❌ planned | ⬜ pending |
| 02-04-03 | 04 | 4 | MEM-01 | integration | `cargo test --workspace --test phase2_memory_limits` | ❌ planned | ⬜ pending |
| 02-08-01 | 08 | 5 | MEM-01 | integration | `cargo test --workspace --test phase2_memory_limits api_memory_policy_threading` | ❌ planned | ⬜ pending |
| 02-08-02 | 08 | 5 | MEM-02 | integration | `cargo test --workspace --test phase2_allocation_failures allocation_failures_are_typed` | ❌ planned | ⬜ pending |
| 02-08-03 | 08 | 5 | RAW-02 | integration | `cargo test --workspace --test phase2_memory_limits raw_query_execute_memory_contract` | ❌ planned | ⬜ pending |
| 02-05-01 | 05 | 6 | COMP-01 | integration | `cargo test --workspace --test phase2_cpu_execution_matrix --test phase2_safe_raw_equivalence` | ❌ planned | ⬜ pending |
| 02-05-02 | 05 | 6 | COMP-01 | integration | `cargo test --workspace --test phase2_oracle_tolerance oracle_tolerance_matrix` | ❌ planned | ⬜ pending |
| 02-05-03 | 05 | 6 | RAW-03, MEM-01, MEM-02 | integration | `cargo test --workspace --test phase2_raw_failure_semantics --test phase2_memory_contracts` | ❌ planned | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing Rust test infrastructure already covers this phase. No framework/bootstrap work is required before execution.

---

## Manual-Only Verifications

All Phase 2 acceptance behaviors have automated verification targets.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verification commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verification
- [x] Wave 0 dependency check completed (no missing framework prerequisites)
- [x] No watch-mode flags in validation commands
- [x] `nyquist_compliant: true` set in frontmatter
- [ ] `cargo test --workspace --all-targets` green on latest phase branch

**Approval:** pending
