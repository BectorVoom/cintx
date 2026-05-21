---
phase: 20-precision-generic-f64-f32-switch
plan: 11
subsystem: oracle / test coverage
tags: [gap-closure, f32-correctness, prec-05, prec-04, f12, ncomp-3, cr-01, cr-02]
dependency_graph:
  requires: [20-10]
  provides: [PREC-05-f12-f32-parity-gate]
  affects: []
tech_stack:
  added: []
  patterns:
    - "evaluate_generic::<f32>() + ExecutionOptions { f12_zeta: Some(1.2_f64) } for f12 safe-API f32 path"
    - "#[cfg(all(has_vendor_libcint, feature = \"with-f12\"))] gate on f32 parity test"
    - "collect_stg_ip1_f32() helper isolates the safe-API f32 call for int2e_stg_ip1_sph"
    - "plan.output_layout.staging_elements for correct spinor staging size in executor test"
key_files:
  created:
    - .planning/phases/20-precision-generic-f64-f32-switch/deferred-items.md
  modified:
    - crates/cintx-oracle/tests/f32_parity.rs
    - crates/cintx-cubecl/src/executor.rs
decisions:
  - "PREC-05 f32 parity test restricted to 3 symmetric quartets ([0,1,0,1], [3,4,3,4], [0,2,0,2]) matching f64 oracle scope — same CR-01/CR-02 regime is exercised (staging_elements = ncomp*base > 2*chunk_len for the old broken code), while avoiding a pre-existing asymmetric-quartet value-swap bug in the f12 ip1 kernel (DI-01) that would produce max_rel_error=3027 unrelated to precision rounding."
  - "max_rel_error for int2e_stg_ip1_sph f32 parity = 3.022e-8 (floor=1e-4). No adjustment to f32_tolerance_for_family('f12') needed; the unified catch-all is sufficient."
  - "Rule 1 auto-fix: executor.rs representation_transforms test had vec![0.0; 8] spinor staging — always undersized (l=1/l=2 shells require 120 f64s). The CR-01 BufferTooSmall guard from 20-10 correctly caught it. Fixed by deriving spinor_staging_elems from plan.output_layout.staging_elements."
metrics:
  duration_minutes: 15
  completed_date: "2026-05-21"
  tasks_completed: 2
  files_modified: 3
  commits: 3
---

# Phase 20 Plan 11: Gap 2 (PREC-05) f12 F32 Oracle Parity Summary

**One-liner:** Vendor-gated f32 parity test for int2e_stg_ip1_sph (ncomp=3) added — 3 symmetric quartets pass at max_rel_error=3.022e-8 (floor 1e-4), confirming CR-01/CR-02 regime is correctly handled post-20-10; Rule 1 fix for spinor staging undersize in executor test exposed by the BufferTooSmall guard.

## Objective

GAP-CLOSURE (PREC-05) — oracle/test half. Add vendor-gated f32 parity coverage for the
multi-component f12 derivative operator (`int2e_stg_ip1_sph`, ncomp=3). This test is
load-bearing: it exercises the exact CR-01/CR-02 regime that was silently corrupting outputs
before Plan 20-10.

## Load-Bearing Data-Flow Analysis (PREC-05)

**Why this test would FAIL on pre-20-10 code:**

The f32 arm of `launch_f12` does:
1. `staging` arrives from `api.rs` with `staging.len() = chunk_len` (true output count)
2. `bytemuck::cast_slice_mut(staging)` → `staging_f32.len() = chunk_len * 2` (doubled due to f32 being half the byte width)
3. **Pre-20-10 (broken):** `launch_f12_typed::<f32>` received the full `staging_f32` (length `chunk_len * 2`). Inside the typed inner, `out_elems = staging.len()` = `chunk_len * 2`. The `staging_f64` temp buffer was allocated at `chunk_len * 2`, but the sub-kernel only filled `chunk_len` f64 slots. The readback loop `staging[..out_elems]` then copied `chunk_len * 2` values from the temp, including `chunk_len` uninitialized zeros.

**For ncomp=3 (int2e_stg_ip1_sph), chunk_len = ncomp * base:**

- For shls=[0,2,0,2] (s-p-s-p): base = 1*3*1*3 = 9 → staging_elements = 3*9 = 27 → staging_f32.len() = 54 → old copy_len = 54 → but sub-kernel only filled 27 f64s → 27 zeros appended → wrong values returned.
- For shls=[0,1,0,1] (s-s-s-s): base = 1*1*1*1 = 1 → staging_elements = 3 → staging_f32.len() = 6 → old copy_len = 6 → sub-kernel filled 3 f64s → 3 zeros appended at tail → 3 extra zeros.

**Post-20-10 (fixed):** The F32 outer arm captures `out_elems = staging.len()` before `bytemuck::cast_slice_mut`, then passes `&mut staging_f32[..out_elems]` to the typed inner. Inside, `staging.len() = out_elems = chunk_len`. The `staging_f64` is allocated at `out_elems`, sub-kernel fills all `out_elems` slots, readback copies exactly `out_elems` values. No zeros appended.

**Empirical confirmation:** Test passes with max_rel_error=3.022e-8 on post-20-10 code.

## Tasks Completed

### Task 1: f12-derivative f32 parity test (int2e_stg_ip1_sph, ncomp=3)

**Commit:** `a63535a`

Added `test_f32_int2e_stg_ip1_sph_parity` to `crates/cintx-oracle/tests/f32_parity.rs`:

```rust
#[test]
#[cfg(all(has_vendor_libcint, feature = "with-f12"))]
fn test_f32_int2e_stg_ip1_sph_parity() { ... }
```

Helper `collect_stg_ip1_f32()` calls `evaluate_generic::<f32>()` with `ExecutionOptions { f12_zeta: Some(1.2_f64), ..Default::default() }` and OperatorId 107 (`int2e_stg_ip1_sph`).

**Shell quartets tested:**
| Quartet          | base | staging_elements (ncomp=3) | max_rel_error |
|------------------|------|---------------------------|---------------|
| [0,1,0,1] s-s-s-s | 1   | 3                         | 0.000e0       |
| [3,4,3,4] s-s-s-s | 1   | 3                         | 3.022e-8      |
| [0,2,0,2] s-p-s-p | 9   | 27                        | 0.000e0       |

**Deferred DI-01:** A pre-existing asymmetric-quartet value-swap bug in the f12 ip1 kernel was discovered when attempting to test all 625 quartets (max_rel_error=3027, 1048 mismatches). This is NOT CR-01/CR-02 — the same swap pattern appears in the f64 path. Restricted the test to the 3 symmetric quartets matching the f64 oracle scope. DI-01 documented in `deferred-items.md`.

### Task 2: Verification — load-bearing confirmation + PREC-04 f64 gate

Full verification results:

- `CINTX_BACKEND=cpu cargo check --workspace --features cpu` — exit 0
- `CINTX_BACKEND=cpu cargo test -p cintx-cubecl --features cpu,with-f12,with-4c1e` — **180 passed** (including the previously-failing `representation_transforms_keep_staging_only_contract` test after Rule 1 fix)
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12 --test f32_parity` — **12 passed** (11 existing + 1 new `test_f32_int2e_stg_ip1_sph_parity`)
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu,with-f12,with-4c1e --test '*'` — all integration test suites pass (14 suites, 0 failed); PREC-04 preserved byte-identical

Pre-existing `compare::tests` lib failures (4 tests, CINTshells_cart_offset[4] mismatch) remain unchanged — confirmed out of scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed spinor staging undersize in executor representation_transforms test**

- **Found during:** Task 2 full verification
- **Issue:** `representation_transforms_keep_staging_only_contract` had `vec![0.0; 8]` for the spinor staging. For shells l=1/l=2, the correct size is 120 f64s. The CR-01 `BufferTooSmall` guard added in 20-10 correctly catches this — the test was always writing beyond its buffer; it just went undetected before the guard was added.
- **Fix:** Replaced `vec![0.0; 8]` with `vec![0.0_f64; plan.output_layout.staging_elements.max(8)]` so the staging self-sizes to the plan's declared output.
- **Files modified:** `crates/cintx-cubecl/src/executor.rs`
- **Commit:** `16b9ac9`

### Out-of-Scope Discovery

**DI-01: Pre-existing f12 ip1 asymmetric-quartet value-swap bug**

- Discovered during test development when testing all 625 quartets.
- Not caused by any change in Phase 20 — same swap in f64 path.
- Documented in `deferred-items.md` for a future fix plan.
- Not fixed here per deviation Rule 3 scope boundary (not caused by current task changes).

## Known Stubs

None.

## Self-Check

### Files Exist

- [x] `crates/cintx-oracle/tests/f32_parity.rs` — modified (test added at line 954+)
- [x] `crates/cintx-cubecl/src/executor.rs` — modified (spinor staging fix)
- [x] `.planning/phases/20-precision-generic-f64-f32-switch/deferred-items.md` — created

### Commits Exist

- [x] `a63535a` — feat(20-11): f32 parity test for int2e_stg_ip1_sph
- [x] `16b9ac9` — fix(20-11): Rule 1 spinor staging undersize fix

## Self-Check: PASSED
