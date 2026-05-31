---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 05
subsystem: kernels
tags: [giao, a01gp, vendor-parity, gap-closure, cr-01, oracle, manifest, math-fix]

# Dependency graph
requires:
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    plan: 04
    provides: "fail-closed UnsupportedApi guard for int1e_a01gp + mod fail_closed contract test (both removed by this plan once parity landed)"
  - phase: 26-group-5-spin-free-giao-nmr-integrals-complex
    plan: 02
    provides: "int1e_a01gp registered/kernel'd/vendor-wrapped with the rank-9 27-s table (correct s-slot map + gout; only the 0.5 family factor was missing)"
provides:
  - "int1e_a01gp byte-identical to libcint 6.1.3 (cart+sph, atol=1e-12) on the non-zero-gauge non-square H1xO block — the 11th and final spin-free 1e GIAO/CG family"
  - "All 11 GIAO-01 1e families now oracle_covered=true (cart+sph); GIAO-01 fully closed"
  - "26-04 fail-closed guard removed; int1e_a01gp rides the normal nuclear-engine dispatch path"
affects: [giao, complex-output, 26-verification-cr-01]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GIAO/CG family scale factors come from libcint `envs.common_factor *= k` per-symbol (a01gp: 0.5, cg/giao_a11part: -0.5, gnuc/ignuc: 0.5, ia01p: 1.0) — transcribe the family factor as well as the s-table/gout"
    - "Guard removal gated on the actual vendor parity test passing, not on belief the math is fixed — the guard is only deleted in the same commit that turns the parity test green"

key-files:
  created:
    - .planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-05-SUMMARY.md
  modified:
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/tests/giao_1e_parity.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv

key-decisions:
  - "BRANCH A (math-fix succeeded): the 26-02 ~2x discrepancy was a MISSING 0.5 common factor, not a ket-derivative s-slot double-count. libcint int1e_a01gp_{cart,sph} set `envs.common_factor *= 0.5` (intor1.c:551/572); the cintx kernel left fam_factor at the 1.0 default for op_kind==3. Setting fam_factor=0.5 made the family byte-identical."
  - "The s-table (27-s -> used 18 slots) and the 9-component gout were ALREADY transcribed verbatim from intor1.c and were correct; component 0 vanishes on the H1xO block (o0 = cy*s23 - cz*s14 - cy*s25 + cz*s16 evaluates to ~0 there), which masked the uniform 0.5 factor and made only components 1..8 look ~2x in 26-02."
  - "int1e_a01gp_spinor row kept oracle_covered=false (NOT flipped to true): spinor reps return UnsupportedApi (D-11) and are not parity-tested; this matches every sibling GIAO family's _spinor row (cg_a11part, giao_a11part, ia01p all stay false). Deviates from the literal Task 2 step-3 wording (which listed all three rows) in favor of D-11 + sibling consistency."
  - "Removed mod fail_closed::test_int1e_a01gp_is_fail_closed (from 26-04): with the guard gone and parity green, a fail-closed contract test would contradict the corrected dispatchable path. The vendor parity test now covers a01gp correctness."

patterns-established:
  - "When a registered family is ~Nx off uniformly but a subset of components looks correct, suspect a missing per-family `common_factor` scale (components that vanish on the test block hide the factor) BEFORE re-deriving the s-slot/gout combination."

requirements-completed: [GIAO-01]

# Metrics
duration: 30min
completed: 2026-05-31
---

# Phase 26 Plan 05: int1e_a01gp Vendor Parity (Branch A — Math Fix Succeeded) Summary

**Root-caused the int1e_a01gp rank-9 ~2x discrepancy to a single missing 0.5 family common factor (libcint `envs.common_factor *= 0.5`), corrected the kernel, and — gated on the now-passing vendor byte-identity parity test (cart+sph, atol=1e-12) — removed the 26-04 fail-closed guard, un-ignored the parity test, and flipped int1e_a01gp_{cart,sph} oracle_covered=true. GIAO-01 is fully closed: all 11 spin-free 1e GIAO/CG families are byte-identical to libcint 6.1.3.**

## Path Taken: BRANCH A (math-fix succeeded)

The plan offered two outcomes. **Branch A was taken**: the math fix was feasible, vendor parity passed, the guard was removed, and oracle_covered was flipped. The public API was never left dispatchable-with-wrong-output — the guard removal and the parity-green commit are the same commit (`37eb969`), and the guard was only deleted after `test_int1e_a01gp_parity` was observed passing under `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu`.

## Performance

- **Duration:** ~30 min
- **Tasks:** 2
- **Files modified:** 5 (0 created besides this SUMMARY, 5 modified)

## Accomplishments

- **Task 1 — Root-cause + correct (`one_electron.rs`):** Diffed the cintx a01gp s-table (lines ~3613-3650) and 9-component gout against `CINTgout1e_int1e_a01gp` (intor1.c:485-540) and the WORKING structurally-identical families. The 27-s slot map (`s[1..3,5..7,10..12,14..16,19..21,23..25]`) and the gout `c[]·s[]` combos were already byte-correct. The divergence was the per-family scale: libcint's `int1e_a01gp_cart`/`_sph` set `envs.common_factor *= 0.5` (intor1.c:551, 572), but the kernel's `fam_factor` left op_kind==3 at the 1.0 default (op_kind 0/1=0.5, 4/5=-0.5 were already set; 2=ia01p=1.0 matches libcint, which sets no factor). Added the `op_kind == 3u32 => fam_factor = 0.5` arm. `cargo build -p cintx-cubecl --features cpu` exits 0; the `giao_nuc_op` table still maps `"a01gp" => Some((3, 9))`. Committed `4af9e28`.
- **Task 2 — Gate guard-removal on parity (Branch A):** Ran `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_1e_parity test_int1e_a01gp_parity` (after temporarily removing the guard + #[ignore] so the kernel could execute). Result: **`test parity::test_int1e_a01gp_parity ... ok`** — byte-identity at atol=1e-12. The full vendor-gated `giao_1e_parity` suite is now **11 passed, 0 failed**. Therefore: removed the 26-04 `op_name == "a01gp"` dispatch guard; un-ignored `test_int1e_a01gp_parity`; flipped `oracle_covered=true` on the int1e_a01gp_cart and int1e_a01gp_sph lock rows; removed `mod fail_closed`; ran `cargo build -p cintx-ops` (regenerated api_manifest.{rs,csv}); `manifest-audit status: ok`. Committed `37eb969`.

## Task Commits

1. **Task 1: restore a01gp 0.5 common factor** — `4af9e28` (fix)
2. **Task 2: a01gp vendor parity green — remove guard, flip oracle_covered (Branch A)** — `37eb969` (feat)

## Deviations from Plan

### Auto-fixed / decision deviations

**1. [Rule 1 - Bug, refined root cause] The defect was a missing common factor, not an s-slot double-count**
- **Found during:** Task 1
- **Issue:** 26-02/26-04 hypothesized a ket-derivative s-slot double-count in the `g2 = D_J + D_I` path. The actual defect was a uniform missing 0.5 family common factor — the s-table and gout were already correct verbatim from intor1.c.
- **Fix:** Added `fam_factor = 0.5` for op_kind == 3 (a01gp), matching libcint `envs.common_factor *= 0.5`.
- **Files modified:** `crates/cintx-cubecl/src/kernels/one_electron.rs`
- **Commit:** `4af9e28`

**2. [Decision] int1e_a01gp_spinor row left oracle_covered=false (literal plan listed all three rows)**
- **Found during:** Task 2 (lock edit)
- **Issue:** Task 2 step 3 listed flipping all three a01gp rows (cart, sph, AND spinor) to true. But the spinor representation returns UnsupportedApi (D-11) and is not parity-tested; every sibling GIAO family (cg_a11part, giao_a11part, ia01p) keeps its `_spinor` row at oracle_covered=false.
- **Fix:** Flipped only cart and sph to true; left spinor false for D-11 + sibling consistency. This is the correct, non-contradictory state — flipping spinor to true would falsely claim oracle coverage for a path that returns UnsupportedApi.
- **Files modified:** `crates/cintx-ops/generated/compiled_manifest.lock.json` (+ regenerated artifacts)
- **Commit:** `37eb969`

---

**Total deviations:** 2 (1 refined root cause, 1 manifest-consistency decision). No scope creep.

## Threat Model Coverage

- **T-26-07 (Information disclosure — wrong data):** mitigated. Guard removal was GATED on vendor parity passing at atol=1e-12; the guard was deleted in the same commit that turned the parity test green. The API was never dispatchable-with-wrong-output.
- **T-26-08 (Tampering — manifest drift):** mitigated. oracle_covered flipped to true ONLY on the parity-pass branch (cart+sph); spinor stays false; manifest-audit auto-syncs from the lock (status: ok).

## Threat Flags

None. No new trust boundaries — the change scales an existing output and REMOVES a fail-closed early return that is no longer needed because the path is now correct. The `caller -> eval_raw` numeric surface is unchanged.

## Known Stubs

None. int1e_a01gp is now byte-identical (cart+sph) and oracle_covered=true. The earlier 26-02/26-04 a01gp known-stub is resolved. (The a01gp `_spinor` rep returns UnsupportedApi by design per D-11 — this is the standing phase-wide spinor policy, not a stub.)

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/one_electron.rs (fam_factor=0.5 for op_kind==3; guard removed; `"a01gp" => Some((3, 9))` intact)
- FOUND: crates/cintx-oracle/tests/giao_1e_parity.rs (test_int1e_a01gp_parity un-ignored; mod fail_closed removed)
- FOUND: crates/cintx-ops/generated/compiled_manifest.lock.json (a01gp cart/sph oracle_covered=true, spinor=false)
- FOUND: .planning/phases/26-group-5-spin-free-giao-nmr-integrals-complex/26-05-SUMMARY.md
- FOUND commit 4af9e28 (Task 1 — fix: 0.5 common factor)
- FOUND commit 37eb969 (Task 2 — feat: parity green, guard removed, oracle_covered flipped)
- VERIFIED: `grep -c 'op_name == "a01gp"' one_electron.rs` == 0 (guard gone)
- VERIFIED: vendor-gated `giao_1e_parity` = 11 passed / 0 failed; `cargo test -p cintx-cubecl --features cpu` all green; manifest-audit status: ok
