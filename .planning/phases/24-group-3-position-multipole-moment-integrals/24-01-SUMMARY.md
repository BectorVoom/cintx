---
phase: 24-group-3-position-multipole-moment-integrals
plan: 01
subsystem: testing
tags: [oracle, vendor-ffi, bindgen, parity, libcint, moment, multipole]

# Dependency graph
requires:
  - phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
    provides: PTR_COMMON_ORIG env-slot plumbing + build_h2o_sto3g_common_orig fixture
  - phase: 23-group-1-remaining-1st-derivative-families-cart-sph
    provides: rank-parameterized vendor_parity + registration recipe (D-05) precedent
provides:
  - 36 vendor FFI wrappers (24 base + 12 _origj moment symbols) over libcint 6.1.3
  - bindgen allowlist extension for all in-scope moment symbols
  - rank-parameterized vendor_parity helper (moment_common.rs) sizing buffers rank*ni*nj
  - 4 moment parity scaffolds (MOM-01..04) — the Nyquist RED→GREEN target for plans 02-05
  - env_with_rinv_origin + non_square_shell_pair shared helpers
  - OQ-2 cart_offset lib-unit failure triaged as pre-existing, Phase 24 gate de-blocked
affects: [24-02, 24-03, 24-04, 24-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rank-parameterized vendor_parity: generalize the hardcoded NCOMP=9 helper by passing rank and sizing every buffer rank*ni*nj (D-08 no-truncation)"
    - "Nyquist RED scaffold: parity #[test] bodies gated on has_vendor_libcint so the crate skips cleanly before the kernel/RawApiId consts exist; helper module compiles in isolation"
    - "Anti-trivial-origin discipline: Cluster-A on non-zero common-orig fixture; rinv/drinv on a non-zero env_with_rinv_origin center (zero origin disallowed)"

key-files:
  created:
    - crates/cintx-oracle/tests/moment_common.rs
    - crates/cintx-oracle/tests/moment_r_parity.rs
    - crates/cintx-oracle/tests/moment_low_parity.rs
    - crates/cintx-oracle/tests/moment_high_parity.rs
    - crates/cintx-oracle/tests/moment_nontensor_parity.rs
  modified:
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - .planning/todos/pending/oracle-cart-offset-vendor-zero.md

key-decisions:
  - "Only the 12 source-confirmed _origj symbols (r/rr/r2/r4/z/zz) are registered — no rrr_origj/rrrr_origj exist in libcint 6.1.3 src/autocode/intor1.c (OQ-3)"
  - "rinv/drinv read PTR_RINV_ORIG (not PTR_COMMON_ORIG) per D-04/OQ-1 correction; tests inject a NON-ZERO [0.5,-0.3,0.8] rinv center"
  - "Non-square bra×ket block fixed at (O-1s shell 0)×(O-2p shell 2) for every test (D-07 transpose gate)"
  - "OQ-2 cart_offset lib-unit failure reproduced at pre-phase-20 commit 8997703 → standalone harness bug, blocks_phase_24_gate=false"

patterns-established:
  - "vendor_parity(rank, api_sph, api_cart, vendor_sph, vendor_cart, atm, bas, env, label): caller passes the prepared fixture so Cluster-B can inject a non-zero rinv origin"
  - "Macro-generated per-family #[test] wrappers (moment_parity_test!) to keep the 4 scaffolds compact across 1/3/9/27/81 ranks"

requirements-completed: [MOM-01, MOM-02, MOM-03, MOM-04]

# Metrics
duration: 18min
completed: 2026-05-30
---

# Phase 24 Plan 01: Moment Validation Surface (Vendor FFI + Parity Scaffolds) Summary

**Wave-0 validation surface for all of Phase 24: 36 libcint moment vendor wrappers + bindgen allowlist, a rank-parameterized vendor_parity helper sizing buffers rank*ni*nj, four MOM-01..04 parity scaffolds (the Nyquist RED→GREEN target for the kernel plans), and the OQ-2 cart_offset lib-unit failure triaged as pre-existing and de-blocked.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-05-30 (Phase 24 execution)
- **Completed:** 2026-05-30
- **Tasks:** 3
- **Files modified:** 8 (5 created, 3 modified)

## Accomplishments
- Extended the bindgen `allowlist_function` regex with all in-scope moment symbols and added 36 safe `vendor_int1e_*_{sph,cart}` wrappers; the vendor build compiles clean under `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu`.
- Built `moment_common.rs` with a RANK-parameterized `vendor_parity` (generalizing the hardcoded NCOMP=9 helper), `env_with_rinv_origin`, and a non-square `non_square_shell_pair` — the helper module compiles in isolation under the vendor gate today.
- Created the four MOM-01..04 parity scaffolds. They reference the per-family `RawApiId` consts that land in plans 02-05, so the parity `#[test]` bodies are RED targets gated on `has_vendor_libcint`; WITHOUT the vendor gate all four files compile and skip cleanly.
- Triaged OQ-2: reproduced `CINTshells_cart_offset[4] cintx=8 vendor=0` at pre-phase-20 commit `8997703`, classified it a standalone oracle-harness bug, and recorded `blocks_phase_24_gate: false`.

## Task Commits

Each task was committed atomically:

1. **Task 1: bindgen allowlist + 36 vendor FFI wrappers** - `31a11fc` (feat)
2. **Task 2: 4 moment parity scaffolds + rank-parameterized helper** - `ab0848e` (test)
3. **Task 3: triage OQ-2 cart_offset lib-unit failure** - `8b5c1b3` (docs)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified
- `crates/cintx-oracle/build.rs` - allowlist regex extended with 24 base + 12 `_origj` moment symbols (no `.c` source change; intor1.c already present).
- `crates/cintx-oracle/src/vendor_ffi.rs` - 36 `vendor_int1e_*_{sph,cart}` wrappers cloned from the iprinv analog.
- `crates/cintx-oracle/tests/moment_common.rs` - rank-parameterized `vendor_parity` + `env_with_rinv_origin` + `non_square_shell_pair` + collectors/asserts at ATOL=1e-12.
- `crates/cintx-oracle/tests/moment_r_parity.rs` - MOM-01: `r`, `r_origj` (rank 3).
- `crates/cintx-oracle/tests/moment_low_parity.rs` - MOM-02: `rr`(9)/`r2`/`z`/`zz` + `_origj`.
- `crates/cintx-oracle/tests/moment_high_parity.rs` - MOM-03: `rrr`(27)/`rrrr`(81)/`r4` + `r4_origj`; NO `rrr/rrrr _origj`.
- `crates/cintx-oracle/tests/moment_nontensor_parity.rs` - MOM-04: `p4`(1)/`irp`(9) on common-orig; `rinv`(1)/`drinv`(3) on a non-zero rinv center.
- `.planning/todos/pending/oracle-cart-offset-vendor-zero.md` - OQ-2 triage disposition (reproduced pre-phase-20, de-blocked).

## Decisions Made
- **`_origj` symbol set (OQ-3):** grep of `libcint-master/src/autocode/intor1.c` confirms exactly `int1e_{r,rr,r2,r4,z,zz}_origj_{cart,sph}` exist (12 symbols). `rrr_origj`/`rrrr_origj` do NOT exist — registered nowhere; only a clarifying comment mentions them.
- **rinv/drinv origin (D-04/OQ-1):** the rinv/drinv parity tests inject a non-zero `[0.5,-0.3,0.8]` center via `env_with_rinv_origin` (PTR_RINV_ORIG=4, in the reserved PTR_ENV_START block — no collision with atom coords at env[20..]). A zero rinv origin is trivially-passing and disallowed.
- **Symbol count:** the plan prose said "base 26"; the explicit pipe-separated list it provided (authoritative) enumerates 24 base symbols (12 families × {cart,sph}). All symbols that exist in libcint 6.1.3 are covered — 24 base + 12 `_origj` = 36 wrappers. The "26" was a benign prose arithmetic slip, not a missing symbol.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Conditional import to silence an introduced unused-import warning**
- **Found during:** Task 2 (moment_common.rs)
- **Issue:** `ATM_SLOTS` is only used inside the `#[cfg(has_vendor_libcint)]` vendor collector, so without the vendor gate it produced a new `unused_imports` warning in all four parity test crates.
- **Fix:** Split the import so `ATM_SLOTS` is `#[cfg(has_vendor_libcint)]`-gated; the four parity files now compile warning-free without the vendor gate.
- **Files modified:** crates/cintx-oracle/tests/moment_common.rs
- **Verification:** `cargo test -p cintx-oracle --features cpu --test moment_common --test moment_r_parity --no-run` finishes with no warnings for these targets.
- **Committed in:** `ab0848e` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — warning hygiene)
**Impact on plan:** Cosmetic warning fix only. No scope creep; all symbols and helpers as specified.

## Issues Encountered
- The plan's "26 base symbols" prose did not match its own explicit symbol list (24). Resolved by following the explicit enumerated list (authoritative) and confirming against libcint source — all existing symbols are covered.
- OQ-2 reproduction required a throwaway detached worktree at `8997703` (added, run, removed cleanly via `git worktree remove --force`); the working branch was never disturbed.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- **Plan 24-02 (Cluster A)** can now wire the manifest lock + `RawApiId` consts + the parameterized moment kernel against `moment_r_parity.rs` / `moment_low_parity.rs` / `moment_high_parity.rs`. Its first task should run `cargo test -p cintx-oracle --features cpu --no-run` to confirm the Cluster-A consts unblock crate compilation.
- **Plans 24-04/24-05 (Clusters B/C/D)** target `moment_nontensor_parity.rs` (`p4`/`irp` + the non-zero-rinv `rinv`/`drinv`).
- **Gate note:** the OQ-2 lib-unit `cart_offset` failure is pre-existing and does NOT block the Phase 24 integration-parity gate (`blocks_phase_24_gate: false`). Use `--test` integration parity, not `--lib`, for the merge contract.

## Self-Check: PASSED

All 5 created files exist; all 3 task commits (`31a11fc`, `ab0848e`, `8b5c1b3`) present in git history.

---
*Phase: 24-group-3-position-multipole-moment-integrals*
*Completed: 2026-05-30*
