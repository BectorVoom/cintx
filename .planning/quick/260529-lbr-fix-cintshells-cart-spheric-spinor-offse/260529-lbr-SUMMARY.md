---
phase: quick-260529-lbr
plan: 01
subsystem: compat
tags: [compat, oracle, libcint-parity, ao-loc, cint_bas]

requires:
  - phase: prior compat helpers work
    provides: CINTshells_{cart,spheric,spinor}_offset wrappers + write_offsets shared core
provides:
  - write_offsets byte-faithful to libcint 6.1.3 shells_cgto_offset (i<nbas, never writes ao_loc[nbas])
  - removal of the pre-existing CINTshells_cart_offset[4] cintx=8 vendor=0 helper-parity bail that masked all downstream oracle comparisons
affects: [oracle helper-parity gate, vendor oracle gate, downstream numeric integral parity]

tech-stack:
  added: []
  patterns:
    - "Helper offset semantics match libcint exactly (shell START offsets, not inclusive prefix sums); grand total lives in CINTtot_cgto_* not ao_loc[nbas]"

key-files:
  created: []
  modified:
    - crates/cintx-compat/src/helpers.rs

key-decisions:
  - "Fix the HELPER to match libcint i<nbas exactly; do NOT relax the oracle comparison harness (compare.rs is correct) — decision made with the user"
  - "Required buffer length is now nbas (was nbas+1); BufferTooSmall guard preserved with corrected length"
  - "nbas==0 returns Ok with no index-0 write (sound, no panic); mirrors that libcint's unconditional ao_loc[0]=0 is only valid for nbas>=1"
  - "i32 overflow guarded via i32::try_from + checked_add → ChunkPlanFailed (no silent wrap, no panic)"

patterns-established:
  - "Pattern: byte-faithful libcint loop replication — accumulate the running sum in i32 (ao_loc[i]=ao_loc[i-1]+count(i-1)) to match libcint's FINT arithmetic exactly"

requirements-completed: [QUICK-260529-lbr]

duration: 12min
completed: 2026-05-29
---

# Phase quick-260529-lbr: CINTshells_*_offset libcint i<nbas parity Summary

**`write_offsets` (shared by CINTshells_{cart,spheric,spinor}_offset) is now byte-faithful to libcint 6.1.3 `shells_cgto_offset` — writes exactly `nbas` shell-start offsets and never `ao_loc[nbas]`, removing the helper-parity bail that masked every downstream oracle comparison on this branch.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2 (TDD: RED test + GREEN fix)
- **Files modified:** 1 (`crates/cintx-compat/src/helpers.rs`)

## Accomplishments

- Flipped the helper unit test (RED) to the libcint-matching expectation `[0, 1, 0]` and confirmed it failed against the old `[0, 1, 7]` code.
- Rewrote `write_offsets` to replicate libcint `cint_bas.c::shells_cgto_offset` exactly: `ao_loc[0]=0`, then `ao_loc[i]=ao_loc[i-1]+count(i-1)` for `i in 1..nbas`. `ao_loc[nbas]` is never touched. Required buffer length is now `nbas`.
- All three `CINTshells_{cart,spheric,spinor}_offset` wrappers fixed by the single shared-core change.
- `cargo test -p cintx-compat --lib`: 40 passed, 0 failed.
- Ran the full vendor oracle gate (all 4 profiles). Confirmed the `CINTshells_*_offset` mismatch is GONE — the gate now reaches the NEXT helper-parity comparison for the first time on this branch.

## Task Commits

1. **Task 1: Update helper unit test to libcint i<nbas expectation (RED)** - `30d7f8c` (test)
2. **Task 2: Rewrite write_offsets to libcint i<nbas semantics (GREEN)** - `3bf0682` (fix)

_Code committed atomically (helpers.rs only). SUMMARY.md / STATE.md / PLAN.md NOT in code commits. ROADMAP.md untouched._

## Files Created/Modified

- `crates/cintx-compat/src/helpers.rs` — `write_offsets` rewritten to libcint i<nbas semantics (required length nbas, no ao_loc[nbas] write, nbas==0 sound, i32 overflow guarded via try_from+checked_add → ChunkPlanFailed); unit test renamed to `helper_offsets_match_libcint_i_lt_nbas` asserting `[0, 1, 0]` + `CINTtot_cgto_cart(&bas, 2) == 7`.

## Decisions Made

- Fix the HELPER, not the harness: `compare.rs` is correct (its len-5/nbas-4 zero-init buffer compares all entries including the trailing slot; matching libcint exactly is what makes that comparison pass). Decision made with the user.
- Accumulate in i32 to match libcint's FINT running sum byte-for-byte.

## Deviations from Plan

None - plan executed exactly as written (RED then GREEN, helpers.rs only, harness untouched).

## Verification — Full Vendor Oracle Gate (MANDATORY FINAL STEP)

Command run verbatim:

```
CINTX_BACKEND=cpu CINTX_ORACLE_BUILD_VENDOR=1 cargo run --locked --manifest-path xtask/Cargo.toml -- oracle-compare --profiles "base,with-f12,with-4c1e,with-f12+with-4c1e" --include-unstable-source false
```

### (a) CINTshells_*_offset mismatch is GONE — CONFIRMED

The previously-bailing `CINTshells_cart_offset[4] cintx=8 vendor=0` line does NOT appear anywhere in the gate output (grep for `CINTshells` returns only the unrelated `non_snake_case` lint warnings on the `vendor_CINTshells_*` FFI shim names, plus the `pub fn` definitions — no parity-mismatch line). The fix landed.

### (b) Gate reaches the NEXT helper-parity comparison for the FIRST time on this branch — and a DIFFERENT pre-existing downstream mismatch surfaces

The helper bail previously masked everything after the CINTshells check. With it removed, the gate now advances to the very next helper-parity comparison (`CINTgto_norm`) and surfaces a NEW, pre-existing, unrelated mismatch. This is a `CINTgto_norm` issue (a separate helper function), NOT a `write_offsets` regression, and it is explicitly OUT OF SCOPE for this task. Per the constraints I did NOT attempt to fix it. Reported VERBATIM for follow-up:

```
xtask gate failed: oracle parity gate failed for 4 profile(s): base: CINTgto_norm(0,0.5) mismatch: cintx=1.3313353638003897 vendor=1.502251088929885 diff=0.17091572512949527 | with-f12: CINTgto_norm(0,0.5) mismatch: cintx=1.3313353638003897 vendor=1.502251088929885 diff=0.17091572512949527 | with-4c1e: CINTgto_norm(0,0.5) mismatch: cintx=1.3313353638003897 vendor=1.502251088929885 diff=0.17091572512949527 | with-f12+with-4c1e: CINTgto_norm(0,0.5) mismatch: cintx=1.3313353638003897 vendor=1.502251088929885 diff=0.17091572512949527
```

Gate exit code: `1` (because the now-reachable `CINTgto_norm` helper mismatch fails the gate). This is honest: the gate is NOT a clean pass. The CINTshells fix succeeded, but it unmasked a separate pre-existing helper-parity defect that was previously hidden behind the CINTshells bail. The numeric INTEGRAL parity comparison is still NOT reached — the gate stops at the next helper-parity failure (`CINTgto_norm`) before integrals.

### Follow-up item (NOT fixed in this task)

- **`CINTgto_norm(0, 0.5)` helper-parity mismatch** vs libcint 6.1.3 across all 4 profiles: cintx=`1.3313353638003897`, vendor=`1.502251088929885`, diff=`0.17091572512949527`. Pre-existing, unrelated to `write_offsets`. The cintx `CINTgto_norm` (helpers.rs ~212-229) likely diverges from libcint `misc.c::CINTgto_norm` (normalization formula / double-factorial convention). Needs its own quick task. This was masked by the CINTshells bail before this fix and is now the first thing the gate trips on.

## Issues Encountered

- The vendor libcint build is slow (full clean compile of the workspace + vendored libcint); ran the gate in the background with a generous timeout and monitored to completion. No abandonment.

## Next Phase Readiness

- The `CINTshells_*_offset` helper-parity blocker is cleared; the oracle gate now progresses past it.
- A new, separate follow-up surfaces: `CINTgto_norm` helper-parity. Until that is fixed, the gate still fails before numeric integral parity. This task deliberately leaves it for a dedicated follow-up per the no-scope-creep constraint.

## Self-Check: PASSED

---
*Phase: quick-260529-lbr*
*Completed: 2026-05-29*
