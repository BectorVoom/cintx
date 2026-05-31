---
phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
plan: 01
subsystem: cubecl-transform
tags: [spinor, c2s_si, cart2spinor, pauli-sigma, relativistic, kappa, a_bra_cart2spinor_si]

requires:
  - phase: 12-real-spinor-transform-c2spinor-replacement
    provides: single-block si transform (cart_to_spinor_si), CG coeff tables (c2spinor_coeffs), spinor_len GT/LT sizing
  - phase: 27-spinor-derivative-transform-gap-b1
    provides: cart_to_spinor_sf_2d structural template, KET→BRA in-transform transpose ownership (D-06)
provides:
  - "apply_bra_si_block: σ-coupled (si) bra step transcribing a_bra_cart2spinor_si signs verbatim (NOT apply_si_block)"
  - "cart_to_spinor_si_2d: host 2D spin-included cart→spinor transform (libcint c2s_si_1e) consuming 4 gc blocks, owning KET→BRA transpose, reusing ordinary ket, sizing via spinor_len, fail-closed"
affects: [28-02 (σ·p #[cube] assembler emits the 4 gc blocks), 28-03/28-04 (fixture + vendor parity), Phase 29 (Group-4 relativistic σ families reuse si_2d), Phase 30 (GIAO×σ), Phase 31 (gauge/Breit-Gaunt)]

tech-stack:
  added: []
  patterns:
    - "si bra step transcribes a_bra_cart2spinor_si signs (+ca_i*vz / -cb_r*vy / +cb_i*vx), distinct from apply_si_block's CINTc2s_ket_spinor_si1 convention"
    - "si_2d owns the KET→BRA transpose per gc block (Phase-27 D-06); ket reuses ordinary apply_ket_transform verbatim (c2s_si_1e ket == c2s_sf_1e ket)"
    - "all spinor buffer sizing via spinor_len; OOM-safe fail-closed guards before any write"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/transform/c2spinor.rs

key-decisions:
  - "Wrote apply_bra_si_block as dispatcher + per-block helper (apply_bra_si_block_one) mirroring apply_bra_sf_block_all_kappa/apply_bra_block, rather than extending apply_si_block — the 2D path requires the a_bra_cart2spinor_si sign convention, not apply_si_block's."
  - "Stage-0 KET→BRA transpose implemented as a local closure applied independently to all four gc blocks."

patterns-established:
  - "SI sign convention landmine: the 2D c2s_si_1e bra step uses a_bra_cart2spinor_si signs; apply_si_block (single-block helper) stays untouched and is NOT reused."
  - "si_2d structural clone of sf_2d with only the bra step swapped (~80% reuse)."

requirements-completed: [FND-05]

duration: 12min
completed: 2026-05-31
---

# Phase 28 Plan 01: Spin-Included si_2d Transform + σ-Coupled Bra Step Summary

**Host-side `cart_to_spinor_si_2d` (libcint `c2s_si_1e`) with a new `apply_bra_si_block` transcribing the `a_bra_cart2spinor_si` Pauli-σ bra convention, an ordinary reused ket, and `spinor_len`-driven fail-closed sizing.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-31
- **Completed:** 2026-05-31
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- `apply_bra_si_block` (+ per-block `apply_bra_si_block_one` helper): the σ-coupled bra step consuming four cart blocks (`gc_x/gc_y/gc_z/gc_1`), transcribing the `a_bra_cart2spinor_si` accumulation (`cart2sph.c:3958-3961`) verbatim — the correct signs (`+ca_i*vz`, `-cb_r*vy`, `+cb_i*vx`), NOT `apply_si_block`'s `CINTc2s_ket_spinor_si1` convention. GT/LT/both-kappa dispatch mirrors `apply_bra_sf_block_all_kappa`.
- `cart_to_spinor_si_2d`: host 2D si transform mirroring `cart_to_spinor_sf_2d`. Owns the KET→BRA transpose internally per gc block (Phase-27 D-06), routes the bra through `apply_bra_si_block`, reuses the ordinary `apply_ket_transform` verbatim, sizes every buffer via `spinor_len` (kappa≠0 → 2l / 2l+2, never 4l+2), and fail-closes (`ChunkPlanFailed` / `BufferTooSmall`) before any write with no partial writes.
- Unit-proven: hand-derived l=1/kappa=−1 bra-si value match to 1e-14 + sign-discrepancy guard against `apply_si_block`; kappa≠0 sizing, undersized-buffer fail-closed guards, and a non-square p×d round-trip.

## Task Commits

Each task was committed atomically:

1. **Task 1: apply_bra_si_block with a_bra_cart2spinor_si signs + unit test** — `1b40e92` (feat)
2. **Task 2: cart_to_spinor_si_2d host transform (4-block, ordinary ket, spinor_len)** — `4c278f7` (feat)

_Note: tasks carried `tdd="true"`; project `tdd_mode` is off, so each task landed as a single feat commit pairing the implementation with its unit tests (which pass, proving the contract)._

## Files Created/Modified
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — added `apply_bra_si_block` + `apply_bra_si_block_one` (si bra step) and `cart_to_spinor_si_2d` (host 2D si transform), plus four unit tests (hand-derived bra-si values + sign guard, kappa sizing, two fail-closed guards, non-square round-trip).

## Decisions Made
- Split `apply_bra_si_block` into a kappa dispatcher + `apply_bra_si_block_one` per-block helper to faithfully mirror the sf bra structure (`apply_bra_sf_block_all_kappa` + `apply_bra_block`) while swapping in the four-block si accumulation.
- KET→BRA transpose realized as a local closure applied to all four gc blocks independently, matching the in-transform 3c2e fix (`c2spinor.rs:1342-1354`).

## Deviations from Plan

None - plan executed exactly as written. The two TDD tasks were implemented with their unit tests included in the same commit (project tdd_mode is disabled), and all tests pass.

## Issues Encountered
None.

## TDD Gate Compliance
Tasks were marked `tdd="true"` but the project config has `tdd_mode: false`, so a separate RED test-only commit was not enforced. Each implementation commit includes passing unit tests that prove the behavioral contract (hand-derived value match + sign-discrepancy guard for Task 1; sizing + fail-closed + round-trip for Task 2).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The host si transform foundation (FND-05 transform half) is in place. Plan 28-02 can now build the σ·p device `#[cube]` assembler that emits the four `gc_x/gc_y/gc_z/gc_1` blocks `cart_to_spinor_si_2d` consumes, and Plans 28-03/28-04 can add the kappa-bearing fixture, vendor FFI (`vendor_int1e_sp_spinor`), and the transform-level byte-identity parity test.
- No σ family is flipped to `oracle_covered` (D-01 infrastructure-only is honored; this plan touches only the transform code).

## Self-Check: PASSED
- `crates/cintx-cubecl/src/transform/c2spinor.rs` — FOUND (`apply_bra_si_block`, `cart_to_spinor_si_2d` present)
- Commit `1b40e92` (Task 1) — FOUND
- Commit `4c278f7` (Task 2) — FOUND
- `cargo test -p cintx-cubecl --lib transform::c2spinor` — 42 passed, 0 failed
- `cargo build -p cintx-cubecl --locked` — succeeds

---
*Phase: 28-spin-included-c2s-si-transform-p-module-gap-b2*
*Completed: 2026-05-31*
