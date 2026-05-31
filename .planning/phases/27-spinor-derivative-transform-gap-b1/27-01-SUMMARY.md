---
phase: 27-spinor-derivative-transform-gap-b1
plan: 01
subsystem: testing
tags: [oracle, vendor-ffi, spinor, derivative, parity, bindgen, libcint, fixtures]

# Dependency graph
requires:
  - phase: 25-group-2-hessian-higher-order-derivs
    provides: nctr>1 contraction-major composition + non-zero rinv-origin landmine patterns reused by the D-08 fixture
  - phase: 26 (1e spinor gradient parity)
    provides: the four rank-3 1e ip-spinor vendor wrappers + cart_to_spinor_sf_2d transform mirrored here
provides:
  - D-08 adversarial spinor-derivative fixture (non-square p×d + nctr=2 + kappa=0 + non-zero rinv origin + aux-k shell)
  - 6 new spinor-derivative vendor FFI wrappers (ipovlpip/ipipipiprinv/int2c2e_ip1/int3c2e_ip1/int3c1e_ip1/int3c1e_iprinv) with aux-k axis sized via CINTcgto_spinor
  - RED Nyquist parity test file spinor_deriv_parity.rs (7 parity + orientation negative control + no-silent-skip)
  - 27-SPIKE-FINDINGS.md (D-11 empirical layout/granularity/nctr decisions + int3c1e launcher file path) [Task 1]
affects: [27-02, 27-03, 27-04, 27-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single adversarial fixture trips all four silent-false-pass landmines at once (orientation, gc-transpose, both-block sizing, non-zero rinv origin)"
    - "Aux-k axis of arity-3 spinor derivative families is SPINOR-sized (CINTcgto_spinor=4l+2), never nsph(lk)"
    - "RED Nyquist scaffold: parity tests compile (--no-run gate) and are intentionally RED until launchers land downstream"

key-files:
  created:
    - crates/cintx-oracle/tests/spinor_deriv_parity.rs
    - .planning/phases/27-spinor-derivative-transform-gap-b1/27-SPIKE-FINDINGS.md
  modified:
    - crates/cintx-oracle/src/fixtures.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/src/compare.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "D-11 spike (Task 1) empirically pinned sf_2d device derivative layout [comp][ket][bra], 3c2e per-(comp,k) [ket][bra] granularity (ip1=ip2 shape), int3c1e THIN SIBLING decision + launcher file center_3c1e.rs, and contraction-major nctr composition — all vendor-evidenced on a non-square block"
  - "Aux-k axis of int3c2e/int3c1e spinor derivative families sized via vendor_CINTcgto_spinor (4l+2=2 at kappa=0), NOT nsph(lk); an nsph-sized buffer trips BufferTooSmall"
  - "Parity tests are intentionally RED (cintx returns UnsupportedApi for non-rank-3 and int3c1e spinor families); only the --no-run compile gate is satisfied by this plan"

patterns-established:
  - "Adversarial-fixture-per-phase: one builder defeats orientation + gc + block-sizing + rinv-origin landmines simultaneously"
  - "Orientation negative control (to_j_fastest) MUST diverge on the non-square block to prove i-fastest orientation"

requirements-completed: [FND-04]

# Metrics
duration: ~8min
completed: 2026-05-31
---

# Phase 27 Plan 01: D-11 spike + RED spinor-derivative parity scaffold Summary

**D-08 adversarial spinor fixture + 6 new spinor-derivative vendor FFI wrappers (aux-k spinor-sized) + a RED Nyquist parity test file that compiles, handed to Plans 03/04/05 — atop the Task 1 D-11 spike that empirically pinned the [comp][ket][bra] layout, 3c2e granularity, int3c1e sibling decision, and nctr composition.**

## Performance

- **Duration:** ~8 min (Task 2 continuation; Task 1 spike committed earlier as bfb2048)
- **Completed:** 2026-05-31
- **Tasks:** 2 (Task 1 spike approved + committed prior; Task 2 executed here)
- **Files modified:** 5 (4 modified + 1 created in Task 2; 1 created in Task 1)

## Accomplishments

- **Task 1 (D-11 spike, approved):** `27-SPIKE-FINDINGS.md` records the vendor-evidenced device derivative cart layout `[comp][ket][bra]` (comp_stride = nci*ncj), the 3c2e per-(comp,k) `[ket][bra]` transpose granularity (ip1 and ip2 share the buffer shape), the int3c1e_ip1/iprinv THIN-SIBLING decision + launcher file path (`crates/cintx-cubecl/src/kernels/center_3c1e.rs`, reject sites L1006-1010 / L1130-1134), and the contraction-major nctr composition (`i_global = ci*di + ic`, COLUMN→ROW coeff transpose). Also surfaced the critical aux-k spinor-sizing caveat (CINTcgto_spinor, not nsph).
- **Task 2 (this continuation):**
  - `build_adversarial_spinor_fixture()` added to `fixtures.rs` — non-square p×d, nctr=2 on the bra (column-major env coeff), kappa=0 on every bas row, non-zero `env[PTR_RINV_ORIG..+3]`, plus an aux-k (s) shell for the arity-3 families.
  - 6 new spinor-derivative vendor FFI wrappers in `vendor_ffi.rs` (rank-9 ipovlpip, rank-81 ipipipiprinv, int2c2e_ip1, int3c2e_ip1, int3c1e_ip1, int3c1e_iprinv); 3c2e/3c1e out buffers sized with the aux-k axis spinor-sized.
  - 6 new bindgen allowlist entries in `build.rs` so the `ffi::int*_spinor` extern decls bind.
  - 5 new symbol→RawApiId rows in `compare.rs` (atol/rtol unchanged at 1e-12).
  - `tests/spinor_deriv_parity.rs` created: 7 parity tests + orientation negative control + no-silent-skip + a non-vendor smoke test; file-gated `#![cfg(any(feature = "cpu", feature = "rocm"))]`.

## Task Commits

1. **Task 1: D-11 design spike** - `bfb2048` (docs) — committed prior to this continuation; checkpoint approved by human ("approved").
2. **Task 2: D-08 fixture + 6 vendor FFI wrappers + RED parity test** - `96e3a87` (test)

## Files Created/Modified

- `crates/cintx-oracle/tests/spinor_deriv_parity.rs` (created) - RED Nyquist parity scaffold: 9 fn bodies (7 parity + orientation negative control + no-silent-skip) over the D-08 adversarial fixture, plus a non-vendor smoke test.
- `crates/cintx-oracle/src/fixtures.rs` (modified) - `build_adversarial_spinor_fixture()` (non-square p×d + nctr=2 + kappa=0 + non-zero rinv origin + aux-k shell).
- `crates/cintx-oracle/src/vendor_ffi.rs` (modified) - 6 new spinor-derivative wrappers; aux-k axis sized via `vendor_CINTcgto_spinor`.
- `crates/cintx-oracle/src/compare.rs` (modified) - symbol→RawApiId rows for the 5 new spinor derivative families.
- `crates/cintx-oracle/build.rs` (modified) - bindgen allowlist for the 6 new `int*_spinor` symbols.

## Decisions Made

- None beyond the plan. The aux-k spinor-sizing (CINTcgto_spinor, not nsph) and the int3c1e sibling-vs-shared decision were resolved by the Task 1 spike and faithfully applied to the wrapper buffer sizing and the test collectors in Task 2.

## Deviations from Plan

None - plan executed exactly as written. The Task 2 deliverables (fixture, wrappers, build.rs allowlist, compare.rs map, test file) were authored to match the plan's `<behavior>`/`<action>` and the approved 27-SPIKE-FINDINGS decisions, then committed atomically. The required RED state (vendor parity tests fail because cintx returns `UnsupportedApi` for non-rank-3 and int3c1e spinor families) is intended and handed to Plans 03/04/05; only the `--no-run` compile gate is satisfied here.

## Issues Encountered

None. The `ffi::int*_spinor` extern decls for the 6 new symbols are bindgen-generated from the `build.rs` allowlist additions; the entire `vendor_ffi.rs` module is gated behind `#![cfg(has_vendor_libcint)]`, so the wrappers are correctly double-gated. The compile gate built the vendored libcint and produced the test executable with no errors (only pre-existing snake-case lint warnings on `CINT*` helper wrappers).

## Verification

- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity --no-run` → exit 0 (test binary COMPILES).
- `grep -c 'fn build_adversarial_spinor_fixture' fixtures.rs` → 1
- `grep -c` the 6 wrapper signatures in `vendor_ffi.rs` → 6
- `grep -c` the 9 test fn names in `spinor_deriv_parity.rs` → 9
- `grep -c 'KAPPA_OF' fixtures.rs` → 5 (fixture sets kappa=0 on every row)
- file gate `#![cfg(any(feature = "cpu", feature = "rocm"))]` present at line 40 (first non-comment line)

## Next Phase Readiness

- Plan 02 finalizes the `cart_to_spinor_sf_derivative_2d` / `_3c2e` wrapper signatures (spike-pinned).
- Plan 03 wires the 1e rank-9/81 + int3c2e_ip1 + int2c2e_ip1 spinor launchers (turns those parity tests GREEN).
- Plan 04 owns `center_3c1e.rs` (int3c1e_ip1/iprinv THIN SIBLING fold; iprinv tests require the non-zero rinv origin the fixture sets).
- Plan 05 flips the manifest lock and wires the `oracle_covered=true` half of `test_no_silent_skip`.

## Self-Check: PASSED

- Files exist: `spinor_deriv_parity.rs`, `27-01-SUMMARY.md`, `27-SPIKE-FINDINGS.md`
- Commits exist: `bfb2048` (Task 1 spike), `96e3a87` (Task 2), `5ad6f54` (plan metadata)
- No STATE.md/ROADMAP.md modifications in this executor's commits (orchestrator owns those)

---
*Phase: 27-spinor-derivative-transform-gap-b1*
*Completed: 2026-05-31*
