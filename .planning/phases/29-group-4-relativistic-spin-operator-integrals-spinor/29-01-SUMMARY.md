---
phase: 29-group-4-relativistic-spin-operator-integrals-spinor
plan: 01
subsystem: testing
tags: [spinor, c2spinor, si_2di, manifest, vendor-ffi, libcint, relativistic, sigma]

# Dependency graph
requires:
  - phase: 28-gap-b2-c2s-si-sigma-p
    provides: cart_to_spinor_si_2d + cart_to_spinor_sf_2d 1e transforms, σ·p assembler, build_kappa_spinor_fixture, vendor_int1e_sp_spinor
  - phase: 12-real-spinor-transform
    provides: cart_to_spinor_iket_si single-block imaginary-ket coupling, spinor_len, interleaved-complex layout
provides:
  - cart_to_spinor_si_2di 1e imaginary-ket bra-σ-mix transform (c2s_si_1ei host analog) for int1e_sr/int1e_sigma
  - 7 1e Group-4 spinor manifest rows (spsp/spnucsp/sprinvsp/srsr/srnucsr/sr/sigma), spinor-only, component_rank=1, oracle_covered=false
  - 7 vendor_int1e_*_spinor FFI shims + bindgen allowlist entries
  - rel_1e_sigma_parity.rs RED parity scaffold (vendor collectors live, cintx arms deferred to 29-02)
affects: [29-02, 29-group-4-wave-2, 29-group-4-wave-3, 30-giao-sigma]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Imaginary-ket transform = ordinary ket then (re,im)->(-im,re) at the zcopy boundary"
    - "RED parity scaffold: real vendor collectors + #[ignore] byte-identity gates + always-on no-silent-skip and row-registration assertions"

key-files:
  created:
    - crates/cintx-oracle/tests/rel_1e_sigma_parity.rs
  modified:
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "si_2di Stage-2 = ordinary apply_ket_transform then i-rotation at the interleaved zcopy (staging[..*2]=-out_i, +1=out_r), matching apply_iket_si_block vs apply_si_block"
  - "All 7 rows component_rank=1 (σ-fold is internal to c2s, not an output axis); sigma confirmed rank-1 per RESEARCH Open Q1 resolution (single CINT1e_spinor_drv call), 29-02 T1 re-checks empirically"
  - "Manifest edited in the lock only; build.rs regenerates api_manifest.rs/.csv from the lock (auto-syncs both audit sides)"
  - "Rows appended at array tail so no positional OperatorId const drift (low ids 0/2/15-23 untouched)"
  - "Byte-identity gates #[ignore] (RED) until 29-02 wires launcher arms; vendor collectors and no-silent-skip/sizing assertions stay live"

patterns-established:
  - "Pattern 1: imaginary-ket 1e transform clones the real si_2d verbatim and i-rotates only at the output store"
  - "Pattern 2: Wave-1 parity scaffold links vendor shims first, stubs cintx collectors with TODO(next-plan), keeps verification-integrity asserts always-on"

requirements-completed: [REL-01, REL-02]

# Metrics
duration: 14min
completed: 2026-05-31
---

# Phase 29 Plan 01: Wave-1 1e Relativistic σ Foundation Summary

**Added the one structurally-new 1e transform `cart_to_spinor_si_2di` (imaginary-ket bra-σ-mix, the c2s_si_1ei host analog), registered all 7 1e Group-4 spinor manifest rows (component_rank=1, oracle_covered=false), added 7 vendor_int1e_*_spinor FFI shims + allowlist entries, and laid down the rel_1e_sigma_parity.rs RED scaffold for 29-02 to wire.**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-05-31T13:56Z
- **Completed:** 2026-05-31
- **Tasks:** 3
- **Files modified:** 6 (1 created, 5 modified)

## Accomplishments
- `cart_to_spinor_si_2di` in c2spinor.rs: clone of `cart_to_spinor_si_2d` with the ONLY change being the Stage-2 ket transform → imaginary-ket (multiply-by-i `(re,im)→(-im,re)` at the interleaved zcopy). Owns its KET→BRA transpose; all sizing via `spinor_len` (no hardcoded 4l+2). Consumed by `int1e_sr`/`int1e_sigma` in 29-02.
- 7 1e Group-4 manifest rows registered: `int1e_{spsp,spnucsp,sprinvsp,srsr,srnucsr,sr,sigma}_spinor`, arity 2, canonical_family 1e, complex_output, component_rank "1", forms ["spinor"], spinor-only RepresentationSupport(false,false,true), oracle_covered false. Existing `int1e_sp_spinor` row stays rank "1". No OperatorId positional drift (workspace build green).
- 7 `vendor_int1e_*_spinor` FFI shims (verbatim clones of `vendor_int1e_sp_spinor`) + all 7 symbols appended to the build.rs bindgen `allowlist_function` regex (intor3.c already built — no `.file()` change for 1e).
- `rel_1e_sigma_parity.rs` RED scaffold: real per-family vendor collectors, RED cintx collector stub with `TODO(29-02)`, per-family byte-identity gates `#[ignore]`-deferred, plus always-on `test_kappa_sizing_non_4l_plus_2`, `test_no_silent_skip`, and a non-vendor row-registration smoke test.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add cart_to_spinor_si_2di imaginary-ket 1e transform** - `9d38f67` (feat)
2. **Task 2: Register 1e Group-4 manifest rows (rank=1, oracle_covered=false)** - `62761ff` (feat)
3. **Task 3: vendor_int1e_*_spinor shims + allowlist + rel_1e parity scaffold** - `bd06ada` (test)

## Files Created/Modified
- `crates/cintx-cubecl/src/transform/c2spinor.rs` - new `cart_to_spinor_si_2di` (imaginary-ket si_2d)
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - 7 new 1e Group-4 spinor rows (source of truth)
- `crates/cintx-ops/src/generated/api_manifest.rs` / `.csv` - regenerated from the lock by build.rs (348→355 entries)
- `crates/cintx-oracle/src/vendor_ffi.rs` - 7 `vendor_int1e_*_spinor` FFI shims
- `crates/cintx-oracle/build.rs` - 7 symbols appended to the bindgen allowlist
- `crates/cintx-oracle/tests/rel_1e_sigma_parity.rs` (NEW) - Wave-1 RED parity scaffold

## Decisions Made
- **si_2di i-rotation at the store, not in a new ket helper:** the imaginary-ket variant is exactly multiply-by-i of the ordinary ket output (verified: `apply_iket_si_block` accumulators are the i-rotation of `apply_si_block`), so the ordinary `apply_ket_transform` is reused and the rotation applied at the Stage-3 zcopy. Minimal divergence from the proven `si_2d` body.
- **component_rank "1" for all 7** including `sigma`: the σ_x/σ_y/σ_z fold is internal to the c2s transform, not an output component axis (RESEARCH Open Q1 RESOLVED — single `CINT1e_spinor_drv(...&c2s_si_1ei...)` call → one fused complex matrix). 29-02 T1 still empirically measures vendor output length vs di*dj*2 as belt-and-suspenders.
- **Lock-only manifest edit:** cintx-ops `build.rs` regenerates `api_manifest.rs`/`.csv` from the lock on build, so only the lock was hand-edited; both manifest-audit sides auto-sync.

## Deviations from Plan

None - plan executed exactly as written. All three tasks' actions, acceptance criteria, and verification commands matched the plan; no Rule 1-4 deviations were needed.

## Issues Encountered
None. The build.rs codegen path meant Task 2 required no hand-editing of the 355-entry generated Rust arrays (regenerated automatically), avoiding the OperatorId positional-shift landmine entirely (new rows append at the tail; low-index hardcoded test consts untouched; workspace build green).

## Known Stubs
- `collect_cintx_rel_1e` in `rel_1e_sigma_parity.rs` is an intentional RED stub (`unimplemented!` with `TODO(29-02)`). Per the plan (D-04 Wave-1 gating), the cintx launcher arms are 29-02 work; the per-family byte-identity gates are `#[ignore]`-deferred until then. This is the planned RED state, not an incomplete deliverable — the transform, manifest, vendor, and test contract are all in place for 29-02 to flip green.

## Next Phase Readiness
- 29-02 can now wire the 7 `one_electron.rs` Spinor launcher arms onto the existing transforms (`spsp`→sf_2d, `spnucsp/sprinvsp/srsr/srnucsr`→si_2d, `sr/sigma`→the new si_2di), replace the `collect_cintx_rel_1e` stub, remove the `#[ignore]` markers, and flip `oracle_covered=true` per family after each byte-identity gate passes under the double gate (`--features cpu` + `CINTX_ORACLE_BUILD_VENDOR=1`).
- No blockers. Vendor shims link, manifest rows registered, scaffold compiles and the always-on assertions pass.

## Self-Check: PASSED

All created/modified files exist on disk; all 4 task/doc commits (9d38f67, 62761ff, bd06ada, 42e2f3a) present in git history.

---
*Phase: 29-group-4-relativistic-spin-operator-integrals-spinor*
*Completed: 2026-05-31*
