---
phase: 27-spinor-derivative-transform-gap-b1
plan: 02a
subsystem: oracle
tags: [spinor, derivative, aux-k, parity, reconciliation, vendor-ffi]
requires:
  - "committed Plan-01 scaffolding (spinor_deriv_parity.rs collectors, fixtures.rs::ao_count_for_rep/dims_for_arity, vendor_ffi.rs 3c2e/3c1e spinor wrappers)"
  - "source-verified aux-k contract (27-SPIKE-FINDINGS CORRECTION NOTICE, cint3c2e.c:631-636)"
provides:
  - "arity-3 spinor aux-k axis sized SPHERICALLY (nsph(lk) = (2lk+1)*nctr_k) in fixtures.rs::dims_for_arity, the parity collectors, and the vendor_ffi.rs doc comments"
  - "runnable assertion pinning the corrected single-contraction p×d×s buffer at 360 (not 720) and the nctr=2 fixture aux-k count at 1 (spherical, not 2 spinor)"
affects:
  - "Plan 03 (3c2e_ip1/ip2 spinor launcher) and Plan 04 (3c1e_ip1/iprinv spinor sibling fold) — their parity tests now compare against correctly-sized collectors"
tech-stack:
  added: []
  patterns:
    - "arity-3 spinor: bra i / ket j spinor-sized (CINTcgto_spinor = 4l+2), aux-k spherical (nsph(lk)); the positional aux-k decision lives in dims_for_arity (knows arity + tail index), ao_count_for_rep stays the per-(shell,representation) primitive"
key-files:
  created:
    - .planning/phases/27-spinor-derivative-transform-gap-b1/27-02a-SUMMARY.md
  modified:
    - crates/cintx-oracle/src/fixtures.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/tests/spinor_deriv_parity.rs
decisions:
  - "The arity-3 spinor aux-k spherical correction belongs in dims_for_arity (positional: it knows arity and the tail index = arity-1), not in ao_count_for_rep which stays the per-(shell,representation) primitive."
metrics:
  duration: ~6m
  completed: 2026-05-31
---

# Phase 27 Plan 02a: Reconcile arity-3 spinor aux-k to SPHERICAL sizing Summary

Corrected three committed Plan-01 scaffolding files so the auxiliary-k axis of arity-3 spinor derivative families is sized SPHERICALLY (`nsph(lk) = (2lk+1)*nctr_k`) rather than the disproven spinor sizing — only bra i and ket j stay spinor-sized (`CINTcgto_spinor = 4l+2`), matching libcint `CINT3c2e_spinor_drv` is_ssc=0 (cint3c2e.c:631-636). The corrected single-contraction p×d×s kappa=0 buffer is now 360, not the over-sized 720.

## What Was Done

### Task 1 — fixtures.rs aux-k sizing + vendor_ffi.rs doc comments (commit be98bb6)
- `dims_for_arity` now sizes the arity-3 spinor tail shell (aux-k, axis == arity-1) with `CINTcgto_spheric` when `representation == Representation::Spinor && arity == 3`; bra i / ket j keep the `ao_count_for_rep` Spinor arm (`CINTcgto_spinor`). A code comment cites cint3c2e.c:631-636 and the 27-SPIKE-FINDINGS CORRECTION NOTICE. `ao_count_for_rep` itself is unchanged (still the per-(shell,representation) primitive) — the positional aux-k decision was made the consumer's responsibility.
- `vendor_ffi.rs`: the arity-3 block comment and the per-wrapper docs for `vendor_int3c2e_ip1_spinor`, `vendor_int3c1e_ip1_spinor`, `vendor_int3c1e_iprinv_spinor` were rewritten from "aux-k SPINOR-sized" to "aux-k SPHERICAL nsph(lk) = (2lk+1)*nctr_k; only bra i and ket j spinor-sized". No signature/code change — the wrappers take a caller-sized `out`; only the doc comments were wrong.
- Arity-2 spinor (1e/2c2e) and Cart/Spheric sizing untouched.

### Task 2 — spinor_deriv_parity.rs collectors + header + assertions (commit 8eae8a1)
- Added `shell_nsph_full(bas, s) = (2l+1)*nctr` next to `shell_nsp_full`.
- `collect_cintx_3c` and `collect_vendor_3c` now size `nk = shell_nsph_full(bas, SK)` (was `shell_nsp_full`); `ni`/`nj` stay on `shell_nsp_full` so the vendor and cintx buffers are sized identically.
- Header doc and the `collect_cintx_3c` doc rewritten to the spherical aux-k rule with the cint3c2e.c:631-636 citation.
- `test_fixture_builds_without_vendor`: SK assertion changed to `assert_eq!(shell_nsph_full(&bas, SK), 1, ...)`; added a runnable assertion pinning the canonical single-contraction buffer at 360 (`3*spinor_len_kappa0(1)*spinor_len_kappa0(2)*1*2`), and an assertion that the committed nctr=2 fixture's arity-3 buffer is `3*12*10*1*2` (spherical k=1, not spinor k=2).

## Verification

- `cargo build -p cintx-oracle --features cpu` — green.
- `cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_fixture_builds_without_vendor` — 1 passed (360 + spherical aux-k assertions hold).
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity --no-run` — exit 0 (corrected collectors compile against the real vendor link).
- Grep acceptance (all met): `CINTcgto_spheric` count in fixtures.rs = 3 (added the arity-3 path); `631-636|is_ssc` present; `CINTcgto_spinor` still present (bra/ket intact); zero stale "aux-k SPINOR-sized" docs in vendor_ffi.rs; 6 spherical aux-k doc mentions; `fn shell_nsph_full` = 1; `shell_nsph_full(bas, SK)` = 2; `shell_nsp_full(bas, SK)` = 0; `360` present; `shell_nsph_full(&bas, SK), 1` = 1.

## Deviations from Plan

None - plan executed exactly as written. The only minor judgment: the SK spherical assertion was kept single-line so it matches the plan's exact acceptance grep `shell_nsph_full(&bas, SK), 1`.

## Notes for Downstream Plans

- The collectors and fixture dims now emit the spherical aux-k (360 for single-contraction p×d×s, 720 for the nctr=2 fixture's full arity-3 buffer — half the disproven 1440). Plans 03/04 launchers must emit the same spherical aux-k so `count_mismatches` length-asserts pass.
- The inner transform `cart_to_spinor_sf_3c2e` already uses `nsk = nsph(lk)` (c2spinor.rs L1293/L1308) and was ALREADY correct — no change needed there.

## Self-Check: PASSED
