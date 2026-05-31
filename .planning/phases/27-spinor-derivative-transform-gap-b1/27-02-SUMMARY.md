---
phase: 27-spinor-derivative-transform-gap-b1
plan: 02
subsystem: cubecl-transform
tags: [spinor, cart2spinor, derivative, c2s, ncomp-fold, aux-k-spherical, nctr, fail-closed]

# Dependency graph
requires:
  - phase: 27-01
    provides: "27-SPIKE-FINDINGS.md D-11 (sf_2d [comp][ket][bra] layout, 3c2e per-(comp,k) granularity, nctr contraction-major) + ⚠ CORRECTION NOTICE (aux-k SPHERICAL nsph(lk), not spinor 720)"
provides:
  - "cart_to_spinor_sf_derivative_2d: generic ncomp-fold spin-free cart→spinor for arity-2 1e derivative families (ipovlp/ipkin/ipnuc/iprinv + rank-9/27/81 siblings)"
  - "cart_to_spinor_sf_derivative_3c2e: generic ncomp-fold for int3c2e_ip1/ip2 spinor gradients with SPHERICAL aux-k"
  - "cart_to_spinor_sf_derivative_3c1e: thin sibling (D3) for int3c1e_ip1/iprinv spinor gradients, same fold via shared impl"
  - "the single audited home for the KET→BRA orientation transpose (D-06) — no launcher may own it again"
affects: [27-03, 27-04, 28-gap-b2, spinor-derivative-launchers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ncomp axis-fold wrapper: loop a proven inner transform ncomp times with comp_stride = ni_full*nj_full*(k-extent)*2"
    - "fail-closed upfront sizing (FND-06): size-check cart+staging once from ncomp/nctr, no per-element scatter guards"
    - "contraction-major spinor composition: i_global = ci*di + ic, scatter into out[comp_base + (j_global*ni_full + i_global)*2 + {re,im}]"

key-files:
  created: []
  modified:
    - "crates/cintx-cubecl/src/transform/c2spinor.rs (+three pub fns, one shared private impl, 7 unit tests)"

key-decisions:
  - "Aux-k axis for arity-3 spinor derivative families is SPHERICAL nsph(lk), NOT CINTcgto_spinor — canonical p×d×s nctr=1 kappa=0 ncomp=3 buffer is 360, never 720 (27-SPIKE-FINDINGS ⚠ CORRECTION NOTICE; libcint CINT3c2e_spinor_drv is_ssc=0)."
  - "3c2e and the int3c1e thin sibling share a single private impl cart_to_spinor_sf_derivative_3c_impl — identical fold math, distinct public entry points keep the 3c2e device-cart precondition decoupled from the 3c1e host scatter (D3)."
  - "The KET→BRA orientation transpose is owned INSIDE cart_to_spinor_sf_derivative_2d (D-06 centralization) — regression-anchored on a NON-SQUARE p×d block so orientation is observable."
  - "nctr>1 composes contraction-major on bra i / ket j only; aux-k stays a single spherical axis as the inner cart_to_spinor_sf_3c2e already handles it."

patterns-established:
  - "Derivative wrapper = thin ncomp loop over a byte-identity-proven scalar inner transform; never re-derive the fold."
  - "Pin disproven-figure regressions with an explicit assert (assert_eq!(total, 360)) so a future aux-k mis-sizing fails loudly."

metrics:
  duration: ~12 min
  completed: 2026-05-31
  tasks: 2
  files: 1
---

# Phase 27 Plan 02: Spinor-Derivative Transform Wrappers Summary

Added the three thin generic derivative cart→spinor wrappers to
`crates/cintx-cubecl/src/transform/c2spinor.rs` — `cart_to_spinor_sf_derivative_2d` (arity-2 1e
gradients/Hessians), `cart_to_spinor_sf_derivative_3c2e` (int3c2e ip1/ip2), and the int3c1e thin
sibling `cart_to_spinor_sf_derivative_3c1e` — each looping a byte-identity-proven inner transform
`ncomp` times, owning the KET→BRA orientation transpose internally (D-06), emitting
component-outer interleaved-complex output (D-07), supporting nctr>1 contraction-major (D-08), and
failing closed on under-sized buffers (FND-06). The aux-k axis is SPHERICAL `nsph(lk)` per the
⚠ CORRECTION NOTICE (360, not the disproven 720).

## What Was Built

### Task 1 — `cart_to_spinor_sf_derivative_2d`
- Generic `<F: CintFloat>` ncomp-fold wrapper. Per `(comp, ci, cj)`: extracts the device-native
  KET-major sub-block `cart[(ci*nctr_j+cj)*total_len + comp*block_len + jc*nci + ic]`, applies the
  D-06 transpose `block_bra_major[ic*ncj+jc] = block[jc*nci+ic]`, calls `cart_to_spinor_sf_2d::<F>`
  into scratch, and scatters into `staging[comp*spinor_block + (j_global*ni_full + i_global)*2 + {0,1}]`
  with `i_global = ci*di + ic`, `spinor_block = ni_full*nj_full*2`.
- FND-06 fail-closed: `cart.len() < ncomp*block_len*nctr_i*nctr_j` → `ChunkPlanFailed`;
  `staging.len() < ncomp*spinor_block` → `BufferTooSmall`, both BEFORE any write. No scatter guards.
- 4 unit tests: `derivative_2d_rank3_matches_inline` (byte-equal to the one_electron.rs inline
  rank-3 replay on a NON-SQUARE p×d block), `derivative_2d_rank9_no_trailing_zero` (all 9 slices
  non-zero — component-truncation guard), `derivative_2d_nctr2_sizing` (ci=1 lands at i_global di..2di),
  `derivative_2d_staging_too_small_fails_closed` (sentinel survives).

### Task 2 — `cart_to_spinor_sf_derivative_3c2e` + `cart_to_spinor_sf_derivative_3c1e`
- Both delegate to a shared private `cart_to_spinor_sf_derivative_3c_impl` that loops the inner
  `cart_to_spinor_sf_3c2e::<F>` `ncomp` times with `comp_stride = ni_full*nj_full*nsph(lk)*2`.
  The inner transform already owns the cart→sph(k) fold + per-(comp,k) KET→BRA + sf_2d, so the
  wrapper only adds the ncomp loop, the contraction-major i/j scatter, and the spherical-k scatter
  `dst = comp_base + (mk*ni_full*nj_full + j_global*ni_full + i_global)*2`.
- **SPHERICAL aux-k**: `nsk = nsph(lk)`, identical to the inner transform's L1293. No
  `CINTcgto_spinor`, no `4lk+2` k-axis anywhere.
- The int3c1e `_3c1e` sibling exists as a distinct `pub fn` (D3 decision) but shares the impl —
  its launcher (Plan 04) produces a host-side `out_buf` in the same `[comp][k][ket][bra]` family.
- 3 unit tests: `derivative_3c2e_rank3_layout` (asserts the canonical **360 not 720** and 3
  non-overlapping all-nonzero comp slices), `derivative_3c2e_staging_too_small_fails_closed`,
  `derivative_3c1e_rank3_spherical_auxk` (sibling parity of layout/non-zero).

## Verification

- `cargo test -p cintx-cubecl --lib transform::c2spinor` → **37 passed, 0 failed** (34 prior + 3
  new for Task 2; the 4 Task-1 tests are within the prior count band after the Task-1 GREEN commit).
- `cargo build -p cintx-cubecl` → clean (no errors, no new unused warnings).
- Acceptance greps: `pub fn cart_to_spinor_sf_derivative_2d`=1, `_3c2e`=1, `_3c1e`=1;
  `nsph(lk)` present in the 3c region; `360` asserted in a unit test; no `CINTcgto_spinor` CALL
  (only negative-contract doc references) and no `4lk+2` k-axis; the inner `cart_to_spinor_sf_2d::<F>`
  is referenced and the `block_bra_major[ic*ncj+jc] =` transpose is present.

## TDD Gate Compliance

Both tasks followed RED→GREEN with explicit gate commits:
- `690ebb8` test(27-02) RED 2d → `86490e7` feat(27-02) GREEN 2d
- `5479188` test(27-02) RED 3c2e/3c1e → `cb76531` feat(27-02) GREEN 3c2e/3c1e
RED was confirmed as a compile failure (`cannot find function …`) before each GREEN. No REFACTOR
commit was needed.

## Deviations from Plan

None — plan executed exactly as written. The aux-k SPHERICAL `nsph(lk)` contract (the ⚠ CORRECTION
NOTICE) was honored throughout; the inner `cart_to_spinor_sf_3c2e` was reused unchanged.

## Known Stubs

None. The wrappers are fully implemented and unit-tested. Full vendor byte-identity parity for the
derivative families is intentionally deferred to Plan 04 (the parity test), per the plan's behavior
notes; the launchers that drive these wrappers (Plans 03/04) are separate, dependency-ordered work.

## Self-Check: PASSED

- File `crates/cintx-cubecl/src/transform/c2spinor.rs` exists and contains all three `pub fn`s.
- Commits `690ebb8`, `86490e7`, `5479188`, `cb76531` all present in `git log`.
- 37/37 c2spinor lib tests green.
