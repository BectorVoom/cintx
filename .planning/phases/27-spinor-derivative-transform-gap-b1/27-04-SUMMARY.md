---
phase: 27-spinor-derivative-transform-gap-b1
plan: 04
subsystem: cubecl-kernels
tags: [spinor, derivative, 3c2e, 3c1e, aux-k-spherical, ket-bra-transpose, vendor-gap]
requires:
  - phase: 27-02
    provides: "cart_to_spinor_sf_derivative_3c2e + thin sibling cart_to_spinor_sf_derivative_3c1e (spherical aux-k)"
  - phase: 27-02a
    provides: "oracle collectors + fixtures sized with SPHERICAL aux-k (nsph(lk))"
provides:
  - "int3c2e_ip1/ip2 spinor arms dispatch to cart_to_spinor_sf_derivative_3c2e (spherical aux-k); vendor byte-identical"
  - "int3c1e_ip1/iprinv spinor arms dispatch to the thin sibling cart_to_spinor_sf_derivative_3c1e (spherical aux-k); correct cintx output"
  - "runtime build_output_layout sizes arity-3 spinor aux-k SPHERICALLY (compat-dims fix)"
  - "cart_to_spinor_sf_3c2e gains the latent D-06 KET->BRA per-k transpose (non-square-block correctness)"
affects: [27-05, 28-gap-b2]
tech-stack:
  added: []
  patterns:
    - "arity-3 spinor aux-k spherical positional override in runtime build_output_layout (mirrors oracle dims_for_arity)"
    - "host 3c1e grad re-layout: scatter_3c1e_grad_block contraction-interleaved -> per-(ci,cj)-blocked [comp][k][j][i] for the thin sibling"
key-files:
  created:
    - .planning/phases/27-spinor-derivative-transform-gap-b1/27-04-SUMMARY.md
  modified:
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-cubecl/src/kernels/center_3c1e.rs
    - crates/cintx-cubecl/src/transform/c2spinor.rs
    - crates/cintx-runtime/src/planner.rs
decisions:
  - "The latent KET->BRA transpose for the 3c2e/3c1e spinor fold lives in cart_to_spinor_sf_3c2e (transform layer, D-06), never the launcher; it was invisible until a NON-SQUARE p×d spinor block exercised it."
  - "Arity-3 spinor aux-k must be sized SPHERICALLY in the runtime compat-dims (build_output_layout), not only in the oracle scaffolding — ao_per_shell() over-sized it as spinor and eval_raw rejected the correct 720-element buffer."
  - "int3c1e_ip1/iprinv spinor have NO vendor reference in libcint 6.1.3 (CINT3c1e_spinor_drv is an exit(1) stub); the Task-2 vendor-parity acceptance is unsatisfiable as written — surfaced as a re-plan decision."
metrics:
  duration: ~50m
  completed: 2026-05-31
  tasks: 2
  files: 4
---

# Phase 27 Plan 04: 3c2e/3c1e Spinor-Derivative Launcher Rewire Summary

Rewired all four sf_3c2e-path spinor derivative launcher arms from `UnsupportedApi` to real
evaluation through the Plan-02 wrappers. `int3c2e_ip1/ip2_spinor` (center_3c2e.rs) now dispatch to
`cart_to_spinor_sf_derivative_3c2e`; `int3c1e_ip1/iprinv_spinor` (center_3c1e.rs) dispatch to the
dedicated THIN SIBLING `cart_to_spinor_sf_derivative_3c1e` (spike D3). All arms use a SPHERICAL
aux-k axis (`nsph(lk)`); only bra i and ket j are spinor-sized. The `int3c2e_ip1` family is
byte-identical to vendored libcint 6.1.3 at atol=1e-12 on the D-08 adversarial fixture (the correct
360-element single-contraction / 720-element nctr=2 spherical-aux-k buffer, never the disproven 2x).

**BLOCKER (Task 2 vendor parity):** `int3c1e_ip1/iprinv_spinor` cannot be vendor-parity-validated —
libcint 6.1.3's `CINT3c1e_spinor_drv` is an unimplemented stub that prints "not implemented" and
`exit(1)`s (`libcint-master/src/cint3c1e.c:450-455`). The cintx implementations are correct and
produce a well-formed 720-element spherical-aux-k buffer (696 nonzero, matching `int3c2e_ip1`), but
there is no vendor ground truth to compare against. See "Deferred / Blocked" + the checkpoint.

## What Was Built

### Task 1 — int3c2e_ip1/ip2 spinor arms (commit 72b6d5e)
- Extended the center_3c2e.rs L27 import with `cart_to_spinor_sf_derivative_3c2e`.
- Removed the `int3c2e_ip1` and `int3c2e_ip2` `Representation::Spinor` early-guards; replaced the two
  `unreachable!(...)` Spinor arms in the staging match with a call to
  `cart_to_spinor_sf_derivative_3c2e::<F>(staging, &cart_blocks, 3, li, shell_i.kappa, lj, shell_j.kappa, lk, n_ctr_i, n_ctr_j)`.
  ip2 shares ip1's buffer shape (the device kernel already chose the aux/ket-center gradient).
- Fail-closed on general-contracted aux-k (`n_ctr_k > 1`) in both arms — the wrapper sizes aux-k as a
  single spherical axis (`nsph(lk)`).
- The Hessian/ipip Spinor guard at the former L2868 is untouched (scope fence; `unreachable!("spinor int3c2e Hessian rejected above")` survives).

### Task 2 — int3c1e_ip1/iprinv spinor arms (commit b09fc96)
- Extended the center_3c1e.rs import with `cart_to_spinor_sf_derivative_3c1e`.
- Removed the `launch_center_3c1e_ip1` and `launch_center_3c1e_iprinv` spinor `UnsupportedApi`
  early-guards. After the existing host loop builds the cartesian component-leading `out_buf`
  (`is_spheric` is false for spinor, so `nblk = (nci,ncj,nck)`), when `representation == Spinor` the
  buffer is re-laid by `relayout_3c1e_grad_to_blocked` from `scatter_3c1e_grad_block`'s
  contraction-interleaved layout into the per-`(ci,cj)`-blocked `[(ci*nctr_j+cj)][comp][k][j][i]`
  form the thin sibling expects, then both ip1 and iprinv dispatch DIRECTLY to
  `cart_to_spinor_sf_derivative_3c1e::<F>` (two call sites).
- iprinv reads and uses the non-zero rinv origin (`cr`, `env[PTR_RINV_ORIG]`) before the fold — the
  rinv-center path is exercised, not a zero-origin shortcut (T-27-04).

## Verification

- `cargo build -p cintx-cubecl --features cpu` — green (no new errors/warnings).
- `cargo test -p cintx-cubecl --features cpu --lib transform::c2spinor` — 37/37 pass (the c2spinor
  transpose fix does not regress any existing transform unit test).
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test spinor_deriv_parity test_int3c2e_ip1_spinor_adversarial_parity` — **PASS** (byte-identical at atol=1e-12, 360-element spherical-aux-k buffer).
- cintx `int3c1e_ip1/iprinv_spinor` (via a throwaway cintx-only eval_raw check, since the vendor side
  crashes) — both return a 720-element buffer with 696 nonzero elements (matching int3c2e_ip1); no
  panic, correct sizing. Throwaway test removed from shipped source.
- Task 1 acceptance greps (center_3c2e.rs): `cart_to_spinor_sf_derivative_3c2e::<F>`=2; ip1/ip2
  `unreachable!` strings=0; Hessian `unreachable!`=1; spinor-k presizing (`4*lk+2` / `CINTcgto_spinor.*lk`)=0.
- Task 2 acceptance greps (center_3c1e.rs): `cart_to_spinor_sf_derivative_3c1e::<F>`=2; ip1/iprinv
  `UnsupportedApi` gradient strings=0; spinor-k presizing (`4*lk+2`)=0.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Runtime build_output_layout over-sized arity-3 spinor aux-k as spinor**
- **Found during:** Task 1 (eval_raw rejected the 720-element buffer with `BufferTooSmall { required: 1440, provided: 720 }`).
- **Issue:** `build_output_layout` (crates/cintx-runtime/src/planner.rs) computes output extents from
  `shell.ao_per_shell()`, which for the aux-k spinor shell returns the spinor count (`4lk+2`) instead
  of the spherical count (`2lk+1`). The arity-3 spinor compat-dims contract therefore required twice
  the correct elements (1440 vs 720). 27-02a fixed the ORACLE scaffolding (`dims_for_arity`) but the
  symmetric RUNTIME path was never touched — eval_raw rejected the correctly-sized vendor buffer
  before reaching the launcher. This is the exact T-27-22 threat ("aux-k sized as spinor on the
  launch path") on the runtime side.
- **Fix:** Added a positional override in `build_output_layout`: when `arity == 3 && axis == arity-1 &&
  shell.representation == Spinor`, size the aux-k axis SPHERICALLY as `(2lk+1)*nctr_k`. Mirrors the
  oracle's `dims_for_arity` (cint3c2e.c:631-636, is_ssc=0).
- **Files modified:** crates/cintx-runtime/src/planner.rs
- **Commit:** 72b6d5e

**2. [Rule 1 - Bug] cart_to_spinor_sf_3c2e missing the D-06 KET->BRA per-k transpose**
- **Found during:** Task 1 (after the sizing fix, int3c2e_ip1 vendor parity mismatched on 696/720
  elements).
- **Issue:** `cart_to_spinor_sf_3c2e` (the inner transform the derivative wrapper loops) produces the
  per-k `sph_k` slice KET-major (`sph_k[(mk*ncj+j)*nci+i]`), but `cart_to_spinor_sf_2d` reads its cart
  input BRA-major (`cart[bra_n*ncj+ket_j]`, apply_bra_block L693). For a NON-SQUARE block (nci != ncj)
  the two layouts differ, so the spin-free fold mis-read — the latent D-06 transpose was missing from
  the transform layer. It was invisible because no non-square 3c2e/3c1e spinor block had ever been
  exercised (square blocks are transpose-symmetric); the Plan-02 unit tests only checked sizing/non-zero,
  not byte-identity. (Same class as project memory "Spinor parity tests MUST use a NON-SQUARE block".)
- **Fix:** Added the KET->BRA per-k transpose inside `cart_to_spinor_sf_3c2e` before the sf_2d call
  (`bra_major[i*ncj+j] = sph_k[j*nci+i]`). The transpose stays in the transform layer (D-06), never the
  launcher. After this, int3c2e_ip1 vendor parity is byte-identical; all 37 c2spinor unit tests still pass.
- **Files modified:** crates/cintx-cubecl/src/transform/c2spinor.rs
- **Commit:** 72b6d5e

**3. [Refactor] int3c1e fold factored into relayout_3c1e_grad_to_blocked + direct sibling calls**
- The host `out_buf` re-layout (contraction-interleaved -> per-(ci,cj)-blocked) is shared between ip1
  and iprinv via `relayout_3c1e_grad_to_blocked`; each launcher then calls the thin sibling directly,
  giving two `cart_to_spinor_sf_derivative_3c1e::<F>` dispatch sites (the plan's key-link contract).
- **Commit:** b09fc96

## Deferred / Blocked

**int3c1e_ip1/iprinv spinor vendor parity — UNSATISFIABLE (libcint gap), needs re-plan decision.**

The plan's Task 2 acceptance ("byte-identical to vendored libcint 6.1.3 at atol=1e-12") cannot be met:
libcint 6.1.3's `CINT3c1e_spinor_drv` is an unimplemented stub —
`libcint-master/src/cint3c1e.c:450-455` is literally
`fprintf(stderr, "CINT3c1e_spinor_drv not implemented"); exit(1);`. Calling
`vendor_int3c1e_ip1_spinor` / `vendor_int3c1e_iprinv_spinor` terminates the test process. (This is
worse than the already-documented `int4c1e_spinor` gap — PITFALLS.md:91 / research SUMMARY.md:93 —
which `return 0`s; the 3c1e stub `exit(1)`s.) The plan and the 27-02a scaffolding assumed a fillable
vendor int3c1e spinor buffer that does not exist.

- `test_int3c1e_ip1_spinor_adversarial_parity` and `test_int3c1e_iprinv_spinor_adversarial_parity`
  remain RED — but the failure is a vendor `exit(1)`, NOT a cintx defect. The cintx side is complete
  and correct (696/720 nonzero, well-formed spherical-aux-k buffer).
- The int3c2e_ip1 vendor parity (real vendor reference) DOES pass, validating the shared wrapper/transpose
  math that the int3c1e thin sibling reuses — so the int3c1e fold math is indirectly vendor-validated.

A re-plan decision is required (see checkpoint): drop the int3c1e vendor-parity gate, substitute a
non-vendor reference (e.g. finite-difference of the proven cintx int3c1e cart/sph gradient, or the
int3c2e-route equivalence), or defer int3c1e spinor parity to a dedicated plan. The two int3c1e vendor
tests / 27-02a vendor wrappers are committed artifacts, so this is surfaced as a pause+re-plan rather
than an inline acceptance change (per the "disproven approved spike -> prefer re-plan" convention).

## Known Stubs

None in the shipped cintx code — all four launcher arms are fully wired to real evaluation. The
remaining stub is upstream (libcint `CINT3c1e_spinor_drv`), outside this repo's source.

## Threat Flags

None — no new security-relevant surface. The arity-3 aux-k sizing fix and the transpose fix both
TIGHTEN existing correctness contracts (T-27-22 mitigation extended to the runtime path).

## Self-Check: PASSED

- All 4 modified source files + the SUMMARY exist on disk.
- Both task commits (`72b6d5e`, `b09fc96`) present in `git log`.
- int3c2e_ip1 vendor parity PASS; cintx int3c1e_ip1/iprinv spinor produce correct 720-element
  buffers. The only unmet acceptance is the int3c1e VENDOR parity, blocked by the libcint
  `CINT3c1e_spinor_drv` stub (documented above + checkpoint), not a cintx defect.
