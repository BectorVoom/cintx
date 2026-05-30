---
phase: 25-group-2-hessian-higher-order-derivatives
plan: 05
subsystem: kernels
tags: [libcint, hessian, multicenter, 2c2e, 3c2e, ipip, rank-9, ket-headroom, cubecl, oracle, vendor-parity]

requires:
  - phase: 25-01
    provides: Rys nroots>=6 host engine (FND-02) — the +2 Hessian raise routes to the host fill_g_tensor_2e path
  - phase: 25-02
    provides: fail-closed rank-9 staging (FND-06) — single upfront BufferTooSmall assertion, unconditional scatter
  - phase: 23
    provides: int3c2e_ip1/ip2 first-order engine + the 3c2e kl slot-mapping (real aux k -> 2e `ll` slot, Pitfall 2)
  - phase: 13
    provides: f12.rs gout_ipip1 rank-9 Hessian gout helper (reused verbatim for bra-side ipip1)
provides:
  - int2c2e_ipip1 (2-center, ∇² on bra center 1) registered + byte-identical to vendor libcint 6.1.3
  - int3c2e_ipip1 (3-center, ∇² on bra center 1) registered + byte-identical
  - int3c2e_ipip2 (3-center, ∇² on the auxiliary k center — KET headroom k_inc=2) registered + byte-identical
  - all 3 families rank-9, cart+sph, atol=1e-12, NON-SQUARE block; oracle_covered=true; manifest-audit green
  - new verbatim gout_ipip2_l helper (ket G2E_D_K via nabla1l_2e) + 6 vendor_ffi wrappers + bindgen allowlist
affects: [25-06, hess]

tech-stack:
  added: []
  patterns:
    - "Multi-center rank-9 Hessian = the verbatim hess.c/int3c2e.c gout (gout_ipip1 column-major reorder) composed atop the plain 2e G-tensor (fill_g_tensor_2e), HOST-routed so the +2 raise reaches nroots 6..12 (FND-02)"
    - "Bra-side ipip1 reuses gout_ipip1 verbatim (nabla1i_2e, li+2 headroom); ket-side ipip2 mirrors it with nabla1l_2e (the real aux k lives in the 2e `ll` slot — Pitfall 2) and ll+2 headroom — IDENTICAL s[] triple product + column-major reorder, only the derivative center differs"
    - "3c2e host launcher MUST include the per-primitive Gaussian-overlap prefactors (pdata.fac) — fac_env = common_factor * pdata_ij.fac * pdata_kl.fac — exactly like the device ip1/ip2 host bridge; bare common_factor scales the 3c2e output wrong (2c2e is immune because its phantom j,l sit on the i,k centers)"

key-files:
  created:
    - crates/cintx-oracle/tests/hess_multicenter_ipip_parity.rs
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-compat/src/raw.rs
    - crates/cintx-cubecl/src/kernels/f12.rs
    - crates/cintx-cubecl/src/kernels/center_2c2e.rs
    - crates/cintx-cubecl/src/kernels/center_3c2e.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/vendor_ffi.rs

key-decisions:
  - "int3c2e_ipip1 == gout_ipip1 verbatim (CINTgout2e_int3c2e_ipip1 has the identical s[] + column-major reorder as int2e_ipip1); only the C source's G2E_D_K vs G2E_D_I distinguishes ipip2 from ipip1, so a single new gout_ipip2_l (nabla1l_2e) covers the ket case"
  - "All 3 families HOST-routed through fill_g_tensor_2e (not the device ip1/ip2 kernels which cap at nroots<=5) — the +2 Hessian raise on the corpus can reach nroots 6, the in-phase FND-02 trigger; ceiling HOST_RYS_NROOTS_CEILING=12, nroots>12 fail-closed"
  - "ipip2 KET headroom = lk+2 applied to the 2e `ll` slot (build_2e_shape(li, lj, 0, lk+2)), distinct from ipip1's li+2 bra raise — the load-bearing ket/bra distinction (T-25-16); gated by the NON-SQUARE bra-i vs aux-k block"
  - "Spinor reps NOT registered (these 3 families have no spinor manifest entry this plan); surface stays manifest + RawApiId + kernel + vendor-FFI + oracle only (D-11). No capi variants, no cint* wrappers"

patterns-established:
  - "Pattern: a shared launch_center_3c2e_hess(HessKind) dispatched by a bra/ket enum covers both ipip1 and ipip2 from one body — avoids two near-duplicate 3c2e Hessian launchers"
  - "Pattern: parametric multi-arity vendor parity (2-shell pairs + 3-shell triples) over a shared spd 3-center fixture, NON-SQUARE, pinning 9*nf and any_nonzero"

requirements-completed: [HESS-03]

duration: 10min
completed: 2026-05-31
---

# Phase 25 Plan 05: HESS-03 multi-center rank-9 Hessian families Summary

**int2c2e_ipip1 (2-center, ∇² on bra center 1), int3c2e_ipip1 (3-center, bra ∇²), and int3c2e_ipip2 (3-center, ∇² on the auxiliary k center — KET headroom k_inc=2) registered with component_rank=9 and byte-identical to vendor libcint 6.1.3 at atol=1e-12 (cart+sph, NON-SQUARE), HOST-routed through fill_g_tensor_2e (FND-02) reusing the verbatim gout_ipip1 for the bra side plus a new ket-side gout_ipip2_l (nabla1l_2e on the 2e `ll` slot).**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-05-31
- **Tasks:** 3 (Task 0 RED scaffold, Task 1 register+implement, Task 2 vendor+parity)
- **Files modified:** 7 (1 created, 6 modified)

## Accomplishments
- 3 multi-center rank-9 Hessian families registered (6 lock entries cart+sph, component_rank=9) + regenerated api_manifest.{rs,csv}; RawApiId consts INT2C2E_IPIP1/INT3C2E_IPIP1/INT3C2E_IPIP2 (cart+sph).
- New `gout_ipip2_l` (ket G2E_D_K via nabla1l_2e, IDENTICAL s[] + column-major reorder to gout_ipip1, copied 1:1 from CINTgout2e_int3c2e_ipip2).
- HOST launchers: `launch_center_2c2e_hess1` (bra-i ∇², li+2) and a shared `launch_center_3c2e_hess(HessKind::{Ipip1,Ipip2})` (bra-i li+2 / ket-aux ll+2) — both route through `fill_g_tensor_2e`, so nroots>=6 Hessian-elevated shells hit FND-02.
- int3c2e_ipip2 raises KET headroom (lk+2 on the `ll` slot), distinct from ipip1's bra li+2 — gated by a NON-SQUARE bra-i vs aux-k block.
- Vendor parity GREEN at atol=1e-12, cart+sph, NON-SQUARE, all 9 components: int2c2e_ipip1 over 9 pairs, int3c2e_ipip1/ipip2 over 27 triples each. oracle_covered=true; manifest-audit status ok, 0 uncovered stable entries.

## Task Commits

1. **Task 0: RED parity scaffold** - `704369a` (test)
2. **Task 1: register + implement 3 families** - `777e9e5` (feat)
3. **Task 2: vendor FFI + parity green + oracle_covered** - `c9d38e7` (feat)

## Files Created/Modified
- `crates/cintx-oracle/tests/hess_multicenter_ipip_parity.rs` - vendor-gated `hess_multicenter_ipip` parity for all 3 families; multi-arity sweep (2-shell pairs for 2c2e, 3-shell triples for 3c2e) over a shared spd 3-center fixture, NON-SQUARE (distinct bra-i vs aux-k l); determinism+shape test pins 9*nf and any_nonzero (catches rank truncation without the vendor gate).
- `crates/cintx-ops/generated/compiled_manifest.lock.json` - 6 stable entries (2c2e_ipip1, 3c2e_ipip1, 3c2e_ipip2 cart+sph, component_rank=9), flipped oracle_covered=true after parity.
- `crates/cintx-compat/src/raw.rs` - 6 RawApiId consts (cart+sph for the 3 families).
- `crates/cintx-cubecl/src/kernels/f12.rs` - new `gout_ipip2_l` (ket-side rank-9, verbatim int3c2e.c).
- `crates/cintx-cubecl/src/kernels/center_2c2e.rs` - `launch_center_2c2e_hess1` + `ipip1` dispatch arm; import gout_ipip1.
- `crates/cintx-cubecl/src/kernels/center_3c2e.rs` - shared `launch_center_3c2e_hess` + HessKind enum + ipip1/ipip2 wrappers + dispatch arms; promoted `fill_g_tensor_2e`/`compute_pdata_host` imports out of `#[cfg(test)]`.
- `crates/cintx-oracle/build.rs` - bindgen allowlist extended with the 6 cart+sph symbols (no new .file(); int3c2e.c already compiles; symbols already in cint_funcs.h so no suppl-header decls).
- `crates/cintx-oracle/src/vendor_ffi.rs` - 6 safe wrappers (int2c2e_ipip1 2-shell arity; int3c2e_ipip1/ipip2 3-shell arity).

## Decisions Made
- The Hessian s[] triple product + column-major reorder is identical across int2e_ipip1 / int2c2e_ipip1 / int3c2e_ipip1 (all bra ∇²) and int3c2e_ipip2 (ket ∇²); only the derivative center differs, so gout_ipip1 was reused verbatim for the bra families and a single new gout_ipip2_l covers the ket case.
- A shared `launch_center_3c2e_hess` dispatched by a `HessKind` bra/ket enum replaces two near-duplicate 3c2e launchers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 3c2e host launcher missing per-primitive Gaussian-overlap prefactors**
- **Found during:** Task 2 (first vendor parity run — int2c2e_ipip1 passed, int3c2e_ipip1 failed with 6561 systematically-scaled mismatches)
- **Issue:** `launch_center_3c2e_hess` passed bare `common_factor` to `fill_g_tensor_2e`, omitting the per-primitive Gaussian-overlap factors (`pdata.fac` = exp(-mu*r²)). The device ip1/ip2 launchers compute these on-device; the host bridge `host_ip1_cart_blocks` builds `fac_env = common_factor * pdata_ij.fac * pdata_kl.fac`. 2c2e was immune because its phantom j,l sit on the i,k centers (overlap factor = 1 for coincident centers), so its real bra/ket pair carries no inter-center exponential — the bug only surfaced once a real j shell on a distinct center entered (3c2e).
- **Fix:** Compute `pdata_ij`/`pdata_kl` via `compute_pdata_host` and pass `fac_env` (promoted `compute_pdata_host` out of `#[cfg(test)]`).
- **Files modified:** crates/cintx-cubecl/src/kernels/center_3c2e.rs
- **Committed in:** `c9d38e7` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (bug). The bra-side gout reuse (int2c2e_ipip1, int3c2e_ipip1) and the ket gout_ipip2_l matched vendor on the first parity once the prefactor bug was fixed.
**Impact on plan:** Required for 3c2e correctness; no scope creep.

## Note on Acceptance Criterion AC2 (Task 1)
The plan's AC2 grep (`grep -A4 'int3c2e_ipip2' ... '"ng".*0,0,2'`) assumed the manifest lock stores the `ng[]` tuple. It does NOT — the lock stores `component_rank` only (same as every prior family, incl. 25-04). The ket headroom `k_inc=2` is encoded in code as `build_2e_shape(li, lj, 0, lk+2)` in `launch_center_3c2e_hess` (HessKind::Ipip2) and is verified end-to-end by the byte-identity parity test (cintx ipip2 must match vendor's `G2E_D_K` ket-derivative path, which it does). The contract is satisfied; only the grep target was mis-specified.

## Known Stubs
None — all 3 cart+sph families are fully wired and vendor-parity green. No spinor entries were registered for these families this plan (the multi-center ipip spinor surface lands when spinor-derivative transforms exist, Phases 27/28).

## Threat Flags
None — the new surface (operator-name dispatch on `ipip1|ipip2` in center_2c2e.rs/center_3c2e.rs) is numerical/component-correctness, fully covered by the threat register: T-25-16 (bra/ket confusion — ipip2 ng raises lk+2 on the `ll` slot, gated by the NON-SQUARE bra-i vs aux-k block, byte-identical to vendor `G2E_D_K`), T-25-17 (component truncation — component_rank=9, 9-component non-square parity green), T-25-18 (transpose — verbatim gout + NON-SQUARE gate), T-25-19 (silent skip — double-gated parity, N=9 pairs / 27 triples > 0). All mitigated.

## Next Phase Readiness
- Cluster C (HESS-03) complete; the four shared Wave-2 files (manifest lock, raw.rs, build.rs, vendor_ffi.rs) were edited additively so 25-06 (HESS-04 3rd/4th-order) appends cleanly.
- The shared `launch_center_3c2e_hess(HessKind)` + verbatim-gout reuse + the multi-arity non-square parity harness are reusable for the remaining Wave-2 / Cluster D work.
- Worktree integration: N/A (sequential executor on the main working tree; D-06 merge-base check applies only to worktree-parallelized clusters).

## Self-Check: PASSED

All created files exist on disk (hess_multicenter_ipip_parity.rs, 25-05-SUMMARY.md); all three task commits (704369a, 777e9e5, c9d38e7) present in git history.

---
*Phase: 25-group-2-hessian-higher-order-derivatives*
*Completed: 2026-05-31*
