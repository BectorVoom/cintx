---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 00
subsystem: testing
tags: [cubecl, spinor, giao, gauge-origin, sigma-p, vendor-parity, libcint, bindgen]

# Dependency graph
requires:
  - phase: 22-gauge-origin-env-slot-gap-a-ptr-common-orig
    provides: PTR_COMMON_ORIG gauge-origin env slot + common_orig plumbing (FND-01)
  - phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
    provides: reusable σ·p #[cube] assembler (sigma_p.rs) + c2s_si transforms (FND-05)
  - phase: 29-group-4-relativistic-spin-operator-integrals-spinor
    provides: kappa spinor fixtures, c2s_si_1ei (cart_to_spinor_si_2di), sigma_1e dispatch pattern
provides:
  - "build_gauge_kappa_spinor_fixture (combined gauge!=0 AND kappa!=0 1e spinor fixture, D-02)"
  - "gauge-origin x1i-with-origin device step in sigma_p.rs (the genuinely-new Phase-30 math)"
  - "launch_int1e_cg_sa10sp_spinor_pair (rank-3 gauge σ·p launcher, c2s_si_1ei, common_factor 0.5)"
  - "vendor_int1e_cg_sa10sp_spinor + vendor_int1e_giao_sa10sp_spinor FFI shims"
  - "giao_sigma_micro Wave-0 byte-identity gate (atol=1e-12) + cg->giao collapse witness"
affects: [phase-30 Wave 1 (all 9 1e GIAO×σ families), phase-30 Wave 2 (6 2e families)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gauge fold = CINTx1i_1e position recurrence f[i]=g[i+1]+origin*g[i], NOT a cross-product"
    - "Separate gauge #[cube] kernel in sigma_p.rs leaves the int1e_sp (tensor_rank==1) path untouched"
    - "cg->giao collapse via common_orig=bra-center (dri=0) as the live-gauge witness"

key-files:
  created:
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
  modified:
    - crates/cintx-cubecl/src/kernels/sigma_p.rs
    - crates/cintx-oracle/src/fixtures.rs
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs

key-decisions:
  - "Implemented the gauge variant as a SEPARATE sigma_p_cg_sa10sp_kernel (rank-3, origin: [f64;3] + #[comptime] variant) rather than overloading sigma_p_kernel, keeping the int1e_sp path byte-for-byte unchanged while honoring the plan's sigma_p.rs extension directive."
  - "Used the larger headroom (nmax=li+lj+2, lj_ext=lj+1) that sigma_1e.rs::sigma_ov_kernel already uses for the srsr R_J/R_I family — the cg_sa10sp ∇_j + x1i build needs the same +1 ket / +1 bra slack."
  - "cg→giao collapse uses common_orig=[0,0,0]; valid because the Wave-0 fixture's bra (i) shell sits at the coordinate origin so dri = ri − 0 = 0 → x1i(g,0)=G1E_R_I=giao."

patterns-established:
  - "Gauge-origin x1i device step: sigma_p_x1i (g[i+1]+origin*g[i]), sigma_p_x1i_of_j, sigma_p_nabla_j helpers"
  - "12-gout → 3 groups × gc 4-block mapping (k = tensor*4 + e1) verified against CINT1e_spinor_drv c2s loop (cint1e.c:269)"

requirements-completed: [GIAO-03]

# Metrics
duration: 35min
completed: 2026-06-01
---

# Phase 30 Plan 00: GIAO×σ Wave-0 Gauge-Gout De-Risk Summary

**Gauge-origin x1i-with-origin device fold for int1e_cg_sa10sp proven byte-identical to vendored libcint at atol=1e-12, plus a cg→giao-at-origin collapse witness — the only genuinely-new Phase-30 device math, de-risked before any family is wired.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-01
- **Completed:** 2026-06-01
- **Tasks:** 3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments
- The combined gauge!=0 AND kappa!=0 1e spinor fixture (`build_gauge_kappa_spinor_fixture`, D-02) — all 5 hard constraints satisfied, Phase-29 fixture untouched.
- The gauge-origin `x1i`-with-origin G-tensor step ported into `sigma_p.rs` as a device recurrence `f[i]=g[i+1]+origin*g[i]` (NOT a cross-product), composed with a ket-nabla (G1E_D_J) into a rank-3 `int1e_cg_sa10sp` gout variant — the highest-rework-risk piece of the phase.
- Two vendor FFI shims (`vendor_int1e_cg_sa10sp_spinor`, `vendor_int1e_giao_sa10sp_spinor`).
- The `giao_sigma_micro` gate: byte-identity vs vendor at atol=1e-12 AND the cg→giao collapse at common_orig=[0,0,0]. The int1e_sp path and the Phase-29 REL suite (10/0) are unregressed.

## Task Commits

1. **Task 1: Combined gauge+kappa 1e spinor fixture (D-02)** - `055429a` (feat)
2. **Task 2: Gauge-origin x1i device step + cg_sa10sp variant** - `2ae0a9a` (feat)
3. **Task 3: Gauge-gout byte-identity micro-test (+ allowlist deviation fix)** - `e90e99d` (test)

_Task 2 was authored GREEN-first (kernel) with Task 3 as its RED→GREEN gate, per the plan's cross-task TDD structure._

## Files Created/Modified
- `crates/cintx-oracle/src/fixtures.rs` - added `build_gauge_kappa_spinor_fixture` (wraps the Phase-29 kappa fixture, sets env[PTR_COMMON_ORIG..+3]=[0.30,-0.45,0.60]).
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` - added `sigma_p_nabla_j`/`sigma_p_x1i`/`sigma_p_x1i_of_j` device helpers, `sigma_p_cg_sa10sp_kernel` (rank 3, runtime `origin`, #[comptime] variant), `run_sigma_p_cg_device`/`run_sigma_p_cg_on_backend` (5-arm), and `launch_int1e_cg_sa10sp_spinor_pair` (c2s_si_1ei fold, fail-closed staging guard, common_factor 0.5).
- `crates/cintx-oracle/src/vendor_ffi.rs` - `vendor_int1e_cg_sa10sp_spinor` + `vendor_int1e_giao_sa10sp_spinor`.
- `crates/cintx-oracle/build.rs` - extended the bindgen `allowlist_function` with the two new vendor symbols (deviation, see below).
- `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` - `giao_sigma_micro` Wave-0 gate + always-on sizing guard + Wave-1 RED stub.

## Decisions Made
- Authored the gauge fold as a self-contained second kernel (`sigma_p_cg_sa10sp_kernel`) so the existing `int1e_sp` (`tensor_rank==1`) kernel and dispatch stay byte-identical (acceptance criterion: no edits inside the `tensor_rank==1` branch). The kernel mirrors `sigma_1e.rs::sigma_ov_kernel`'s headroom (nmax=li+lj+2, lj_ext=lj+1) and the `srsr` family's g0/g1/g2/g3 composition structure, swapping the bra `R_I` for the gauge `x1i`-with-origin.
- The 12-component gout → 3 groups × gc 4-block layout (`k = tensor*4 + e1`) was verified against `CINT1e_spinor_drv` (`cint1e.c:269` loops c2s `ncomp_tensor` times over `nc=nf*ctr*ncomp_e1` slices after the `CINTdmat_transpose` to component-leading), matching the existing rank-3 `int1e_sigma` grouping.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Extended the bindgen function allowlist for the two new vendor symbols**
- **Found during:** Task 3 (running the micro-test)
- **Issue:** RESEARCH §Registration step 3 stated "all 15 symbols are in cint_funcs.h → no suppl-header extern decls needed", which is true but incomplete: the oracle `build.rs` runs a SECOND bindgen pass (line 383) with an explicit `allowlist_function(...)` regex. Symbols absent from that regex (even though present in cint_funcs.h) are NOT emitted as `pub fn`, so `ffi::int1e_cg_sa10sp_spinor` / `ffi::int1e_giao_sa10sp_spinor` failed with E0425 "cannot find function in module ffi".
- **Fix:** Appended `int1e_cg_sa10sp_spinor|int1e_giao_sa10sp_spinor` to the `allowlist_function` regex in `crates/cintx-oracle/build.rs`.
- **Files modified:** crates/cintx-oracle/build.rs
- **Verification:** `cargo build -p cintx-oracle --features cpu` (with CINTX_ORACLE_BUILD_VENDOR=1) compiles; `giao_sigma_micro` passes.
- **Committed in:** e90e99d (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking, build-config only)
**Impact on plan:** The fix is a pure build-config addition (no math, no architecture change); it surfaces an under-specification in RESEARCH that Wave 1 must also apply (it will add the other 13 family symbols to the same allowlist). No scope creep.

## Issues Encountered
- The plan's literal "common_orig=[0,0,0]" collapse only equals giao when the bra center is also at the origin (because cg uses dri=ri−common_orig, not dri=common_orig). The Wave-0 fixture's bra (i) shell IS at [0,0,0], so the literal [0,0,0] collapse is valid; the test asserts the bra-at-origin precondition explicitly to keep this honest for any future fixture edit.

## Known Stubs
- `giao_sigma_1e_full_parity_red` in `tests/giao_sigma_1e_parity.rs` is an intentional `#[ignore]`d RED stub for Wave 1 (Plan 30-01) — it `unimplemented!()`s the full 9-family parity + no-silent-skip gate. This is deliberate scaffolding (the file is the one Wave 1 extends), not a coverage gap for GIAO-03's Wave-0 deliverable. It does not run and cannot mask a regression (the live `giao_sigma_micro` gate is the Wave-0 proof).

## Next Phase Readiness
- The gauge `x1i`-with-origin fold is proven byte-identical. Wave 1 (Plan 30-01) can now wire the remaining 8 1e families (`spgsp`, `spgnucsp`, `spgsa01`, `cg_sa10nucsp`, `cg_sa10sa01`, `giao_sa10sp`, `giao_sa10nucsp`, `giao_sa10sa01`) onto this proven fold, extend the allowlist for their vendor symbols, and add manifest rows (watch the OperatorId positional-shift landmine + sa01 rank=9).
- No blockers. The micro-test gate command for re-verification:
  `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity giao_sigma_micro`

## Self-Check: PASSED

- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
- FOUND: crates/cintx-oracle/src/fixtures.rs::build_gauge_kappa_spinor_fixture
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::sigma_p_cg_sa10sp_kernel
- FOUND commits: 055429a, 2ae0a9a, e90e99d

---
*Phase: 30-group-5-giao-slice-spin-giao-integrals-spinor*
*Completed: 2026-06-01*
