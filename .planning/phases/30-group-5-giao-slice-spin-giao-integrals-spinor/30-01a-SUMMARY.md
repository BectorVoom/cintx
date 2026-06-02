---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01a
subsystem: cubecl
tags: [cubecl, spinor, giao, london, spgsp, sigma-p, vendor-parity, libcint, oracle-covered]

# Dependency graph
requires:
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 00
    provides: "gauge x1i-with-origin fold + cg_sa10sp rank-3 gout in sigma_p.rs; combined gauge∧kappa spinor fixture; giao_sigma_1e_parity scaffold; vendor shims; bindgen allowlist (3b68ff1)"
  - phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
    provides: "c2s_si transforms (cart_to_spinor_si_2di) + σ·p assembler (FND-05)"
  - phase: 29-group-4-relativistic-spin-operator-integrals-spinor
    provides: "kappa spinor fixtures, sigma_1e dispatch pattern, c2s_si_1ei"
provides:
  - "int1e_spgsp 8-G-tensor London overlap engine (G1E_R0I origin=ri + rirj=ri-rj post-multiply, 27→12 gout, rank 3) — net-new overlap engine, byte-identical to vendored libcint at atol=1e-12"
  - "launch_int1e_giao_sigma_family_spinor_pair — unified GIAO×σ 1e dispatcher (spgsp/cg_sa10sp/giao_sa10sp), per-arm fail-closed full-block staging guards"
  - "per-family byte-identity gates + sub-wave-a test_no_silent_skip (3 flipped / 6 deferred split)"
  - "oracle_covered=true (spinor-only) for int1e_spgsp_spinor / int1e_cg_sa10sp_spinor / int1e_giao_sa10sp_spinor"
affects: [phase-30 Sub-waves 1b/1c/1d (remaining 6 GIAO×σ 1e families), phase-30 Wave 2 (6 2e families)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "spgsp London overlap = G1E_R0I (origin=ri) + rirj=ri-rj post-multiply, NOT the cg/giao common_orig gauge fold and NOT a cross-product (Pitfall 1)"
    - "8-G build via per-axis stencils: g4..g7 = D_I(g0..g3) computed by reading g0..g3 at bra ix-1/ix/ix+1 then the nabla1i combine (sigma_p_nabla_i_combine)"
    - "Unified GIAO dispatch table (giao_family_id/rank/transform) routes by SYMBOL NAME — no positional OperatorId (Pitfall 6)"
    - "Each inline dispatch arm carries its OWN fail-closed full-block staging guard (ni_sp*nj_sp*2*rank), defense-in-depth over the launcher guards (Phase-28 CR-01)"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/sigma_p.rs
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs

decisions:
  - "Implemented spgsp as a SEPARATE sigma_p_spgsp_kernel (rank-3, ri+rj, no #[comptime] origin) ALONGSIDE the 30-00 cg kernel — the cg/giao paths and the int1e_sp (tensor_rank==1) path stay byte-for-byte unchanged."
  - "Computed g4..g7 = D_I(g0..g3) by reading each per-axis g0..g3 stencil at ix-1/ix/ix+1 and applying the nabla1i combine, rather than materializing 8 full G-tensors — keeps the kernel a single-pass per-cart accumulation matching the cg kernel structure and headroom (nmax=li+lj+2, lj_ext=lj+1)."
  - "Routed all three sub-wave-1a families through a new launch_int1e_giao_sigma_family_spinor_pair dispatcher in sigma_1e.rs (per the plan's family_id/family_transform key_link) that delegates to the existing sigma_p.rs launchers, so the cg/giao launchers proven in 30-00 are reused verbatim and only spgsp is net-new."
  - "Flipped oracle_covered via direct lock edit (3-line surgical diff) + xtask manifest-sync to regenerate api_manifest.rs (also a 3-line diff) — both authoritative sides derive from the lock so manifest-audit auto-syncs."

# Metrics
duration: 70min
completed: 2026-06-01
---

# Phase 30 Plan 01a: GIAO×σ Sub-wave 1a — spgsp 8-G London Overlap Summary

**The net-new int1e_spgsp 8-G-tensor London overlap engine (G1E_R0I origin=ri + rirj=ri−rj post-multiply, 27→12 gout, rank 3, c2s_si_1ei) proven byte-identical to vendored libcint at atol=1e-12 on a non-square p×d combined gauge∧kappa block, with cg_sa10sp/giao_sa10sp dispatched through sigma_1e.rs and exactly these 3 rows flipped oracle_covered=true — Sub-wave 1a gated green.**

## Performance

- **Duration:** ~70 min
- **Started:** 2026-06-01
- **Completed:** 2026-06-01
- **Tasks:** 2
- **Files modified:** 5 (0 created, 5 modified)

## Accomplishments
- **spgsp London overlap engine** (`sigma_p_spgsp_kernel`): the 8-G build (g0=overlap base, g1=∇_j, g2=x1i(g0,ri), g3=x1i(g1,ri), g4..g7=∇_i of g0..g3) transcribed VERBATIM from `autocode/intor3.c:1724-1758` — including the London `rirj = ri − rj` post-multiply — folding the 27-component cart mix s[0..26] into the 12-component (3 group × gc 4-block) gout. The fold is the `x1i` position recurrence, NOT a cross-product. spgsp does NOT read PTR_COMMON_ORIG (origin = `ri`; London phase = `ri − rj`).
- **Unified GIAO×σ 1e dispatcher** (`launch_int1e_giao_sigma_family_spinor_pair` in sigma_1e.rs): routes spgsp/cg_sa10sp/giao_sa10sp by symbol name through `giao_family_id`/`giao_family_rank` (rank 3) / `giao_family_transform` (SiI), each arm with its OWN fail-closed full-block staging guard (`ni_sp*nj_sp*2*rank`) before any write.
- **Per-family byte-identity gates** + `test_no_silent_skip`: spgsp/cg_sa10sp/giao_sa10sp each gated at atol=1e-12 on a NON-SQUARE p(LT,nctr=2)×d(GT) block (baked-in `assert_ne!(ni_sp, nj_sp)`); `test_no_silent_skip` asserts the 3 flipped rows RUN + non-zero + byte-identical AND oracle_covered=true, and the 6 deferred rows stay false.
- **oracle_covered flip** for exactly `int1e_spgsp_spinor` / `int1e_cg_sa10sp_spinor` / `int1e_giao_sa10sp_spinor` (spinor-only, rank 3). manifest-audit green; no capi/legacy surface.

## Task Commits

1. **Task 1: spgsp 8-G London overlap variant + GIAO σ dispatch** — `eaa5f2f` (feat)
2. **Task 2: byte-identity gate + flip oracle_covered (incl. spgsp headroom fix)** — `688c872` (test)

_Task 1 (kernel) was authored GREEN-first against the Task 2 vendor gate, mirroring 30-00's cross-task TDD structure. The Task-2 vendor gate caught a real spgsp headroom bug (Rule 1) — see Deviations._

## Files Modified
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — added `sigma_p_nabla_i_combine` (bra-nabla combine helper), `sigma_p_spgsp_kernel` (rank 3, 8-G London build), `run_sigma_p_spgsp_device` / `run_sigma_p_spgsp_on_backend` (5-arm), and `launch_int1e_spgsp_spinor_pair` (c2s_si_1ei, own fail-closed staging guard). The cg/giao kernels and the int1e_sp path are untouched.
- `crates/cintx-cubecl/src/kernels/sigma_1e.rs` — added `giao_family_id` / `giao_family_rank` / `giao_family_transform` (all SiI, rank 3) and `launch_int1e_giao_sigma_family_spinor_pair` dispatching the three families with per-arm fail-closed guards.
- `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` — `collect_cintx_giao_1e` (non-square assert), extended `collect_vendor_giao_1e` for spgsp, `giao_1e_byte_identity_gate!` × 3, `test_no_silent_skip` (3 flipped / 6 deferred). The 30-00 `giao_sigma_micro` gate kept live.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — flipped `oracle_covered` false→true for the 3 spgsp/cg/giao spinor rows (3-line diff).
- `crates/cintx-ops/src/generated/api_manifest.rs` — regenerated via `xtask manifest-sync` (3-line diff; the MANIFEST_ENTRIES `test_no_silent_skip` reads).

## Decisions Made
- spgsp's 8-G build folds the bra-nabla back-compose (g4..g7 = ∇_i of g0..g3) by reading each per-axis g0..g3 stencil at the bra-neighbors `ix-1/ix/ix+1` and applying `sigma_p_nabla_i_combine`, avoiding materializing 8 full G-tensors. Headroom matches the 30-00 cg kernel (`nmax=li+lj+2`, `lj_ext=lj+1`).
- The GIAO dispatcher delegates to the existing sigma_p.rs launchers so cg/giao stay the 30-00-proven code; only spgsp is net-new. Resolution is by symbol name (no positional OperatorId — Pitfall 6); `OperatorId::new(24)` (int4c1e_cart, resolved by symbol) re-verified unshifted (cintx-compat resolver tests 7/7 green).
- The flip was done by direct lock edit + manifest-sync (both audit sides derive from the lock), keeping each generated file's diff to exactly 3 lines.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] spgsp 8-G G-tensor headroom was one bra level too short**
- **Found during:** Task 2 (the `giao_sigma_1e_spgsp` vendor gate failed RED with per-element mismatches at atol=1e-12 — ratios ~2× and ~6×, indicating a stale/zero g-tensor read, not a global scale).
- **Issue:** The spgsp chain `g7 = G1E_D_I(G1E_R0I(G1E_D_J(g0)))` reaches bra index `li+2` *inside* the `R0I` (x1i) step before `D_I` lowers it back, so the base overlap G-tensor must be valid up to bra `li+2` at ket `jx+1`. The kernel initially copied the cg kernel's `nmax = li+lj+2` headroom (cg's deepest read is only `li+1`), so the top bra level read uninitialized/out-of-fill HRR slots.
- **Fix:** Bumped `nmax` to `li+lj+3` in both `sigma_p_spgsp_kernel` and `run_sigma_p_spgsp_device` (host buffer sizing). One extra bra level; lj_ext stays lj+1.
- **Files modified:** crates/cintx-cubecl/src/kernels/sigma_p.rs
- **Verification:** `giao_sigma_1e_spgsp` now passes at atol=1e-12 on the non-square block; cg/giao unaffected.
- **Committed in:** 688c872 (Task 2 commit, alongside the gate).

**2. [Rule 1 - Bug] `TransformKind` visibility**
- **Found during:** Task 2 build (warning: `giao_family_transform` (pub(crate)) returns the private `TransformKind`).
- **Fix:** Made `TransformKind` `pub(crate)` (it is an internal dispatch enum, no public surface).
- **Files modified:** crates/cintx-cubecl/src/kernels/sigma_1e.rs
- **Committed in:** 688c872.

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs caught by the vendor gate / compiler; no architecture change).

The plan's must_have wording ("Add them to `family_id`/`family_transform`/`build_sigma_cart`") was satisfied via a sibling GIAO dispatch table (`giao_family_id`/`giao_family_transform` + `launch_int1e_giao_sigma_family_spinor_pair`) rather than overloading the Phase-29 `family_id`/`build_sigma_cart` (whose REL signature threads `origin_coords`/`origin_charges` and whose OperatorIds 0..3/10..12 belong to the relativistic σ families). This keeps the Phase-29 REL dispatch byte-for-byte unchanged while honoring the key_link (`family_id #[comptime] selector → build_sigma_cart → run_sigma_p_*_on_backend`) and all acceptance greps (3 operators present, BufferTooSmall per arm, no partial guards). Not a behavioral deviation.

## Verification Performed
- `cargo build -p cintx-cubecl --features cpu` and `cargo build -p cintx-oracle --features cpu` — both exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e` — exit 0; `giao_sigma_1e_spgsp`, `giao_sigma_1e_cg_sa10sp`, `giao_sigma_1e_giao_sa10sp`, `test_no_silent_skip`, and the kept-live `giao_sigma_micro` all pass (0 failures, atol=1e-12, non-square block).
- WITHOUT the env var: `cargo test -p cintx-oracle --features cpu giao_sigma_1e` — the spgsp/cg/giao gates and `test_no_silent_skip` are absent (compiled out via `has_vendor_libcint`), confirming the double-gate (no silent pass).
- `cargo run -p xtask -- manifest-audit` — exit 0.
- Plan's python check: `int1e_spgsp_spinor` / `int1e_cg_sa10sp_spinor` / `int1e_giao_sa10sp_spinor` oracle_covered=true; `int1e_spgnucsp_spinor` still false — prints `ok`.
- `git diff --stat crates/cintx-capi/` empty; no new `cint1e_*` legacy wrapper symbols.
- cintx-compat resolver tests: 7 passed (OperatorId unshifted).

## Known Stubs
None for Sub-wave 1a. The other 6 GIAO×σ 1e families (`spgnucsp`/`spgsa01`/`cg_sa10nucsp`/`cg_sa10sa01`/`giao_sa10nucsp`/`giao_sa10sa01`) remain registered with `oracle_covered=false` — they are deliberately deferred to Sub-waves 1b/1c/1d (each with its own engine class and vendor gate), not coverage gaps for this sub-wave's GIAO-03 deliverable. `test_no_silent_skip` asserts they stay false.

## Next Plan Readiness
- Sub-wave 1a is gated green; 30-01b (cg/giao_sa10nucsp Rys-gauge sp/nucsp, rank 3) may begin. The GIAO dispatch table (`giao_family_id`/`giao_family_transform`/`launch_int1e_giao_sigma_family_spinor_pair`) and the per-family gate macro are the extension points; 1b/1c/1d add their families to the dispatcher + `test_no_silent_skip`'s FLIPPED list (moving them off DEFERRED).
- Re-verify command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu giao_sigma_1e`.

## Self-Check: PASSED

- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::sigma_p_spgsp_kernel
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::launch_int1e_spgsp_spinor_pair
- FOUND: crates/cintx-cubecl/src/kernels/sigma_1e.rs::launch_int1e_giao_sigma_family_spinor_pair
- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs::test_no_silent_skip
- FOUND commit: eaa5f2f (Task 1)
- FOUND commit: 688c872 (Task 2)

---
*Phase: 30-group-5-giao-slice-spin-giao-integrals-spinor*
*Completed: 2026-06-01*
