---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01c
subsystem: cubecl
tags: [cubecl, spinor, giao, sa01, rys, gauge, rank-9, vendor-parity, libcint, resolved]

# Dependency graph
requires:
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 00
    provides: "gauge x1i-with-origin fold helpers (sigma_p_x1i, sigma_p_nabla_j) + combined gauge∧kappa spinor fixture + giao_sigma_1e_parity scaffold + vendor shims + bindgen allowlist (3b68ff1/30-00)"
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 01a
    provides: "GIAO×σ dispatch table (giao_family_id/rank/transform + launch_int1e_giao_sigma_family_spinor_pair) in sigma_1e.rs"
  - phase: 29-group-4-relativistic-spin-operator-integrals-spinor
    provides: "Phase-29 nuclear Rys path (sigma_1e_nuc: nuc_vrr_axis/rys_root1..5 pattern, MAX_DEVICE_NROOTS fail-closed), c2s_si_1e (cart_to_spinor_si_2d, REAL)"
provides:
  - "sa01_rys_kernel — net-new rank-9 Rys+gauge sa01 device kernel (rinv nuclear base, g1=∇_j+∇_i both-side nabla, x1i-with-origin gauge fold, 36-component 9×4 gout) in sigma_p.rs — BUILDS GREEN; NOT yet byte-identical to vendor"
  - "launch_int1e_sa10sa01_spinor_pair — rank-9 launcher with fail-closed full-block (×9) staging guard + fail-closed nroots guard (no clamp), REAL c2s_si_1e transform"
  - "cg_sa10sa01 / giao_sa10sa01 dispatch arms in launch_int1e_giao_sigma_family_spinor_pair (giao_family_rank→9, giao_family_transform→Si REAL)"
  - "rank-9 byte-identity gates + no-silent-skip (RED/#[ignore]d pending the gout-layout fix)"
affects: [phase-30 Sub-wave 1d (spgnucsp/spgsa01 — closes the full 9-family 1e gate), phase-30 Wave 2]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "rank-9 sa01 engine = rinv Rys nuclear base (single center, charge +1) + g1=∇_j(g0)+∇_i(g0) both-side nabla + x1i-with-origin gauge fold inside the Rys root loop"
    - "cg vs giao differ ONLY in the x1i origin threaded by the dispatcher: dri=ri−common_orig (G2E_RCI) vs ri (G2E_R_I); gout body byte-identical (intor3.c:998 == :1323)"
    - "rank-9 sa01 routes through the REAL c2s_si_1e (cart_to_spinor_si_2d), NOT the imaginary si_2di the sp/nucsp arms use (Pitfall 2 — applied correctly)"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/sigma_p.rs
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs

decisions:
  - "sa01 Rys+gauge engine implemented as a SELF-CONTAINED net-new kernel (sa01_rys_kernel + sa01_gout + sa01_g1_bothside + sa01_x1i_of_g1 + sa01_nuc_vrr_axis) reusing the 30-00 gauge helpers (sigma_p_x1i, sigma_p_nabla_j) and the overlap HRR (sigma_p_hrr_axis); the int1e_sp / cg-overlap / spgsp / nucsp paths are untouched."
  - "Sub-wave 1b (cg/giao_sa10nucsp) was NEVER executed (no 30-01b-SUMMARY, no sa10nucsp code anywhere) — the prompt's claim that 1b landed a nucsp Rys+gauge engine is factually wrong. 1c is self-contained on 30-00 + 30-01a + Phase-29, so it was executed regardless (see Deviations)."
  - "oracle_covered was NOT flipped — byte-identity is not yet achieved; the 3 rank-9 byte-identity gates are #[ignore]d and test_no_silent_skip enforces the honest oracle_covered=false state."

# Metrics
duration: 95min
completed: 2026-06-01
status: RESOLVED (root cause = hardcoded rinv center; byte-identity achieved 2026-06-01)
---

> ## ⚠️ RESOLUTION (2026-06-01) — the BLOCKED diagnosis below was WRONG
>
> The "gout → gc-block layout" hypothesis in this SUMMARY was **disproven**. The
> cart-path discriminator (`crates/cintx-oracle/tests/sa01_cart_discriminator.rs`)
> proved the g-tensor AND the `cart_to_spinor_si_2d` transform both **byte-correct**.
>
> **Actual root cause:** the `cg_sa10sa01` / `giao_sa10sa01` dispatch arms in
> `sigma_1e.rs` hardcoded the rinv center as `[0,0,0]` instead of reading
> `env[PTR_RINV_ORIG]` (which the Phase-30 fixture sets to `[0.30,-0.45,0.60]`).
> Vendor evaluates at that env center; cintx used `[0,0,0]` → wrong values. The
> "structural zeros" were the symmetric-center artifact.
>
> **Fix:** threaded `rinv_orig` through `launch_int1e_giao_sigma_family_spinor_pair`
> + the sa01 arms; added the sa01 symbols to `is_giao_rinv_center_symbol` (raw.rs).
> The 3 byte-identity gates are now GREEN, `test_no_silent_skip` asserts byte-identity,
> and `oracle_covered=true` for both sa01 rows. **Sub-wave 1c is now gated green.**
>
> Full evidence chain + the cart-discriminator harness are documented in
> `30-01c-DEBUG.md` (status: root-caused-and-fixed). Everything below this banner is
> the original (incorrect-conclusion) execution record, preserved for history.

# Phase 30 Plan 01c: GIAO×σ Sub-wave 1c — rank-9 Rys+gauge sa01 engine (originally BLOCKED — see RESOLUTION above)

**The net-new rank-9 Rys+gauge `int1e_cg_sa10sa01` / `int1e_giao_sa10sa01` engine (rinv nuclear base, `g1 = ∇_j(g0)+∇_i(g0)` both-side nabla, `x1i`-with-origin gauge fold, 36-component 9×4 gout, REAL `c2s_si_1e`) is implemented and BUILDS GREEN with all guards in place, but is NOT yet byte-identical to vendored libcint — the 36-gout → 9×4 gc-block layout consumed by the c2s transform is mismatched. The byte-identity gates are RED/#[ignore]d and `oracle_covered` is NOT flipped. Sub-wave 1c is NOT gated green.**

## Status: BLOCKED

The engine is structurally complete and compiles cleanly (`cargo build -p cintx-cubecl --features cpu` and `-p cintx-oracle` both exit 0). Both arms run, produce all-9-components-non-zero output, and the staging/nroots guards are correct. **But the output diverges from vendor** at atol=1e-12 with a systematic pattern (vendor non-zero where cintx is exactly 0.0 at gout component slots 1,2,5,6,8,11,12,15,17,18, plus scaled differences elsewhere). This indicates the **36-component `gout[n*36+k]` → 9-group × 4-gc-block layout** that `cart_to_spinor_si_2d` consumes does not match libcint's actual within-group gc-block ordering / axis-fold for these rank-9 families. RESEARCH Open Q1 explicitly flagged the full 36-component gout transcription as MEDIUM confidence and deferred it to execution; the structural transcription landed but the c2s consumption mapping needs numerical reverse-engineering against vendor (the same kind of dual hand-derived + vendor verification the spike-findings skill prescribes).

This is a genuine numerical/layout blocker, not a quick fix. I reached the fix-attempt limit during the structural reconstruction (which was itself complicated by repeated Edit-anchor failures and a build that had to be recovered twice). The honest outcome is recorded here rather than fabricating a green gate.

## Performance

- **Duration:** ~95 min (including two build recoveries from failed Edit-anchor inserts)
- **Tasks:** 2 (Task 1 engine — done/green-build; Task 2 gate+flip — gates landed RED/ignored, flip withheld)
- **Files modified:** 3

## Accomplishments (what landed and builds green)

- **`sa01_rys_kernel`** (sigma_p.rs, net-new): rinv Rys nuclear base (single center, charge +1), comptime nroots 1..=5, `g1 = ∇_j(g0) + ∇_i(g0)` both-side nabla (`sa01_g1_bothside`), `g2 = x1i(g0, origin)` and `g3 = x1i(g1, origin)` gauge folds (`sa01_x1i_of_g1`) inside the Rys root loop, and `sa01_gout` writing the full 36-component (9 tensor × 4 gc) mix transcribed from intor3.c:998 — **no truncation to rank 3** (Pitfall 3 avoided; all 9 components are non-zero in the output).
- **`launch_int1e_sa10sa01_spinor_pair`**: folds each of the 9 σ-groups through the **REAL** `cart_to_spinor_si_2d` (c2s_si_1e — NOT the imaginary si_2di; Pitfall 2 applied), with its OWN fail-closed full-block **×9** staging guard (`ni_sp*nj_sp*2*9`) and a fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` guard (**no clamp**).
- **sigma_1e.rs dispatch**: `giao_family_id` +cg/giao_sa10sa01; `giao_family_rank` → **9**; `giao_family_transform` → **Si** (REAL) for sa01; two dispatch arms threading the x1i origin (cg `dri=ri−common_orig`, giao `origin=ri`), each with its own ×9 staging guard.
- **cg→giao reduction is correct in principle**: cg with `origin=dri` and giao with `origin=ri` share the identical gout body, exactly as libcint (the gout-body identity is verified against intor3.c:998 vs :1323).
- **Rank-9 byte-identity gates + no-silent-skip** added to giao_sigma_1e_parity.rs on the NON-SQUARE p×d block (Pitfall 6 / T-30-01c-06 assertion baked in), with the all-9-components-non-zero assertion (Pitfall 3) and the cg→giao collapse witness. They are `#[ignore]`d pending the layout fix; `test_no_silent_skip` enforces the honest `oracle_covered=false` state.

## Task Commits

1. **Task 1: rank-9 Rys+gauge sa01 engine + dispatch arms** — `783d392` (feat) — builds green
2. **Task 2: rank-9 byte-identity gates (RED/ignored) + honest no-silent-skip** — `a39f9f7` (test) — oracle_covered NOT flipped

## Files Modified

- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — +743 lines: `SA01_PIE4`, `SA01_MAX_DEVICE_NROOTS`, `sa01_nuc_vrr_axis`, `sa01_nabla_i`, `sa01_g1_bothside`, `sa01_x1i_of_g1`, `sa01_gout`, `sa01_rys_kernel` (#[cube(launch)]), `run_sa01_rys_device`/`run_sa01_rys_on_backend` (5-arm), `launch_int1e_sa10sa01_spinor_pair`; added `use crate::math::rys::{rys_root1..5}`. Existing kernels untouched.
- `crates/cintx-cubecl/src/kernels/sigma_1e.rs` — extended `giao_family_id`/`giao_family_rank`(→9)/`giao_family_transform`(→Si) and added the two sa01 dispatch arms.
- `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` — rank-9 collectors, 3 byte-identity gates (#[ignore]d WIP), cg→giao collapse witness, `manifest_lock_entry`, honest `test_no_silent_skip`.

## Deviations from Plan

### RULE 4 / Blocker — vendor byte-identity not achieved (gout layout)

- **What:** The rank-9 sa01 output is not byte-identical to vendor. The systematic zero-vs-nonzero mismatch pattern points at the `gout[n*36+k]` → 9-group × 4-gc-block mapping that `cart_to_spinor_si_2d` consumes.
- **Why it's a blocker, not a quick fix:** RESEARCH Open Q1 flagged the full 36-gout transcription as MEDIUM confidence. The structural math (s[0..8], the 9 group rows) was transcribed verbatim from intor3.c, but how libcint packs those 36 values into the 9×(gc_x,gc_y,gc_z,gc_1) blocks that the REAL si transform reads — versus the `k = group*4 + gc_block` scheme I assumed from the rank-3 spgsp/cg precedent — needs numerical reverse-engineering against vendor (per the spike-findings dual-verification method). I reached the fix-attempt limit during the structural reconstruction.
- **Disposition:** Did NOT flip `oracle_covered` (no coverage claim on a non-byte-identical family); marked the 3 byte-identity gates `#[ignore]` with a clear WIP reason; `test_no_silent_skip` enforces the honest false state. No fabricated green.

### [Blocker — missing dependency] Sub-wave 1b was never executed

- **Found during:** execution start (context load).
- **Issue:** The plan declares `depends_on: [01b]` and the prompt states "1b landed the Rys+gauge nuclear engine (commits c3f9a14, d8b2e07)". **Those commits do not exist in the repo, no `30-01b-SUMMARY.md` exists, and there is no `sa10nucsp` code anywhere** (sigma_p.rs / sigma_1e.rs / sigma_1e_nuc.rs / the test file all return 0 for `sa10nucsp`). The last landed work was 30-01a (f067b3a). The prompt's stated 1a/1b commit hashes (b4e85e9/a7c1f2d, c3f9a14/d8b2e07) are also wrong — the actual 1a commits are eaa5f2f/688c872.
- **Resolution (no Rule-4 stop):** 1c's actual code dependencies are 30-00 (gauge helpers + fixture), 30-01a (the dispatch table), and Phase-29 (the nuclear Rys pattern) — NOT 01b's nucsp kernel. The sa01 engine is net-new and imports none of 01b. So 1c was executed regardless. The `depends_on: [01b]` is a sequencing constraint; the per-sub-wave gating means 1b will still need to be executed separately before the full 9-family Wave-1 gate (30-01d) can close.

### [Process] Two build recoveries

- Two large Edit inserts into sigma_p.rs failed silently because the anchor text did not match the file exactly (the real test-module guard is `#[cfg(all(test, feature = "cpu"))]\nmod tests {`, not `#[cfg(test)]\nmod tests {\n use cubecl::Runtime;`). Each was caught by the build (E0432 unresolved import) and a premature commit was undone via `git reset --soft HEAD~1` (working tree preserved). The final insert used the verified exact anchor and built green. No work was lost; the lock was never erroneously modified (the one stray python flip was reverted).

## Verification Performed

- `cargo build -p cintx-cubecl --features cpu` → exit 0 (32 pre-existing warnings, no errors).
- `cargo build -p cintx-oracle --features cpu` → exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity` → exit 0: **3 passed, 3 ignored** (`test_kappa_sizing_non_4l_plus_2`, `test_no_silent_skip`, `giao_sigma_micro` pass; the 3 sa01 byte-identity gates are ignored WIP). The 30-00 `giao_sigma_micro` gate and the 30-01a gates are unregressed.
- WITHOUT the env var: only `test_kappa_sizing_non_4l_plus_2` runs (1 passed) — double-gate confirmed (vendor tests compiled out).
- Acceptance greps: `sa10sa01` present in sigma_p.rs (40 hits); cg/giao_sa10sa01 in sigma_1e.rs (6); no partial `if dst < staging.len()` guards (0); no nroots clamp (0); `UnsupportedApi` present (3); `BufferTooSmall` per arm including the ×9 sa01 guard (`ni_sp*nj_sp*2*RANK`); `OperatorId::new(24)` for int4c1e_cart unshifted (resolver tests 12/12 pass).
- `git diff --stat crates/cintx-capi/` empty; lock + api_manifest UNCHANGED (oracle_covered NOT flipped).

## Known Stubs / Incomplete

- **`int1e_cg_sa10sa01` / `int1e_giao_sa10sa01` are NOT byte-identical** to vendor and remain `oracle_covered=false`. The engine produces non-zero rank-9 output but the gout→gc-block layout is wrong. The 3 byte-identity gates are `#[ignore]`d (`30-01c WIP: rank-9 sa01 gout->gc-block layout not yet byte-identical`).

## Next Plan Readiness

- **Sub-wave 1c is NOT gated green.** Before 30-01d:
  1. **Resolve the rank-9 gout→gc-block layout** (the blocker): reverse-engineer the correct mapping from `gout[n*36+k]` to the 9×(gc_x,gc_y,gc_z,gc_1) blocks `cart_to_spinor_si_2d` reads, using vendor as oracle (the spike-findings dual hand-derived + vendor method). Likely candidates: the within-group gc-block order differs from the rank-3 `k=group*4+e1` assumption, OR the si transform expects a different group/component permutation for the GIAO sa01 g-factor. Un-ignore the 3 gates, confirm byte-identity, then flip `oracle_covered=true` for the 2 rows and regenerate api_manifest via the cintx-ops build.rs.
  2. **Execute the missing Sub-wave 1b** (cg/giao_sa10nucsp) — it was never done.
- Re-verify command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity -- --include-ignored`.

## Self-Check: PASSED (with BLOCKED status)

- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::sa01_rys_kernel
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::launch_int1e_sa10sa01_spinor_pair
- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs::giao_sigma_1e_cg_sa10sa01 (#[ignore]d)
- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs::test_no_silent_skip
- FOUND commit: 783d392 (Task 1, builds green)
- FOUND commit: a39f9f7 (Task 2, gates RED/ignored, no flip)
- NOTE: parity NOT byte-identical; oracle_covered NOT flipped; sub-wave NOT gated green.

---
*Phase: 30-group-5-giao-slice-spin-giao-integrals-spinor*
*Completed (engine build): 2026-06-01 — BLOCKED on vendor byte-identity*
