---
phase: 24-group-3-position-multipole-moment-integrals
reviewed: 2026-05-30T07:44:33Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - crates/cintx-compat/src/raw.rs
  - crates/cintx-cubecl/src/kernels/one_electron.rs
  - crates/cintx-oracle/build.rs
  - crates/cintx-oracle/src/vendor_ffi.rs
  - crates/cintx-oracle/tests/moment_common.rs
  - crates/cintx-oracle/tests/moment_high_parity.rs
  - crates/cintx-oracle/tests/moment_low_parity.rs
  - crates/cintx-oracle/tests/moment_nontensor_parity.rs
  - crates/cintx-oracle/tests/moment_r_parity.rs
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 24: Code Review Report

**Reviewed:** 2026-05-30T07:44:33Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 24 adds the position / multipole-moment integral families (`int1e_r/rr/rrr/rrrr/r2/r4/z/zz` and their `_origj` variants, plus `rinv/drinv/p4/irp`) as on-device `#[cube]` kernels, their `RawApiId` registrations, vendor FFI wrappers, and rank-parameterized parity scaffolds. The bulk of the change is the 5,160-line edit to `one_electron.rs`.

The diff is internally well-disciplined: the four new device kernels (`one_electron_moment_kernel`, `one_electron_irp_kernel`, `one_electron_rinv_kernel`, `one_electron_drinv_kernel`, `one_electron_p4_kernel`) carry consistent rank wiring between the device buffer sizing (`total_len = rank * block_len`) and the host-side staging copy loops, the fail-closed `li+lj+headroom > 8` and `nroots > MAX_DEVICE_NROOTS` guards are present on every Rys/headroom path, the `rinv/drinv` gate correctly reads `PTR_RINV_ORIG` (env[4..6]) rather than the gauge origin, the `_origj` vs base origin-source branch is realized cleanly host-side as `drj = rj - origin`, and the env coefficient-layout (`coeff[pi*nctr_i+ci]`, prim-major) matches the Phase-23 corrected convention. The math (RCJ ket shift, `D_I + D_J` drinv gradient, base-3 digit tensor emission) traces correctly against the cited libcint sources. `cargo build -p cintx-cubecl --features cpu` is clean (the sole warning is pre-existing in `f12.rs`).

No correctness or security BLOCKERs were proven. The findings below are latent-hazard / robustness / coverage concerns plus minor quality items.

## Warnings

### WR-01: Moment rank/order tuple is duplicated between dispatcher and device runner — silent buffer-overrun hazard if they ever drift

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:4986-4995` (with `:4935` and dispatcher `:5844-5854`)
**Issue:** `run_1e_moment_device` receives `moment_order` and `rank` as parameters but then **ignores them for the kernel launch**, re-deriving the comptime `(mode, order, rank)` triple from `op_mode` in a hardcoded `match` (lines 4986-4995). Meanwhile the output buffer length `out_len = nctr_i*nctr_j*(rank)*nci*ncj` (line 4935) and the g-tensor sizing `nmax_u = li+lj+mo` (lines 4928-4931) use the **passed** `rank`/`moment_order`. The same `(op_mode → order, rank)` mapping is also independently encoded in the dispatcher's `moment_dispatch` (lines 5844-5854). Three copies of the same source-of-truth must stay in lockstep. If any copy drifts (e.g. a future family edit updates `moment_dispatch` but not the device `match`), the device kernel would write `comptime_rank * block_len` elements into an `out_h` buffer allocated for `passed_rank * block_len` — a heap-side buffer overrun (or silent truncation), with no compile-time check catching it. They agree today, so this is latent, not live.
**Fix:** Drive the comptime selection and the `out_len`/`nmax` sizing from a single tuple. Either drop the redundant `match` and pass the triple through (CubeCL comptime requires the `match`, so instead) collapse the dispatcher + device into one `const fn moment_params(op_mode) -> (u32,u32,u32)` used by both `moment_dispatch` and `run_1e_moment_device`, and `debug_assert_eq!((moment_order, rank), moment_params(op_mode).1_2)` at the top of `run_1e_moment_device`.

### WR-02: Output-copy loops silently drop writes past `staging.len()` — masks an undersized staging buffer instead of failing closed

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:5973`, `5997`, `6135`, `6159`, `6270`, `6294`, `6411`, `6436` (every Phase-24 staging copy)
**Issue:** Each Phase-24 family's staging write is guarded `if dst < staging.len() { staging[dst] = ... }`. If the planner ever sizes `staging` too small for a family's `component_rank` (exactly the off-by-rank / truncation failure mode called out for this phase), the guard makes the kernel **silently drop the out-of-range components** and return `Ok(...)` with a partial result, rather than erroring. This contradicts the project's OOM-safe "no best-effort partial writes / typed failure" contract (CLAUDE.md). The component-rank correctness then rests entirely on the separate planner manifest (not in this diff's scope), with no defensive check at the write site.
**Fix:** Before the copy loops, assert the staging buffer is large enough for the full rank and return a typed error otherwise, e.g. `let needed = rank_us * <comp_block>; if staging.len() < needed { return Err(cintxRsError::ChunkPlanFailed { from: "cubecl_1e", detail: format!("staging too small: have {}, need {needed}", staging.len()) }); }`, then drop the per-element `if dst < staging.len()` guard so an undersized buffer fails loudly.

### WR-03: No `nctr > 1` (general-contraction) parity case for any Phase-24 family

**File:** `crates/cintx-oracle/tests/moment_common.rs:42-43, 76-94` and all `moment_*_parity.rs`
**Issue:** Every Phase-24 parity test runs on the H2O/STO-3G corpus, where all shells are `nctr == 1`. The project's own hard-won lesson (the raw `eval_raw` column-major↔row-major coefficient transpose bug, latent because *all prior fixtures were `nctr==1`*) is that a new family's `nctr>1` path is exercised by **no** test until one is deliberately added. The kernels here read `coeff[(pi*nctr_i+ci)]` and emit per-contraction blocks via `base = (ci*nctr_j+cj)*total_len`, but that contraction-blocking + staging stride (`ii = ci*nci+ic`, `jj = cj*ncj+jc`) is never verified against libcint for these 12+ new symbols.
**Fix:** Add at least one `nctr>1` (general-contraction) vendor parity case for a representative moment family (e.g. `int1e_rr` on a contracted basis), mirroring the `int3c1e_genctr_parity` precedent, so the contraction-blocked staging stride is byte-checked.

### WR-04: Parity helpers panic-unwrap and the non-zero guard can mask a stuck-at-zero family

**File:** `crates/cintx-oracle/tests/moment_common.rs:116-120, 170-173`
**Issue:** `collect_cintx_block` calls `eval_raw(...).unwrap_or_else(|e| panic!(...))`; an `UnsupportedApi`/`InvalidEnvParam` regression surfaces as an opaque panic rather than a comparison. More substantively, `assert_any_nonzero` only checks that *at least one* element exceeds `1e-14` — for a high-rank family (`rrrr`, rank 81; `irp`, rank 9) a kernel that correctly fills one component but zeroes the rest would pass both `assert_any_nonzero` *and* `count_mismatches` only insofar as the dropped components also read zero in the vendor block. Combined with WR-02 (silent component drop), a partial-rank emission could pass the gate if the vendor happens to be zero in the dropped slots for the chosen shell pair.
**Fix:** Keep the `unwrap` (test code, `anyhow`/panic is acceptable per CLAUDE.md) but strengthen the gate: assert a minimum count of non-zero elements proportional to `rank`, or assert per-component that at least one element per component block is non-zero, so a zeroed trailing component cannot slip through.

## Info

### IN-01: `_origj` origin-source decision relies on `op_name.ends_with("_origj")` string suffix

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:5859`
**Issue:** `is_origj = op_name.ends_with("_origj")` selects the ket-center origin branch by string suffix. This couples a numerical branch (origin = `rj` vs `common_orig`) to an operator-name spelling convention. A future operator named with a coincidental `_origj` suffix, or a rename, would silently change the origin source.
**Fix:** Acceptable as-is given the `moment_dispatch` match already enumerates the exact `*_origj` names; optionally fold the origin-source choice into the `moment_dispatch` tuple so it is data-driven rather than re-derived from the string.

### IN-02: `run_1e_moment_device` invalid-`op_mode` arm silently falls through to `zz`

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:4994`
**Issue:** The `match op_mode` uses `_ => launch_with!(7u32, ...)` (zz) as the catch-all. An out-of-range `op_mode` (only reachable via a future dispatcher bug) would silently compute `zz` instead of erroring. `op_mode` is always set correctly by the caller today.
**Fix:** Make arm 7 explicit (`7u32 => ...`) and `_ => unreachable!("invalid moment op_mode {op_mode}")` to surface a future miswire.

### IN-03: Large verbatim duplication of the four staging-copy blocks across moment/rinv/p4/irp/rank9 paths

**File:** `crates/cintx-cubecl/src/kernels/one_electron.rs:5951-6007, 6113-6169, 6248-6304, 6389-6445, 6542-6577…`
**Issue:** The cart/sph staging-copy + `common_fac_sp` scale + `not0` count + `ExecutionStats` return block is copy-pasted nearly verbatim across five family paths (~70 lines each). This is the kind of duplication where a fix to one (e.g. the WR-02 hardening) is easy to forget in the others.
**Fix:** Extract a helper `write_component_leading_staging::<F>(rep, rank, n_ctr_i, n_ctr_j, nci, ncj, nsi, nsj, li, lj, &cart_comp, staging) -> not0` and call it from each path.

### IN-04: Doc comment in `RawApiId` references unimplemented future plans as the rationale for declaring p4/irp consts

**File:** `crates/cintx-compat/src/raw.rs:201-214`
**Issue:** The comment block states the p4/irp manifest+kernel "land in plans 24-04 / 24-05" and that "until 24-04/24-05 register the manifest rows, a p4/irp dispatch fails closed at resolver (MissingSymbol)". As of this diff the p4/irp kernels and vendor wrappers are in fact present, so the comment is stale relative to the merged state and could mislead a future reader into thinking these are still unwired stubs.
**Fix:** Update the comment to reflect that the p4/irp families are now fully registered (kernel + vendor FFI + parity), removing the "until 24-04/24-05" forward reference.

---

_Reviewed: 2026-05-30T07:44:33Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
