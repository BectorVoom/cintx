---
phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
plan: 01b
subsystem: cubecl
tags: [cubecl, spinor, giao, nucsp, rys, gauge, sigma-p, vendor-parity, libcint, oracle-covered]

# Dependency graph
requires:
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 00
    provides: "gauge x1i-with-origin fold + combined gauge∧kappa spinor fixture; vendor shims (3b68ff1); bindgen allowlist incl. both nucsp symbols"
  - phase: 30-group-5-giao-slice-spin-giao-integrals-spinor
    plan: 01a
    provides: "unified GIAO×σ 1e dispatcher (launch_int1e_giao_sigma_family_spinor_pair) + giao_family_id/rank/transform table + per-family gate macro + test_no_silent_skip"
  - phase: 29-group-4-relativistic-spin-operator-integrals-spinor
    provides: "Phase-29 σ·p-nuc Rys path (run_sigma_nuc_on_backend), kappa spinor fixtures, c2s_si_1ei"
provides:
  - "NEW Rys+gauge nuclear engine (sigma_nuc_gauge_kernel + run_sigma_nuc_gauge_on_backend): nuclear G2E_* base + gauge x1i-with-origin fold INSIDE the Rys root loop, 12-comp (3 tensor × 4 gc) gout, fail-closed nroots guard (no clamp)"
  - "launch_int1e_sa10nucsp_spinor_pair (rank 3, c2s_si_1ei imaginary, common_factor 0.5, own fail-closed full-block staging guard)"
  - "cg_sa10nucsp/giao_sa10nucsp wired into the unified GIAO dispatcher (origin_coords/origin_charges threaded; dri vs [0,0,0] gauge origin)"
  - "int1e_cg_sa10nucsp_spinor / int1e_giao_sa10nucsp_spinor byte-identical to vendored libcint 6.1.3 at atol=1e-12 on a NON-SQUARE p×d block; oracle_covered=true (spinor-only, rank 3)"
  - "fixed manifest_lock_entry parser (anchored on \"arity\") — resolves the pre-existing 30-01c test_no_silent_skip RED without weakening any sa01 assertion"
affects: [phase-30 Sub-wave 1c (rank-9 sa01, still BLOCKED), Sub-wave 1d (spgnucsp+spgsa01), phase-30 Wave 2]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rys+gauge nuclear engine = Phase-29 nuclear G2E_* base + the 30-00 x1i-with-origin recurrence (f[i]=g[i+1]+origin·g[i]) applied INSIDE the Rys root loop; origin=dri (cg) / [0,0,0] (giao); cg and giao share the gout body byte-for-byte"
    - "nucsp launcher mirrors the cg_sa10sp launcher: rank 3, c2s_si_1ei (imaginary ket), common_factor 0.5, own fail-closed full-block staging guard; nroots fail-closed in run_sigma_nuc_gauge_on_backend (never clamp)"
    - "manifest_lock_entry must anchor entry windows on the entry's FIRST field (\"arity\") — component_rank precedes the id sub-object and oracle_covered follows it, so rfind('{') lands on the id object and misses both"

key-files:
  created: []
  modified:
    - crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs
    - crates/cintx-cubecl/src/kernels/sigma_p.rs
    - crates/cintx-cubecl/src/kernels/sigma_1e.rs
    - crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv

decisions:
  - "Recovered the Rys+gauge nuclear kernel from git stash@{0} (sigma_nuc_gauge_kernel + nuc_x1i/nuc_x1i_of_j helpers + run_sigma_nuc_gauge_on_backend) — it applied cleanly and was structurally sound (fail-closed nroots, no clamp), preserved verbatim."
  - "Added cg_sa10nucsp/giao_sa10nucsp to the EXISTING unified GIAO dispatcher (giao_family_id 5/6, rank 3, SiI) rather than the Phase-29 family_id/build_sigma_cart REL path; extended the dispatcher signature with origin_coords/origin_charges (it was the sole caller, the parity collector)."
  - "Flipped oracle_covered via direct lock edit + cintx-ops build regeneration of api_manifest.rs/.csv (the build derives both mirrors from the lock); manifest-audit auto-syncs. component_rank was already \"3\" from the 3b68ff1 scaffolding — only the two oracle_covered booleans changed."
  - "Fixed the manifest_lock_entry parser (Rule 1 bug) instead of working around it — it was the root cause of the pre-existing 30-01c test_no_silent_skip RED. The fix is conservative (re-anchor only) and leaves every sa01 assertion intact (rank=9, covered=false)."

# Metrics
duration: 55min
completed: 2026-06-01
---

# Phase 30 Plan 01b: GIAO×σ Sub-wave 1b — Rys+gauge nucsp engine Summary

**The NEW Rys+gauge nuclear engine (gauge x1i-with-origin fold INSIDE the Rys root loop, 12-comp rank-3 gout, c2s_si_1ei) proven byte-identical to vendored libcint 6.1.3 at atol=1e-12 on a NON-SQUARE p×d combined gauge∧kappa block for both int1e_cg_sa10nucsp_spinor and int1e_giao_sa10nucsp_spinor, with the cg→giao collapse witness, exactly these 2 rows flipped oracle_covered=true, and the pre-existing 30-01c test_no_silent_skip parser bug fixed — Sub-wave 1b gated green.**

## Performance
- **Duration:** ~55 min (resumed execution; kernel recovered from stash)
- **Completed:** 2026-06-01
- **Tasks:** 2
- **Files modified:** 7 (0 created)

## Accomplishments
- **Rys+gauge nuclear engine** (`sigma_nuc_gauge_kernel` + `run_sigma_nuc_gauge_on_backend`, recovered from `stash@{0}`): builds the Phase-29 nuclear G2E_* base, then applies the gauge `x1i`-with-origin fold (`g2 = x1i(g0,origin)`, `g3 = x1i(g1,origin)`, `g1 = ∇_j(g0)`) INSIDE the Rys root loop, folding 9 cart products `s[0..8]` into the 12-component (3 tensor × gc 4-block) gout transcribed verbatim from `intor3.c:1230` (cg) / `:1547` (giao). `origin = dri` (cg) vs `[0,0,0]` (giao); cg/giao share the gout body byte-for-byte. Fail-closed `nroots > MAX_DEVICE_NROOTS → UnsupportedApi` (never clamps — CR-01).
- **nucsp launcher** (`launch_int1e_sa10nucsp_spinor_pair`): rank 3, `cart_to_spinor_si_2di` (imaginary ket), `common_fac_sp·0.5`, OWN fail-closed full-block `BufferTooSmall` guard (`ni_sp*nj_sp*2*3`) before any write.
- **Dispatch wiring** (`sigma_1e.rs`): `cg_sa10nucsp`/`giao_sa10nucsp` added to `giao_family_id` (5/6), `giao_family_rank` (3), `giao_family_transform` (SiI default); 2 dispatch arms thread `dri` (cg) / `[0,0,0]` (giao) + `origin_coords`/`origin_charges`. The dispatcher signature gained the two nuclear-center params (its sole caller is the parity collector).
- **Byte-identity gates** (`giao_sigma_1e_parity.rs`): `giao_sigma_1e_cg_sa10nucsp`, `giao_sigma_1e_giao_sa10nucsp`, and `giao_sigma_1e_sa10nucsp_cg_giao_collapse` — all at atol=1e-12 on a NON-SQUARE p(LT,nctr=2)×d(GT) block (`assert_ne!(ni_sp,nj_sp)`), all-3-components-non-zero. `test_no_silent_skip` extended: nucsp RUN + non-zero + byte-identical + oracle_covered=true (rank 3); sa01 stay covered=false/rank 9; spgsa01/spgnucsp false.
- **oracle_covered flip** for exactly `int1e_cg_sa10nucsp_spinor` / `int1e_giao_sa10nucsp_spinor` (spinor-only, rank 3 already present). manifest-audit exit 0; no capi/legacy surface.

## Task Commits
1. **Task 1: Rys+gauge nucsp engine + cg/giao_sa10nucsp dispatch arms** — `f7c879c` (feat)
2. **Task 2: nucsp byte-identity gate + flip oracle_covered + fix lock parser** — `93e7109` (test)

## Files Modified
- `crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs` — recovered `nuc_x1i`/`nuc_x1i_of_j` gauge helpers, `sigma_nuc_gauge_kernel` (#[cube(launch)]), `run_sigma_nuc_gauge_device`/`run_sigma_nuc_gauge_on_backend` (fail-closed nroots) from `stash@{0}`.
- `crates/cintx-cubecl/src/kernels/sigma_p.rs` — `launch_int1e_sa10nucsp_spinor_pair` (rank 3, c2s_si_1ei, own staging guard).
- `crates/cintx-cubecl/src/kernels/sigma_1e.rs` — nucsp added to the GIAO dispatch table + 2 dispatch arms; dispatcher signature extended with `origin_coords`/`origin_charges`.
- `crates/cintx-oracle/tests/giao_sigma_1e_parity.rs` — nucsp collectors (`nucsp_nuclear_origins`, `collect_cintx/vendor_sa10nucsp`), 3 nucsp gates, extended `test_no_silent_skip`, **fixed `manifest_lock_entry` parser**; sa01 caller updated for the new origin params.
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — flipped `oracle_covered` false→true for the 2 nucsp rows (2-field diff).
- `crates/cintx-ops/src/generated/api_manifest.rs` / `.csv` — regenerated by the cintx-ops build from the lock.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `manifest_lock_entry` scanned the wrong window → component_rank/oracle_covered always missed**
- **Found during:** Task 2 (`test_no_silent_skip` failed RED: `int1e_cg_sa10nucsp_spinor must carry component_rank 3 — left: None`).
- **Issue:** The helper did `block_start = lock[..sym_pos].rfind('{')`, which lands on the nested `id` sub-object's `{` (the object holding `"symbol"`). But `"component_rank"` is a sibling field serialized BEFORE `id`, and `"oracle_covered"` is serialized AFTER `id` — both fall outside that window. The helper therefore returned `rank=None` (and would have mis-read `covered`) for EVERY entry. This is the same parser the 30-01c code shipped, and is the **root cause of the pre-existing 30-01c `test_no_silent_skip` RED** called out in the resume brief.
- **Fix:** Re-anchor the entry window on `"arity":` — the FIRST field serialized in every entry — spanning from this entry's `"arity":` to the next entry's `"arity":` (or file end). Verified each window holds exactly one `oracle_covered` and the entry's `component_rank`.
- **Files modified:** crates/cintx-oracle/tests/giao_sigma_1e_parity.rs
- **No sa01 assertion was weakened:** sa01 rows are still asserted `rank == Some(9)` and `covered == false`; spgsa01/spgnucsp still `false`. The fix only makes the (previously broken) reads correct, so the honest 30-01c state is now actually enforced rather than crashing the test.
- **Committed in:** `93e7109` (Task 2).

---

**Total deviations:** 1 auto-fixed (Rule 1 bug in the test's lock parser, caught by the nucsp gate). No architecture change; no plan deviation in the kernel/dispatch (the stash kernel matched the plan's intor3.c transcription).

The plan's must_have wording ("add them to `family_id`/`family_transform`/`build_sigma_cart`") was satisfied via the 30-01a sibling GIAO dispatch table (`giao_family_id`/`giao_family_transform` + `launch_int1e_giao_sigma_family_spinor_pair`), consistent with 30-01a's documented resolution. Not a behavioral deviation.

The resume brief flagged a `vendor_ffi.rs` "wrong-symbol bug" (`vendor_int1e_giao_sa10nucsp_spinor` calling the cg symbol). **Ground-truth: the symbol was already correct** in the tree (`ffi::int1e_giao_sa10nucsp_spinor`, vendor_ffi.rs:4589) — no fix needed. Verified by grep before and after. The brief's report was stale.

## 30-01c / Phase status (HONEST)
- **30-01c (rank-9 sa01) remains BLOCKED**, NOT in this plan's scope. Its 3 byte-identity gates stay `#[ignore]`d WIP (gout→gc-block layout, RESEARCH Open Q1); `int1e_cg_sa10sa01_spinor`/`int1e_giao_sa10sa01_spinor` stay `oracle_covered=false` (rank 9). `test_no_silent_skip` enforces this honest state — and now PASSES (the parser fix made the previously-crashing assertions evaluate correctly). The sa01 families still RUN under the double gate (all-9-non-zero) but are not asserted byte-identical.
- **30-01d (spgnucsp + spgsa01) and Wave 2 (6 × 2e) are NOT started.** The phase is NOT complete and GIAO-03 is NOT closed. ROADMAP Phase 30 stays incomplete; STATE stays `executing`.

## Known Stubs
None for Sub-wave 1b. The remaining 4 GIAO×σ 1e families (`spgnucsp`/`spgsa01`/`cg_sa10sa01`/`giao_sa10sa01`) stay `oracle_covered=false` — deliberately deferred to Sub-waves 1c/1d (each with its own engine class + vendor gate), enforced by `test_no_silent_skip`.

## Verification Performed
- `cargo build -p cintx-cubecl --features cpu` / `cargo build -p cintx-oracle --features cpu` — exit 0.
- `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity` — 6 passed / 0 failed / 3 ignored (sa01 WIP). `giao_sigma_1e_cg_sa10nucsp`, `giao_sigma_1e_giao_sa10nucsp`, `giao_sigma_1e_sa10nucsp_cg_giao_collapse`, `giao_sigma_micro`, `test_no_silent_skip`, `test_kappa_sizing_non_4l_plus_2` all green at atol=1e-12 on a non-square block.
- WITHOUT the env var: only `test_kappa_sizing_non_4l_plus_2` runs (1 passed); the vendor gates AND `test_no_silent_skip` are compiled out (no silent pass — double-gate confirmed, T-30-01b-04).
- `cargo run -p xtask --manifest-path xtask/Cargo.toml -- manifest-audit` — exit 0.
- Python check: `int1e_cg_sa10nucsp_spinor`/`int1e_giao_sa10nucsp_spinor` oracle_covered=true; `int1e_cg_sa10sa01_spinor`/`int1e_spgnucsp_spinor` false — `ok`.
- `git diff --stat crates/cintx-capi/` empty; no new `cint1e_*` legacy wrappers. OperatorId resolved by symbol (`int4c1e_cart`) — no positional shift.
- Acceptance greps: `nucsp` in sigma_p.rs (5), `cg_sa10nucsp|giao_sa10nucsp` in sigma_1e.rs (5), `BufferTooSmall` per nucsp arm with no `if dst < staging.len()` partial guard (0), no nroots clamp (0), `UnsupportedApi` nroots guard in sigma_1e_nuc.rs (2).

## Next Plan Readiness
- Sub-wave 1b gated green. 30-01c (rank-9 sa01) still needs its gout→gc-block layout reverse-engineered (RESEARCH Open Q1) before its gates un-ignore and flip; 30-01d (spgnucsp + spgsa01) closes the full 9-family 1e Wave-1 gate. The unified GIAO dispatcher + per-family gate macro + `test_no_silent_skip` (now with a correct lock parser) are the extension points.
- Re-verify command: `CINTX_ORACLE_BUILD_VENDOR=1 cargo test -p cintx-oracle --features cpu --test giao_sigma_1e_parity`.

## Self-Check: PASSED
- FOUND: crates/cintx-cubecl/src/kernels/sigma_1e_nuc.rs::run_sigma_nuc_gauge_on_backend
- FOUND: crates/cintx-cubecl/src/kernels/sigma_p.rs::launch_int1e_sa10nucsp_spinor_pair
- FOUND: crates/cintx-cubecl/src/kernels/sigma_1e.rs cg_sa10nucsp/giao_sa10nucsp dispatch arms
- FOUND: crates/cintx-oracle/tests/giao_sigma_1e_parity.rs::giao_sigma_1e_cg_sa10nucsp
- FOUND commit: f7c879c (Task 1)
- FOUND commit: 93e7109 (Task 2)

---
*Phase: 30-group-5-giao-slice-spin-giao-integrals-spinor*
*Completed: 2026-06-01*
