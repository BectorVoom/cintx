---
phase: 28-spin-included-c2s-si-transform-p-module-gap-b2
plan: 03
subsystem: testing
tags: [manifest, vendor-ffi, libcint, spinor, oracle, bindgen, sigma-p, int1e_sp_spinor]

# Dependency graph
requires:
  - phase: 27-spinor-derivative-transform-gap-b1
    provides: skipped-fixture SC#4 guard pattern (oracle_covered_update.rs) + no-silent-skip vendor parity
  - phase: 28-spin-included-c2s-si-transform-p-module-gap-b2 (28-01/28-02)
    provides: si_2d transform + σ·p #[cube] assembler (the runtime pieces int1e_sp_spinor proves)
provides:
  - int1e_sp_spinor manifest row (oracle_covered=false, component_rank=1, appended last → OperatorId 347, no positional shift)
  - vendor_int1e_sp_spinor FFI shim + bindgen extern (byte-identity reference for Plan 04 FND-05)
  - SC#4 enforcement: int1e_sp_spinor recorded as a skipped spinor fixture so oracle-covered-update refuses to flip it
affects: [28-04, 29-group-4-relativistic-sigma]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Infrastructure-only σ family: manifest row + vendor FFI registered, but oracle_covered=false and skipped in the generic parity matrix (proven by a dedicated transform test, flip deferred to Phase 29)"
    - "Append-last manifest registration to preserve all positional OperatorId consts"

key-files:
  created:
    - .planning/phases/28-spin-included-c2s-si-transform-p-module-gap-b2/deferred-items.md
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-oracle/src/vendor_ffi.rs
    - crates/cintx-oracle/build.rs
    - crates/cintx-oracle/src/compare.rs
    - xtask/src/oracle_covered_update.rs
    - xtask/Cargo.toml

key-decisions:
  - "Append int1e_sp_spinor LAST in the lock entries array (OperatorId 347) so no existing positional OperatorId test const shifts (int4c1e_cart=24, int1e_ovlp_spinor=2 verified green)"
  - "component_rank=1 (ng[7]=1, tensor rank 1) — the 4 gc_x/gc_y/gc_z/gc_1 blocks are ncomp_e1, NOT tensor components"
  - "int1e_sp_spinor recorded as a skipped spinor fixture via is_skipped_spinor_fixture (no RawApiId; FND-05 proven by dedicated si_transform_parity test) — this is the SC#4 mechanism (D-01)"
  - "cint_funcs.h already declares int1e_sp_spinor; only the bindgen allowlist needed extending (no supplemental-header edit)"
  - "Added xtask cpu feature passthrough so the vendor parity path can drive real CPU kernels under CINTX_ORACLE_BUILD_VENDOR=1"

patterns-established:
  - "Infrastructure-only σ-family registration: lock-as-source-of-truth row (oracle_covered=false) + vendor FFI + skipped-fixture guard, no capi/legacy surface, no RawApiId, no kernel"
  - "is_skipped_spinor_fixture predicate centralizes the matrix-skip decision for both rank-3 spinor gradients (R5/D-03) and Phase-28 σ infrastructure-only families (D-01)"

requirements-completed: [FND-05]

# Metrics
duration: 38min
completed: 2026-05-31
---

# Phase 28 Plan 03: int1e_sp_spinor Manifest + Vendor FFI + SC#4 Guard Summary

**Registered the `int1e_sp_spinor` σ family as infrastructure-only — manifest row (oracle_covered=false), `vendor_int1e_sp_spinor` byte-identity FFI shim, and an SC#4 skipped-fixture guard that provably refuses to flip its coverage — all while honoring D-01 (no σ flips this phase) and adding zero capi/legacy surface.**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-05-31T11:40Z (approx)
- **Completed:** 2026-05-31
- **Tasks:** 2
- **Files modified:** 8 (1 created)

## Accomplishments
- `int1e_sp_spinor` manifest row in `compiled_manifest.lock.json` (the source of truth; build.rs auto-regenerates `api_manifest.rs` + `.csv`): `oracle_covered=false`, `component_rank=1`, arity 2, complex spinor, stable. Appended last → OperatorId 347, so NO positional OperatorId const shifted (Pitfall 6 avoided structurally).
- `vendor_int1e_sp_spinor` FFI shim in `vendor_ffi.rs` (mirrors `vendor_int1e_ovlp_spinor`, calls `ffi::int1e_sp_spinor`, sized `ni_sp*nj_sp*2` f64); `int1e_sp_spinor` added to the bindgen allowlist (`cint_funcs.h:365` already declares the extern). Compiles under the `CINTX_ORACLE_BUILD_VENDOR=1 --features cpu` gate.
- SC#4 (D-01) enforced: `int1e_sp_spinor` enters the parity matrix as a stable 1e operator but has no `RawApiId`, so `is_skipped_spinor_fixture` records it `skipped` → it never enters `covered_symbols` → the existing `if fixture.skipped { continue; }` guard in `oracle-covered-update` refuses to flip it. Verified the lock stays `oracle_covered=false` after a run, plus a dedicated unit test `sc4_int1e_sp_spinor_is_skipped_not_covered`.
- `manifest-audit` green; `cargo build --workspace --locked` green; `cintx-ops`/`cintx-rs`/`cintx-runtime` lib tests (incl. positional OperatorId invariants) green.

## Task Commits

1. **Task 1: Add int1e_sp_spinor manifest row (oracle_covered=false) + re-grep OperatorId consts** — `f96870c` (feat)
2. **Task 2: vendor_int1e_sp_spinor FFI shim + assert/extend SC#4 skipped-fixture guard** — `dbe38c9` (feat)

## Files Created/Modified
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — appended int1e_sp_spinor row (source of truth)
- `crates/cintx-ops/src/generated/api_manifest.rs` / `.csv` — build.rs-regenerated mirror (OperatorId 347)
- `crates/cintx-oracle/src/vendor_ffi.rs` — `vendor_int1e_sp_spinor` shim
- `crates/cintx-oracle/build.rs` — added `int1e_sp_spinor` to the bindgen allowlist_function regex
- `crates/cintx-oracle/src/compare.rs` — `is_skipped_spinor_fixture` predicate + SC#4 unit test
- `xtask/src/oracle_covered_update.rs` — extended guard comment (Phase-28 σ families deferred to Phase 29)
- `xtask/Cargo.toml` — `cpu` feature passthrough (cintx-oracle/cpu + cintx-compat/cpu)
- `.planning/phases/28-.../deferred-items.md` — logged pre-existing GIAO parity noise (out of scope)

## Decisions Made
- **Append-last registration** to preserve all positional OperatorId consts — the cleanest defense against Pitfall 6 (no symbol-by-position re-pointing occurs at all).
- **Skip int1e_sp_spinor in the generic parity matrix** rather than wiring a RawApiId — D-01 mandates it stays UnsupportedApi; the FND-05 proof is the dedicated transform test (si_transform_parity.rs, Plan 04), not a flag flip.
- **Bindgen allowlist only** (no supplemental-header edit) because `cint_funcs.h` already declares `int1e_sp_spinor`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] int1e_sp_spinor (component_rank=1) entered the parity matrix and would bail oracle-covered-update**
- **Found during:** Task 2 (SC#4 guard verification)
- **Issue:** The plan's interfaces note assumed the existing `if fixture.skipped { continue; }` guard already covered σ families. In fact `int1e_sp_spinor` is tensor-rank-1 (component_count=1), so it does NOT hit the existing `component_count == 3 && spinor` matrix-skip branch — it fell through to `raw_api_for_fixture`, hit `missing_raw_api_mapping` (no RawApiId), pushed a mismatch, and would make `generate_profile_parity_report` bail. The skipped-fixture guard alone was insufficient.
- **Fix:** Added `is_skipped_spinor_fixture(fixture)` in `compare.rs` (refactored the old rank-3 condition into it and added the `int1e_sp_spinor` symbol arm) so the σ family is recorded `skipped` BEFORE the raw-api lookup. This is what actually makes the SC#4 guard apply to it.
- **Files modified:** `crates/cintx-oracle/src/compare.rs`
- **Verification:** `sc4_int1e_sp_spinor_is_skipped_not_covered` unit test passes; `int1e_sp_spinor` absent from the parity mismatch list; lock stays `oracle_covered=false`.
- **Committed in:** `dbe38c9`

**2. [Rule 3 - Blocking] xtask had no way to forward the `cpu` feature for the vendor parity path**
- **Found during:** Task 2 (running oracle-covered-update under the vendor gate)
- **Issue:** `cargo run -p xtask --features cpu` failed — xtask exposed no `cpu` feature, so the vendor parity path could not drive the real CPU kernels (reference_oracle_vendor_parity_invocation double-gate).
- **Fix:** Added a `cpu` feature to `xtask/Cargo.toml` forwarding to `cintx-oracle/cpu` + `cintx-compat/cpu`.
- **Files modified:** `xtask/Cargo.toml`
- **Verification:** `CINTX_ORACLE_BUILD_VENDOR=1 cargo run --features cpu -- oracle-covered-update` now builds the real path and runs parity (it bails only on the pre-existing GIAO noise — see Issues).
- **Committed in:** `dbe38c9`

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking).
**Impact on plan:** Both were necessary to actually satisfy the SC#4 acceptance criterion. The first corrects a wrong assumption in the plan's interfaces note (the existing guard did NOT already cover rank-1 σ families); the second is plumbing for the documented vendor-gate invocation. No scope creep, no capi/legacy surface.

## Issues Encountered
- **Pre-existing 158-mismatch parity baseline noise (out of scope).** The full `cintx-oracle` parity tests and `oracle-covered-update` fail on the current working tree with 158 mismatches (154 × `missing_raw_api_mapping` for GIAO/origj/giao families like `int1e_a01gp_*`, `int1e_giao_*`, `int1e_cg_*`; 4 × `legacy_eval`). Confirmed pre-existing: these manifest rows are present at `HEAD~1` and had no `raw_api_for_symbol` arm before this plan. `int1e_sp_spinor` is NOT among them (correctly skipped). Per the SCOPE BOUNDARY rule these were logged to `deferred-items.md` and not fixed — they belong to the in-flight Phase-26 GIAO / Phase-28 spin family wiring. SC#4 was instead proven via the dedicated unit test + the lock-stays-false check after a run.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 04 has the `int1e_sp_spinor` manifest row + `vendor_int1e_sp_spinor` reference it needs to run the FND-05 transform-level byte-identity proof (si_2d transform + σ·p assembler through the int1e_sp path).
- The SC#4 guard provably keeps every σ family (incl. int1e_sp_spinor) `oracle_covered=false`; the full σ-group flips remain queued for Phase 29.
- Note for Phase 29 / Plan 04: the pre-existing GIAO `missing_raw_api_mapping` parity noise will keep `oracle-covered-update` from completing cleanly until those families are wired — see deferred-items.md.

## Self-Check: PASSED

- Created/modified files verified present: 28-03-SUMMARY.md, deferred-items.md, vendor_ffi.rs (with `vendor_int1e_sp_spinor`).
- Commits verified in git log: f96870c (Task 1), dbe38c9 (Task 2).
- int1e_sp_spinor row confirmed `oracle_covered=false` (D-01 honored).

---
*Phase: 28-spin-included-c2s-si-transform-p-module-gap-b2*
*Completed: 2026-05-31*
