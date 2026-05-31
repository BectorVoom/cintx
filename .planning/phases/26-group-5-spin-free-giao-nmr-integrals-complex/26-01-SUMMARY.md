---
phase: 26-group-5-spin-free-giao-nmr-integrals-complex
plan: 01
subsystem: api
tags: [manifest, complex-output, giao, planner, oracle, cubecl, num-complex]

# Dependency graph
requires:
  - phase: 22-gauge-origin-env-slot
    provides: "PTR_COMMON_ORIG gauge-origin slot + build_h2o_sto3g_common_orig non-zero fixture"
  - phase: 12-real-spinor-transform
    provides: "complex-interleaved spinor staging + IntegralTensor.complex_interleaved / complex_values() safe-API view"
provides:
  - "Per-family `complex_output: bool` manifest field end-to-end (lock.json → build.rs codegen → ManifestEntry → generated api_manifest.{rs,csv}); 56 spinor operator rows backfilled true"
  - "Manifest-data-driven complex staging: planner build_output_layout sizes 2×ncomp from descriptor.entry.complex_output, NOT Representation::Spinor"
  - "Always-on fail-closed assert_flat_buffer_contract gated on complex_interleaved for ANY representation (cart/sph GIAO families honored, not just spinor)"
  - "Comptime complex_output kernel hint threaded through the 1e moment kernel + launchers (inert on-device today; GIAO device path can specialize)"
  - "FND-03 safe-API Complex<f64> round-trip proof (giao_complex_roundtrip.rs)"
affects: [26-02-giao-cluster-a, 26-03-giao-cluster-b, 30-giao-sigma, giao, complex-output]

# Tech tracking
tech-stack:
  added: [num-complex (dev-dep on cintx-oracle)]
  patterns:
    - "Manifest-flag-driven routing: re-key behavior off a per-family lock.json bool instead of the Representation string"
    - "Comptime device hint plumbed from plan.descriptor.entry.<flag> (inert today, specializable later)"

key-files:
  created:
    - crates/cintx-oracle/tests/giao_complex_roundtrip.rs
  modified:
    - crates/cintx-ops/generated/compiled_manifest.lock.json
    - crates/cintx-ops/build.rs
    - crates/cintx-ops/src/resolver.rs
    - crates/cintx-ops/src/generated/api_manifest.rs
    - crates/cintx-ops/src/generated/api_manifest.csv
    - crates/cintx-runtime/src/planner.rs
    - crates/cintx-oracle/src/compare.rs
    - crates/cintx-cubecl/src/kernels/one_electron.rs
    - crates/cintx-oracle/Cargo.toml

key-decisions:
  - "complex_output is a per-entry lock.json bool that defaults false in build.rs (mirrors component_rank); only spinor operator rows are backfilled true"
  - "build_output_layout takes the OperatorDescriptor (not just Representation) so it reads descriptor.entry.complex_output — same data-driven access as component_multiplier_for_descriptor"
  - "assert_flat_buffer_contract gates on complex_interleaved (always-on, no debug gate) so a complex cart/sph family staged real-only fails in release"
  - "RED test predicate for spinor-row backfill filters HelperKind::Operator + _spinor suffix — length/offset helpers (CINTlen_spinor, CINTcgto_spinor, …) support spinor representation but are NOT complex integral output"
  - "FND-03 round-trip proven on the first registered complex_output family (int1e_ovlp_spinor); the int1e_igovlp D-07 imag/real upgrade is Plan 26-02's seam"

patterns-established:
  - "New schema fields land at four linked codegen sites: LockEntry (Option<T>) → GeneratedEntry (T) → From impl default → RS literal writer + CSV header/row"
  - "Complex-output families size 2× from manifest data; spinor stays 2× via backfill, GIAO cart/sph gain 2× without touching the rep string"

requirements-completed: [FND-03]

# Metrics
duration: 12min
completed: 2026-05-31
---

# Phase 26 Plan 01: FND-03 Complex Output Foundation Summary

**Re-keyed the complex/imaginary output path from `Representation::Spinor` to a per-family manifest `complex_output: bool` flag — threaded end-to-end through lock.json/codegen, the planner's 2× staging SET, an always-on fail-closed flat-buffer contract, a comptime 1e kernel hint, and a safe-API `Complex<f64>` round-trip proof.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-31T02:02Z
- **Completed:** 2026-05-31T02:14Z
- **Tasks:** 3
- **Files modified:** 9 (1 created, 8 modified)

## Accomplishments
- `complex_output: bool` manifest field exists end-to-end (lock.json schema → build.rs codegen → `ManifestEntry` struct → generated `api_manifest.{rs,csv}`); 56 spinor operator rows backfilled `true`, all non-spinor rows default `false`.
- Planner `build_output_layout` now sizes complex staging from `descriptor.entry.complex_output` (D-01) — spinor families preserved via the Task-1 backfill, and GIAO cart/sph families will size 2× from manifest data (no rep-string keying).
- `assert_flat_buffer_contract` is always-on fail-closed gated on `complex_interleaved` for ANY representation (D-04, threat T-26-01) — a complex cart/sph family staged real-only is rejected, not silently accepted.
- The manifest `complex_output` flag is threaded as a comptime hint through the 1e moment `#[cube]` kernel and its `run_1e_moment_device`/`run_1e_moment_on_backend` launchers (D-02), derived from `plan.descriptor.entry.complex_output`.
- `giao_complex_roundtrip.rs` proves the FND-03 manifest-flag → `complex_interleaved` → safe-API `Complex<f64>` view (`complex_values()` returns `Some`) on `int1e_ovlp_spinor`.

## Task Commits

1. **Task 1: complex_output manifest field end-to-end** — `2d6f932` (test, RED) → `4fae610` (feat, GREEN)
2. **Task 2: re-key planner SET + generalize fail-closed contract** — `2470260` (test, RED) → `8134507` (feat, GREEN)
3. **Task 3: comptime kernel hint + FND-03 round-trip proof** — `b2088cc` (feat)

_TDD: Tasks 1 and 2 followed RED→GREEN with separate test/feat commits. Task 3's round-trip test and kernel-hint plumbing landed in one commit (the test first failed on a zero-fill assertion during development, then passed once the assembled-matrix non-zero scan was used)._

## Files Created/Modified
- `crates/cintx-ops/generated/compiled_manifest.lock.json` — backfilled `"complex_output": true` on 56 spinor operator rows (56 pure insertions, no reformatting).
- `crates/cintx-ops/build.rs` — threaded `complex_output` through `LockEntry` (`Option<bool>`), `GeneratedEntry` (`bool`), `From<&LockEntry>` default (`unwrap_or(false)`), the RS literal writer, and the CSV header + row.
- `crates/cintx-ops/src/resolver.rs` — added `pub complex_output: bool` to `ManifestEntry`; added two RED→GREEN tests (spinor-operator backfill + real cart default).
- `crates/cintx-ops/src/generated/api_manifest.{rs,csv}` — regenerated with the `complex_output` column.
- `crates/cintx-runtime/src/planner.rs` — `build_output_layout(descriptor, shells, component_count)` reads `descriptor.entry.complex_output`; 3 manifest-sizing unit tests.
- `crates/cintx-oracle/src/compare.rs` — `assert_flat_buffer_contract` gates on `complex_interleaved`; 2 fail-closed unit tests.
- `crates/cintx-cubecl/src/kernels/one_electron.rs` — comptime `complex_output` hint on the moment kernel + both launchers, derived from `plan.descriptor.entry.complex_output`.
- `crates/cintx-oracle/tests/giao_complex_roundtrip.rs` — new FND-03 safe-API round-trip proof.
- `crates/cintx-oracle/Cargo.toml` — `num-complex` dev-dep (for the `Complex<f64>` view type).

## Decisions Made
- `complex_output` defaults `false` in build.rs and is only set `true` on backfilled spinor operator rows — keeps the schema change additive and non-spinor rows untouched.
- `build_output_layout` was changed to accept the `OperatorDescriptor` (dropping the unused `Representation` arg from its own logic) so it reads the manifest the same way `component_multiplier_for_descriptor` does.
- The contract gate is always-on (no `debug_assert`) so a complex family staged real-only fails in release builds too.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RED test predicate for spinor backfill was too broad**
- **Found during:** Task 1 (GREEN verification)
- **Issue:** The initial `spinor_rows_have_complex_output_true` test filtered on `entry.representation.spinor` (the representation-SUPPORT flag), which also matched 5 length/offset HELPER rows (`CINTlen_spinor`, `CINTcgtos_spinor`, `CINTcgto_spinor`, `CINTtot_pgto_spinor`, `CINTtot_cgto_spinor`). Those helpers support spinor representation but are NOT complex integral output and are correctly NOT backfilled, so the test failed.
- **Fix:** Re-scoped the predicate to `HelperKind::Operator && symbol_name.ends_with("_spinor")` — the set the plan's "every `"representation": "spinor"` block" intent actually targets.
- **Files modified:** `crates/cintx-ops/src/resolver.rs`
- **Verification:** Both Task-1 tests green; manifest-audit-relevant cintx-ops lib suite green (13 passed).
- **Committed in:** `4fae610` (Task 1 GREEN commit)

**2. [Rule 2 - Missing Critical] Added num-complex dev-dependency to cintx-oracle**
- **Found during:** Task 3 (round-trip test compilation)
- **Issue:** The test names the `num_complex::Complex<f64>` view type returned by `complex_values()`, but `num-complex` was not a (dev-)dependency of cintx-oracle and is not re-exported by cintx-rs.
- **Fix:** Added `num-complex = "0.4"` to `[dev-dependencies]` (same line as cintx-rs; single resolved graph, Cargo.lock unchanged version).
- **Files modified:** `crates/cintx-oracle/Cargo.toml`, `Cargo.lock`
- **Verification:** Test compiles and passes under `--features cpu`.
- **Committed in:** `b2088cc` (Task 3 commit)

**3. [Rule 2 - Missing Critical] Widened test cfg gate to include rocm**
- **Found during:** Task 3 (test authoring)
- **Issue:** The plan specified `#![cfg(feature = "cpu")]`, but the established safe-API test pattern (Phase 16-04) gates on `any(feature = "cpu", feature = "rocm")` so the suite also runs under `--features rocm`.
- **Fix:** Used `#![cfg(any(feature = "cpu", feature = "rocm"))]` to match `safe_api_arity2_parity.rs`. The cpu-gated intent (runs under `--features cpu`) is fully satisfied.
- **Files modified:** `crates/cintx-oracle/tests/giao_complex_roundtrip.rs`
- **Verification:** `cargo test -p cintx-oracle --features cpu --test giao_complex_roundtrip` passes.
- **Committed in:** `b2088cc` (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 missing-critical)
**Impact on plan:** All necessary for correctness and to match established codebase conventions. No scope creep.

## Known Stubs

- `int1e_igovlp` is NOT registered in this plan — the GIAO kernel + host re=0 materialization land in Plan 26-02 (Cluster A). The `giao_complex_roundtrip.rs` test therefore proves the flag→view path on the first registered complex_output family (`int1e_ovlp_spinor`) and is UPGRADED to the full D-07 imag/real assertion on `int1e_igovlp` by Plan 26-02. This seam is documented in the test header — it is an intentional plan boundary, not an unresolved stub.
- The comptime `complex_output` kernel hint is inert on-device today (RESEARCH Open-Q1: the device emits real components; host materializes re=0/im=value before `complex_values()`). The GIAO device path in Cluster A consumes the hint.

## Issues Encountered
- The Task-3 round-trip non-zero check initially asserted on a single off-diagonal spinor block (O-2p × O-1s), which is physically zero. Resolved by scanning the full assembled matrix of pairs (which includes the non-zero self-overlap diagonal) for at least one non-zero component, while still confirming the non-square block's `complex_values()` returns `Some` for the core flag→view assertion.

## Next Phase Readiness
- FND-03 foundation is complete and MUST merge to main before Plan 26-02 / 26-03 start (D-09). After merge, verify `git merge-base --is-ancestor <this-wave-branch> main`; merge manually if not an ancestor (worktree auto-merge is inconsistent — project memory).
- Plan 26-02 (Cluster A): register `int1e_igovlp` + GIAO kernels, set their manifest `complex_output=true`, consume the comptime hint, add host re=0 materialization, and upgrade `giao_complex_roundtrip.rs` to the full D-07 imag/real assertion.

## Self-Check: PENDING

---
*Phase: 26-group-5-spin-free-giao-nmr-integrals-complex*
*Completed: 2026-05-31*
